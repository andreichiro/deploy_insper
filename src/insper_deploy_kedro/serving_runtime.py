"""Shared Kedro runtime helpers for API and dashboard serving."""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from kedro.framework.project import configure_project
from kedro.framework.project import pipelines as kedro_pipelines
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import DatasetError, MemoryDataset
from kedro.runner import SequentialRunner

from insper_deploy_kedro import ops_store
from insper_deploy_kedro.logging_utils import configure_project_logging

configure_project_logging()
logger = logging.getLogger(__name__)

PROJECT_PATH = Path(__file__).resolve().parents[2]
PACKAGE_NAME = Path(__file__).resolve().parent.name

TRAIN_PIPELINES = ["data_engineering", "modelling", "refit"]
BATCH_INFERENCE_PIPELINES = ["inference"]
PRODUCTION_ARTIFACT_DATASETS = {
    "encoders": "production_encoders",
    "scalers": "production_scalers",
    "model": "production_model",
}
INFERENCE_MEMORY_DATASETS = (
    "raw_data_inference",
    "cleaned_inference",
    "featured_inference",
    "encoded_inference",
    "scaled_inference",
    "predictions",
    "risk_report",
)

_bootstrap_lock = threading.Lock()
_bootstrapped = False

_runs: dict[str, dict[str, Any]] = {}
_runs_lock = threading.Lock()


def ensure_bootstrap() -> None:
    """Bootstrap the Kedro project exactly once, safely across threads."""
    global _bootstrapped  # noqa: PLW0603

    if _bootstrapped:
        return

    with _bootstrap_lock:
        if not _bootstrapped:
            bootstrap_project(PROJECT_PATH)
            configure_project(PACKAGE_NAME)
            _bootstrapped = True


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _set_run(run_id: str, **fields: Any) -> None:
    with _runs_lock:
        run_state = _runs.setdefault(run_id, {})
        run_state.update(fields)
        run_state["updated_at"] = _now()
        persisted_state = run_state.copy()
    try:
        ops_store.upsert_background_run(run_id, persisted_state)
    except Exception:
        logger.exception("Failed to persist run state for %s", run_id)


def get_run_status(run_id: str) -> dict[str, Any] | None:
    """Return a shallow copy of the tracked background run state."""
    with _runs_lock:
        run_state = _runs.get(run_id)
        if run_state is not None:
            return run_state.copy()

    try:
        persisted_run = ops_store.get_background_run(run_id)
    except Exception:
        logger.exception("Failed to load persisted run state for %s", run_id)
        return None
    if persisted_run is None:
        return None

    with _runs_lock:
        _runs[run_id] = persisted_run.copy()
    return persisted_run


def load_catalog_dataset(dataset_name: str) -> Any:
    """Load a dataset through Kedro so versioned resolution stays centralized."""
    ensure_bootstrap()

    with KedroSession.create(project_path=PROJECT_PATH) as session:
        context = session.load_context()
        return context.catalog.load(dataset_name)


def load_production_artifacts() -> dict[str, Any] | None:
    """Load all artifacts required for production inference."""
    ensure_bootstrap()

    try:
        with KedroSession.create(project_path=PROJECT_PATH) as session:
            context = session.load_context()
            catalog = context.catalog
            return {
                artifact_key: catalog.load(dataset_name)
                for artifact_key, dataset_name in PRODUCTION_ARTIFACT_DATASETS.items()
            }
    except (DatasetError, FileNotFoundError, OSError):
        logger.warning(
            "Production inference artifacts are not available", exc_info=True
        )
        return None
    except Exception:
        logger.exception("Unexpected error while loading production artifacts")
        return None


def load_production_model_artifact() -> dict[str, Any] | None:
    """Load only the production model artifact when it is available."""
    artifacts = load_production_artifacts()
    model_artifact = artifacts.get("model") if artifacts is not None else None
    return model_artifact if isinstance(model_artifact, dict) else None


def get_production_status() -> dict[str, Any]:
    """Report whether production inference artifacts are available."""
    artifacts = load_production_artifacts()
    if artifacts is None:
        return {"model_loaded": False, "model_version": None}

    encoders = artifacts.get("encoders")
    scalers = artifacts.get("scalers")
    model_artifact = artifacts.get("model")
    if not isinstance(encoders, dict):
        return {"model_loaded": False, "model_version": None}
    if not isinstance(scalers, dict):
        return {"model_loaded": False, "model_version": None}
    if not isinstance(model_artifact, dict):
        return {"model_loaded": False, "model_version": None}

    model_version = model_artifact.get("class_path", "unknown")
    return {
        "model_loaded": True,
        "model_version": str(model_version) if model_version is not None else None,
    }


def run_online_inference(
    instances: list[dict[str, Any]],
    *,
    output_dataset: str = "predictions",
) -> pd.DataFrame:
    """Run the real Kedro inference pipeline fully in memory for request payloads."""
    ensure_bootstrap()

    if output_dataset not in {"predictions", "risk_report"}:
        raise ValueError(
            "output_dataset must be one of {'predictions', 'risk_report'}"
        )
    resolved_output_dataset = (
        "risk_report" if output_dataset == "predictions" else output_dataset
    )

    with KedroSession.create(project_path=PROJECT_PATH) as session:
        context = session.load_context()
        catalog = context.catalog

        raw_dataframe = pd.DataFrame(instances)
        catalog["raw_data_inference"] = MemoryDataset(data=raw_dataframe)

        # Keep request-scoped inference fully in memory instead of persisting
        # intermediate artifacts on disk for every API/dashboard invocation.
        for dataset_name in INFERENCE_MEMORY_DATASETS[1:]:
            catalog[dataset_name] = MemoryDataset()

        pipeline = kedro_pipelines["inference"]
        SequentialRunner().run(pipeline, catalog)
        predictions = catalog.load(resolved_output_dataset)

    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.DataFrame(predictions)

    if output_dataset == "predictions":
        prediction_columns = [
            column
            for column in ["prediction", "prediction_proba", "risk_score", "risk_band"]
            if column in predictions.columns
        ]
        return predictions[prediction_columns].copy()
    return predictions


def _build_run_result(pipeline: str, pipeline_names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"pipeline_names": pipeline_names}

    if pipeline == "train":
        result.update(get_production_status())
    elif pipeline == "batch-inference":
        result["output_datasets"] = ["predictions", "risk_report"]

    return result


def _run_pipelines_background(
    run_id: str,
    pipeline: str,
    pipeline_names: list[str],
) -> None:
    """Execute one or more Kedro pipelines in a background thread."""
    _set_run(run_id, status="running", started_at=_now())

    try:
        ensure_bootstrap()
        with KedroSession.create(project_path=PROJECT_PATH) as session:
            session.run(pipeline_names=pipeline_names)

        _set_run(
            run_id,
            status="completed",
            finished_at=_now(),
            result=_build_run_result(pipeline, pipeline_names),
        )
    except Exception as exc:
        logger.exception("Background pipeline run %s failed", run_id)
        _set_run(
            run_id,
            status="failed",
            finished_at=_now(),
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _start_background_run(pipeline: str, pipeline_names: list[str]) -> str:
    run_id = str(uuid.uuid4())
    _set_run(
        run_id,
        status="pending",
        pipeline=pipeline,
        started_at=None,
        finished_at=None,
        error=None,
        result=None,
    )

    thread = threading.Thread(
        target=_run_pipelines_background,
        args=(run_id, pipeline, pipeline_names),
        daemon=True,
        name=f"kedro-{pipeline}-{run_id[:8]}",
    )
    thread.start()
    return run_id


def start_training_run() -> str:
    """Start the train pipeline chain in the background."""
    return _start_background_run("train", TRAIN_PIPELINES)


def start_batch_inference_run() -> str:
    """Start catalog-based batch inference in the background."""
    return _start_background_run("batch-inference", BATCH_INFERENCE_PIPELINES)
