"""Helpers to persist experiment records plus governance/serving manifests."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from insper_deploy_kedro import ops_store
from insper_deploy_kedro.constants import ModelArtifact
from insper_deploy_kedro.logging_utils import configure_project_logging

configure_project_logging()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "data" / "06_models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "07_model_output"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _artifact_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_name in (
        "production_imputers.pkl",
        "production_outlier_cappers.pkl",
        "production_encoders.pkl",
        "production_scalers.pkl",
        "raw_production_model.pkl",
        "production_model.pkl",
    ):
        path = MODELS_DIR / dataset_name
        if path.exists():
            stats = path.stat()
            rows.append(
                {
                    "path": str(path),
                    "exists": True,
                    "size_bytes": int(stats.st_size),
                    "modified_at": datetime.fromtimestamp(
                        stats.st_mtime, tz=UTC
                    ).isoformat(),
                }
            )
        else:
            rows.append({"path": str(path), "exists": False})
    return rows


def _output_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_name in (
        "model_frontier.parquet",
        "model_selection_scorecard.parquet",
        "cv_fold_metrics.parquet",
        "cv_metric_summary.parquet",
        "bootstrap_metric_intervals.parquet",
        "permutation_feature_importance.parquet",
        "threshold_metrics.parquet",
        "split_comparison_report.parquet",
        "nested_cv_audit.parquet",
        "modelling_design_audit.parquet",
        "perturbation_sensitivity_audit.parquet",
        "perturbation_sensitivity_summary.parquet",
    ):
        path = OUTPUT_DIR / dataset_name
        if path.exists():
            stats = path.stat()
            rows.append(
                {
                    "path": str(path),
                    "exists": True,
                    "size_bytes": int(stats.st_size),
                    "modified_at": datetime.fromtimestamp(
                        stats.st_mtime, tz=UTC
                    ).isoformat(),
                }
            )
        else:
            rows.append({"path": str(path), "exists": False})
    return rows


def record_experiment_run(  # noqa: PLR0913
    model_frontier: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    cv_metric_summary: pd.DataFrame,
    selected_deployment_policy: dict[str, Any],
    best_model_config: dict[str, Any],
    model_selection: dict[str, Any],
) -> dict[str, Any]:
    """Persist a structured experiment summary in the local registry."""
    frontier_rows = (
        model_frontier.to_dict(orient="records") if not model_frontier.empty else []
    )
    threshold_rows = (
        threshold_metrics.to_dict(orient="records")
        if not threshold_metrics.empty
        else []
    )
    cv_summary_rows = (
        cv_metric_summary.to_dict(orient="records")
        if not cv_metric_summary.empty
        else []
    )

    selected_model_name = selected_deployment_policy.get("model_name")
    selected_row = next(
        (row for row in frontier_rows if row.get("model_name") == selected_model_name),
        {},
    )
    selection_metric = best_model_config.get(
        "selection_metric",
        model_selection.get("metric", "roc_auc"),
    )

    record = {
        "experiment_id": str(uuid.uuid4()),
        "created_at": _now(),
        "selected_model_name": selected_model_name,
        "selected_class_path": best_model_config.get("class_path"),
        "selection_metric": selection_metric,
        "selection_score": best_model_config.get(
            "selection_score",
            selected_row.get(f"validation_{selection_metric}"),
        ),
        "selected_policy_name": selected_deployment_policy.get("decision_policy_name"),
        "selected_threshold": selected_deployment_policy.get("decision_threshold"),
        "best_model_config": best_model_config,
        "frontier": frontier_rows,
        "threshold_metrics": threshold_rows,
        "cv_metric_summary": cv_summary_rows,
    }
    try:
        return ops_store.record_experiment_run(record)
    except Exception as exc:
        logger.exception("Failed to persist experiment registry entry")
        return {
            **record,
            "registry_status": "failed",
            "registry_error": f"{type(exc).__name__}: {exc}",
        }


def build_training_run_manifest(  # noqa: PLR0913
    latest_experiment_run: dict[str, Any],
    bootstrap_metric_intervals: pd.DataFrame,
    permutation_feature_importance: pd.DataFrame,
    perturbation_sensitivity_summary: pd.DataFrame,
    split_strategy_report: pd.DataFrame,
    feature_selection_frontier: pd.DataFrame,
    feature_selection_stability: pd.DataFrame,
    feature_selection_manifest: dict[str, Any] | None,
    model_selection_scorecard: pd.DataFrame | None = None,
    split_comparison_report: pd.DataFrame | None = None,
    nested_cv_audit: pd.DataFrame | None = None,
    modelling_design_audit: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a compact, dashboard-friendly training manifest."""
    manifest = {
        "manifest_type": "training_run",
        "generated_at": _now(),
        "experiment_id": latest_experiment_run.get("experiment_id"),
        "created_at": latest_experiment_run.get("created_at"),
        "selected_model_name": latest_experiment_run.get("selected_model_name"),
        "selected_class_path": latest_experiment_run.get("selected_class_path"),
        "selection_metric": latest_experiment_run.get("selection_metric"),
        "selection_score": latest_experiment_run.get("selection_score"),
        "selected_policy_name": latest_experiment_run.get("selected_policy_name"),
        "selected_threshold": latest_experiment_run.get("selected_threshold"),
        "best_model_config": latest_experiment_run.get("best_model_config", {}),
        "frontier": latest_experiment_run.get("frontier", []),
        "threshold_metrics": latest_experiment_run.get("threshold_metrics", []),
        "cv_metric_summary": latest_experiment_run.get("cv_metric_summary", []),
        "bootstrap_metric_intervals": (
            bootstrap_metric_intervals.to_dict(orient="records")
            if not bootstrap_metric_intervals.empty
            else []
        ),
        "permutation_feature_importance": (
            permutation_feature_importance.to_dict(orient="records")
            if not permutation_feature_importance.empty
            else []
        ),
        "perturbation_sensitivity_summary": (
            perturbation_sensitivity_summary.to_dict(orient="records")
            if not perturbation_sensitivity_summary.empty
            else []
        ),
        "model_selection_scorecard": (
            model_selection_scorecard.to_dict(orient="records")
            if model_selection_scorecard is not None
            and not model_selection_scorecard.empty
            else []
        ),
        "split_comparison_report": (
            split_comparison_report.to_dict(orient="records")
            if split_comparison_report is not None and not split_comparison_report.empty
            else []
        ),
        "nested_cv_audit": (
            nested_cv_audit.to_dict(orient="records")
            if nested_cv_audit is not None and not nested_cv_audit.empty
            else []
        ),
        "modelling_design_audit": (
            modelling_design_audit.to_dict(orient="records")
            if modelling_design_audit is not None and not modelling_design_audit.empty
            else []
        ),
        "split_strategy_report": (
            split_strategy_report.to_dict(orient="records")
            if not split_strategy_report.empty
            else []
        ),
        "feature_selection_frontier": (
            feature_selection_frontier.to_dict(orient="records")
            if not feature_selection_frontier.empty
            else []
        ),
        "feature_selection_stability": (
            feature_selection_stability.to_dict(orient="records")
            if not feature_selection_stability.empty
            else []
        ),
        "feature_selection_manifest": feature_selection_manifest or {},
        "artifact_inventory": _output_inventory(),
        "registry_status": latest_experiment_run.get("registry_status", "persisted"),
        "registry_error": latest_experiment_run.get("registry_error"),
    }
    return manifest


def record_model_registry_entry(production_model: ModelArtifact) -> dict[str, Any]:
    """Persist the current production model metadata in the local registry."""
    record = {
        "registry_id": str(uuid.uuid4()),
        "created_at": _now(),
        "stage": "production",
        "class_path": production_model.get("class_path"),
        "train_splits": list(production_model.get("train_splits", [])),
        "decision_policy_name": production_model.get("decision_policy_name"),
        "decision_threshold": production_model.get("decision_threshold"),
        "risk_bands": production_model.get("risk_bands", []),
        "artifact_inventory": _artifact_inventory(),
    }
    try:
        return ops_store.record_model_registry_entry(record)
    except Exception as exc:
        logger.exception("Failed to persist production model registry entry")
        return {
            **record,
            "registry_status": "failed",
            "registry_error": f"{type(exc).__name__}: {exc}",
        }


def build_serving_manifest(
    production_model: ModelArtifact,
    latest_model_registry_entry: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the current serving bundle and policy in a stable manifest."""
    return {
        "manifest_type": "serving_manifest",
        "generated_at": _now(),
        "registry_id": latest_model_registry_entry.get("registry_id"),
        "created_at": latest_model_registry_entry.get("created_at"),
        "stage": latest_model_registry_entry.get("stage", "production"),
        "class_path": production_model.get("class_path"),
        "train_splits": list(production_model.get("train_splits", [])),
        "decision_policy_name": production_model.get("decision_policy_name"),
        "decision_policy_description": production_model.get(
            "decision_policy_description",
            "",
        ),
        "decision_threshold": production_model.get("decision_threshold"),
        "risk_bands": list(production_model.get("risk_bands", [])),
        "policy_catalog": list(production_model.get("policy_catalog", [])),
        "artifact_inventory": latest_model_registry_entry.get(
            "artifact_inventory",
            _artifact_inventory(),
        ),
        "registry_status": latest_model_registry_entry.get(
            "registry_status",
            "persisted",
        ),
        "registry_error": latest_model_registry_entry.get("registry_error"),
    }


def build_inference_contract(
    production_model: ModelArtifact,
    raw_columns: dict[str, list[str]],
    columns: dict[str, list[str]],
) -> dict[str, Any]:
    """Describe the runtime inference contract for API and dashboard consumers."""
    raw_input_names = raw_columns.get("categorical", []) + raw_columns.get(
        "numerical", []
    )
    input_fields = []
    for field_name in raw_input_names:
        input_fields.append(
            {
                "name": field_name,
                "required": True,
                "kind": (
                    "categorical"
                    if field_name in raw_columns.get("categorical", [])
                    else "numerical"
                ),
            }
        )

    return {
        "manifest_type": "inference_contract",
        "generated_at": _now(),
        "input_fields": input_fields,
        "derived_feature_columns": [
            column_name
            for column_name in production_model.get("feature_columns", [])
            if column_name not in raw_input_names
        ],
        "model_feature_columns": list(production_model.get("feature_columns", [])),
        "model_class_path": production_model.get("class_path"),
        "train_splits": list(production_model.get("train_splits", [])),
        "decision_policy_name": production_model.get("decision_policy_name"),
        "decision_threshold": production_model.get("decision_threshold"),
        "risk_bands": list(production_model.get("risk_bands", [])),
        "pipeline_steps": [
            "clean_data",
            "transform_zero_imputers",
            "transform_outlier_cappers",
            "add_features",
            "transform_encoders",
            "transform_scalers",
            "predict",
        ],
        "outputs": [
            {
                "name": "prediction",
                "type": "int",
                "description": "Binary decision using the deployed policy threshold.",
            },
            {
                "name": "prediction_proba",
                "type": "float",
                "description": "Estimated probability for the positive class.",
            },
            {
                "name": "risk_score",
                "type": "float",
                "description": "Probability rescaled to a 0-100 risk score.",
            },
            {
                "name": "risk_band",
                "type": "string",
                "description": "Configured policy risk band for the score.",
            },
        ],
        "target_column": columns.get("target", [None])[0],
    }
