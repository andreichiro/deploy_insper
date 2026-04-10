"""Focused tests for production artifact readiness in serving runtime."""

from __future__ import annotations

from unittest import mock

import pandas as pd

from insper_deploy_kedro import serving_runtime


def test_load_production_artifacts_requires_all_three_datasets():
    expected = {
        "encoders": {"age": "encoder"},
        "scalers": {"glucose": "scaler"},
        "model": {"class_path": "sklearn.linear_model.LogisticRegression"},
    }

    def fake_load(dataset_name: str):
        mapping = {
            "production_encoders": expected["encoders"],
            "production_scalers": expected["scalers"],
            "production_model": expected["model"],
        }
        return mapping[dataset_name]

    class _Catalog:
        def load(self, dataset_name: str):
            return fake_load(dataset_name)

    class _Context:
        catalog = _Catalog()

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def load_context(self):
            return _Context()

    with (
        mock.patch.object(serving_runtime, "ensure_bootstrap"),
        mock.patch.object(
            serving_runtime.KedroSession, "create", return_value=_Session()
        ),
    ):
        artifacts = serving_runtime.load_production_artifacts()

    assert artifacts == expected


def test_get_production_status_returns_false_when_artifacts_are_incomplete():
    with mock.patch.object(
        serving_runtime,
        "load_production_artifacts",
        return_value=None,
    ):
        status = serving_runtime.get_production_status()

    assert status == {"model_loaded": False, "model_version": None}


def test_get_production_status_returns_false_when_model_artifact_shape_is_invalid():
    with mock.patch.object(
        serving_runtime,
        "load_production_artifacts",
        return_value={
            "encoders": {"age": "encoder"},
            "scalers": {"glucose": "scaler"},
            "model": "not-a-dict",
        },
    ):
        status = serving_runtime.get_production_status()

    assert status == {"model_loaded": False, "model_version": None}


def test_get_production_status_returns_false_when_scalers_are_invalid():
    with mock.patch.object(
        serving_runtime,
        "load_production_artifacts",
        return_value={
            "encoders": {"age": "encoder"},
            "scalers": "not-a-dict",
            "model": {"class_path": "catboost.CatBoostClassifier"},
        },
    ):
        status = serving_runtime.get_production_status()

    assert status == {"model_loaded": False, "model_version": None}


def test_get_production_status_returns_false_when_scalers_are_missing():
    with mock.patch.object(
        serving_runtime,
        "load_production_artifacts",
        return_value={
            "encoders": {"age": "encoder"},
            "model": {"class_path": "catboost.CatBoostClassifier"},
        },
    ):
        status = serving_runtime.get_production_status()

    assert status == {"model_loaded": False, "model_version": None}


def test_get_production_status_uses_model_version_only_when_all_artifacts_exist():
    with mock.patch.object(
        serving_runtime,
        "load_production_artifacts",
        return_value={
            "encoders": {"age": "encoder"},
            "scalers": {"glucose": "scaler"},
            "model": {"class_path": "catboost.CatBoostClassifier"},
        },
    ):
        status = serving_runtime.get_production_status()

    assert status == {
        "model_loaded": True,
        "model_version": "catboost.CatBoostClassifier",
    }


def test_get_run_status_falls_back_to_persisted_store():
    serving_runtime._runs.clear()

    with mock.patch.object(
        serving_runtime.ops_store,
        "get_background_run",
        return_value={
            "pipeline": "train",
            "status": "completed",
            "started_at": "2026-04-01T00:00:00+00:00",
            "finished_at": "2026-04-01T00:01:00+00:00",
            "error": None,
            "result": {"model_loaded": True},
        },
    ):
        payload = serving_runtime.get_run_status("run-123")

    assert payload is not None
    assert payload["pipeline"] == "train"
    assert serving_runtime._runs["run-123"]["status"] == "completed"


def test_set_run_logs_persistence_failures_without_raising():
    serving_runtime._runs.clear()

    with (
        mock.patch.object(
            serving_runtime.ops_store,
            "upsert_background_run",
            side_effect=OSError("db unavailable"),
        ),
        mock.patch.object(serving_runtime.logger, "exception") as mocked_exception,
    ):
        serving_runtime._set_run("run-123", pipeline="train", status="pending")

    assert serving_runtime._runs["run-123"]["status"] == "pending"
    mocked_exception.assert_called_once()


def test_build_run_result_includes_model_status_for_training():
    with mock.patch.object(
        serving_runtime,
        "get_production_status",
        return_value={"model_loaded": True, "model_version": "catboost"},
    ):
        result = serving_runtime._build_run_result(
            "train",
            ["data_engineering", "modelling", "refit"],
        )

    assert result == {
        "pipeline_names": ["data_engineering", "modelling", "refit"],
        "model_loaded": True,
        "model_version": "catboost",
    }


def test_run_pipelines_background_persists_traceback_on_failure():
    serving_runtime._runs.clear()

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, *, pipeline_names):
            raise RuntimeError(f"boom for {pipeline_names}")

    with (
        mock.patch.object(serving_runtime, "ensure_bootstrap"),
        mock.patch.object(
            serving_runtime.KedroSession, "create", return_value=_Session()
        ),
        mock.patch.object(serving_runtime.ops_store, "upsert_background_run"),
    ):
        serving_runtime._run_pipelines_background(
            "run-err",
            "train",
            ["data_engineering"],
        )

    failed = serving_runtime._runs["run-err"]
    assert failed["status"] == "failed"
    assert "RuntimeError: boom for ['data_engineering']" in failed["error"]
    assert "Traceback" in failed["error"]


def test_run_online_inference_runs_pipeline_in_memory():
    class _Catalog(dict):
        def load(self, dataset_name: str):
            return self[dataset_name]._data

    class _Context:
        def __init__(self):
            self.catalog = _Catalog()

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def load_context(self):
            return _Context()

    def fake_runner_run(_pipeline, catalog):
        catalog["risk_report"] = mock.Mock(
            _data=pd.DataFrame(
                [
                    {
                        "case_id": 1,
                        "prediction": 1,
                        "prediction_proba": 0.9,
                        "risk_score": 90.0,
                        "risk_band": "Alto risco",
                    }
                ]
            )
        )

    with (
        mock.patch.object(serving_runtime, "ensure_bootstrap"),
        mock.patch.object(
            serving_runtime.KedroSession, "create", return_value=_Session()
        ),
        mock.patch.object(
            serving_runtime, "kedro_pipelines", {"inference": object()}
        ),
        mock.patch.object(
            serving_runtime.SequentialRunner,
            "run",
            side_effect=fake_runner_run,
        ),
    ):
        predictions = serving_runtime.run_online_inference(
            [{"Pregnancies": 1, "Glucose": 90}]
        )

    assert isinstance(predictions, pd.DataFrame)
    assert float(predictions.iloc[0]["prediction_proba"]) == 0.9
    assert list(predictions.columns) == [
        "prediction",
        "prediction_proba",
        "risk_score",
        "risk_band",
    ]


def test_run_online_inference_returns_full_risk_report_when_requested():
    class _Catalog(dict):
        def load(self, dataset_name: str):
            return self[dataset_name]._data

    class _Context:
        def __init__(self):
            self.catalog = _Catalog()

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def load_context(self):
            return _Context()

    def fake_runner_run(_pipeline, catalog):
        catalog["risk_report"] = mock.Mock(
            _data=pd.DataFrame(
                [
                    {
                        "case_id": 1,
                        "prediction": 1,
                        "prediction_label": "Positivo",
                        "prediction_proba": 0.9,
                        "risk_score": 90.0,
                        "risk_band": "Alto risco",
                        "recommended_action": "Priorizar revisão clínica",
                    }
                ]
            )
        )

    with (
        mock.patch.object(serving_runtime, "ensure_bootstrap"),
        mock.patch.object(
            serving_runtime.KedroSession, "create", return_value=_Session()
        ),
        mock.patch.object(
            serving_runtime, "kedro_pipelines", {"inference": object()}
        ),
        mock.patch.object(
            serving_runtime.SequentialRunner,
            "run",
            side_effect=fake_runner_run,
        ),
    ):
        report = serving_runtime.run_online_inference(
            [{"Pregnancies": 1, "Glucose": 90}],
            output_dataset="risk_report",
        )

    assert isinstance(report, pd.DataFrame)
    assert "recommended_action" in report.columns
