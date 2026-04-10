"""Testes da API FastAPI cobrindo serving Kedro e jobs assíncronos."""

from __future__ import annotations

import os
from unittest import mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from insper_deploy_kedro import api


@pytest.fixture()
def client():
    """TestClient com bootstrap Kedro mockado no lifespan."""
    with mock.patch.object(api.serving_runtime, "ensure_bootstrap"):
        with TestClient(api.app, raise_server_exceptions=False) as tc:
            yield tc


VALID_PAYLOAD = {
    "instances": [
        {
            "Pregnancies": 6,
            "Glucose": 148,
            "BloodPressure": 72,
            "SkinThickness": 35,
            "Insulin": 0,
            "BMI": 33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age": 50,
        }
    ]
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        with mock.patch.object(
            api.serving_runtime,
            "get_production_status",
            return_value={
                "model_loaded": True,
                "model_version": "sklearn.linear_model.LogisticRegression",
            },
        ):
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True

    def test_health_returns_503_when_model_is_unavailable(self, client):
        with mock.patch.object(
            api.serving_runtime,
            "get_production_status",
            return_value={"model_loaded": False, "model_version": None},
        ):
            resp = client.get("/health")

        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "unhealthy"
        assert body["model_loaded"] is False

    def test_health_includes_model_version(self, client):
        with mock.patch.object(
            api.serving_runtime,
            "get_production_status",
            return_value={
                "model_loaded": True,
                "model_version": "sklearn.linear_model.LogisticRegression",
            },
        ):
            resp = client.get("/health")

        body = resp.json()
        assert body["model_version"] is not None

    def test_health_recomputes_status_on_each_request(self, client):
        with mock.patch.object(
            api.serving_runtime,
            "get_production_status",
            side_effect=[
                {"model_loaded": False, "model_version": None},
                {
                    "model_loaded": True,
                    "model_version": "catboost.CatBoostClassifier",
                },
            ],
        ):
            first = client.get("/health")
            second = client.get("/health")

        assert first.status_code == 503
        assert first.json()["status"] == "unhealthy"
        assert first.json()["model_loaded"] is False
        assert second.status_code == 200
        assert second.json()["model_loaded"] is True
        assert second.json()["model_version"] == "catboost.CatBoostClassifier"


class TestInferenceEndpoint:
    def test_valid_request_returns_predictions(self, client):
        predictions = pd.DataFrame(
            [
                {
                    "prediction": 1,
                    "prediction_proba": 0.91,
                    "risk_score": 91.0,
                    "risk_band": "Alto risco",
                }
            ]
        )
        with mock.patch.object(
            api.serving_runtime,
            "run_online_inference",
            return_value=predictions,
        ) as mocked_run:
            resp = client.post("/inference", json=VALID_PAYLOAD)

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["predictions"]) == 1
        assert body["predictions"][0]["prediction"] == "1"
        assert body["predictions"][0]["prediction_proba"] == pytest.approx(0.91)
        assert body["predictions"][0]["risk_score"] == pytest.approx(91.0)
        assert body["predictions"][0]["risk_band"] == "Alto risco"
        mocked_run.assert_called_once_with(VALID_PAYLOAD["instances"])

    def test_batch_request(self, client):
        payload = {"instances": VALID_PAYLOAD["instances"] * 3}
        predictions = pd.DataFrame([{"prediction": 0, "prediction_proba": 0.12}] * 3)
        with mock.patch.object(
            api.serving_runtime,
            "run_online_inference",
            return_value=predictions,
        ) as mocked_run:
            resp = client.post("/inference", json=payload)

        assert resp.status_code == 200
        assert len(resp.json()["predictions"]) == 3
        mocked_run.assert_called_once_with(payload["instances"])

    def test_missing_field_returns_422(self, client):
        bad = {"instances": [{"Pregnancies": 6}]}
        resp = client.post("/inference", json=bad)
        assert resp.status_code == 422

    def test_empty_instances_returns_422(self, client):
        resp = client.post("/inference", json={"instances": []})
        assert resp.status_code == 422

    def test_extra_field_rejected(self, client):
        bad = {"instances": [{**VALID_PAYLOAD["instances"][0], "extra": 99}]}
        resp = client.post("/inference", json=bad)
        assert resp.status_code == 422

    def test_missing_model_artifacts_returns_503(self, client):
        with mock.patch.object(
            api.serving_runtime,
            "run_online_inference",
            side_effect=FileNotFoundError("production model missing"),
        ):
            resp = client.post("/inference", json=VALID_PAYLOAD)

        assert resp.status_code == 503
        assert "production model missing" in resp.json()["detail"]

    def test_pipeline_value_error_returns_422(self, client):
        with mock.patch.object(
            api.serving_runtime,
            "run_online_inference",
            side_effect=ValueError("bad payload"),
        ):
            resp = client.post("/inference", json=VALID_PAYLOAD)

        assert resp.status_code == 422
        assert resp.json()["detail"] == "bad payload"

    def test_pipeline_runtime_error_returns_500(self, client):
        with mock.patch.object(
            api.serving_runtime,
            "run_online_inference",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.post("/inference", json=VALID_PAYLOAD)

        assert resp.status_code == 500
        assert "RuntimeError: boom" == resp.json()["detail"]


class TestAPISecurity:
    def test_no_key_required_when_env_unset(self, client):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(
                api.serving_runtime,
                "run_online_inference",
                return_value=pd.DataFrame(
                    [{"prediction": 1, "prediction_proba": 0.88}]
                ),
            ):
                resp = client.post("/inference", json=VALID_PAYLOAD)
                assert resp.status_code == 200

    def test_invalid_key_returns_401_when_env_set(self, client):
        with mock.patch.dict(os.environ, {"API_KEY": "secret"}, clear=True):
            resp = client.post(
                "/inference",
                json=VALID_PAYLOAD,
                headers={"X-API-Key": "wrong"},
            )

        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or missing API key"


class TestTrainingEndpoints:
    def test_start_training_returns_run_id(self, client):
        with mock.patch.object(
            api.serving_runtime,
            "start_training_run",
            return_value="train-123",
        ):
            resp = client.post("/train")

        assert resp.status_code == 200
        assert resp.json() == {"run_id": "train-123", "status": "pending"}

    def test_get_training_status_returns_payload(self, client):
        entry = {
            "status": "completed",
            "pipeline": "train",
            "started_at": "2026-03-30T00:00:00+00:00",
            "finished_at": "2026-03-30T00:01:00+00:00",
            "error": None,
            "result": {
                "pipeline_names": ["data_engineering", "modelling", "refit"],
                "model_loaded": True,
                "model_version": "catboost.CatBoostClassifier",
            },
        }
        with mock.patch.object(
            api.serving_runtime,
            "get_run_status",
            return_value=entry,
        ):
            resp = client.get("/train/train-123")

        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "train-123"
        assert body["status"] == "completed"
        assert body["result"]["model_version"] == "catboost.CatBoostClassifier"

    def test_get_training_status_redacts_traceback_details(self, client):
        entry = {
            "status": "failed",
            "pipeline": "train",
            "started_at": "2026-03-30T00:00:00+00:00",
            "finished_at": "2026-03-30T00:01:00+00:00",
            "error": "RuntimeError: boom\nTraceback (most recent call last):\n...",
            "result": None,
        }
        with mock.patch.object(
            api.serving_runtime,
            "get_run_status",
            return_value=entry,
        ):
            resp = client.get("/train/train-123")

        assert resp.status_code == 200
        assert resp.json()["error"] == "RuntimeError: boom"

    def test_get_training_status_returns_404_for_missing_run(self, client):
        with mock.patch.object(
            api.serving_runtime,
            "get_run_status",
            return_value=None,
        ):
            resp = client.get("/train/missing")

        assert resp.status_code == 404

    def test_get_training_status_requires_api_key_when_configured(self, client):
        with mock.patch.dict(os.environ, {"API_KEY": "secret"}, clear=True):
            resp = client.get("/train/train-123", headers={"X-API-Key": "wrong"})

        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or missing API key"


class TestBatchInferenceEndpoints:
    def test_start_batch_inference_returns_run_id(self, client):
        with mock.patch.object(
            api.serving_runtime,
            "start_batch_inference_run",
            return_value="batch-123",
        ):
            resp = client.post("/batch-inference")

        assert resp.status_code == 200
        assert resp.json() == {"run_id": "batch-123", "status": "pending"}

    def test_get_batch_inference_status_returns_payload(self, client):
        entry = {
            "status": "completed",
            "pipeline": "batch-inference",
            "started_at": "2026-03-30T00:00:00+00:00",
            "finished_at": "2026-03-30T00:00:05+00:00",
            "error": None,
            "result": {
                "pipeline_names": ["inference"],
                "output_dataset": "predictions",
            },
        }
        with mock.patch.object(
            api.serving_runtime,
            "get_run_status",
            return_value=entry,
        ):
            resp = client.get("/batch-inference/batch-123")

        assert resp.status_code == 200
        body = resp.json()
        assert body["run_id"] == "batch-123"
        assert body["result"]["output_dataset"] == "predictions"

    def test_get_batch_inference_status_redacts_traceback_details(self, client):
        entry = {
            "status": "failed",
            "pipeline": "batch-inference",
            "started_at": "2026-03-30T00:00:00+00:00",
            "finished_at": "2026-03-30T00:00:05+00:00",
            "error": "ValueError: bad batch\nTraceback (most recent call last):\n...",
            "result": None,
        }
        with mock.patch.object(
            api.serving_runtime,
            "get_run_status",
            return_value=entry,
        ):
            resp = client.get("/batch-inference/batch-123")

        assert resp.status_code == 200
        assert resp.json()["error"] == "ValueError: bad batch"

    def test_get_batch_inference_status_returns_404_for_missing_run(self, client):
        with mock.patch.object(
            api.serving_runtime,
            "get_run_status",
            return_value=None,
        ):
            resp = client.get("/batch-inference/missing")

        assert resp.status_code == 404

    def test_get_batch_inference_status_requires_api_key_when_configured(
        self,
        client,
    ):
        with mock.patch.dict(os.environ, {"API_KEY": "secret"}, clear=True):
            resp = client.get(
                "/batch-inference/batch-123",
                headers={"X-API-Key": "wrong"},
            )

        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or missing API key"
