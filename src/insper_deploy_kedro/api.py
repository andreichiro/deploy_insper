"""Camada de serving FastAPI, com inferência Kedro e jobs assíncronos."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Response, Security, status
from fastapi.security import APIKeyHeader
from kedro.io import DatasetError
from pydantic import BaseModel, ConfigDict, Field

from insper_deploy_kedro import serving_runtime
from insper_deploy_kedro.logging_utils import configure_project_logging

configure_project_logging()
logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(
    api_key: str | None = Security(API_KEY_HEADER),
) -> str | None:
    """Valida API key quando a env var API_KEY tá setada."""
    expected = os.getenv("API_KEY")
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


class RunResponse(BaseModel):
    """Response de disparo de job assíncrono."""

    run_id: str
    status: str


class RunStatus(BaseModel):
    """Status de execução de job assíncrono."""

    run_id: str
    status: str
    pipeline: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: Any | None = None


class DiabetesFeatures(BaseModel):
    """Features de entrada pra um paciente."""

    model_config = ConfigDict(extra="forbid")

    Pregnancies: float = Field(..., ge=0, examples=[6])
    Glucose: float = Field(..., ge=0, examples=[148])
    BloodPressure: float = Field(..., ge=0, examples=[72])
    SkinThickness: float = Field(..., ge=0, examples=[35])
    Insulin: float = Field(..., ge=0, examples=[0])
    BMI: float = Field(..., ge=0, examples=[33.6])
    DiabetesPedigreeFunction: float = Field(..., ge=0, examples=[0.627])
    Age: float = Field(..., ge=0, examples=[50])


class InferenceRequest(BaseModel):
    """Request de inferência em batch."""

    model_config = ConfigDict(extra="forbid")

    instances: list[DiabetesFeatures] = Field(..., min_length=1)


class PredictionResult(BaseModel):
    """Resultado de uma predição."""

    prediction: str
    prediction_proba: float | None = None
    risk_score: float | None = None
    risk_band: str | None = None


class InferenceResponse(BaseModel):
    """Response de inferência em batch."""

    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    """Response do health check."""

    status: str
    model_loaded: bool
    model_version: str | None = None


def _sanitize_run_error(error: str | None) -> str | None:
    if error is None:
        return None
    return error.splitlines()[0].strip()


def _public_run_status(run_id: str, entry: dict[str, Any]) -> RunStatus:
    public_entry = entry.copy()
    public_entry["error"] = _sanitize_run_error(public_entry.get("error"))
    return RunStatus(run_id=run_id, **public_entry)


def _serialize_predictions(predictions_frame: Any) -> InferenceResponse:
    records = predictions_frame.to_dict(orient="records")
    predictions = [
        PredictionResult(
            prediction=str(record["prediction"]),
            prediction_proba=(
                float(record["prediction_proba"])
                if record.get("prediction_proba") is not None
                else None
            ),
            risk_score=(
                float(record["risk_score"])
                if record.get("risk_score") is not None
                else None
            ),
            risk_band=(
                str(record["risk_band"])
                if record.get("risk_band") is not None
                else None
            ),
        )
        for record in records
    ]
    return InferenceResponse(predictions=predictions)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Inicializa bootstrap Kedro uma vez no startup."""
    serving_runtime.ensure_bootstrap()
    yield


app = FastAPI(
    title="API de Predição de Diabetes",
    description="Prever diabetes usando um pipeline ML de produção.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
)
def health_check(response: Response) -> HealthResponse:
    """Readiness probe for production inference artifacts."""
    production_status = serving_runtime.get_production_status()
    model_loaded = bool(production_status.get("model_loaded"))
    response.status_code = (
        status.HTTP_200_OK if model_loaded else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_version=production_status.get("model_version"),
    )


@app.post("/train", response_model=RunResponse)
def start_training(
    _key: str | None = Depends(_verify_api_key),
) -> RunResponse:
    """Dispara treino completo (DE + modelling + refit) em background."""
    run_id = serving_runtime.start_training_run()
    return RunResponse(run_id=run_id, status="pending")


@app.get("/train/{run_id}", response_model=RunStatus)
def get_training_status(
    run_id: str,
    _key: str | None = Depends(_verify_api_key),
) -> RunStatus:
    """Consulta o status de um treino em background."""
    entry = serving_runtime.get_run_status(run_id)
    if entry is None or entry.get("pipeline") != "train":
        raise HTTPException(status_code=404, detail="Run not found")

    return _public_run_status(run_id, entry)


@app.post("/batch-inference", response_model=RunResponse)
def start_batch_inference(
    _key: str | None = Depends(_verify_api_key),
) -> RunResponse:
    """Dispara a pipeline Kedro de inferência em background usando o catálogo."""
    run_id = serving_runtime.start_batch_inference_run()
    return RunResponse(run_id=run_id, status="pending")


@app.get("/batch-inference/{run_id}", response_model=RunStatus)
def get_batch_inference_status(
    run_id: str,
    _key: str | None = Depends(_verify_api_key),
) -> RunStatus:
    """Consulta o status de uma inferência batch em background."""
    entry = serving_runtime.get_run_status(run_id)
    if entry is None or entry.get("pipeline") != "batch-inference":
        raise HTTPException(status_code=404, detail="Run not found")
    return _public_run_status(run_id, entry)


@app.post("/inference", response_model=InferenceResponse)
def run_inference(
    request: InferenceRequest,
    _key: str | None = Depends(_verify_api_key),
) -> InferenceResponse:
    """Roda a pipeline Kedro real de inferência num batch de pacientes."""
    instances = [inst.model_dump() for inst in request.instances]

    try:
        predictions_frame = serving_runtime.run_online_inference(instances)
        return _serialize_predictions(predictions_frame)
    except (DatasetError, FileNotFoundError, OSError) as exc:
        logger.exception("Online inference unavailable")
        raise HTTPException(
            status_code=503,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
    except (KeyError, ValueError) as exc:
        logger.exception("Online inference rejected the payload")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Online inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
