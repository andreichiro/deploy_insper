"""Nodes de inferência — só funções novas (to_dataframe, predict)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from insper_deploy_kedro.constants import ModelArtifact

logger = logging.getLogger(__name__)


def _risk_band_label(
    probability: float, risk_bands: list[dict[str, Any]]
) -> str | None:
    for band in risk_bands:
        lower = float(band.get("min_probability", 0.0))
        upper = float(band.get("max_probability", 1.0))
        if lower <= probability < upper:
            return str(band.get("label", band.get("name", "unknown")))
    return None


def to_dataframe(raw_input: dict[str, Any] | list[dict[str, Any]]) -> pd.DataFrame:
    """Converte input bruto (dict ou lista de dicts) em DataFrame."""
    if isinstance(raw_input, dict):
        raw_input = [raw_input]
    dataframe = pd.DataFrame(raw_input)
    logger.info(
        "to_dataframe: %d rows, %d columns", len(dataframe), len(dataframe.columns)
    )
    return dataframe


def predict(
    features_dataframe: pd.DataFrame,
    model_artifact: ModelArtifact,
) -> pd.DataFrame:
    """Roda o modelo nas features preparadas e decodifica os labels"""
    estimator = model_artifact["estimator"]
    target_encoder = model_artifact["target_encoder"]
    feature_columns = model_artifact["feature_columns"]

    x_inference = features_dataframe[feature_columns]

    predictions = features_dataframe.copy()

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_inference)[:, 1]
        predictions["prediction_proba"] = probabilities
        predictions["risk_score"] = probabilities * 100.0
        risk_bands = model_artifact.get("risk_bands", [])
        if risk_bands:
            predictions["risk_band"] = [
                _risk_band_label(float(probability), risk_bands)
                for probability in probabilities
            ]
        threshold = float(model_artifact.get("decision_threshold", 0.5))
        predicted_codes = (probabilities >= threshold).astype(int)
    else:
        predicted_codes = estimator.predict(x_inference)

    predicted_labels = target_encoder.inverse_transform(predicted_codes)
    predictions["prediction"] = predicted_labels

    logger.info("predict: %d predictions made", len(predictions))
    return predictions


def _recommended_action(risk_band: str | None) -> str:
    band = (risk_band or "").strip().lower()
    if band == "risco critico":
        return "Encaminhar avaliação clínica imediata"
    if band == "alto risco":
        return "Priorizar revisão clínica"
    if band == "risco moderado":
        return "Solicitar triagem complementar"
    if band == "baixo risco":
        return "Manter monitoramento de rotina"
    return "Revisar contexto clínico"


def build_risk_report(
    source_dataframe: pd.DataFrame,
    predictions: pd.DataFrame,
    model_artifact: ModelArtifact,
) -> pd.DataFrame:
    """Materialize a user-facing risk report with readable fields and ranking."""
    report = source_dataframe.reset_index(drop=True).copy()
    scored = predictions.reset_index(drop=True).copy()
    report.insert(0, "case_id", np.arange(1, len(report) + 1))

    report["prediction"] = scored["prediction"]
    report["prediction_label"] = np.where(
        pd.Series(scored["prediction"]).astype(str) == "1",
        "Positivo",
        "Negativo",
    )
    if "prediction_proba" in scored.columns:
        report["prediction_proba"] = scored["prediction_proba"].astype(float)
        report["risk_score"] = scored.get("risk_score", scored["prediction_proba"] * 100.0)
    if "risk_band" in scored.columns:
        report["risk_band"] = scored["risk_band"]
    report["decision_policy_name"] = model_artifact.get("decision_policy_name")
    report["decision_threshold"] = float(model_artifact.get("decision_threshold", 0.5))
    report["recommended_action"] = [
        _recommended_action(report.get("risk_band", pd.Series([None] * len(report))).iloc[idx])
        for idx in range(len(report))
    ]
    if "prediction_proba" in report.columns:
        report["risk_priority_rank"] = (
            report["prediction_proba"].rank(method="first", ascending=False).astype(int)
        )
        report = report.sort_values(
            by=["risk_priority_rank", "risk_score"],
            ascending=[True, False],
            kind="mergesort",
        )
    else:
        report["risk_priority_rank"] = np.arange(1, len(report) + 1)
    report = report.reset_index(drop=True)
    logger.info("build_risk_report: %d report rows materialized", len(report))
    return report
