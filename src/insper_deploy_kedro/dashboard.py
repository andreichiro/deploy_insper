"""Clinical ML dashboard with operations, robustness, policy, and inference views."""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from insper_deploy_kedro import ops_store, serving_runtime
from insper_deploy_kedro.logging_utils import configure_project_logging

configure_project_logging()
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "06_models"
OUTPUT_DIR = DATA_DIR / "07_model_output"
OPS_DIR = DATA_DIR / "09_ops"

FEATURE_DESCRIPTIONS: dict[str, tuple[str, float, float, float]] = {
    "Pregnancies": ("Gestações", 0.0, 20.0, 6.0),
    "Glucose": ("Glicose (mg/dL)", 0.0, 250.0, 148.0),
    "BloodPressure": ("Pressão arterial (mm Hg)", 0.0, 140.0, 72.0),
    "SkinThickness": ("Espessura da pele (mm)", 0.0, 100.0, 35.0),
    "Insulin": ("Insulina (mu U/ml)", 0.0, 900.0, 0.0),
    "BMI": ("IMC (kg/m²)", 0.0, 70.0, 33.6),
    "DiabetesPedigreeFunction": ("Função pedigree", 0.0, 2.5, 0.627),
    "Age": ("Idade", 18.0, 100.0, 50.0),
}

MODEL_DISPLAY_NAMES = {
    "baseline": "Logistic Regression",
    "optimized": "CatBoost (Optuna)",
    "xgboost": "XGBoost (Optuna)",
}

ROBUSTNESS_METRIC_LABELS = {
    "roc_auc": "ROC AUC",
    "brier": "Brier score",
    "log_loss": "Log loss",
    "recall": "Recall",
    "precision": "Precisão",
    "f1": "F1",
    "expected_cost": "Custo esperado",
    "false_negative_rate": "Taxa de falso negativo",
    "false_positive_rate": "Taxa de falso positivo",
    "calibration_slope_error": "Erro de slope de calibração",
    "calibration_intercept_abs": "Intercepto absoluto de calibração",
    "tp_share": "Proporção de verdadeiros positivos",
    "fp_share": "Proporção de falsos positivos",
    "fn_share": "Proporção de falsos negativos",
    "tn_share": "Proporção de verdadeiros negativos",
    "mean_risk_score": "Score médio de risco",
}

IMPORTANCE_METRIC_LABELS = {
    "roc_auc": "Delta ROC AUC",
    "brier": "Delta Brier",
    "log_loss": "Delta Log loss",
    "recall": "Delta Recall",
    "expected_cost": "Delta Custo esperado",
}

PIPELINE_ACTIONS: list[tuple[str, list[str], str]] = [
    (
        "Treinar tudo",
        [sys.executable, "-m", "kedro", "run"],
        "Executa data engineering, modelagem e refit de ponta a ponta.",
    ),
    (
        "Refit produção",
        [sys.executable, "-m", "kedro", "run", "--pipeline", "refit"],
        "Regera os artefatos de produção usando todos os dados.",
    ),
    (
        "Inferência batch",
        [sys.executable, "-m", "kedro", "run", "--pipeline", "inference"],
        "Roda a pipeline batch usando o catálogo atual.",
    ),
]


def _find_artifact(path: Path) -> Path | None:
    """Resolve artefato Kedro, incluindo layouts versionados."""
    if path.is_file():
        return path
    if path.is_dir():
        versions = sorted(path.iterdir(), reverse=True)
        for version_dir in versions:
            candidate = version_dir / path.name
            if candidate.is_file():
                return candidate
    return None


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)  # noqa: S301


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_parquet(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_parquet(path)


def _artifact_signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for path in paths:
        resolved = _find_artifact(path)
        if resolved is None:
            continue
        stats = resolved.stat()
        signature.append((str(resolved), stats.st_mtime_ns, stats.st_size))
    return tuple(signature)


def _production_artifact_fingerprint() -> tuple[tuple[str, int, int], ...]:
    return _artifact_signature(
        [
            MODELS_DIR / "production_encoders.pkl",
            MODELS_DIR / "production_scalers.pkl",
            MODELS_DIR / "production_model.pkl",
        ]
    )


def _dashboard_artifact_fingerprint() -> tuple[tuple[str, int, int], ...]:
    return _artifact_signature(
        [
            OUTPUT_DIR / "baseline_metrics.pkl",
            OUTPUT_DIR / "optimized_metrics.pkl",
            OUTPUT_DIR / "xgboost_metrics.pkl",
            OUTPUT_DIR / "test_report.pkl",
            OUTPUT_DIR / "model_frontier.parquet",
            OUTPUT_DIR / "model_selection_scorecard.parquet",
            OUTPUT_DIR / "feature_selection_frontier.parquet",
            OUTPUT_DIR / "feature_selection_stability.parquet",
            OUTPUT_DIR / "feature_selection_manifest.json",
            OUTPUT_DIR / "cv_fold_metrics.parquet",
            OUTPUT_DIR / "cv_metric_summary.parquet",
            OUTPUT_DIR / "bootstrap_metric_intervals.parquet",
            OUTPUT_DIR / "permutation_feature_importance.parquet",
            OUTPUT_DIR / "perturbation_sensitivity_audit.parquet",
            OUTPUT_DIR / "perturbation_sensitivity_summary.parquet",
            OUTPUT_DIR / "threshold_metrics.parquet",
            OUTPUT_DIR / "split_comparison_report.parquet",
            OUTPUT_DIR / "nested_cv_audit.parquet",
            OUTPUT_DIR / "modelling_design_audit.parquet",
            OUTPUT_DIR / "selected_deployment_policy.pkl",
            OUTPUT_DIR / "predictions.parquet",
            OUTPUT_DIR / "risk_report.parquet",
            OPS_DIR / "split_strategy_report.parquet",
            OPS_DIR / "latest_training_run_manifest.json",
            OPS_DIR / "latest_serving_manifest.json",
            OPS_DIR / "latest_inference_contract.json",
        ]
    )


@st.cache_resource
def _load_production_artifacts_cached(
    _fingerprint: tuple[tuple[str, int, int], ...],
) -> dict[str, Any] | None:
    return serving_runtime.load_production_artifacts()


def load_production_artifacts() -> dict[str, Any] | None:
    return _load_production_artifacts_cached(_production_artifact_fingerprint())


def _fallback_model_frontier(
    report: dict[str, Any] | None,
    selected_policy: dict[str, Any] | None,
) -> pd.DataFrame | None:
    if report is None:
        return None

    selected_model_name = selected_policy.get("model_name") if selected_policy else None
    rows: list[dict[str, Any]] = []
    for model_name, metrics in report.items():
        rows.append(
            {
                "model_name": model_name,
                "class_path": model_name,
                "selected_for_refit": model_name == selected_model_name,
                "best_cv_score": None,
                "validation_accuracy": None,
                "validation_precision": None,
                "validation_recall": None,
                "validation_f1": None,
                "validation_roc_auc": None,
                "validation_mape": None,
                "validation_r2": None,
                "deployment_policy_name": (
                    selected_policy.get("decision_policy_name")
                    if model_name == selected_model_name and selected_policy
                    else None
                ),
                "deployment_threshold": (
                    selected_policy.get("decision_threshold")
                    if model_name == selected_model_name and selected_policy
                    else None
                ),
                "test_accuracy": metrics.get("accuracy"),
                "test_precision": metrics.get("precision"),
                "test_recall": metrics.get("recall"),
                "test_f1": metrics.get("f1"),
                "test_roc_auc": metrics.get("roc_auc"),
            }
        )
    return pd.DataFrame(rows)


def _selected_policy_from_artifacts(
    production_artifacts: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if production_artifacts is None:
        return None

    model_artifact = production_artifacts["model"]
    if "decision_policy_name" not in model_artifact:
        return None

    return {
        "model_name": "production",
        "class_path": model_artifact.get("class_path"),
        "decision_policy_name": model_artifact.get("decision_policy_name"),
        "decision_policy_description": model_artifact.get(
            "decision_policy_description",
            "",
        ),
        "decision_threshold": model_artifact.get("decision_threshold"),
        "policy_catalog": model_artifact.get("policy_catalog", []),
        "risk_bands": model_artifact.get("risk_bands", []),
    }


@st.cache_data
def _load_dashboard_state_cached(
    _fingerprint: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    report_path = _find_artifact(OUTPUT_DIR / "test_report.pkl")
    policy_path = _find_artifact(OUTPUT_DIR / "selected_deployment_policy.pkl")
    state = {
        "test_report": _load_pickle(report_path) if report_path else None,
        "model_frontier": _load_parquet(
            _find_artifact(OUTPUT_DIR / "model_frontier.parquet")
        ),
        "model_selection_scorecard": _load_parquet(
            _find_artifact(OUTPUT_DIR / "model_selection_scorecard.parquet")
        ),
        "feature_selection_frontier": _load_parquet(
            _find_artifact(OUTPUT_DIR / "feature_selection_frontier.parquet")
        ),
        "feature_selection_stability": _load_parquet(
            _find_artifact(OUTPUT_DIR / "feature_selection_stability.parquet")
        ),
        "feature_selection_manifest": _load_json(
            _find_artifact(OUTPUT_DIR / "feature_selection_manifest.json")
        ),
        "cv_fold_metrics": _load_parquet(
            _find_artifact(OUTPUT_DIR / "cv_fold_metrics.parquet")
        ),
        "cv_metric_summary": _load_parquet(
            _find_artifact(OUTPUT_DIR / "cv_metric_summary.parquet")
        ),
        "bootstrap_metric_intervals": _load_parquet(
            _find_artifact(OUTPUT_DIR / "bootstrap_metric_intervals.parquet")
        ),
        "permutation_feature_importance": _load_parquet(
            _find_artifact(OUTPUT_DIR / "permutation_feature_importance.parquet")
        ),
        "perturbation_sensitivity_audit": _load_parquet(
            _find_artifact(OUTPUT_DIR / "perturbation_sensitivity_audit.parquet")
        ),
        "perturbation_sensitivity_summary": _load_parquet(
            _find_artifact(OUTPUT_DIR / "perturbation_sensitivity_summary.parquet")
        ),
        "threshold_metrics": _load_parquet(
            _find_artifact(OUTPUT_DIR / "threshold_metrics.parquet")
        ),
        "split_comparison_report": _load_parquet(
            _find_artifact(OUTPUT_DIR / "split_comparison_report.parquet")
        ),
        "nested_cv_audit": _load_parquet(
            _find_artifact(OUTPUT_DIR / "nested_cv_audit.parquet")
        ),
        "modelling_design_audit": _load_parquet(
            _find_artifact(OUTPUT_DIR / "modelling_design_audit.parquet")
        ),
        "selected_deployment_policy": _load_pickle(policy_path)
        if policy_path
        else None,
        "predictions": _load_parquet(
            _find_artifact(OUTPUT_DIR / "predictions.parquet")
        ),
        "risk_report": _load_parquet(
            _find_artifact(OUTPUT_DIR / "risk_report.parquet")
        ),
        "training_run_manifest": _load_json(
            _find_artifact(OPS_DIR / "latest_training_run_manifest.json")
        ),
        "split_strategy_report": _load_parquet(
            _find_artifact(OPS_DIR / "split_strategy_report.parquet")
        ),
        "serving_manifest": _load_json(
            _find_artifact(OPS_DIR / "latest_serving_manifest.json")
        ),
        "inference_contract": _load_json(
            _find_artifact(OPS_DIR / "latest_inference_contract.json")
        ),
    }
    return state


def load_dashboard_state(
    production_artifacts: dict[str, Any] | None,
) -> dict[str, Any]:
    state = _load_dashboard_state_cached(_dashboard_artifact_fingerprint())
    selected_policy = state[
        "selected_deployment_policy"
    ] or _selected_policy_from_artifacts(production_artifacts)
    state["selected_deployment_policy"] = selected_policy
    if state["model_frontier"] is None:
        state["model_frontier"] = _fallback_model_frontier(
            state["test_report"],
            selected_policy,
        )
    return state


def _selected_model_name(
    frontier: pd.DataFrame | None,
    selected_policy: dict[str, Any] | None,
) -> str | None:
    selected_model_name: str | None = None
    if frontier is None or frontier.empty:
        if selected_policy is not None and selected_policy.get("model_name"):
            selected_model_name = str(selected_policy["model_name"])
        return selected_model_name

    if selected_policy is not None and selected_policy.get("model_name") and selected_model_name is None:
        candidate_name = str(selected_policy["model_name"])
        if "model_name" in frontier.columns and candidate_name in set(
            frontier["model_name"].astype(str)
        ):
            selected_model_name = candidate_name
        class_path = selected_policy.get("class_path") if selected_model_name is None else None
        if class_path and "class_path" in frontier.columns and selected_model_name is None:
            matches = frontier[frontier["class_path"].astype(str) == str(class_path)]
            if not matches.empty:
                selected_model_name = str(matches.iloc[0]["model_name"])

    if selected_model_name is None and "selected_for_refit" in frontier.columns:
        selected_rows = frontier[frontier["selected_for_refit"] == True]  # noqa: E712
        if not selected_rows.empty:
            selected_model_name = str(selected_rows.iloc[0]["model_name"])
    return selected_model_name


def _format_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def _class_short_name(class_path: str | None) -> str:
    if not class_path:
        return "-"
    return str(class_path).split(".")[-1]


def _display_model_name(model_name: str | None, class_path: str | None = None) -> str:
    if model_name and model_name in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[model_name]
    short_name = _class_short_name(class_path)
    return short_name if short_name != "-" else (model_name or "-")


def _policy_display_name(selected_policy: dict[str, Any] | None) -> str:
    if selected_policy is None:
        return "-"
    policy_name = str(selected_policy.get("decision_policy_name", "-"))
    for policy in selected_policy.get("policy_catalog", []):
        if policy.get("policy_name") == policy_name:
            return str(policy.get("policy_label", policy_name))
    return policy_name


def build_overview_metrics(
    frontier: pd.DataFrame | None,
    selected_policy: dict[str, Any] | None,
    production_artifacts: dict[str, Any] | None,
) -> dict[str, str]:
    selected_model_name = _selected_model_name(frontier, selected_policy)
    selected_class_path = None
    if frontier is not None and not frontier.empty and selected_model_name is not None:
        matching_rows = frontier[frontier["model_name"].astype(str) == selected_model_name]
        if not matching_rows.empty and "class_path" in matching_rows.columns:
            selected_class_path = str(matching_rows.iloc[0]["class_path"])
    if selected_class_path is None and selected_policy is not None:
        selected_class_path = selected_policy.get("class_path")
    return {
        "model_name": _display_model_name(
            selected_model_name,
            selected_class_path,
        ),
        "policy_name": (
            _policy_display_name(selected_policy)
            if selected_policy
            else "-"
        ),
        "threshold": (
            f"{float(selected_policy['decision_threshold']):.2f}"
            if selected_policy and selected_policy.get("decision_threshold") is not None
            else "-"
        ),
        "refit_all_data": (
            "Sim"
            if production_artifacts
            and production_artifacts["model"].get("train_splits")
            == ["train", "validation", "test"]
            else "N/A"
        ),
    }


def prepare_frontier_view(frontier: pd.DataFrame | None) -> pd.DataFrame | None:
    if frontier is None or frontier.empty:
        return None

    frontier_view = frontier.copy()
    frontier_view["status"] = np.where(
        frontier_view.get("selected_for_refit", False),
        "Selecionado",
        "",
    )
    frontier_view["model_name"] = frontier_view["model_name"].map(
        lambda name: MODEL_DISPLAY_NAMES.get(name, name)
    )
    if (
        "best_cv_score" in frontier_view.columns
        and "validation_roc_auc" in frontier_view.columns
    ):
        frontier_view["validation_minus_cv_gap"] = (
            frontier_view["validation_roc_auc"] - frontier_view["best_cv_score"]
        )
    if "selected_for_refit" in frontier_view.columns:
        frontier_view = frontier_view.sort_values(
            by=[
                "selected_for_refit",
                "selection_rank" if "selection_rank" in frontier_view.columns else "model_name",
                "validation_roc_auc" if "validation_roc_auc" in frontier_view.columns else "model_name",
            ],
            ascending=[False, True, False],
            kind="mergesort",
        )
    ordered_columns = [
        "status",
        "model_name",
        "selection_rank",
        "selection_composite_score",
        "best_cv_score",
        "validation_roc_auc",
        "validation_brier",
        "validation_log_loss",
        "validation_calibration_slope_error",
        "policy_recall",
        "policy_precision",
        "policy_false_negative_rate",
        "policy_expected_cost_per_sample",
        "deployment_policy_name",
        "deployment_threshold",
        "validation_minus_cv_gap",
    ]
    available_columns = [column for column in ordered_columns if column in frontier_view.columns]
    return frontier_view[available_columns].reset_index(drop=True)


def prepare_model_selection_scorecard_view(
    model_selection_scorecard: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if model_selection_scorecard is None or model_selection_scorecard.empty:
        return None

    view = model_selection_scorecard.copy()
    if "selection_rank" in view.columns:
        view["status"] = np.where(view["selection_rank"] == 1, "Selecionado", "")
    else:
        view["status"] = ""
    view["model_name"] = view["model_name"].map(
        lambda name: MODEL_DISPLAY_NAMES.get(name, name)
    )
    sort_columns = ["model_name"]
    ascending = [True]
    if "selection_rank" in view.columns:
        sort_columns = ["selection_rank", *sort_columns]
        ascending = [True, *ascending]
    view = view.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")
    ordered_columns = [
        "status",
        "selection_rank",
        "model_name",
        "selection_composite_score",
        "selection_policy_label",
        "selection_policy_threshold",
        "validation_roc_auc",
        "validation_brier",
        "validation_log_loss",
        "validation_calibration_slope_error",
        "policy_recall",
        "policy_false_negative_rate",
        "policy_precision",
        "policy_expected_cost_per_sample",
    ]
    available_columns = [column for column in ordered_columns if column in view.columns]
    return view[available_columns].copy()


def prepare_split_comparison_view(
    split_comparison_report: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if split_comparison_report is None or split_comparison_report.empty:
        return None

    split_order = {"train": 0, "validation": 1, "test": 2}
    view = split_comparison_report.copy()
    view["split_order"] = view["split"].map(lambda name: split_order.get(name, 999))
    ordered_columns = [
        "split",
        "n_samples",
        "policy_threshold",
        "roc_auc",
        "brier",
        "log_loss",
        "recall",
        "precision",
        "f1",
        "false_negative_rate",
        "false_positive_rate",
        "expected_cost",
        "roc_auc_gap_vs_train",
        "brier_gap_vs_train",
    ]
    available_columns = [column for column in ordered_columns if column in view.columns]
    return view.sort_values("split_order", kind="mergesort")[available_columns].reset_index(
        drop=True
    )


def prepare_nested_cv_audit_view(
    nested_cv_audit: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if nested_cv_audit is None or nested_cv_audit.empty:
        return None
    return nested_cv_audit.sort_values("fold_id", kind="mergesort").reset_index(
        drop=True
    )


def prepare_modelling_design_audit_view(
    modelling_design_audit: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if modelling_design_audit is None or modelling_design_audit.empty:
        return None
    return modelling_design_audit.copy()


def build_risk_report_preview(
    predictions: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if (
        predictions is None
        or predictions.empty
        or "prediction_proba" not in predictions.columns
    ):
        return None, None

    risk_counts = None
    if "risk_band" in predictions.columns:
        risk_counts = (
            predictions["risk_band"]
            .fillna("Sem faixa")
            .value_counts()
            .rename_axis("Faixa")
            .reset_index(name="Casos")
        )

    ordered_columns = [
        "case_id",
        "risk_priority_rank",
        "prediction_label",
        "risk_band",
        "risk_score",
        "prediction_proba",
        "recommended_action",
        "prediction",
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "glucose_bmi_interaction",
    ]
    available_columns = [column for column in ordered_columns if column in predictions.columns]
    ranked = predictions.copy()
    if "risk_priority_rank" in ranked.columns:
        ranked = ranked.sort_values(
            by=["risk_priority_rank", "prediction_proba"],
            ascending=[True, False],
            kind="mergesort",
        )
    else:
        ranked = ranked.sort_values(by="prediction_proba", ascending=False)
    return risk_counts, ranked[available_columns].reset_index(drop=True)


def build_test_report_summary(
    report: dict[str, Any] | None,
    frontier: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not report:
        return pd.DataFrame()

    selected_model_name = _selected_model_name(frontier, None)
    rows = []
    for model_name, metrics in report.items():
        rows.append(
            {
                "status": "Selecionado" if model_name == selected_model_name else "",
                "model_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                "test_roc_auc": metrics.get("roc_auc"),
                "test_recall": metrics.get("recall"),
                "test_precision": metrics.get("precision"),
                "test_f1": metrics.get("f1"),
            }
        )
    summary = pd.DataFrame(rows)
    summary["selected_order"] = np.where(summary["status"] == "Selecionado", 0, 1)
    return summary.sort_values(
        by=["selected_order", "test_roc_auc", "model_name"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop(columns=["selected_order"])


def build_robustness_chart_frame(
    cv_fold_metrics: pd.DataFrame | None,
    metric_name: str,
) -> pd.DataFrame:
    if cv_fold_metrics is None or cv_fold_metrics.empty:
        return pd.DataFrame()

    return cv_fold_metrics.pivot_table(
        index="fold_id",
        columns="model_name",
        values=metric_name,
    ).rename(columns=MODEL_DISPLAY_NAMES)


def prepare_cv_metric_summary_view(
    cv_metric_summary: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if cv_metric_summary is None or cv_metric_summary.empty:
        return None

    summary = cv_metric_summary.copy()
    summary["model_name"] = summary["model_name"].map(
        lambda name: MODEL_DISPLAY_NAMES.get(name, name)
    )
    return summary


def prepare_bootstrap_metric_intervals_view(
    bootstrap_metric_intervals: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if bootstrap_metric_intervals is None or bootstrap_metric_intervals.empty:
        return None
    view = bootstrap_metric_intervals.copy()
    view["model_name"] = view["model_name"].map(
        lambda name: MODEL_DISPLAY_NAMES.get(name, name)
    )
    return view.sort_values(by=["metric_name", "ci_width", "model_name"]).reset_index(
        drop=True
    )


def prepare_permutation_feature_importance_view(
    permutation_feature_importance: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if permutation_feature_importance is None or permutation_feature_importance.empty:
        return None
    return permutation_feature_importance.sort_values(
        by=["metric_name", "importance_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)


def prepare_feature_selection_frontier_view(
    feature_selection_frontier: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if feature_selection_frontier is None or feature_selection_frontier.empty:
        return None
    view = feature_selection_frontier.copy()
    if "model_name" in view.columns:
        view["model_name"] = view["model_name"].map(
            lambda name: MODEL_DISPLAY_NAMES.get(name, name)
        )
    ordered_columns = [
        "model_name",
        "selected_flag",
        "selected_model_flag",
        "within_one_se",
        "feature_count",
        "block_count",
        "feature_names_text",
        "mean_brier",
        "sem_brier",
        "mean_roc_auc",
        "mean_log_loss",
        "mean_calibration_slope_error",
        "mean_calibration_intercept_abs",
    ]
    available_columns = [
        column for column in ordered_columns if column in view.columns
    ]
    sort_columns = [
        column_name
        for column_name in [
            "selected_model_flag",
            "selected_flag",
            "within_one_se",
            "feature_count",
            "mean_brier",
            "model_name",
        ]
        if column_name in view.columns
    ]
    ascending = [
        False if column_name in {"selected_model_flag", "selected_flag", "within_one_se"} else True
        for column_name in sort_columns
    ]
    return view.sort_values(
        by=sort_columns,
        ascending=ascending,
        kind="mergesort",
    )[available_columns].reset_index(drop=True)


def prepare_feature_selection_stability_view(
    feature_selection_stability: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if feature_selection_stability is None or feature_selection_stability.empty:
        return None
    return feature_selection_stability.sort_values(
        by=["entity_type", "selection_frequency", "entity_name"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def prepare_perturbation_sensitivity_summary_view(
    perturbation_sensitivity_summary: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if (
        perturbation_sensitivity_summary is None
        or perturbation_sensitivity_summary.empty
    ):
        return None
    return perturbation_sensitivity_summary.sort_values(
        by=["max_decision_flip_rate", "max_sensitivity_ratio"],
        ascending=False,
    ).reset_index(drop=True)


def prepare_threshold_metrics_view(
    threshold_metrics: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if threshold_metrics is None or threshold_metrics.empty:
        return None

    columns = [
        "policy_label",
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_negative_rate",
        "false_positive_rate",
        "expected_cost",
    ]
    available_columns = [column for column in columns if column in threshold_metrics]
    return threshold_metrics[available_columns].copy()


def select_active_policy_row(
    threshold_metrics: pd.DataFrame | None,
    selected_policy: dict[str, Any] | None,
) -> pd.Series | None:
    if threshold_metrics is None or threshold_metrics.empty:
        return None

    active_policy_name = (
        selected_policy.get("decision_policy_name")
        if selected_policy is not None
        else threshold_metrics.iloc[0]["policy_name"]
    )
    active_rows = threshold_metrics[threshold_metrics["policy_name"] == active_policy_name]
    if active_rows.empty:
        return threshold_metrics.iloc[0]
    return active_rows.iloc[0]


def build_sidebar_snapshot(
    production_status: dict[str, Any],
    production_artifacts: dict[str, Any] | None,
    training_manifest: dict[str, Any] | None = None,
    selected_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"production_status": production_status}
    if training_manifest is not None:
        snapshot["selected_model_name"] = _display_model_name(
            training_manifest.get("selected_model_name"),
            training_manifest.get("selected_class_path"),
        )
        snapshot["selected_class_path"] = training_manifest.get("selected_class_path")
        snapshot["selected_policy_name"] = _policy_display_name(
            {
                "decision_policy_name": training_manifest.get("selected_policy_name"),
                "policy_catalog": (training_manifest.get("best_model_config") or {}).get(
                    "policy_catalog",
                    [],
                ),
            }
        )
        snapshot["selected_threshold"] = training_manifest.get("selected_threshold")
        feature_manifest = training_manifest.get("feature_selection_manifest") or {}
        selected_features = feature_manifest.get("selected_feature_names_text") or ", ".join(
            feature_manifest.get("selected_feature_names", [])
        )
        if selected_features:
            snapshot["selected_feature_names_text"] = selected_features
    if production_artifacts is None:
        return snapshot

    model_artifact = production_artifacts["model"]
    snapshot.update(
        {
            "class_path": model_artifact.get("class_path", "?"),
            "train_splits": model_artifact.get("train_splits", []),
            "decision_threshold": model_artifact.get("decision_threshold", "n/a"),
        }
    )
    if selected_policy is not None and "selected_policy_name" not in snapshot:
        snapshot["selected_policy_name"] = _policy_display_name(selected_policy)
        snapshot["selected_threshold"] = selected_policy.get("decision_threshold")
    return snapshot


def load_registry_state() -> dict[str, Any]:
    try:
        return {
            "experiment_runs": ops_store.list_experiment_runs(limit=5),
            "model_registry": ops_store.list_model_registry_entries(limit=5),
            "error": None,
        }
    except Exception as exc:
        logger.exception("Failed to load dashboard registry state")
        return {
            "experiment_runs": [],
            "model_registry": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_cli(command: list[str], cwd: Path, caption: str) -> tuple[int, list[str]]:
    st.caption(caption)
    status = st.empty()
    log_box = st.empty()
    lines: list[str] = []
    status.info("Executando comando local do projeto...")
    try:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault(
            "KEDRO_LOGGING_CONFIG",
            str(PROJECT_ROOT / "conf" / "logging.yml"),
        )
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        logger.exception("Failed to launch dashboard CLI command: %s", command)
        status.error(f"Falha ao iniciar comando: {type(exc).__name__}: {exc}")
        return 1, [f"{type(exc).__name__}: {exc}"]
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        lines.append(line)
        log_box.code("\n".join(lines[-200:]), language="text")
    return_code = proc.wait()
    if return_code == 0:
        status.success("Comando concluído com sucesso. Limpando caches do dashboard.")
        st.cache_data.clear()
        st.cache_resource.clear()
    else:
        status.error(f"Comando falhou com código {return_code}.")
    return return_code, lines


def _render_confusion_matrix(tp: int, fp: int, tn: int, fn: int, title: str) -> None:
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negativo", "Positivo"])
    ax.set_yticklabels(["Negativo", "Positivo"])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.66)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _policy_decisions(
    probability: float, policy_catalog: list[dict[str, Any]]
) -> pd.DataFrame:
    if not policy_catalog:
        return pd.DataFrame()
    rows = []
    for policy in policy_catalog:
        threshold = float(policy.get("threshold", 0.5))
        rows.append(
            {
                "policy": policy.get("policy_label", policy.get("policy_name", "?")),
                "threshold": threshold,
                "decision": "Positivo" if probability >= threshold else "Negativo",
                "expected_cost": policy.get("expected_cost"),
                "recall": policy.get("recall"),
                "precision": policy.get("precision"),
            }
        )
    return pd.DataFrame(rows)


def render_overview_tab(
    state: dict[str, Any],
    production_artifacts: dict[str, Any] | None,
) -> None:
    selected_policy = state["selected_deployment_policy"]
    frontier = state["model_frontier"]
    predictions = state["risk_report"]
    overview_metrics = build_overview_metrics(
        frontier,
        selected_policy,
        production_artifacts,
    )

    st.markdown(
        """
        <div style="padding: 1.2rem 1.4rem; border-radius: 18px;
                    background: linear-gradient(135deg, #edf6f9 0%, #fdfcf7 100%);
                    border: 1px solid #d6e3ea;">
          <h2 style="margin: 0 0 0.3rem 0; color: #144552;">Clinical ML Control Tower</h2>
          <p style="margin: 0; color: #3d5a68;">
            Este painel junta tuning, validação, robustez por fold, política clínica,
            relatórios de risco e serving do modelo de diabetes em uma única superfície operacional.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Modelo em foco", overview_metrics["model_name"])
    col2.metric("Politica de deploy", overview_metrics["policy_name"])
    col3.metric("Threshold atual", overview_metrics["threshold"])
    col4.metric("Refit com todos os dados", overview_metrics["refit_all_data"])

    frontier_view = prepare_frontier_view(frontier)
    if frontier_view is not None:
        st.subheader("Otimização vs. validação")
        st.dataframe(frontier_view, width="stretch", hide_index=True)

    risk_counts, ranked_predictions = build_risk_report_preview(predictions)
    if ranked_predictions is not None:
        st.subheader("Relatório de risco materializado")
        col1, col2 = st.columns([1.1, 1.4])
        with col1:
            if risk_counts is not None:
                st.bar_chart(risk_counts.set_index("Faixa"))
            else:
                st.metric(
                    "Score médio",
                    f"{float(predictions['prediction_proba'].mean()):.1%}",
                )
        with col2:
            st.dataframe(
                ranked_predictions,
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Baixar relatório CSV",
                data=ranked_predictions.to_csv(index=False).encode("utf-8"),
                file_name="risk_report.csv",
                mime="text/csv",
            )
    else:
        st.info(
            "Nenhum relatório de risco materializado foi encontrado. Rode a inferência batch para gerar um relatório ordenado por risco."
        )


def render_actions_tab() -> None:
    st.header("Ações do Pipeline")
    st.caption(
        "Os botões abaixo executam as pipelines reais do projeto e recarregam os artefatos do painel quando o comando termina."
    )
    columns = st.columns(len(PIPELINE_ACTIONS))
    for column, (label, command, caption) in zip(columns, PIPELINE_ACTIONS):
        if column.button(label, width="stretch", key=f"pipeline-action::{label}"):
            _run_cli(command, PROJECT_ROOT, caption)


def render_model_comparison_tab(
    frontier: pd.DataFrame | None,
    model_selection_scorecard: pd.DataFrame | None,
    split_comparison_report: pd.DataFrame | None,
    nested_cv_audit: pd.DataFrame | None,
    report: dict[str, Any] | None,
) -> None:
    st.header("Comparação de Modelos")
    if frontier is None or frontier.empty:
        st.warning("Rode `kedro run` para gerar `model_frontier.parquet`.")
        return

    chart_frame = prepare_frontier_view(frontier)
    if chart_frame is None:
        st.warning("Rode `kedro run` para gerar a comparação dos modelos.")
        return
    st.dataframe(chart_frame, width="stretch", hide_index=True)

    metric_options = [
        column
        for column in [
            "best_cv_score",
            "validation_roc_auc",
            "validation_recall",
            "validation_precision",
            "validation_f1",
        ]
        if column in chart_frame.columns
    ]
    if metric_options:
        selected_metric = st.selectbox("Métrica para comparar", metric_options)
        st.bar_chart(chart_frame.set_index("model_name")[[selected_metric]])

    summary_frame = build_test_report_summary(report, frontier)
    if not summary_frame.empty:
        st.subheader("Leitura no split de teste")
        st.dataframe(summary_frame, width="stretch", hide_index=True)

    scorecard_view = prepare_model_selection_scorecard_view(model_selection_scorecard)
    if scorecard_view is not None:
        st.subheader("Scorecard composto de seleção")
        st.caption(
            "A seleção do modelo combina qualidade probabilística na validação com utilidade da política clínica de deploy, em vez de depender só de ROC AUC."
        )
        st.dataframe(scorecard_view, width="stretch", hide_index=True)

    split_comparison_view = prepare_split_comparison_view(split_comparison_report)
    if split_comparison_view is not None:
        st.subheader("Comparação treino vs validação vs teste")
        st.caption(
            "Este artefato usa o modelo de desenvolvimento selecionado e o threshold final de deploy para mostrar gaps entre splits e ler sobreajuste com mais honestidade."
        )
        st.dataframe(split_comparison_view, width="stretch", hide_index=True)
        if {"split", "roc_auc", "brier"}.issubset(split_comparison_view.columns):
            st.line_chart(
                split_comparison_view.set_index("split")[["roc_auc", "brier"]]
            )

    nested_cv_view = prepare_nested_cv_audit_view(nested_cv_audit)
    if nested_cv_view is not None:
        st.subheader("Auditoria com outer nested CV")
        st.caption(
            "Cada fold externo repete seleção de features, tuning, política e refit antes de medir fora da amostra."
        )
        st.dataframe(nested_cv_view, width="stretch", hide_index=True)


def render_robustness_tab(  # noqa: PLR0913, PLR0915
    feature_selection_frontier: pd.DataFrame | None,
    feature_selection_stability: pd.DataFrame | None,
    feature_selection_manifest: dict[str, Any] | None,
    cv_fold_metrics: pd.DataFrame | None,
    cv_metric_summary: pd.DataFrame | None,
    bootstrap_metric_intervals: pd.DataFrame | None,
    permutation_feature_importance: pd.DataFrame | None,
    perturbation_sensitivity_audit: pd.DataFrame | None,
    perturbation_sensitivity_summary: pd.DataFrame | None,
) -> None:
    st.header("Robustez do Modelo")
    st.caption(
        "Aqui a pergunta não é apenas quem venceu, mas quão estável esse comportamento fica entre folds e quando fazemos pequenas perturbações nos inputs."
    )
    if cv_fold_metrics is None or cv_fold_metrics.empty:
        st.warning(
            "Rode `kedro run` novamente para gerar os artefatos de robustez por fold."
        )
        return

    st.info(
        "As métricas por fold abaixo usam a política e o threshold de deploy selecionados, não um cutoff fixo de 0.50."
    )

    feature_selection_view = prepare_feature_selection_frontier_view(
        feature_selection_frontier
    )
    if feature_selection_view is not None:
        st.subheader("Seleção de features")
        if feature_selection_manifest:
            selected_features = feature_selection_manifest.get(
                "selected_feature_names_text"
            ) or ", ".join(feature_selection_manifest.get("selected_feature_names", []))
            st.caption(
                "A seleção roda só no split de treino com CV interna, "
                "usa uma regra de 1 erro-padrão para preferir parcimônia, "
                "e deixa a validação oficial livre para comparar modelos."
            )
            if selected_features:
                st.markdown(f"**Features escolhidas:** {selected_features}")
        st.dataframe(feature_selection_view, width="stretch", hide_index=True)

    feature_selection_stability_view = prepare_feature_selection_stability_view(
        feature_selection_stability
    )
    if feature_selection_stability_view is not None:
        st.subheader("Estabilidade da seleção")
        st.dataframe(
            feature_selection_stability_view,
            width="stretch",
            hide_index=True,
        )

    metric_options = list(ROBUSTNESS_METRIC_LABELS)
    metric_name = st.selectbox(
        "Métrica por fold",
        metric_options,
        format_func=lambda name: ROBUSTNESS_METRIC_LABELS.get(name, name),
    )
    chart = build_robustness_chart_frame(cv_fold_metrics, metric_name)
    st.line_chart(chart)
    st.caption(f"Visualizando: {ROBUSTNESS_METRIC_LABELS.get(metric_name, metric_name)}")

    summary = prepare_cv_metric_summary_view(cv_metric_summary)
    if summary is not None:
        st.subheader("Resumo de variação relativa")
        st.dataframe(summary, width="stretch", hide_index=True)

    bootstrap_view = prepare_bootstrap_metric_intervals_view(bootstrap_metric_intervals)
    if bootstrap_view is not None:
        st.subheader("Intervalos de confiança por bootstrap")
        st.caption(
            "Este quadro mostra a largura do intervalo de confiança das métricas no split de validação. Intervalos estreitos sugerem comportamento mais confiável."
        )
        st.dataframe(bootstrap_view, width="stretch", hide_index=True)

    importance_view = prepare_permutation_feature_importance_view(
        permutation_feature_importance
    )
    if importance_view is not None:
        st.subheader("Permutation feature importance")
        importance_metric = st.selectbox(
            "Métrica da importance",
            sorted(importance_view["metric_name"].dropna().unique().tolist()),
            format_func=lambda name: IMPORTANCE_METRIC_LABELS.get(name, name),
        )
        metric_view = importance_view[
            importance_view["metric_name"] == importance_metric
        ].copy()
        if not metric_view.empty:
            st.bar_chart(
                metric_view.set_index("feature_name")[["importance_mean"]]
            )
            st.dataframe(metric_view, width="stretch", hide_index=True)

    sensitivity_summary = prepare_perturbation_sensitivity_summary_view(
        perturbation_sensitivity_summary
    )
    if sensitivity_summary is not None:
        st.subheader("Auditoria de resposta proporcional")
        too_insensitive = int(
            (sensitivity_summary["sensitivity_label"] == "too_insensitive").sum()
        )
        proportional = int(
            (sensitivity_summary["sensitivity_label"] == "proportional").sum()
        )
        too_sensitive = int(
            (sensitivity_summary["sensitivity_label"] == "too_sensitive").sum()
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Features pouco responsivas", str(too_insensitive))
        col2.metric("Features proporcionais", str(proportional))
        col3.metric("Features super-reativas", str(too_sensitive))
        st.caption(
            "Pouco responsiva significa que o score quase não reage a mudanças no input; super-reativa significa que reage além do proporcional."
        )
        st.dataframe(sensitivity_summary, width="stretch", hide_index=True)

    if (
        perturbation_sensitivity_audit is not None
        and not perturbation_sensitivity_audit.empty
    ):
        st.subheader("Detalhe por feature e perturbação")
        st.dataframe(
            perturbation_sensitivity_audit.sort_values(
                by=["mean_sensitivity_ratio", "decision_flip_rate"],
                ascending=False,
            ),
            width="stretch",
            hide_index=True,
        )


def render_policy_tab(
    threshold_metrics: pd.DataFrame | None,
    selected_policy: dict[str, Any] | None,
) -> None:
    st.header("Políticas Clínicas de FP vs FN")
    if threshold_metrics is None or threshold_metrics.empty:
        st.warning(
            "Nenhuma tabela de política encontrada. Rode `kedro run` para materializar o comparativo de thresholds."
        )
        return

    view = prepare_threshold_metrics_view(threshold_metrics)
    if view is not None:
        st.dataframe(view, width="stretch", hide_index=True)

    active = select_active_policy_row(threshold_metrics, selected_policy)
    if active is None:
        st.warning("Nenhuma política ativa encontrada.")
        return

    st.info(
        f"Política ativa: {active['policy_label']} com threshold {float(active['threshold']):.2f}. "
        "Neste fluxo clínico, estamos assumindo que deixar passar um caso positivo é mais grave do que gerar um alerta extra."
    )
    st.caption(
        "Por isso, a política vencedora reduz o threshold para capturar mais casos positivos, mesmo aceitando mais falsos positivos."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Recall", _format_pct(active["recall"]))
    col2.metric("Precisão", _format_pct(active["precision"]))
    col3.metric("Custo esperado", f"{float(active['expected_cost']):.1f}")

    _render_confusion_matrix(
        tp=int(active["tp"]),
        fp=int(active["fp"]),
        tn=int(active["tn"]),
        fn=int(active["fn"]),
        title=f"Matriz da política {active['policy_label']}",
    )


def render_inference_tab(
    production_artifacts: dict[str, Any] | None,
    selected_policy: dict[str, Any] | None,
) -> None:
    st.header("Predição ao Vivo")
    if production_artifacts is None:
        st.error(
            "Artefatos de produção não encontrados. Rode `kedro run` ou use o container."
        )
        return

    col1, col2 = st.columns(2)
    inputs: dict[str, float] = {}
    half = len(FEATURE_DESCRIPTIONS) // 2
    for index, (feature, (label, minimum, maximum, default)) in enumerate(
        FEATURE_DESCRIPTIONS.items()
    ):
        column = col1 if index < half else col2
        inputs[feature] = column.number_input(
            label,
            min_value=minimum,
            max_value=maximum,
            value=default,
            step=1.0 if feature in ("Pregnancies", "Age") else 0.1,
            key=f"inference::{feature}",
        )

    if st.button("Gerar score clínico", type="primary", width="stretch"):
        try:
            live_report = serving_runtime.run_online_inference(
                [inputs],
                output_dataset="risk_report",
            )
        except Exception as exc:
            logger.exception("Dashboard live inference failed")
            st.error("Falha ao executar a pipeline Kedro de inferência.")
            st.exception(exc)
            return

        row = live_report.iloc[0]
        probability = float(row.get("prediction_proba", 0.0))
        risk_score = float(row.get("risk_score", probability * 100.0))
        risk_band = row.get("risk_band", "Sem faixa")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Predição", "Positivo" if str(row["prediction"]) == "1" else "Negativo"
        )
        col2.metric("Probabilidade", f"{probability:.1%}")
        col3.metric("Faixa de risco", str(risk_band))
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(f"Score de risco: {risk_score:.1f} / 100")

        policy_catalog = []
        if selected_policy is not None:
            policy_catalog = list(selected_policy.get("policy_catalog", []))
        elif production_artifacts["model"].get("policy_catalog"):
            policy_catalog = list(production_artifacts["model"]["policy_catalog"])
        policy_view = _policy_decisions(probability, policy_catalog)
        if not policy_view.empty:
            st.subheader("Como cada política reagiria a este caso")
            st.dataframe(policy_view, width="stretch", hide_index=True)
        st.subheader("Relatório deste caso")
        st.dataframe(live_report, width="stretch", hide_index=True)
        st.download_button(
            "Baixar relatório deste caso",
            data=live_report.to_csv(index=False).encode("utf-8"),
            file_name="live_risk_report.csv",
            mime="text/csv",
        )


def render_manifests_tab(state: dict[str, Any]) -> None:  # noqa: PLR0912, PLR0915
    st.header("Manifestos e Contratos")

    training_manifest = state.get("training_run_manifest")
    split_strategy_report = state.get("split_strategy_report")
    serving_manifest = state.get("serving_manifest")
    inference_contract = state.get("inference_contract")
    feature_selection_manifest = state.get("feature_selection_manifest")
    model_selection_scorecard = state.get("model_selection_scorecard")
    split_comparison_report = state.get("split_comparison_report")
    nested_cv_audit = state.get("nested_cv_audit")
    modelling_design_audit = state.get("modelling_design_audit")

    if training_manifest:
        st.subheader("Resumo do treino")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Modelo vencedor",
            _display_model_name(
                training_manifest.get("selected_model_name"),
                training_manifest.get("selected_class_path"),
            ),
        )
        col2.metric(
            "Classe",
            _class_short_name(training_manifest.get("selected_class_path")),
        )
        col3.metric(
            "Threshold escolhido",
            f"{float(training_manifest.get('selected_threshold', 0.0)):.2f}",
        )
        col4.metric(
            "Política escolhida",
            _policy_display_name(
                {
                    "decision_policy_name": training_manifest.get(
                        "selected_policy_name"
                    ),
                    "policy_catalog": training_manifest.get("threshold_metrics", []),
                }
            ),
        )
        selected_features = (
            (training_manifest.get("feature_selection_manifest") or {}).get(
                "selected_feature_names_text"
            )
            or ", ".join(
                (training_manifest.get("feature_selection_manifest") or {}).get(
                    "selected_feature_names",
                    [],
                )
            )
        )
        if selected_features:
            st.caption(f"Features finais do modelo vencedor: {selected_features}")
        selection_metric = training_manifest.get("selection_metric")
        selection_score = training_manifest.get("selection_score")
        if selection_metric is not None and selection_score is not None:
            st.caption(
                f"Critério de seleção: {selection_metric} = {float(selection_score):.3f}"
            )

    if serving_manifest:
        st.subheader("Resumo de serving")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Modelo em produção",
            _display_model_name(
                training_manifest.get("selected_model_name") if training_manifest else None,
                serving_manifest.get("class_path"),
            ),
        )
        col2.metric(
            "Classe",
            _class_short_name(serving_manifest.get("class_path")),
        )
        col3.metric(
            "Threshold ativo",
            f"{float(serving_manifest.get('decision_threshold', 0.0)):.2f}",
        )
        col4.metric(
            "Features ativas",
            str(len((inference_contract or {}).get("model_feature_columns", []))),
        )
        model_features = ", ".join(
            (inference_contract or {}).get("model_feature_columns", [])
        )
        if model_features:
            st.caption(f"Features usadas em produção: {model_features}")

    if inference_contract:
        st.subheader("Resumo do contrato de inferência")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Campos de entrada", str(len(inference_contract.get("input_fields", []))))
        col2.metric(
            "Features usadas pelo modelo",
            str(len(inference_contract.get("model_feature_columns", []))),
        )
        col3.metric(
            "Features derivadas usadas",
            str(len(inference_contract.get("derived_feature_columns", []))),
        )
        col4.metric("Saídas", str(len(inference_contract.get("outputs", []))))
        input_names = ", ".join(field["name"] for field in inference_contract.get("input_fields", []))
        model_features = ", ".join(inference_contract.get("model_feature_columns", []))
        derived_features = ", ".join(inference_contract.get("derived_feature_columns", []))
        output_names = ", ".join(output["name"] for output in inference_contract.get("outputs", []))
        st.caption(f"Entradas esperadas: {input_names or '-'}")
        st.caption(f"Features do modelo: {model_features or '-'}")
        st.caption(f"Features derivadas usadas: {derived_features or 'Nenhuma'}")
        st.caption(f"Saídas produzidas: {output_names or '-'}")

    if split_strategy_report is not None and not split_strategy_report.empty:
        st.subheader("Resumo da estrategia de split")
        primary_row = split_strategy_report.sort_values("split_order").iloc[0]
        ratio_text = " | ".join(
            f"{row['split_name']}: {float(row['observed_ratio']):.1%}"
            for _, row in split_strategy_report.sort_values("split_order").iterrows()
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Estratégia aplicada",
            str(primary_row.get("resolved_strategy_kind", "-")),
        )
        col2.metric(
            "Estratificação",
            "Sim" if bool(primary_row.get("stratified_flag")) else "Nao",
        )
        col3.metric(
            "Fallback aplicado",
            "Sim" if bool(primary_row.get("fallback_applied")) else "Nao",
        )
        col4.metric(
            "Linhas totais",
            str(int(split_strategy_report["rows"].sum())),
        )
        st.caption(f"Distribuição observada: {ratio_text}")
        st.dataframe(split_strategy_report, width="stretch", hide_index=True)

    if feature_selection_manifest:
        st.subheader("Resumo da seleção de features")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Features escolhidas",
            str(feature_selection_manifest.get("selected_feature_count", "-")),
        )
        col2.metric(
            "Métrica primária",
            str(feature_selection_manifest.get("primary_metric", "-")),
        )
        col3.metric(
            "Candidatas avaliadas",
            str(feature_selection_manifest.get("candidate_count", "-")),
        )
        selected_features = feature_selection_manifest.get(
            "selected_feature_names_text"
        ) or ", ".join(feature_selection_manifest.get("selected_feature_names", []))
        if selected_features:
            st.markdown(f"**Features finais:** {selected_features}")

    design_audit_view = prepare_modelling_design_audit_view(modelling_design_audit)
    if design_audit_view is not None:
        st.subheader("Checklist de leakage e desenho")
        st.caption(
            "Este checklist torna explícitos os principais guardrails contra definição circular, leakage e avaliação otimista."
        )
        st.dataframe(design_audit_view, width="stretch", hide_index=True)

    if model_selection_scorecard is not None and not model_selection_scorecard.empty:
        st.subheader("Seleção composta")
        st.dataframe(
            prepare_model_selection_scorecard_view(model_selection_scorecard),
            width="stretch",
            hide_index=True,
        )

    if split_comparison_report is not None and not split_comparison_report.empty:
        st.subheader("Artefato de comparação entre splits")
        st.dataframe(
            prepare_split_comparison_view(split_comparison_report),
            width="stretch",
            hide_index=True,
        )

    if nested_cv_audit is not None and not nested_cv_audit.empty:
        st.subheader("Artefato de nested CV")
        st.dataframe(
            prepare_nested_cv_audit_view(nested_cv_audit),
            width="stretch",
            hide_index=True,
        )

    sections = [
        ("Manifesto de treino", training_manifest),
        ("Manifesto de serving", serving_manifest),
        ("Contrato de inferência", inference_contract),
        ("Manifesto de seleção de features", feature_selection_manifest),
    ]
    rendered_any = False
    for title, payload in sections:
        if payload:
            rendered_any = True
            st.subheader(title)
            st.json(payload)
    if not rendered_any:
        st.warning(
            "Nenhum manifesto encontrado ainda. Rode `kedro run` para materializar treino, refit e contrato de inferência."
        )


def main() -> None:  # noqa: PLR0915
    st.set_page_config(
        page_title="Diabetes Clinical ML",
        page_icon="🩺",
        layout="wide",
    )
    st.title("Diabetes Clinical ML Dashboard")

    production_artifacts = load_production_artifacts()
    state = load_dashboard_state(production_artifacts)

    if state["test_report"] is None and production_artifacts is None:
        st.error(
            "Nenhum artefato encontrado. Rode `uv run kedro run` ou use `docker compose up --build`."
        )
        return

    tabs = st.tabs(
        [
            "Visão Geral",
            "Ações",
            "Modelos",
            "Robustez",
            "Políticas",
            "Manifestos",
            "Predição",
        ]
    )

    with tabs[0]:
        render_overview_tab(state, production_artifacts)
    with tabs[1]:
        render_actions_tab()
    with tabs[2]:
        render_model_comparison_tab(
            state["model_frontier"],
            state["model_selection_scorecard"],
            state["split_comparison_report"],
            state["nested_cv_audit"],
            state["test_report"],
        )
    with tabs[3]:
        render_robustness_tab(
            state["feature_selection_frontier"],
            state["feature_selection_stability"],
            state["feature_selection_manifest"],
            state["cv_fold_metrics"],
            state["cv_metric_summary"],
            state["bootstrap_metric_intervals"],
            state["permutation_feature_importance"],
            state["perturbation_sensitivity_audit"],
            state["perturbation_sensitivity_summary"],
        )
    with tabs[4]:
        render_policy_tab(
            state["threshold_metrics"],
            state["selected_deployment_policy"],
        )
    with tabs[5]:
        render_manifests_tab(state)
    with tabs[6]:
        render_inference_tab(
            production_artifacts,
            state["selected_deployment_policy"],
        )

    with st.sidebar:
        st.markdown("### Estado atual")
        sidebar_snapshot = build_sidebar_snapshot(
            serving_runtime.get_production_status(),
            production_artifacts,
            state.get("training_run_manifest"),
            state.get("selected_deployment_policy"),
        )
        if "selected_model_name" in sidebar_snapshot:
            st.metric("Modelo vencedor", sidebar_snapshot["selected_model_name"])
        if "selected_class_path" in sidebar_snapshot:
            st.markdown(
                f"**Classe vencedora:** `{_class_short_name(sidebar_snapshot['selected_class_path'])}`"
            )
        if "selected_policy_name" in sidebar_snapshot:
            st.markdown(f"**Política ativa:** `{sidebar_snapshot['selected_policy_name']}`")
        if "selected_threshold" in sidebar_snapshot:
            st.markdown(
                f"**Threshold selecionado:** `{float(sidebar_snapshot['selected_threshold']):.2f}`"
            )
        if "selected_feature_names_text" in sidebar_snapshot:
            st.caption(
                f"Features ativas no modelo vencedor: {sidebar_snapshot['selected_feature_names_text']}"
            )
        st.json(sidebar_snapshot["production_status"])
        if "class_path" in sidebar_snapshot:
            st.markdown(
                f"**Classe em produção:** `{_class_short_name(sidebar_snapshot['class_path'])}`"
            )
            st.markdown(
                f"**Train splits do artefato:** `{sidebar_snapshot['train_splits']}`"
            )
            st.markdown(
                f"**Threshold de produção:** `{sidebar_snapshot['decision_threshold']}`"
            )
        registry_state = load_registry_state()
        if registry_state.get("error"):
            st.warning(
                "Não foi possível carregar o histórico operacional. Veja os logs para detalhes."
            )
        if registry_state["model_registry"]:
            st.markdown(
                f"**Último registro de produção:** `{registry_state['model_registry'][0]['created_at']}`"
            )
        if registry_state["experiment_runs"]:
            st.markdown(
                f"**Último experimento:** `{registry_state['experiment_runs'][0]['created_at']}`"
            )
        st.divider()
        if st.button("Limpar cache e recarregar"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()


if __name__ == "__main__":
    main()
