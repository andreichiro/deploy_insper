"""Nodes de modelagem — Optuna, métricas e calibração guiados por YAML (class_path / callables)."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from itertools import combinations
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from insper_deploy_kedro.class_loading import load_callable, load_class
from insper_deploy_kedro.constants import SPLIT_COLUMN, ModelArtifact

logger = logging.getLogger(__name__)

_MODEL_SEQUENCE = ("baseline", "optimized", "xgboost")
_MINIMUM_CV_SPLITS = 2
_MINIMUM_CLASSES_FOR_CALIBRATION = 2
_ROBUSTNESS_STABLE_MAX = 10.0
_ROBUSTNESS_MODERATE_MAX = 20.0
_PROBABILITY_EPSILON = 1e-6
_MAX_EXHAUSTIVE_FEATURE_SELECTION_CANDIDATES = 2048
_MINIMUM_NESTED_OUTER_TRAIN_ROWS = 4
_MODEL_SELECTION_NEUTRAL_SCORE = 0.5
_KNOWN_LBFGSB_DEPRECATION = (
    r"scipy\.optimize: The `disp` and `iprint` options of the L-BFGS-B solver "
    r"are deprecated and will be removed in SciPy 1\.18\.0\."
)


def _resolved_model_init_args(
    class_path: str,
    init_args: dict[str, Any] | None,
) -> dict[str, Any]:
    """Inject safe runtime defaults without forcing them into YAML."""
    resolved = dict(init_args or {})
    if class_path.startswith("catboost.") and "allow_writing_files" not in resolved:
        resolved["allow_writing_files"] = False
    return resolved


def _get_feature_and_target_columns(
    columns: dict[str, list[str]],
) -> tuple[list[str], str]:
    """Extrai lista de features e nome da coluna target do config."""
    feature_columns = columns["categorical"] + columns["numerical"]
    target_column = columns["target"][0]
    return feature_columns, target_column


def _build_target_encoder(ml_runtime: dict[str, Any]) -> Any:
    cfg = ml_runtime["target_encoder"]
    cls = load_class(cfg["class_path"])
    return cls(**dict(cfg.get("init_args") or {}))


def _merged_model_init_args(model_artifact: ModelArtifact) -> dict[str, Any]:
    merged = dict(model_artifact.get("init_args", {}))
    if "best_params" in model_artifact:
        merged.update(model_artifact["best_params"])
    return _resolved_model_init_args(model_artifact["class_path"], merged)


def _model_candidates(
    baseline_model: ModelArtifact,
    optimized_model: ModelArtifact,
    xgboost_model: ModelArtifact,
) -> list[tuple[str, ModelArtifact]]:
    return [
        ("baseline", baseline_model),
        ("optimized", optimized_model),
        ("xgboost", xgboost_model),
    ]


def _build_estimator_from_artifact(model_artifact: ModelArtifact) -> Any:
    model_class = load_class(model_artifact["class_path"])
    return model_class(**_merged_model_init_args(model_artifact))


def _artifact_calibration_params(
    model_artifact: ModelArtifact,
) -> dict[str, Any] | None:
    calibration = dict(model_artifact.get("calibration") or {})
    if not calibration or not bool(calibration.get("enabled", True)):
        return None
    return {
        "enabled": True,
        "class_path": calibration.get(
            "class_path",
            "sklearn.calibration.CalibratedClassifierCV",
        ),
        "init_args": dict(calibration.get("init_args") or {}),
    }


def _fit_time_model_init_args(
    class_path: str,
    init_args: dict[str, Any] | None,
    y_encoded: np.ndarray | None = None,
) -> dict[str, Any]:
    resolved = _resolved_model_init_args(class_path, init_args)
    if class_path == "sklearn.linear_model.LogisticRegression":
        if resolved.get("penalty") == "none":
            resolved["penalty"] = None
    if class_path.startswith("xgboost.") and str(
        resolved.get("scale_pos_weight", "")
    ).lower() == "auto":
        if y_encoded is None:
            resolved["scale_pos_weight"] = 1.0
        else:
            positives = int(np.sum(y_encoded == 1))
            negatives = int(np.sum(y_encoded == 0))
            resolved["scale_pos_weight"] = (
                float(negatives / positives) if positives > 0 else 1.0
            )
    return resolved


def _search_param_is_active(
    sampled_params: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    active_if = dict(config.get("active_if") or {})
    for parent_name, allowed_values in active_if.items():
        if sampled_params.get(parent_name) not in list(allowed_values):
            return False
    return True


def _resolve_conditional_choices(
    sampled_params: dict[str, Any],
    config: dict[str, Any],
) -> list[Any]:
    parent_name = str(config["parent"])
    parent_value = sampled_params.get(parent_name)
    choices_by_parent = dict(config.get("choices_by_parent") or {})
    lookup_key = "null" if parent_value is None else str(parent_value)
    choices = choices_by_parent.get(lookup_key, choices_by_parent.get(parent_value))
    if not choices:
        raise ValueError(
            f"conditional_search_space_missing_choices:{parent_name}:{lookup_key}"
        )
    return list(choices)


def _sample_search_params(
    trial: optuna.Trial,
    search_space: dict[str, Any],
    class_path: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, config in search_space.items():
        if not _search_param_is_active(params, config):
            continue
        stype = str(config["type"])
        if stype == "int":
            params[name] = trial.suggest_int(name, config["low"], config["high"])
        elif stype == "float":
            params[name] = trial.suggest_float(
                name,
                config["low"],
                config["high"],
                log=config.get("log", False),
            )
        elif stype == "categorical":
            params[name] = trial.suggest_categorical(name, list(config["choices"]))
        elif stype == "conditional_categorical":
            params[name] = trial.suggest_categorical(
                name,
                _resolve_conditional_choices(params, config),
            )
        else:
            raise ValueError(f"unsupported_search_space_type:{stype}:{name}")
    if class_path == "sklearn.linear_model.LogisticRegression" and (
        "solver_penalty_combo" in params
    ):
        penalty_name, solver_name = str(params.pop("solver_penalty_combo")).split("__", 1)
        params["penalty"] = penalty_name
        params["solver"] = solver_name
    return _fit_time_model_init_args(class_path, params)


def _resolve_calibration_cv(
    y_encoded: np.ndarray,
    calibration_params: dict[str, Any],
) -> int | str | None:
    cal_kwargs = dict(calibration_params.get("init_args") or {})
    cv_value = cal_kwargs.get("cv", 5)
    if isinstance(cv_value, str):
        if cv_value.lower() == "prefit":
            return max(_safe_n_splits(y_encoded, _MINIMUM_CV_SPLITS), _MINIMUM_CV_SPLITS)
        return cv_value
    n_splits = _safe_n_splits(y_encoded, int(cv_value))
    return n_splits if n_splits >= _MINIMUM_CV_SPLITS else None


def _fit_estimator_with_optional_calibration(
    class_path: str,
    init_args: dict[str, Any],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    calibration_params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None]:
    resolved_init_args = _fit_time_model_init_args(class_path, init_args, y_train)
    model_class = load_class(class_path)

    if not calibration_params or not bool(calibration_params.get("enabled", True)):
        estimator = model_class(**resolved_init_args)
        with _suppress_known_training_warnings():
            estimator.fit(x_train, y_train)
        return estimator, resolved_init_args, None

    calibration_class_path = calibration_params.get(
        "class_path",
        "sklearn.calibration.CalibratedClassifierCV",
    )
    calibration_kwargs = dict(calibration_params.get("init_args") or {})
    calibration_cv = _resolve_calibration_cv(y_train, calibration_params)
    if calibration_cv is None:
        estimator = model_class(**resolved_init_args)
        with _suppress_known_training_warnings():
            estimator.fit(x_train, y_train)
        return estimator, resolved_init_args, {
            "enabled": False,
            "reason": "insufficient_class_support_for_calibration",
        }

    calibration_kwargs["cv"] = calibration_cv
    calibration_class = load_class(calibration_class_path)
    calibrated_estimator = calibration_class(
        estimator=model_class(**resolved_init_args),
        **calibration_kwargs,
    )
    with _suppress_known_training_warnings():
        calibrated_estimator.fit(x_train, y_train)
    return calibrated_estimator, resolved_init_args, {
        "enabled": True,
        "class_path": calibration_class_path,
        "init_args": calibration_kwargs,
    }


@contextmanager
def _suppress_known_training_warnings():
    """Silence upstream sklearn/scipy deprecations that we cannot fix locally."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=_KNOWN_LBFGSB_DEPRECATION,
            category=DeprecationWarning,
        )
        yield


def _predict_scores(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(features)[:, 1], dtype=float)
    predictions = estimator.predict(features)
    return np.asarray(predictions, dtype=float)


def _predict_labels(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)


def _calibration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    if (
        len(probabilities) == 0
        or len(np.unique(y_true)) < _MINIMUM_CLASSES_FOR_CALIBRATION
    ):
        return {
            "calibration_slope": 1.0,
            "calibration_intercept": 0.0,
            "calibration_slope_error": 0.0,
            "calibration_intercept_abs": 0.0,
        }

    logits = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1e6,
        fit_intercept=True,
        max_iter=1000,
        solver="lbfgs",
    )
    try:
        with _suppress_known_training_warnings():
            calibrator.fit(logits, y_true)
    except ValueError:
        return {
            "calibration_slope": 1.0,
            "calibration_intercept": 0.0,
            "calibration_slope_error": 0.0,
            "calibration_intercept_abs": 0.0,
        }

    slope = float(calibrator.coef_[0][0])
    intercept = float(calibrator.intercept_[0])
    return {
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "calibration_slope_error": float(abs(slope - 1.0)),
        "calibration_intercept_abs": float(abs(intercept)),
    }


def _probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    clipped_probabilities = _clip_probabilities(probabilities)
    roc_auc = float("nan")
    if len(np.unique(y_true)) >= _MINIMUM_CLASSES_FOR_CALIBRATION:
        try:
            roc_auc = float(roc_auc_score(y_true, clipped_probabilities))
        except ValueError:
            roc_auc = float("nan")
    return {
        "roc_auc": roc_auc,
        "brier": float(brier_score_loss(y_true, clipped_probabilities)),
        "log_loss": float(log_loss(y_true, clipped_probabilities, labels=[0, 1])),
        **_calibration_metrics(y_true, clipped_probabilities),
    }


def _threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = _predict_labels(probabilities, threshold)
    counts = _confusion_counts(y_true, y_pred)
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    total = tp + fp + tn + fn
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    accuracy = _safe_divide(tp + tn, total)
    false_positive_rate = _safe_divide(fp, fp + tn)
    false_negative_rate = _safe_divide(fn, fn + tp)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return {
        **counts,
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "positive_prediction_rate": _safe_divide(tp + fp, total),
        "positive_prevalence": _safe_divide(tp + fn, total),
        "tn_share": _safe_divide(tn, total),
        "fp_share": _safe_divide(fp, total),
        "fn_share": _safe_divide(fn, total),
        "tp_share": _safe_divide(tp, total),
        "mean_risk_score": float(np.mean(probabilities)) if len(probabilities) else 0.0,
    }


def _candidate_thresholds(policy_params: dict[str, Any]) -> list[float]:
    cfg = policy_params.get("candidate_thresholds") or {}
    start = float(cfg.get("start", 0.1))
    stop = float(cfg.get("stop", 0.9))
    step = float(cfg.get("step", 0.05))
    thresholds = np.arange(start, stop + step / 2, step, dtype=float)
    return [float(np.clip(value, 0.0, 1.0)) for value in thresholds]


def _expected_cost(
    metrics: dict[str, float],
    false_negative_cost: float,
    false_positive_cost: float,
) -> float:
    return metrics["fn"] * float(false_negative_cost) + metrics["fp"] * float(
        false_positive_cost
    )


def _merged_metric_payload(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, float]:
    threshold_metrics = _threshold_metrics(y_true, probabilities, threshold)
    threshold_metrics["expected_cost"] = _expected_cost(
        threshold_metrics,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    return {**threshold_metrics, **_probability_metrics(y_true, probabilities)}


def _metric_is_higher_better(metric_name: str) -> bool:
    return metric_name in {
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "tn_share",
        "tp_share",
        "mean_risk_score",
    }


def _bootstrap_bounds(
    values: Sequence[float],
    confidence_level: float,
) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    alpha = max(0.0, min(1.0, 1.0 - float(confidence_level)))
    lower = float(np.quantile(values, alpha / 2))
    upper = float(np.quantile(values, 1.0 - alpha / 2))
    return lower, upper, float(upper - lower)


def _feature_group_lookup(
    feature_groups: dict[str, list[str]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group_name, feature_names in (feature_groups or {}).items():
        for feature_name in feature_names:
            mapping[str(feature_name)] = str(group_name)
    return mapping


def _select_policy_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    policy_name: str,
    policy_cfg: dict[str, Any],
    policy_params: dict[str, Any],
) -> dict[str, Any]:
    strategy = policy_cfg.get("strategy", "fixed_threshold")
    label = policy_cfg.get("label", policy_name)
    description = policy_cfg.get("description", "")

    if strategy == "fixed_threshold":
        threshold = float(policy_cfg.get("threshold", 0.5))
        metrics = _threshold_metrics(y_true, probabilities, threshold)
        metrics["expected_cost"] = _expected_cost(
            metrics,
            float(policy_cfg.get("false_negative_cost", 1.0)),
            float(policy_cfg.get("false_positive_cost", 1.0)),
        )
        return {
            "policy_name": policy_name,
            "policy_label": label,
            "policy_description": description,
            "strategy": strategy,
            "constraint_satisfied": True,
            **metrics,
        }

    false_negative_cost = float(policy_cfg.get("false_negative_cost", 1.0))
    false_positive_cost = float(policy_cfg.get("false_positive_cost", 1.0))
    min_recall = float(policy_cfg.get("min_recall", 0.0))
    min_precision = float(policy_cfg.get("min_precision", 0.0))
    min_specificity = float(policy_cfg.get("min_specificity", 0.0))

    candidates: list[dict[str, Any]] = []
    for threshold in _candidate_thresholds(policy_params):
        metrics = _threshold_metrics(y_true, probabilities, threshold)
        metrics["expected_cost"] = _expected_cost(
            metrics,
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
        metrics["constraint_satisfied"] = (
            metrics["recall"] >= min_recall
            and metrics["precision"] >= min_precision
            and metrics["specificity"] >= min_specificity
        )
        candidates.append(metrics)

    eligible = [row for row in candidates if row["constraint_satisfied"]]
    ranked = eligible or candidates
    best = min(
        ranked,
        key=lambda row: (
            row["expected_cost"],
            row["false_negative_rate"],
            -row["recall"],
            row["threshold"],
        ),
    )
    return {
        "policy_name": policy_name,
        "policy_label": label,
        "policy_description": description,
        "strategy": strategy,
        **best,
    }


def _diagnostic_policy_threshold(  # noqa: PLR0913
    y_true: np.ndarray,
    probabilities: np.ndarray,
    policy_name: str,
    policy_cfg: dict[str, Any],
    policy_params: dict[str, Any],
    selected_deployment_policy: dict[str, Any] | None,
) -> float:
    if (
        selected_deployment_policy is not None
        and selected_deployment_policy.get("decision_threshold") is not None
    ):
        return float(selected_deployment_policy["decision_threshold"])
    if "threshold" in policy_cfg:
        return float(policy_cfg["threshold"])
    selected_policy = _select_policy_threshold(
        y_true,
        probabilities,
        policy_name,
        policy_cfg,
        policy_params,
    )
    return float(selected_policy["threshold"])


def _active_deployment_policy(
    decision_policy_params: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    policy_name = decision_policy_params.get("deployment_policy", "default_050")
    policy_cfg = dict((decision_policy_params.get("policies") or {}).get(policy_name, {}))
    return str(policy_name), policy_cfg


def _resolve_selection_components(
    selection_params: dict[str, Any],
) -> list[dict[str, Any]]:
    configured = list(selection_params.get("score_components") or [])
    if not configured:
        metric_name = str(selection_params.get("metric", "roc_auc"))
        configured = [
            {
                "metric": metric_name,
                "source": "validation",
                "weight": 1.0,
                "direction": (
                    "maximize" if _metric_is_higher_better(metric_name) else "minimize"
                ),
            }
        ]

    components: list[dict[str, Any]] = []
    for raw_component in configured:
        metric_name = str(raw_component["metric"])
        source = str(raw_component.get("source", "validation"))
        if source not in {"validation", "policy"}:
            raise ValueError(f"model_selection_invalid_component_source:{source}")
        direction = str(
            raw_component.get(
                "direction",
                "maximize" if _metric_is_higher_better(metric_name) else "minimize",
            )
        ).lower()
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"model_selection_invalid_component_direction:{direction}")
        weight = float(raw_component.get("weight", 1.0))
        if weight < 0.0:
            raise ValueError(f"model_selection_negative_component_weight:{metric_name}")
        components.append(
            {
                "metric": metric_name,
                "source": source,
                "weight": weight,
                "direction": direction,
                "label": str(raw_component.get("label", metric_name)),
            }
        )

    if np.isclose(sum(component["weight"] for component in components), 0.0):
        raise ValueError("model_selection_zero_total_weight")
    return components


def _selection_component_column(component: dict[str, Any]) -> str:
    return f"{component['source']}_{component['metric']}"


def _component_utility(
    metric_name: str,
    raw_value: float | int | None,
    *,
    direction: str,
) -> float:
    if raw_value is None or pd.isna(raw_value):
        return _MODEL_SELECTION_NEUTRAL_SCORE

    value = float(raw_value)
    if metric_name in {
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
        "brier",
        "calibration_slope_error",
        "calibration_intercept_abs",
        "tn_share",
        "fp_share",
        "fn_share",
        "tp_share",
        "positive_prediction_rate",
        "positive_prevalence",
    }:
        clipped = float(np.clip(value, 0.0, 1.0))
        return clipped if direction == "maximize" else 1.0 - clipped
    if metric_name in {"log_loss", "expected_cost", "expected_cost_per_sample", "mape"}:
        scaled = 1.0 / (1.0 + max(value, 0.0))
        return scaled if direction == "minimize" else 1.0 - scaled
    if metric_name == "r2":
        shifted = float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))
        return shifted if direction == "maximize" else 1.0 - shifted
    if metric_name == "mean_risk_score":
        shifted = float(np.clip(value / 100.0, 0.0, 1.0))
        return shifted if direction == "maximize" else 1.0 - shifted

    scaled = 1.0 / (1.0 + abs(value))
    return scaled if direction == "minimize" else 1.0 - scaled


def _selection_component_value(
    component: dict[str, Any],
    validation_payload: dict[str, Any],
    policy_payload: dict[str, Any],
) -> float | None:
    payload = validation_payload if component["source"] == "validation" else policy_payload
    return payload.get(component["metric"])


def _selection_composite_score(
    validation_payload: dict[str, Any],
    policy_payload: dict[str, Any],
    selection_params: dict[str, Any],
) -> tuple[float, dict[str, float], dict[str, float]]:
    components = _resolve_selection_components(selection_params)
    total_weight = sum(component["weight"] for component in components)
    utility_map: dict[str, float] = {}
    raw_map: dict[str, float] = {}
    score = 0.0

    for component in components:
        column_name = _selection_component_column(component)
        raw_value = _selection_component_value(component, validation_payload, policy_payload)
        raw_map[column_name] = (
            float(raw_value) if raw_value is not None and not pd.isna(raw_value) else float("nan")
        )
        utility = _component_utility(
            component["metric"],
            raw_value,
            direction=component["direction"],
        )
        utility_map[f"{column_name}_utility"] = utility
        score += utility * float(component["weight"])

    return float(score / total_weight), utility_map, raw_map


def _policy_payload_for_model(
    model_artifact: ModelArtifact,
    split_frame: pd.DataFrame,
    target_column: str,
    decision_policy_params: dict[str, Any],
) -> dict[str, Any]:
    policy_name, policy_cfg = _active_deployment_policy(decision_policy_params)
    target_encoder = model_artifact["target_encoder"]
    estimator = model_artifact["estimator"]
    feature_columns = model_artifact["feature_columns"]
    y_true = target_encoder.transform(split_frame[target_column])
    probabilities = _predict_scores(estimator, split_frame[feature_columns])
    return _selection_payloads(
        y_true,
        probabilities,
        decision_policy_params,
    )[1]


def _selection_payloads(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    decision_policy_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_name, policy_cfg = _active_deployment_policy(decision_policy_params)
    selected_policy = _select_policy_threshold(
        y_true,
        probabilities,
        policy_name,
        policy_cfg,
        decision_policy_params,
    )
    false_negative_cost = float(policy_cfg.get("false_negative_cost", 1.0))
    false_positive_cost = float(policy_cfg.get("false_positive_cost", 1.0))
    validation_payload = _merged_metric_payload(
        y_true,
        probabilities,
        threshold=float(selected_policy["threshold"]),
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    validation_payload["selection_policy_threshold"] = float(selected_policy["threshold"])
    validation_payload["selection_policy_name"] = policy_name
    validation_payload["selection_policy_label"] = selected_policy.get(
        "policy_label",
        policy_name,
    )
    return {
        **validation_payload,
    }, {
        "selection_policy_name": policy_name,
        "selection_policy_label": selected_policy.get("policy_label", policy_name),
        "selection_policy_threshold": float(selected_policy["threshold"]),
        "policy_accuracy": float(selected_policy["accuracy"]),
        "policy_precision": float(selected_policy["precision"]),
        "policy_recall": float(selected_policy["recall"]),
        "policy_specificity": float(selected_policy["specificity"]),
        "policy_f1": float(selected_policy["f1"]),
        "policy_false_positive_rate": float(selected_policy["false_positive_rate"]),
        "policy_false_negative_rate": float(selected_policy["false_negative_rate"]),
        "policy_expected_cost": float(selected_policy["expected_cost"]),
        "policy_expected_cost_per_sample": float(
            selected_policy["expected_cost"] / max(len(y_true), 1)
        ),
        "policy_tp": int(selected_policy["tp"]),
        "policy_fp": int(selected_policy["fp"]),
        "policy_tn": int(selected_policy["tn"]),
        "policy_fn": int(selected_policy["fn"]),
    }


def _build_working_split_table(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
) -> pd.DataFrame:
    train_copy = train_frame.copy()
    validation_copy = validation_frame.copy()
    train_copy[SPLIT_COLUMN] = "train"
    validation_copy[SPLIT_COLUMN] = "validation"
    return pd.concat([train_copy, validation_copy], ignore_index=True)


def _split_outer_train_for_nested_audit(
    outer_train_frame: pd.DataFrame,
    target_column: str,
    validation_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(outer_train_frame) < _MINIMUM_NESTED_OUTER_TRAIN_ROWS:
        raise ValueError("nested_cv_outer_train_too_small")

    indices = np.arange(len(outer_train_frame))
    stratify: pd.Series | None = outer_train_frame[target_column]
    if outer_train_frame[target_column].nunique() < _MINIMUM_CLASSES_FOR_CALIBRATION:
        stratify = None
    elif outer_train_frame[target_column].value_counts().min() < _MINIMUM_CV_SPLITS:
        stratify = None

    try:
        train_idx, valid_idx = train_test_split(
            indices,
            test_size=float(validation_fraction),
            random_state=int(random_state),
            stratify=stratify,
        )
    except ValueError:
        train_idx, valid_idx = train_test_split(
            indices,
            test_size=float(validation_fraction),
            random_state=int(random_state),
            stratify=None,
        )

    return (
        outer_train_frame.iloc[train_idx].reset_index(drop=True),
        outer_train_frame.iloc[valid_idx].reset_index(drop=True),
    )


def _scaled_nested_optimization_params(
    optimization_params: dict[str, Any],
    nested_cv_params: dict[str, Any],
) -> dict[str, Any]:
    scaled = dict(optimization_params)
    trial_overrides = dict(nested_cv_params.get("trial_overrides") or {})
    class_path = str(optimization_params.get("class_path", ""))
    override = None
    for key, value in trial_overrides.items():
        if key in class_path.lower():
            override = int(value)
            break
    if override is not None:
        scaled["n_trials"] = override
        return scaled

    trial_scaling = float(nested_cv_params.get("trial_scaling", 1.0))
    min_trials = int(nested_cv_params.get("min_trials", 4))
    original_trials = int(optimization_params.get("n_trials", min_trials))
    scaled["n_trials"] = max(min_trials, int(round(original_trials * trial_scaling)))
    return scaled


def _safe_n_splits(y_encoded: np.ndarray, desired_splits: int) -> int:
    if len(y_encoded) == 0:
        return 0

    _, counts = np.unique(y_encoded, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    return max(0, min(int(desired_splits), min_count))


def _feature_selection_metric_is_higher_better(metric_name: str) -> bool:
    return metric_name in {"roc_auc"}


def _build_feature_selection_blocks(
    columns: dict[str, list[str]],
    feature_selection_params: dict[str, Any],
) -> tuple[list[str], dict[str, list[str]]]:
    feature_order, _target = _get_feature_and_target_columns(columns)
    configured_blocks = feature_selection_params.get("feature_blocks") or {}
    if not configured_blocks:
        return feature_order, {feature_name: [feature_name] for feature_name in feature_order}

    blocks: dict[str, list[str]] = {}
    known_features = set(feature_order)
    for block_name, feature_names in configured_blocks.items():
        resolved = [str(feature_name) for feature_name in feature_names]
        unknown = sorted(set(resolved) - known_features)
        if unknown:
            raise ValueError(
                f"feature_selection_unknown_features:{block_name}:{unknown}"
            )
        blocks[str(block_name)] = [
            feature_name for feature_name in feature_order if feature_name in resolved
        ]
    return feature_order, blocks


def _build_feature_selector_estimator(
    feature_selection_params: dict[str, Any],
    selector_model_params: dict[str, Any] | None,
    y_train: np.ndarray,
) -> Any:
    selector_source = str(feature_selection_params.get("selector_source", "explicit"))
    selector_cfg: dict[str, Any]
    if selector_source == "candidate_model" and selector_model_params is not None:
        selector_cfg = dict(selector_model_params.get("feature_selection_selector") or {})
        if "class_path" not in selector_cfg:
            selector_cfg["class_path"] = selector_model_params.get(
                "class_path",
                "sklearn.linear_model.LogisticRegression",
            )
        if "init_args" not in selector_cfg:
            selector_cfg["init_args"] = dict(selector_model_params.get("init_args") or {})
    else:
        selector_cfg = dict(feature_selection_params.get("selector_model") or {})
    class_path = selector_cfg.get("class_path", "sklearn.linear_model.LogisticRegression")
    model_class = load_class(class_path)
    return model_class(
        **_fit_time_model_init_args(class_path, selector_cfg.get("init_args"), y_train)
    )


def _feature_selection_metrics(
    feature_selection_params: dict[str, Any],
) -> tuple[str, list[str]]:
    primary_metric = str(feature_selection_params.get("primary_metric", "brier"))
    secondary_metrics = [
        str(metric_name)
        for metric_name in feature_selection_params.get(
            "secondary_metrics",
            [
                "roc_auc",
                "log_loss",
                "calibration_slope_error",
                "calibration_intercept_abs",
            ],
        )
    ]
    metrics = list(dict.fromkeys([primary_metric, *secondary_metrics]))
    supported_metrics = {
        "roc_auc",
        "brier",
        "log_loss",
        "calibration_slope",
        "calibration_intercept",
        "calibration_slope_error",
        "calibration_intercept_abs",
    }
    unsupported = sorted(
        metric_name for metric_name in metrics if metric_name not in supported_metrics
    )
    if unsupported:
        raise ValueError(
            "feature_selection_unsupported_metrics:"
            + ",".join(unsupported)
        )
    return primary_metric, secondary_metrics


def _feature_selection_cv(
    y_encoded: np.ndarray,
    ml_runtime: dict[str, Any],
    feature_selection_params: dict[str, Any],
) -> Any | None:
    cv_cfg = ml_runtime["cross_validation"]
    cv_overrides = dict(feature_selection_params.get("cv") or {})
    desired_splits = int(cv_overrides.pop("n_splits", cv_cfg.get("init_args", {}).get("n_splits", 5)))
    n_splits = _safe_n_splits(y_encoded, desired_splits)
    if n_splits < _MINIMUM_CV_SPLITS:
        return None
    cv_class = load_class(cv_cfg["class_path"])
    cv_kwargs = {**dict(cv_cfg.get("init_args") or {}), **cv_overrides, "n_splits": n_splits}
    return cv_class(**cv_kwargs)


def _enumerate_feature_selection_candidates(
    feature_blocks: dict[str, list[str]],
    feature_order: list[str],
    feature_selection_params: dict[str, Any],
) -> list[dict[str, Any]]:
    block_names = list(feature_blocks.keys())
    always_include = [
        str(block_name)
        for block_name in feature_selection_params.get("always_include_blocks", [])
    ]
    unknown_blocks = sorted(set(always_include) - set(block_names))
    if unknown_blocks:
        raise ValueError(
            "feature_selection_unknown_always_include_blocks:"
            + ",".join(unknown_blocks)
        )
    required_blocks_map = {
        str(block_name): [str(required) for required in required_blocks]
        for block_name, required_blocks in (
            feature_selection_params.get("required_blocks") or {}
        ).items()
    }
    unknown_required_blocks = sorted(set(required_blocks_map) - set(block_names))
    if unknown_required_blocks:
        raise ValueError(
            "feature_selection_unknown_required_blocks:"
            + ",".join(unknown_required_blocks)
        )
    for block_name, required_blocks in required_blocks_map.items():
        unknown_children = sorted(set(required_blocks) - set(block_names))
        if unknown_children:
            raise ValueError(
                f"feature_selection_unknown_required_children:{block_name}:{unknown_children}"
            )

    free_blocks = [block_name for block_name in block_names if block_name not in always_include]
    min_blocks = int(feature_selection_params.get("min_blocks", 1))
    max_blocks = int(feature_selection_params.get("max_blocks", len(block_names)))
    min_free = max(0, min_blocks - len(always_include))
    max_free = min(len(free_blocks), max_blocks - len(always_include))
    if max_free < min_free:
        raise ValueError("feature_selection_invalid_block_bounds")

    candidate_records: list[dict[str, Any]] = []
    candidate_counter = 0
    seen_candidates: set[tuple[str, ...]] = set()
    candidate_limit = int(
        feature_selection_params.get(
            "max_candidates",
            _MAX_EXHAUSTIVE_FEATURE_SELECTION_CANDIDATES,
        )
    )

    def expand_required_blocks(block_combo: list[str]) -> list[str]:
        expanded = list(dict.fromkeys(block_combo))
        changed = True
        while changed:
            changed = False
            for block_name in list(expanded):
                for required_block in required_blocks_map.get(block_name, []):
                    if required_block not in expanded:
                        expanded.append(required_block)
                        changed = True
        return [block_name for block_name in block_names if block_name in expanded]

    for free_count in range(min_free, max_free + 1):
        for combo in combinations(free_blocks, free_count):
            block_combo = expand_required_blocks([*always_include, *combo])
            block_combo_key = tuple(block_combo)
            if block_combo_key in seen_candidates:
                continue
            seen_candidates.add(block_combo_key)
            feature_names = [
                feature_name
                for feature_name in feature_order
                if any(feature_name in feature_blocks[block_name] for block_name in block_combo)
            ]
            if not feature_names:
                continue
            candidate_counter += 1
            if candidate_counter > candidate_limit:
                raise ValueError(
                    f"feature_selection_candidate_limit_exceeded:{candidate_limit}"
                )
            candidate_records.append(
                {
                    "candidate_id": candidate_counter,
                    "block_names": list(block_combo),
                    "block_names_text": " + ".join(block_combo),
                    "feature_names": list(feature_names),
                    "feature_names_text": ", ".join(feature_names),
                    "feature_count": len(feature_names),
                    "block_count": len(block_combo),
                }
            )
    return candidate_records


def _feature_selection_sort_key(
    row: pd.Series,
    primary_metric: str,
    secondary_metrics: list[str],
    prefer_fewer_features: bool,
) -> tuple[Any, ...]:
    ordering: list[Any] = []
    if prefer_fewer_features:
        ordering.append(int(row["feature_count"]))
    for metric_name in secondary_metrics:
        value = float(row.get(f"mean_{metric_name}", np.nan))
        if pd.isna(value):
            ordering.append(float("inf"))
            continue
        ordering.append(
            -value
            if _feature_selection_metric_is_higher_better(metric_name)
            else value
        )
    primary_value = float(row.get(f"mean_{primary_metric}", np.nan))
    ordering.append(
        -primary_value
        if _feature_selection_metric_is_higher_better(primary_metric)
        else primary_value
    )
    ordering.append(int(row["candidate_id"]))
    return tuple(ordering)


def _build_feature_selection_summary(
    fold_results: pd.DataFrame,
    primary_metric: str,
    secondary_metrics: list[str],
    prefer_fewer_features: bool,
) -> pd.DataFrame:
    grouped = (
        fold_results.groupby(
            [
                "candidate_id",
                "block_names_text",
                "feature_names_text",
                "feature_count",
                "block_count",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_brier=("brier", "mean"),
            std_brier=("brier", "std"),
            mean_log_loss=("log_loss", "mean"),
            std_log_loss=("log_loss", "std"),
            mean_calibration_slope_error=("calibration_slope_error", "mean"),
            std_calibration_slope_error=("calibration_slope_error", "std"),
            mean_calibration_intercept_abs=("calibration_intercept_abs", "mean"),
            std_calibration_intercept_abs=("calibration_intercept_abs", "std"),
            folds=("fold_id", "nunique"),
        )
        .sort_values("candidate_id", kind="mergesort")
        .reset_index(drop=True)
    )
    for metric_name in [
        "roc_auc",
        "brier",
        "log_loss",
        "calibration_slope_error",
        "calibration_intercept_abs",
    ]:
        std_column = f"std_{metric_name}"
        grouped[std_column] = pd.to_numeric(
            grouped[std_column], errors="coerce"
        ).fillna(0.0)
        grouped[f"sem_{metric_name}"] = grouped[std_column] / np.sqrt(
            grouped["folds"].clip(lower=1)
        )

    higher_is_better = _feature_selection_metric_is_higher_better(primary_metric)
    best_row = grouped.sort_values(
        by=f"mean_{primary_metric}",
        ascending=not higher_is_better,
        kind="mergesort",
    ).iloc[0]
    best_mean = float(best_row[f"mean_{primary_metric}"])
    best_sem = float(best_row[f"sem_{primary_metric}"])
    if higher_is_better:
        grouped["within_one_se"] = (
            pd.to_numeric(grouped[f"mean_{primary_metric}"], errors="coerce")
            >= (best_mean - best_sem)
        )
    else:
        grouped["within_one_se"] = (
            pd.to_numeric(grouped[f"mean_{primary_metric}"], errors="coerce")
            <= (best_mean + best_sem)
        )

    candidate_pool = grouped[grouped["within_one_se"]].copy()
    if candidate_pool.empty:
        candidate_pool = grouped.copy()
    selected_index = min(
        candidate_pool.index.tolist(),
        key=lambda idx: _feature_selection_sort_key(
            grouped.loc[idx],
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            prefer_fewer_features=prefer_fewer_features,
        ),
    )
    grouped["selected_flag"] = 0
    grouped.loc[selected_index, "selected_flag"] = 1
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    grouped["primary_metric_name"] = primary_metric
    return grouped


def _build_feature_selection_stability(
    fold_results: pd.DataFrame,
    candidate_lookup: dict[int, dict[str, Any]],
    primary_metric: str,
    secondary_metrics: list[str],
    prefer_fewer_features: bool,
) -> pd.DataFrame:
    winner_rows: list[pd.Series] = []
    for _fold_id, group in fold_results.groupby("fold_id", sort=True):
        winner_index = min(
            group.index.tolist(),
            key=lambda idx: _feature_selection_sort_key(
                group.loc[idx],
                primary_metric=primary_metric,
                secondary_metrics=secondary_metrics,
                prefer_fewer_features=prefer_fewer_features,
            ),
        )
        winner_rows.append(group.loc[winner_index])

    if not winner_rows:
        return pd.DataFrame()

    winners = pd.DataFrame(winner_rows)
    winner_count = max(1, winners["fold_id"].nunique())
    candidate_counts: dict[int, int] = {}
    block_counts: dict[str, int] = {}
    feature_counts: dict[str, int] = {}

    for candidate_id in winners["candidate_id"].astype(int):
        candidate = candidate_lookup[candidate_id]
        candidate_counts[candidate_id] = candidate_counts.get(candidate_id, 0) + 1
        for block_name in candidate["block_names"]:
            block_counts[block_name] = block_counts.get(block_name, 0) + 1
        for feature_name in candidate["feature_names"]:
            feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1

    rows: list[dict[str, Any]] = []
    for candidate_id, candidate in candidate_lookup.items():
        winner_hits = candidate_counts.get(candidate_id, 0)
        rows.append(
            {
                "entity_type": "candidate",
                "entity_name": candidate["feature_names_text"],
                "selection_frequency": float(winner_hits / winner_count),
                "winner_folds": int(winner_hits),
            }
        )
    all_blocks = sorted(
        {
            block_name
            for candidate in candidate_lookup.values()
            for block_name in candidate["block_names"]
        }
    )
    for block_name in all_blocks:
        winner_hits = block_counts.get(block_name, 0)
        rows.append(
            {
                "entity_type": "block",
                "entity_name": block_name,
                "selection_frequency": float(winner_hits / winner_count),
                "winner_folds": int(winner_hits),
            }
        )
    all_features = sorted(
        {
            feature_name
            for candidate in candidate_lookup.values()
            for feature_name in candidate["feature_names"]
        }
    )
    for feature_name in all_features:
        winner_hits = feature_counts.get(feature_name, 0)
        rows.append(
            {
                "entity_type": "feature",
                "entity_name": feature_name,
                "selection_frequency": float(winner_hits / winner_count),
                "winner_folds": int(winner_hits),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["entity_type", "selection_frequency", "entity_name"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def select_feature_columns(  # noqa: PLR0913, PLR0915
    master_table: pd.DataFrame,
    columns: dict[str, list[str]],
    ml_runtime: dict[str, Any],
    feature_selection_params: dict[str, Any],
    selector_model_params: dict[str, Any] | None = None,
    model_name: str | None = None,
) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select a parsimonious feature subset using train-only nested CV diagnostics."""
    feature_order, target_column = _get_feature_and_target_columns(columns)
    if not bool(feature_selection_params.get("enabled", False)):
        manifest = {
            "manifest_type": "feature_selection",
            "enabled": False,
            "selection_reason": "disabled_in_yaml",
            "selected_feature_names": list(feature_order),
            "selected_feature_count": len(feature_order),
        }
        return columns, pd.DataFrame(), pd.DataFrame(), manifest

    selection_splits = feature_selection_params.get("selection_splits", ["train"])
    selection_frame = master_table[
        master_table[SPLIT_COLUMN].isin(selection_splits)
    ].reset_index(drop=True)
    if selection_frame.empty:
        raise ValueError("feature_selection_no_rows_in_selection_splits")

    feature_order, feature_blocks = _build_feature_selection_blocks(
        columns,
        feature_selection_params,
    )
    candidate_records = _enumerate_feature_selection_candidates(
        feature_blocks,
        feature_order,
        feature_selection_params,
    )
    if not candidate_records:
        raise ValueError("feature_selection_no_candidates")

    selector_source = str(feature_selection_params.get("selector_source", "explicit"))
    if selector_source == "candidate_model" and selector_model_params is not None:
        selector_model = dict(selector_model_params.get("feature_selection_selector") or {})
        selector_model.setdefault(
            "class_path",
            selector_model_params.get(
                "class_path",
                "sklearn.linear_model.LogisticRegression",
            ),
        )
        selector_model.setdefault(
            "init_args",
            dict(selector_model_params.get("init_args") or {}),
        )
    else:
        selector_model = dict(feature_selection_params.get("selector_model") or {})
    primary_metric, secondary_metrics = _feature_selection_metrics(
        feature_selection_params
    )
    target_encoder = _build_target_encoder(ml_runtime)
    target_encoder.fit(selection_frame[target_column])
    y_encoded = target_encoder.transform(selection_frame[target_column])
    cv = _feature_selection_cv(y_encoded, ml_runtime, feature_selection_params)
    if cv is None:
        manifest = {
            "manifest_type": "feature_selection",
            "enabled": True,
            "selection_reason": "insufficient_class_support_for_cv",
            "selected_feature_names": list(feature_order),
            "selected_feature_count": len(feature_order),
        }
        return columns, pd.DataFrame(), pd.DataFrame(), manifest

    fold_rows: list[dict[str, Any]] = []
    for candidate in candidate_records:
        feature_names = candidate["feature_names"]
        x_candidate = selection_frame[feature_names]
        for fold_id, (train_idx, valid_idx) in enumerate(
            cv.split(x_candidate, y_encoded),
            start=1,
        ):
            x_train = x_candidate.iloc[train_idx]
            y_train = y_encoded[train_idx]
            x_valid = x_candidate.iloc[valid_idx]
            y_valid = y_encoded[valid_idx]
            estimator = _build_feature_selector_estimator(
                feature_selection_params,
                selector_model_params,
                y_train,
            )
            with _suppress_known_training_warnings():
                estimator.fit(x_train, y_train)
            probabilities = _predict_scores(estimator, x_valid)
            payload = _probability_metrics(y_valid, probabilities)
            fold_rows.append(
                {
                    "candidate_id": int(candidate["candidate_id"]),
                    "fold_id": int(fold_id),
                    "block_names_text": candidate["block_names_text"],
                    "feature_names_text": candidate["feature_names_text"],
                    "feature_count": int(candidate["feature_count"]),
                    "block_count": int(candidate["block_count"]),
                    **payload,
                }
            )

    fold_results = pd.DataFrame(fold_rows)
    frontier = _build_feature_selection_summary(
        fold_results,
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
        prefer_fewer_features=bool(
            feature_selection_params.get("prefer_fewer_features", True)
        ),
    )
    candidate_lookup = {
        int(candidate["candidate_id"]): candidate for candidate in candidate_records
    }
    stability = _build_feature_selection_stability(
        fold_results,
        candidate_lookup=candidate_lookup,
        primary_metric=primary_metric,
        secondary_metrics=secondary_metrics,
        prefer_fewer_features=bool(
            feature_selection_params.get("prefer_fewer_features", True)
        ),
    )
    selected_row = frontier[frontier["selected_flag"] == 1].iloc[0]
    selected_candidate = candidate_lookup[int(selected_row["candidate_id"])]
    selected_features = selected_candidate["feature_names"]

    selected_columns = {
        "target": list(columns["target"]),
        "categorical": [
            feature_name
            for feature_name in columns["categorical"]
            if feature_name in selected_features
        ],
        "numerical": [
            feature_name
            for feature_name in columns["numerical"]
            if feature_name in selected_features
        ],
    }
    manifest = {
        "manifest_type": "feature_selection",
        "model_name": model_name,
        "enabled": True,
        "selection_splits": list(selection_splits),
        "selector_model": {
            "class_path": selector_model.get(
                "class_path",
                "sklearn.linear_model.LogisticRegression",
            ),
            "init_args": dict(selector_model.get("init_args") or {}),
        },
        "primary_metric": primary_metric,
        "secondary_metrics": list(secondary_metrics),
        "candidate_count": int(len(candidate_records)),
        "selected_candidate_id": int(selected_candidate["candidate_id"]),
        "selected_block_names": list(selected_candidate["block_names"]),
        "selected_feature_names": list(selected_features),
        "selected_feature_count": int(len(selected_features)),
        "selected_feature_names_text": selected_candidate["feature_names_text"],
        "best_primary_mean": float(frontier[f"mean_{primary_metric}"].min())
        if not _feature_selection_metric_is_higher_better(primary_metric)
        else float(frontier[f"mean_{primary_metric}"].max()),
        "selected_primary_mean": float(selected_row[f"mean_{primary_metric}"]),
        "selected_primary_sem": float(selected_row[f"sem_{primary_metric}"]),
        "selected_within_one_se": bool(selected_row["within_one_se"]),
        "selection_policy": {
            "prefer_fewer_features": bool(
                feature_selection_params.get("prefer_fewer_features", True)
            ),
            "one_standard_error_rule": True,
            "candidate_generation": "exhaustive_feature_block_combinations",
            "candidate_blocks": json.loads(
                json.dumps(feature_blocks, ensure_ascii=True)
            ),
        },
    }
    if model_name:
        frontier = frontier.assign(model_name=model_name)
        stability = stability.assign(model_name=model_name)
    return selected_columns, frontier, stability, manifest


def _find_model_name_and_artifact(
    best_model_config: dict[str, Any],
    baseline_model: ModelArtifact,
    optimized_model: ModelArtifact,
    xgboost_model: ModelArtifact,
) -> tuple[str, ModelArtifact]:
    target_class_path = best_model_config["class_path"]
    target_init_args = dict(best_model_config.get("init_args", {}))

    for model_name, model_artifact in _model_candidates(
        baseline_model, optimized_model, xgboost_model
    ):
        if model_artifact["class_path"] != target_class_path:
            continue
        if _merged_model_init_args(model_artifact) == target_init_args:
            return model_name, model_artifact

    raise ValueError("Could not map best_model_config back to a trained model artifact")


def train_model(
    master_table: pd.DataFrame,
    columns: dict[str, list[str]],
    model_params: dict[str, Any],
    ml_runtime: dict[str, Any],
) -> ModelArtifact:
    """Treina modelo com classe e hiperparâmetros do YAML."""
    feature_columns, target_column = _get_feature_and_target_columns(columns)

    train_splits: list[str] = model_params.get("train_splits", ["train"])
    train_data = master_table[master_table[SPLIT_COLUMN].isin(train_splits)]

    target_encoder = _build_target_encoder(ml_runtime)
    target_encoder.fit(train_data[target_column])

    x_train = train_data[feature_columns]
    y_train = target_encoder.transform(train_data[target_column])

    model_class = load_class(model_params["class_path"])
    resolved_init_args = _fit_time_model_init_args(
        model_params["class_path"],
        model_params.get("init_args", {}),
        y_train,
    )
    estimator = model_class(**resolved_init_args)
    with _suppress_known_training_warnings():
        estimator.fit(x_train, y_train)

    logger.info(
        "train_model: %s on %d rows (%s)",
        model_params["class_path"],
        len(x_train),
        train_splits,
    )

    artifact: ModelArtifact = {
        "estimator": estimator,
        "target_encoder": target_encoder,
        "feature_columns": feature_columns,
        "class_path": model_params["class_path"],
        "init_args": resolved_init_args,
        "train_splits": train_splits,
    }
    for optional_key in (
        "decision_threshold",
        "decision_policy_name",
        "decision_policy_description",
        "policy_catalog",
        "risk_bands",
    ):
        if optional_key in model_params:
            artifact[optional_key] = model_params[optional_key]
    return artifact


def optimize_model(  # noqa: PLR0913, PLR0915
    master_table: pd.DataFrame,
    columns: dict[str, list[str]],
    optimization_params: dict[str, Any],
    ml_runtime: dict[str, Any],
    decision_policy_params: dict[str, Any] | None = None,
    selection_params: dict[str, Any] | None = None,
    calibration_params: dict[str, Any] | None = None,
) -> ModelArtifact:
    """Optimize each model family against the same composite policy-aware objective."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    decision_policy_params = decision_policy_params or {
        "deployment_policy": "default_050",
        "policies": {
            "default_050": {
                "strategy": "fixed_threshold",
                "threshold": 0.5,
            }
        },
    }
    selection_params = selection_params or {
        "metric": optimization_params.get("scoring", "roc_auc"),
        "score_components": [
            {
                "metric": optimization_params.get("scoring", "roc_auc"),
                "source": "validation",
                "direction": (
                    "maximize"
                    if _metric_is_higher_better(
                        str(optimization_params.get("scoring", "roc_auc"))
                    )
                    else "minimize"
                ),
                "weight": 1.0,
            }
        ],
    }

    search_space = optimization_params.get("search_space", {})
    if not search_space:
        trained_artifact = train_model(master_table, columns, optimization_params, ml_runtime)
        train_splits = list(optimization_params.get("train_splits", ["train"]))
        training_frame = master_table[master_table[SPLIT_COLUMN].isin(train_splits)]
        calibrated_estimator, resolved_init_args, calibration_state = (
            _fit_estimator_with_optional_calibration(
                trained_artifact["class_path"],
                trained_artifact["init_args"],
                training_frame[trained_artifact["feature_columns"]],
                trained_artifact["target_encoder"].transform(
                    training_frame[columns["target"][0]]
                ),
                calibration_params,
            )
        )
        trained_artifact["estimator"] = calibrated_estimator
        trained_artifact["init_args"] = resolved_init_args
        if calibration_state is not None:
            trained_artifact["calibration"] = calibration_state
        return trained_artifact

    feature_columns, target_column = _get_feature_and_target_columns(columns)
    train_splits = list(optimization_params.get("train_splits", ["train"]))
    train_data = master_table[master_table[SPLIT_COLUMN].isin(train_splits)]

    target_encoder = _build_target_encoder(ml_runtime)
    target_encoder.fit(train_data[target_column])

    x_train = train_data[feature_columns]
    y_train = target_encoder.transform(train_data[target_column])

    class_path = optimization_params["class_path"]
    init_args = _fit_time_model_init_args(
        class_path,
        optimization_params.get("init_args", {}),
        y_train,
    )
    n_trials = optimization_params.get("n_trials", 30)
    cv_folds = optimization_params.get("cv", 5)
    optimization_objective = optimization_params.get(
        "optimization_objective",
        "composite_probability_policy_score",
    )
    seed = optimization_params.get("random_state", 42)

    resolved_cv_folds = _safe_n_splits(y_train, int(cv_folds))
    if resolved_cv_folds < _MINIMUM_CV_SPLITS:
        logger.warning(
            "optimize_model: insufficient class support for CV on %s, using direct training",
            class_path,
        )
        direct_artifact = train_model(master_table, columns, optimization_params, ml_runtime)
        calibrated_estimator, resolved_init_args, calibration_state = (
            _fit_estimator_with_optional_calibration(
                class_path,
                direct_artifact["init_args"],
                x_train,
                y_train,
                calibration_params,
            )
        )
        direct_artifact["estimator"] = calibrated_estimator
        direct_artifact["init_args"] = resolved_init_args
        direct_artifact["train_splits"] = train_splits
        direct_artifact["best_cv_score"] = float("nan")
        direct_artifact["cv_folds"] = int(resolved_cv_folds)
        direct_artifact["optimization_objective"] = optimization_objective
        if calibration_state is not None:
            direct_artifact["calibration"] = calibration_state
        return direct_artifact

    cv_cfg = ml_runtime["cross_validation"]
    cv_class = load_class(cv_cfg["class_path"])
    cv_kwargs = {
        **dict(cv_cfg.get("init_args") or {}),
        "n_splits": resolved_cv_folds,
    }
    cv = cv_class(**cv_kwargs)

    sampler_cfg = ml_runtime["optuna_sampler"]
    sampler_class = load_class(sampler_cfg["class_path"])
    sampler_kwargs = {**dict(sampler_cfg.get("init_args") or {}), "seed": seed}
    sampler = sampler_class(**sampler_kwargs)

    study_cfg = ml_runtime.get("optuna_study") or {}
    direction = study_cfg.get("direction", "maximize")

    def objective(trial: optuna.Trial) -> float:
        params = _sample_search_params(trial, search_space, class_path)
        try:
            fold_scores: list[float] = []
            for fold_train_idx, fold_valid_idx in cv.split(x_train, y_train):
                x_fold_train = x_train.iloc[fold_train_idx]
                y_fold_train = y_train[fold_train_idx]
                x_fold_valid = x_train.iloc[fold_valid_idx]
                y_fold_valid = y_train[fold_valid_idx]
                estimator, _resolved_params, _calibration_state = (
                    _fit_estimator_with_optional_calibration(
                        class_path,
                        {**init_args, **params},
                        x_fold_train,
                        y_fold_train,
                        calibration_params,
                    )
                )
                probabilities = _predict_scores(estimator, x_fold_valid)
                validation_payload, policy_payload = _selection_payloads(
                    y_fold_valid,
                    probabilities,
                    decision_policy_params,
                )
                selection_score, _utility_map, _raw_map = _selection_composite_score(
                    validation_payload,
                    policy_payload,
                    selection_params,
                )
                fold_scores.append(float(selection_score))
            return float(np.mean(fold_scores)) if fold_scores else 0.0
        except ValueError:
            return 0.0

    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning(
            "optimize_model: all %d trials failed for %s, falling back to train_model",
            n_trials,
            class_path,
        )
        fallback_artifact = train_model(master_table, columns, optimization_params, ml_runtime)
        calibrated_estimator, resolved_init_args, calibration_state = (
            _fit_estimator_with_optional_calibration(
                class_path,
                fallback_artifact["init_args"],
                x_train,
                y_train,
                calibration_params,
            )
        )
        fallback_artifact["estimator"] = calibrated_estimator
        fallback_artifact["init_args"] = resolved_init_args
        if calibration_state is not None:
            fallback_artifact["calibration"] = calibration_state
        return fallback_artifact

    best_params = _sample_search_params(
        optuna.trial.FixedTrial(study.best_params),
        search_space,
        class_path,
    )
    estimator, resolved_init_args, calibration_state = (
        _fit_estimator_with_optional_calibration(
            class_path,
            {**init_args, **best_params},
            x_train,
            y_train,
            calibration_params,
        )
    )

    logger.info(
        "optimize_model: %s best_%s=%.4f, best_params=%s (n_trials=%d)",
        class_path,
        optimization_objective,
        study.best_value,
        best_params,
        n_trials,
    )

    artifact: ModelArtifact = {
        "estimator": estimator,
        "target_encoder": target_encoder,
        "feature_columns": feature_columns,
        "class_path": class_path,
        "init_args": init_args,
        "train_splits": train_splits,
        "best_params": best_params,
        "best_cv_score": float(study.best_value),
        "cv_folds": int(cv_folds),
        "optimization_objective": optimization_objective,
    }
    if calibration_state is not None:
        artifact["calibration"] = calibration_state
    return artifact


def _metric_keys(evaluation_params: dict[str, Any]) -> list[str]:
    keys = [m["key"] for m in evaluation_params.get("metrics", [])]
    derived = evaluation_params.get("derived") or {}
    if isinstance(derived, dict):
        if derived.get("r2"):
            keys.append("r2")
        if derived.get("mape"):
            keys.append("mape")
        if derived.get("calibration"):
            keys.extend(
                [
                    "calibration_slope",
                    "calibration_intercept",
                    "calibration_slope_error",
                    "calibration_intercept_abs",
                ]
            )
    return list(dict.fromkeys(keys))


def _risk_band_label(probability: float, risk_bands: list[dict[str, Any]]) -> str:
    for band in risk_bands:
        if float(band["min_probability"]) <= probability < float(
            band["max_probability"]
        ):
            return str(band["label"])
    return "Sem faixa"


def evaluate_model(  # noqa: PLR0915
    master_table: pd.DataFrame,
    model_artifact: ModelArtifact,
    columns: dict[str, list[str]],
    evaluation_params: dict[str, Any],
) -> dict[str, Any]:
    """Métricas e matriz de confusão declaradas no YAML (`evaluation`)."""
    split_name: str = evaluation_params["split"]
    _, target_column = _get_feature_and_target_columns(columns)

    feature_columns = model_artifact["feature_columns"]
    target_encoder = model_artifact["target_encoder"]
    estimator = model_artifact["estimator"]

    split_data = master_table[master_table[SPLIT_COLUMN] == split_name]
    metric_keys = _metric_keys(evaluation_params)

    if len(split_data) == 0:
        logger.warning("evaluate_model (%s): no samples in split", split_name)
        empty: dict[str, Any] = {
            "split": split_name,
            "n_samples": 0,
            "confusion_matrix": [],
        }
        for k in metric_keys:
            empty[k] = 0.0
        return empty

    x_split = split_data[feature_columns]
    y_true = target_encoder.transform(split_data[target_column])

    y_pred = estimator.predict(x_split)
    y_proba = (
        estimator.predict_proba(x_split)[:, 1]
        if hasattr(estimator, "predict_proba")
        else y_pred.astype(float)
    )

    n_classes = len(np.unique(y_true))

    cm_cfg = evaluation_params.get("confusion_matrix") or {}
    cm_fn = load_callable(
        cm_cfg.get("function_path", "sklearn.metrics.confusion_matrix")
    )
    cm_kwargs = dict(cm_cfg.get("kwargs") or {})
    cm = cm_fn(y_true, y_pred, **cm_kwargs)

    metrics: dict[str, Any] = {
        "split": split_name,
        "n_samples": len(x_split),
        "confusion_matrix": cm.tolist(),
    }

    for m in evaluation_params.get("metrics", []) or []:
        fn = load_callable(m["function_path"])
        pred_kind = m.get("prediction_input", "y_pred")
        y_second = y_proba if pred_kind == "y_proba" else y_pred
        kwargs = dict(m.get("kwargs") or {})
        try:
            metrics[m["key"]] = float(fn(y_true, y_second, **kwargs))
        except ValueError:
            logger.warning(
                "evaluate_model (%s): metric %s unavailable for current class support",
                split_name,
                m["key"],
            )
            metrics[m["key"]] = float("nan")

    metrics.update(_probability_metrics(y_true, y_proba))

    derived = evaluation_params.get("derived") or {}
    if isinstance(derived, dict):
        r2_cfg = derived.get("r2")
        if r2_cfg and n_classes > 1:
            r2_fn = load_callable(r2_cfg["function_path"])
            pred_kind = r2_cfg.get("prediction_input", "y_proba")
            y_second = y_proba if pred_kind == "y_proba" else y_pred
            rk = dict(r2_cfg.get("kwargs") or {})
            metrics["r2"] = float(r2_fn(y_true, y_second, **rk))
        elif "r2" in metric_keys and "r2" not in metrics:
            metrics["r2"] = 0.0

        mape_cfg = derived.get("mape")
        if mape_cfg and mape_cfg.get("type") == "mae_as_percent_of_mean_label":
            mae = float(mean_absolute_error(y_true, y_proba))
            metrics["mape"] = float(mae / max(y_true.mean(), 1e-8)) * 100.0
        elif "mape" in metric_keys and "mape" not in metrics:
            metrics["mape"] = 0.0

        if not derived.get("calibration"):
            metrics.pop("calibration_slope", None)
            metrics.pop("calibration_intercept", None)
            metrics.pop("calibration_slope_error", None)
            metrics.pop("calibration_intercept_abs", None)

    logger.info(
        "evaluate_model (%s): f1=%.4f, roc_auc=%.4f, brier=%.4f, log_loss=%.4f, r2=%.4f, mape=%.2f%%, cm=%s",
        split_name,
        metrics.get("f1", 0),
        metrics.get("roc_auc", 0),
        metrics.get("brier", 0),
        metrics.get("log_loss", 0),
        metrics.get("r2", 0),
        metrics.get("mape", 0),
        cm.tolist(),
    )
    return metrics


def build_model_selection_scorecard(  # noqa: PLR0913
    master_table: pd.DataFrame,
    baseline_model: ModelArtifact,
    baseline_metrics: dict[str, float],
    optimized_model: ModelArtifact,
    optimized_metrics: dict[str, float],
    xgboost_model: ModelArtifact,
    xgboost_metrics: dict[str, float],
    columns: dict[str, list[str]],
    decision_policy_params: dict[str, Any],
    selection_params: dict[str, Any],
) -> pd.DataFrame:
    """Build a policy-aware, multi-metric scorecard for model selection."""
    _feature_columns, target_column = _get_feature_and_target_columns(columns)
    split_name = decision_policy_params.get("policy_selection_split", "validation")
    split_frame = master_table[master_table[SPLIT_COLUMN] == split_name].reset_index(
        drop=True
    )
    if split_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    components = _resolve_selection_components(selection_params)
    for model_name, model_artifact, validation_metrics in [
        ("baseline", baseline_model, baseline_metrics),
        ("optimized", optimized_model, optimized_metrics),
        ("xgboost", xgboost_model, xgboost_metrics),
    ]:
        policy_payload = _policy_payload_for_model(
            model_artifact,
            split_frame,
            target_column,
            decision_policy_params,
        )
        row = {
            "model_name": model_name,
            "class_path": model_artifact["class_path"],
            "best_cv_score": float(model_artifact.get("best_cv_score", np.nan)),
            "validation_accuracy": float(validation_metrics.get("accuracy", np.nan)),
            "validation_precision": float(validation_metrics.get("precision", np.nan)),
            "validation_recall": float(validation_metrics.get("recall", np.nan)),
            "validation_f1": float(validation_metrics.get("f1", np.nan)),
            "validation_roc_auc": float(validation_metrics.get("roc_auc", np.nan)),
            "validation_brier": float(validation_metrics.get("brier", np.nan)),
            "validation_log_loss": float(validation_metrics.get("log_loss", np.nan)),
            "validation_calibration_slope_error": float(
                validation_metrics.get("calibration_slope_error", np.nan)
            ),
            "validation_calibration_intercept_abs": float(
                validation_metrics.get("calibration_intercept_abs", np.nan)
            ),
            "validation_r2": float(validation_metrics.get("r2", np.nan)),
            "validation_mape": float(validation_metrics.get("mape", np.nan)),
            "selection_split": split_name,
            "selected_feature_count": int(len(model_artifact.get("feature_columns", []))),
            "selected_feature_names_text": ", ".join(
                model_artifact.get("feature_columns", [])
            ),
            **policy_payload,
        }
        validation_payload = {
            metric_name.removeprefix("validation_"): metric_value
            for metric_name, metric_value in row.items()
            if metric_name.startswith("validation_")
        }
        selection_score, utility_map, raw_map = _selection_composite_score(
            validation_payload,
            policy_payload,
            selection_params,
        )
        row["selection_metric"] = "composite_probability_policy_score"
        row["selection_composite_score"] = selection_score
        row.update(utility_map)
        row.update(raw_map)
        rows.append(row)

    scorecard = pd.DataFrame(rows)
    if scorecard.empty:
        return scorecard

    for component in components:
        column_name = _selection_component_column(component)
        utility_column = f"{column_name}_utility"
        weighted_column = f"{column_name}_weighted_score"
        if utility_column not in scorecard.columns:
            scorecard[utility_column] = _MODEL_SELECTION_NEUTRAL_SCORE
        scorecard[weighted_column] = scorecard[utility_column] * float(
            component["weight"]
        )
    scorecard = scorecard.sort_values(
        by=[
            "selection_composite_score",
            "policy_expected_cost_per_sample",
            "validation_brier",
            "validation_log_loss",
            "validation_calibration_slope_error",
            "model_name",
        ],
        ascending=[False, True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    scorecard["selection_rank"] = np.arange(1, len(scorecard) + 1)
    return scorecard


def select_best_model(  # noqa: PLR0913
    baseline_model: ModelArtifact,
    baseline_metrics: dict[str, float],
    optimized_model: ModelArtifact,
    optimized_metrics: dict[str, float],
    xgboost_model: ModelArtifact,
    xgboost_metrics: dict[str, float],
    selection_params: dict[str, Any],
    model_selection_scorecard: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Escolhe vencedor e monta config de refit (splits também no YAML)."""
    metric = selection_params.get("metric", "roc_auc")
    refit_splits = selection_params.get(
        "refit_train_splits",
        ["train", "validation", "test"],
    )

    candidates = [
        ("baseline", baseline_model, baseline_metrics),
        ("optimized", optimized_model, optimized_metrics),
        ("xgboost", xgboost_model, xgboost_metrics),
    ]

    for name, _model, metrics in candidates:
        logger.info(
            "select_best_model: %-10s  %s=%.4f  f1=%.4f  recall=%.4f  accuracy=%.4f",
            name,
            metric,
            metrics.get(metric, 0),
            metrics.get("f1", 0),
            metrics.get("recall", 0),
            metrics.get("accuracy", 0),
        )

    best_name: str
    best_model: ModelArtifact
    best_metrics: dict[str, float]
    selection_metric_label = str(metric)
    selection_score_value = float("nan")
    selection_rank = None

    if model_selection_scorecard is not None and not model_selection_scorecard.empty:
        best_row = model_selection_scorecard.iloc[0]
        best_name = str(best_row["model_name"])
        best_model, best_metrics = next(
            (model, metrics)
            for name, model, metrics in candidates
            if name == best_name
        )
        selection_metric_label = str(
            best_row.get("selection_metric", "composite_probability_policy_score")
        )
        selection_score_value = float(best_row.get("selection_composite_score", np.nan))
        selection_rank = int(best_row.get("selection_rank", 1))
    else:
        best_name, best_model, best_metrics = max(
            candidates, key=lambda c: c[2].get(metric, 0)
        )
        selection_score_value = float(best_metrics.get(metric, np.nan))

    init_args = dict(best_model.get("init_args", {}))
    if "best_params" in best_model:
        init_args.update(best_model["best_params"])

    config: dict[str, Any] = {
        "class_path": best_model["class_path"],
        "train_splits": list(refit_splits),
        "init_args": init_args,
        "model_name": best_name,
        "selection_metric": selection_metric_label,
        "selection_score": selection_score_value,
    }
    if selection_rank is not None:
        config["selection_rank"] = selection_rank

    logger.info(
        "select_best_model: WINNER = %s (%s=%.4f) → refit with %s %s",
        best_name,
        selection_metric_label,
        selection_score_value,
        config["class_path"],
        init_args,
    )

    return config


def build_feature_selection_bundle(  # noqa: PLR0913
    best_model_config: dict[str, Any],
    baseline_selected_feature_columns: dict[str, list[str]],
    baseline_feature_selection_frontier: pd.DataFrame,
    baseline_feature_selection_stability: pd.DataFrame,
    baseline_feature_selection_manifest: dict[str, Any] | None,
    optimized_selected_feature_columns: dict[str, list[str]],
    optimized_feature_selection_frontier: pd.DataFrame,
    optimized_feature_selection_stability: pd.DataFrame,
    optimized_feature_selection_manifest: dict[str, Any] | None,
    xgboost_selected_feature_columns: dict[str, list[str]],
    xgboost_feature_selection_frontier: pd.DataFrame,
    xgboost_feature_selection_stability: pd.DataFrame,
    xgboost_feature_selection_manifest: dict[str, Any] | None,
) -> tuple[dict[str, list[str]], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected_model_name = str(best_model_config.get("model_name", "baseline"))
    bundle_lookup = {
        "baseline": (
            baseline_selected_feature_columns,
            baseline_feature_selection_frontier,
            baseline_feature_selection_stability,
            baseline_feature_selection_manifest or {},
        ),
        "optimized": (
            optimized_selected_feature_columns,
            optimized_feature_selection_frontier,
            optimized_feature_selection_stability,
            optimized_feature_selection_manifest or {},
        ),
        "xgboost": (
            xgboost_selected_feature_columns,
            xgboost_feature_selection_frontier,
            xgboost_feature_selection_stability,
            xgboost_feature_selection_manifest or {},
        ),
    }
    (
        selected_feature_columns,
        _selected_frontier,
        _selected_stability,
        selected_manifest,
    ) = bundle_lookup.get(selected_model_name, bundle_lookup["baseline"])

    frontier_frames: list[pd.DataFrame] = []
    stability_frames: list[pd.DataFrame] = []
    per_model_manifests: dict[str, Any] = {}
    total_candidate_count = 0
    for model_name, (
        _feature_columns,
        frontier,
        stability,
        manifest,
    ) in bundle_lookup.items():
        per_model_manifests[model_name] = manifest
        total_candidate_count += int(manifest.get("candidate_count", 0))
        if frontier is not None and not frontier.empty:
            frontier_frames.append(
                frontier.assign(
                    model_name=model_name,
                    selected_model_flag=int(model_name == selected_model_name),
                )
            )
        if stability is not None and not stability.empty:
            stability_frames.append(
                stability.assign(
                    model_name=model_name,
                    selected_model_flag=int(model_name == selected_model_name),
                )
            )

    aggregate_manifest = {
        "manifest_type": "feature_selection",
        "selection_scope": "per_model_family",
        "enabled": True,
        "selected_model_name": selected_model_name,
        "selected_feature_names": list(selected_manifest.get("selected_feature_names", [])),
        "selected_feature_names_text": selected_manifest.get(
            "selected_feature_names_text",
            "",
        ),
        "selected_feature_count": int(selected_manifest.get("selected_feature_count", 0)),
        "selection_splits": list(selected_manifest.get("selection_splits", [])),
        "primary_metric": selected_manifest.get("primary_metric"),
        "secondary_metrics": list(selected_manifest.get("secondary_metrics", [])),
        "candidate_count": total_candidate_count,
        "selected_model_candidate_count": int(
            selected_manifest.get("candidate_count", 0)
        ),
        "selector_source": "candidate_model",
        "per_model_manifests": per_model_manifests,
    }
    return (
        selected_feature_columns,
        pd.concat(frontier_frames, ignore_index=True) if frontier_frames else pd.DataFrame(),
        pd.concat(stability_frames, ignore_index=True) if stability_frames else pd.DataFrame(),
        aggregate_manifest,
    )


def build_model_frontier(  # noqa: PLR0913
    baseline_model: ModelArtifact,
    baseline_metrics: dict[str, float],
    optimized_model: ModelArtifact,
    optimized_metrics: dict[str, float],
    xgboost_model: ModelArtifact,
    xgboost_metrics: dict[str, float],
    best_model_config: dict[str, Any],
    selected_deployment_policy: dict[str, Any] | None = None,
    model_selection_scorecard: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Materializa uma visão comparativa dos modelos para o dashboard."""
    selected_model_name = None
    try:
        selected_model_name, _ = _find_model_name_and_artifact(
            best_model_config,
            baseline_model,
            optimized_model,
            xgboost_model,
        )
    except ValueError:
        logger.warning("build_model_frontier: could not identify selected model")

    deployment_policy_name = None
    deployment_threshold = None
    if selected_deployment_policy:
        deployment_policy_name = selected_deployment_policy.get("decision_policy_name")
        deployment_threshold = selected_deployment_policy.get("decision_threshold")

    rows: list[dict[str, Any]] = []
    for model_name, model_artifact, metrics in [
        ("baseline", baseline_model, baseline_metrics),
        ("optimized", optimized_model, optimized_metrics),
        ("xgboost", xgboost_model, xgboost_metrics),
    ]:
        rows.append(
            {
                "model_name": model_name,
                "class_path": model_artifact["class_path"],
                "selected_for_refit": model_name == selected_model_name,
                "best_cv_score": float(model_artifact.get("best_cv_score", 0.0)),
                "validation_accuracy": float(metrics.get("accuracy", 0.0)),
                "validation_precision": float(metrics.get("precision", 0.0)),
                "validation_recall": float(metrics.get("recall", 0.0)),
                "validation_f1": float(metrics.get("f1", 0.0)),
                "validation_roc_auc": float(metrics.get("roc_auc", 0.0)),
                "validation_brier": float(metrics.get("brier", 0.0)),
                "validation_log_loss": float(metrics.get("log_loss", 0.0)),
                "validation_calibration_slope_error": float(
                    metrics.get("calibration_slope_error", 0.0)
                ),
                "validation_calibration_intercept_abs": float(
                    metrics.get("calibration_intercept_abs", 0.0)
                ),
                "validation_mape": float(metrics.get("mape", 0.0)),
                "validation_r2": float(metrics.get("r2", 0.0)),
                "deployment_policy_name": (
                    deployment_policy_name
                    if model_name == selected_model_name
                    else None
                ),
                "deployment_threshold": (
                    float(deployment_threshold)
                    if model_name == selected_model_name
                    and deployment_threshold is not None
                    else None
                ),
            }
        )

    frontier = pd.DataFrame(rows)
    if model_selection_scorecard is None or model_selection_scorecard.empty:
        return frontier

    merge_columns = [
        column_name
        for column_name in model_selection_scorecard.columns
        if column_name
        not in {
            "class_path",
            "best_cv_score",
            "validation_accuracy",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "validation_roc_auc",
            "validation_brier",
            "validation_log_loss",
            "validation_calibration_slope_error",
            "validation_calibration_intercept_abs",
            "validation_mape",
            "validation_r2",
        }
    ]
    return frontier.merge(
        model_selection_scorecard[merge_columns],
        on="model_name",
        how="left",
    )


def build_cv_fold_metrics(  # noqa: PLR0913
    master_table: pd.DataFrame,
    baseline_model: ModelArtifact,
    optimized_model: ModelArtifact,
    xgboost_model: ModelArtifact,
    columns: dict[str, list[str]],
    ml_runtime: dict[str, Any],
    decision_policy_params: dict[str, Any],
    selected_deployment_policy: dict[str, Any] | None,
) -> pd.DataFrame:
    """Calcula estabilidade das métricas por fold usando o melhor setup de cada modelo."""
    _feature_columns, target_column = _get_feature_and_target_columns(columns)
    development_splits = decision_policy_params.get(
        "development_splits",
        ["train", "validation"],
    )
    development_frame = master_table[
        master_table[SPLIT_COLUMN].isin(development_splits)
    ].reset_index(drop=True)

    if development_frame.empty:
        return pd.DataFrame()

    y_raw = development_frame[target_column]

    target_encoder = _build_target_encoder(ml_runtime)
    target_encoder.fit(y_raw)
    y_development = target_encoder.transform(y_raw)

    desired_splits = int(
        decision_policy_params.get("cv_folds", baseline_model.get("cv_folds", 5))
    )
    n_splits = _safe_n_splits(y_development, desired_splits)
    if n_splits < _MINIMUM_CV_SPLITS:
        logger.warning(
            "build_cv_fold_metrics: not enough positive/negative samples for CV"
        )
        return pd.DataFrame()

    cv_cfg = ml_runtime["cross_validation"]
    cv_class = load_class(cv_cfg["class_path"])
    cv_kwargs = {**dict(cv_cfg.get("init_args") or {}), "n_splits": n_splits}
    cv = cv_class(**cv_kwargs)

    selected_policy_name = (
        selected_deployment_policy.get("decision_policy_name")
        if selected_deployment_policy
        else decision_policy_params.get("deployment_policy", "default_050")
    )
    selected_policy_cfg = dict(
        (decision_policy_params.get("policies") or {}).get(selected_policy_name, {})
    )
    policy_threshold = (
        float(selected_deployment_policy.get("decision_threshold"))
        if selected_deployment_policy
        and selected_deployment_policy.get("decision_threshold") is not None
        else float(selected_policy_cfg.get("threshold", 0.5))
    )
    false_negative_cost = float(selected_policy_cfg.get("false_negative_cost", 1.0))
    false_positive_cost = float(selected_policy_cfg.get("false_positive_cost", 1.0))

    rows: list[dict[str, Any]] = []
    for model_name, model_artifact in _model_candidates(
        baseline_model, optimized_model, xgboost_model
    ):
        model_features = development_frame[model_artifact["feature_columns"]]
        calibration_params = _artifact_calibration_params(model_artifact)
        for fold_id, (train_idx, valid_idx) in enumerate(
            cv.split(model_features, y_development),
            start=1,
        ):
            x_train = model_features.iloc[train_idx]
            y_train = y_development[train_idx]
            x_valid = model_features.iloc[valid_idx]
            y_valid = y_development[valid_idx]
            estimator, _resolved_init_args, _calibration_state = (
                _fit_estimator_with_optional_calibration(
                    model_artifact["class_path"],
                    _merged_model_init_args(model_artifact),
                    x_train,
                    y_train,
                    calibration_params,
                )
            )
            probabilities = _predict_scores(estimator, x_valid)
            metrics = _threshold_metrics(
                y_valid,
                probabilities,
                threshold=policy_threshold,
            )
            metrics["expected_cost"] = _expected_cost(
                metrics,
                false_negative_cost=false_negative_cost,
                false_positive_cost=false_positive_cost,
            )
            probability_metrics = _probability_metrics(y_valid, probabilities)

            rows.append(
                {
                    "model_name": model_name,
                    "fold_id": fold_id,
                    "policy_name": selected_policy_name,
                    "policy_threshold": policy_threshold,
                    **metrics,
                    **probability_metrics,
                }
            )

    return pd.DataFrame(rows)


def summarize_cv_fold_metrics(cv_fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Resume média, desvio e variação relativa entre folds."""
    if cv_fold_metrics.empty:
        return pd.DataFrame()

    metric_columns = [
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "expected_cost",
        "brier",
        "log_loss",
        "calibration_slope",
        "calibration_intercept",
        "calibration_slope_error",
        "calibration_intercept_abs",
        "false_positive_rate",
        "false_negative_rate",
        "positive_prediction_rate",
        "positive_prevalence",
        "tn_share",
        "fp_share",
        "fn_share",
        "tp_share",
        "mean_risk_score",
    ]
    rows: list[dict[str, Any]] = []

    for model_name, group in cv_fold_metrics.groupby("model_name", sort=False):
        for metric_name in metric_columns:
            values = group[metric_name].dropna().astype(float).to_numpy()
            if len(values) == 0:
                continue

            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=0))
            relative_std_pct = (
                float(abs(std_value / mean_value) * 100.0)
                if not np.isclose(mean_value, 0.0)
                else 0.0
            )
            max_jump = (
                float(np.max(np.abs(np.diff(values)))) if len(values) > 1 else 0.0
            )
            if relative_std_pct <= _ROBUSTNESS_STABLE_MAX:
                robustness_label = "stable"
            elif relative_std_pct <= _ROBUSTNESS_MODERATE_MAX:
                robustness_label = "moderate"
            else:
                robustness_label = "volatile"

            rows.append(
                {
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "folds": int(len(values)),
                    "mean_value": mean_value,
                    "std_value": std_value,
                    "variation_pct_of_mean": relative_std_pct,
                    "min_value": float(np.min(values)),
                    "max_value": float(np.max(values)),
                    "max_fold_to_fold_jump": max_jump,
                    "robustness_label": robustness_label,
                }
            )

    return pd.DataFrame(rows)


def build_bootstrap_metric_intervals(  # noqa: PLR0913
    master_table: pd.DataFrame,
    baseline_model: ModelArtifact,
    optimized_model: ModelArtifact,
    xgboost_model: ModelArtifact,
    columns: dict[str, list[str]],
    evaluation_params: dict[str, Any],
    decision_policy_params: dict[str, Any],
    selected_deployment_policy: dict[str, Any] | None,
    bootstrap_ci_params: dict[str, Any],
) -> pd.DataFrame:
    """Bootstrap confidence intervals for model probability and policy metrics."""
    _feature_columns, target_column = _get_feature_and_target_columns(columns)
    split_name = bootstrap_ci_params.get("split", evaluation_params.get("split", "validation"))
    split_frame = master_table[master_table[SPLIT_COLUMN] == split_name].reset_index(drop=True)
    if split_frame.empty:
        return pd.DataFrame()

    metric_names = list(
        dict.fromkeys(
            bootstrap_ci_params.get(
                "metrics",
                [
                    "roc_auc",
                    "brier",
                    "log_loss",
                    "recall",
                    "precision",
                    "f1",
                    "false_negative_rate",
                    "false_positive_rate",
                    "expected_cost",
                ],
            )
        )
    )
    iterations = int(bootstrap_ci_params.get("iterations", 250))
    random_state = int(bootstrap_ci_params.get("random_state", 42))
    confidence_level = float(bootstrap_ci_params.get("confidence_level", 0.95))
    selected_policy_name = (
        selected_deployment_policy.get("decision_policy_name")
        if selected_deployment_policy
        else decision_policy_params.get("deployment_policy", "default_050")
    )
    selected_policy_cfg = dict(
        (decision_policy_params.get("policies") or {}).get(selected_policy_name, {})
    )
    rows: list[dict[str, Any]] = []

    for model_name, model_artifact in _model_candidates(
        baseline_model,
        optimized_model,
        xgboost_model,
    ):
        estimator = model_artifact["estimator"]
        target_encoder = model_artifact["target_encoder"]
        y_true = target_encoder.transform(split_frame[target_column])
        probabilities = _predict_scores(
            estimator,
            split_frame[model_artifact["feature_columns"]],
        )
        threshold = _diagnostic_policy_threshold(
            y_true,
            probabilities,
            selected_policy_name,
            selected_policy_cfg,
            decision_policy_params,
            selected_deployment_policy,
        )
        false_negative_cost = float(selected_policy_cfg.get("false_negative_cost", 1.0))
        false_positive_cost = float(selected_policy_cfg.get("false_positive_cost", 1.0))
        baseline_payload = _merged_metric_payload(
            y_true,
            probabilities,
            threshold=threshold,
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
        sample_values: dict[str, list[float]] = {metric_name: [] for metric_name in metric_names}
        rng = np.random.default_rng(random_state)
        sample_size = len(split_frame)
        for _ in range(iterations):
            take = rng.integers(0, sample_size, sample_size)
            sample_y = y_true[take]
            sample_probabilities = probabilities[take]
            sample_payload = _merged_metric_payload(
                sample_y,
                sample_probabilities,
                threshold=threshold,
                false_negative_cost=false_negative_cost,
                false_positive_cost=false_positive_cost,
            )
            for metric_name in metric_names:
                metric_value = sample_payload.get(metric_name)
                if metric_value is None or pd.isna(metric_value):
                    continue
                sample_values[metric_name].append(float(metric_value))

        for metric_name in metric_names:
            ci_low, ci_high, ci_width = _bootstrap_bounds(
                sample_values[metric_name],
                confidence_level=confidence_level,
            )
            rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    "policy_name": selected_policy_name,
                    "policy_threshold": threshold,
                    "metric_name": metric_name,
                    "baseline_value": float(baseline_payload.get(metric_name, np.nan)),
                    "bootstrap_mean": (
                        float(np.mean(sample_values[metric_name]))
                        if sample_values[metric_name]
                        else float("nan")
                    ),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_width": ci_width,
                    "bootstrap_samples": int(len(sample_values[metric_name])),
                }
            )

    return pd.DataFrame(rows)


def build_permutation_feature_importance(  # noqa: PLR0913
    best_model_config: dict[str, Any],
    baseline_model: ModelArtifact,
    optimized_model: ModelArtifact,
    xgboost_model: ModelArtifact,
    master_table: pd.DataFrame,
    columns: dict[str, list[str]],
    decision_policy_params: dict[str, Any],
    selected_deployment_policy: dict[str, Any] | None,
    permutation_importance_params: dict[str, Any],
) -> pd.DataFrame:
    """Permutation importance for the selected model on the configured evaluation split."""
    model_name, model_artifact = _find_model_name_and_artifact(
        best_model_config,
        baseline_model,
        optimized_model,
        xgboost_model,
    )
    split_name = permutation_importance_params.get("split", "validation")
    split_frame = master_table[master_table[SPLIT_COLUMN] == split_name].reset_index(drop=True)
    if split_frame.empty:
        return pd.DataFrame()

    metric_names = list(
        dict.fromkeys(
            permutation_importance_params.get(
                "metrics",
                ["roc_auc", "brier", "log_loss"],
            )
        )
    )
    repeats = int(permutation_importance_params.get("repeats", 8))
    random_state = int(permutation_importance_params.get("random_state", 42))
    feature_group_map = _feature_group_lookup(
        permutation_importance_params.get("feature_groups") or {}
    )
    _feature_columns, target_column = _get_feature_and_target_columns(columns)
    estimator = model_artifact["estimator"]
    target_encoder = model_artifact["target_encoder"]
    y_true = target_encoder.transform(split_frame[target_column])
    features = split_frame[model_artifact["feature_columns"]].copy()
    baseline_probabilities = _predict_scores(estimator, features)

    policy_name = (
        selected_deployment_policy.get("decision_policy_name")
        if selected_deployment_policy
        else decision_policy_params.get("deployment_policy", "default_050")
    )
    policy_cfg = dict((decision_policy_params.get("policies") or {}).get(policy_name, {}))
    threshold = _diagnostic_policy_threshold(
        y_true,
        baseline_probabilities,
        policy_name,
        policy_cfg,
        decision_policy_params,
        selected_deployment_policy,
    )
    false_negative_cost = float(policy_cfg.get("false_negative_cost", 1.0))
    false_positive_cost = float(policy_cfg.get("false_positive_cost", 1.0))
    baseline_payload = _merged_metric_payload(
        y_true,
        baseline_probabilities,
        threshold=threshold,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []

    for feature_name in model_artifact["feature_columns"]:
        if feature_name not in features.columns:
            continue
        metric_samples: dict[str, list[float]] = {metric_name: [] for metric_name in metric_names}
        permuted_metric_values: dict[str, list[float]] = {
            metric_name: [] for metric_name in metric_names
        }
        for _ in range(repeats):
            take = rng.permutation(len(features))
            permuted_features = features.copy()
            permuted_features.loc[:, feature_name] = features[feature_name].to_numpy()[take]
            permuted_probabilities = _predict_scores(estimator, permuted_features)
            permuted_payload = _merged_metric_payload(
                y_true,
                permuted_probabilities,
                threshold=threshold,
                false_negative_cost=false_negative_cost,
                false_positive_cost=false_positive_cost,
            )
            for metric_name in metric_names:
                baseline_value = baseline_payload.get(metric_name)
                permuted_value = permuted_payload.get(metric_name)
                if baseline_value is None or permuted_value is None:
                    continue
                if pd.isna(baseline_value) or pd.isna(permuted_value):
                    continue
                permuted_metric_values[metric_name].append(float(permuted_value))
                delta = (
                    float(baseline_value - permuted_value)
                    if _metric_is_higher_better(metric_name)
                    else float(permuted_value - baseline_value)
                )
                metric_samples[metric_name].append(delta)

        for metric_name in metric_names:
            samples = metric_samples[metric_name]
            if not samples:
                continue
            importance_mean = float(np.mean(samples))
            rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    "policy_name": policy_name,
                    "policy_threshold": threshold,
                    "feature_name": feature_name,
                    "feature_group": feature_group_map.get(feature_name, "ungrouped"),
                    "metric_name": metric_name,
                    "baseline_metric_value": float(baseline_payload[metric_name]),
                    "permuted_metric_mean": float(np.mean(permuted_metric_values[metric_name])),
                    "importance_mean": importance_mean,
                    "importance_std": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
                    "permutation_repeats": int(len(samples)),
                    "positive_means_feature_helps_flag": int(importance_mean > 0.0),
                }
            )

    return pd.DataFrame(rows)


def build_perturbation_sensitivity_audit(  # noqa: PLR0912, PLR0913, PLR0915
    best_model_config: dict[str, Any],
    baseline_model: ModelArtifact,
    optimized_model: ModelArtifact,
    xgboost_model: ModelArtifact,
    master_table: pd.DataFrame,
    columns: dict[str, list[str]],
    selected_deployment_policy: dict[str, Any] | None,
    sensitivity_audit_params: dict[str, Any],
) -> pd.DataFrame:
    """Audita se pequenas perturbacoes geram resposta proporcional do score."""
    model_name, model_artifact = _find_model_name_and_artifact(
        best_model_config,
        baseline_model,
        optimized_model,
        xgboost_model,
    )
    split_name = sensitivity_audit_params.get("split", "validation")
    working = master_table[master_table[SPLIT_COLUMN] == split_name].reset_index(
        drop=True
    )
    if working.empty:
        return pd.DataFrame()

    sample_size = int(sensitivity_audit_params.get("sample_size", 128))
    random_state = int(sensitivity_audit_params.get("random_state", 42))
    if sample_size > 0 and len(working) > sample_size:
        working = working.sample(n=sample_size, random_state=random_state).reset_index(
            drop=True
        )

    baseline_features = working[model_artifact["feature_columns"]].copy()
    estimator = model_artifact["estimator"]
    baseline_probabilities = _predict_scores(estimator, baseline_features)

    risk_bands = []
    if selected_deployment_policy is not None:
        risk_bands = list(selected_deployment_policy.get("risk_bands", []))
    if not risk_bands:
        risk_bands = list(model_artifact.get("risk_bands", []))

    policy_threshold = (
        float(selected_deployment_policy.get("decision_threshold"))
        if selected_deployment_policy
        and selected_deployment_policy.get("decision_threshold") is not None
        else 0.5
    )
    baseline_decisions = baseline_probabilities >= policy_threshold
    baseline_risk_bands = (
        np.asarray(
            [
                _risk_band_label(float(probability), risk_bands)
                for probability in baseline_probabilities
            ],
            dtype=object,
        )
        if risk_bands
        else None
    )

    min_baseline_abs = float(sensitivity_audit_params.get("min_baseline_abs", 1.0))
    insensitive_max_ratio = float(
        sensitivity_audit_params.get("insensitive_max_ratio", 0.10)
    )
    overreactive_min_ratio = float(
        sensitivity_audit_params.get("overreactive_min_ratio", 1.00)
    )
    perturbation_percents = [
        float(value)
        for value in sensitivity_audit_params.get(
            "perturbation_percents",
            [0.05, 0.10, 0.20],
        )
    ]

    rows: list[dict[str, Any]] = []
    model_numerical_features = [
        feature_name
        for feature_name in columns.get("numerical", [])
        if feature_name in model_artifact["feature_columns"]
    ]
    for feature_name in model_numerical_features:
        if feature_name not in baseline_features.columns:
            continue

        feature_values = baseline_features[feature_name].astype(float).to_numpy()
        feature_min = float(np.min(feature_values))
        feature_max = float(np.max(feature_values))

        for perturbation_pct in perturbation_percents:
            for direction, multiplier in (
                ("down", 1.0 - perturbation_pct),
                ("up", 1.0 + perturbation_pct),
            ):
                candidate_values = np.clip(
                    feature_values * multiplier,
                    feature_min,
                    feature_max,
                )
                relative_change = np.abs(candidate_values - feature_values) / np.maximum(
                    np.abs(feature_values),
                    min_baseline_abs,
                )
                valid_mask = relative_change > 0.0
                if not np.any(valid_mask):
                    continue

                perturbed_features = baseline_features.copy()
                perturbed_features.loc[:, feature_name] = candidate_values
                perturbed_probabilities = _predict_scores(estimator, perturbed_features)
                probability_delta = np.abs(
                    perturbed_probabilities - baseline_probabilities
                )
                sensitivity_ratio = probability_delta[valid_mask] / relative_change[
                    valid_mask
                ]

                if len(sensitivity_ratio) == 0:
                    continue

                mean_ratio = float(np.mean(sensitivity_ratio))
                if mean_ratio < insensitive_max_ratio:
                    sensitivity_label = "too_insensitive"
                elif mean_ratio > overreactive_min_ratio:
                    sensitivity_label = "too_sensitive"
                else:
                    sensitivity_label = "proportional"

                perturbed_decisions = perturbed_probabilities >= policy_threshold
                decision_flip_rate = float(
                    np.mean(
                        baseline_decisions[valid_mask]
                        != perturbed_decisions[valid_mask]
                    )
                )

                risk_band_change_rate: float | None = None
                if baseline_risk_bands is not None:
                    perturbed_risk_bands = np.asarray(
                        [
                            _risk_band_label(float(probability), risk_bands)
                            for probability in perturbed_probabilities
                        ],
                        dtype=object,
                    )
                    risk_band_change_rate = float(
                        np.mean(
                            baseline_risk_bands[valid_mask]
                            != perturbed_risk_bands[valid_mask]
                        )
                    )

                rows.append(
                    {
                        "model_name": model_name,
                        "split": split_name,
                        "policy_name": (
                            selected_deployment_policy.get("decision_policy_name")
                            if selected_deployment_policy
                            else None
                        ),
                        "policy_threshold": policy_threshold,
                        "feature_name": feature_name,
                        "direction": direction,
                        "perturbation_pct": float(perturbation_pct),
                        "sampled_rows": int(np.sum(valid_mask)),
                        "mean_relative_input_change": float(
                            np.mean(relative_change[valid_mask])
                        ),
                        "mean_abs_probability_delta": float(
                            np.mean(probability_delta[valid_mask])
                        ),
                        "median_abs_probability_delta": float(
                            np.median(probability_delta[valid_mask])
                        ),
                        "max_abs_probability_delta": float(
                            np.max(probability_delta[valid_mask])
                        ),
                        "mean_sensitivity_ratio": mean_ratio,
                        "max_sensitivity_ratio": float(np.max(sensitivity_ratio)),
                        "decision_flip_rate": decision_flip_rate,
                        "risk_band_change_rate": risk_band_change_rate,
                        "sensitivity_label": sensitivity_label,
                    }
                )

    return pd.DataFrame(rows)


def summarize_perturbation_sensitivity_audit(
    perturbation_sensitivity_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Resume a auditoria de sensibilidade por feature."""
    if perturbation_sensitivity_audit.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for feature_name, group in perturbation_sensitivity_audit.groupby(
        "feature_name", sort=False
    ):
        labels = set(group["sensitivity_label"].dropna().astype(str))
        if "too_sensitive" in labels and "too_insensitive" in labels:
            summary_label = "mixed"
        elif "too_sensitive" in labels:
            summary_label = "too_sensitive"
        elif "too_insensitive" in labels:
            summary_label = "too_insensitive"
        else:
            summary_label = "proportional"

        rows.append(
            {
                "feature_name": feature_name,
                "directions_tested": int(group["direction"].nunique()),
                "perturbations_tested": int(len(group)),
                "mean_sensitivity_ratio": float(
                    pd.to_numeric(
                        group["mean_sensitivity_ratio"],
                        errors="coerce",
                    ).mean()
                ),
                "max_sensitivity_ratio": float(
                    pd.to_numeric(
                        group["max_sensitivity_ratio"],
                        errors="coerce",
                    ).max()
                ),
                "max_decision_flip_rate": float(
                    pd.to_numeric(group["decision_flip_rate"], errors="coerce").max()
                ),
                "max_risk_band_change_rate": float(
                    pd.to_numeric(
                        group["risk_band_change_rate"],
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .max()
                ),
                "mean_abs_probability_delta": float(
                    pd.to_numeric(
                        group["mean_abs_probability_delta"],
                        errors="coerce",
                    ).mean()
                ),
                "sensitivity_label": summary_label,
            }
        )

    return pd.DataFrame(rows)


def build_split_comparison_report(  # noqa: PLR0913
    best_model_config: dict[str, Any],
    baseline_model: ModelArtifact,
    optimized_model: ModelArtifact,
    xgboost_model: ModelArtifact,
    master_table: pd.DataFrame,
    columns: dict[str, list[str]],
    decision_policy_params: dict[str, Any],
    selected_deployment_policy: dict[str, Any] | None,
    model_selection_params: dict[str, Any],
) -> pd.DataFrame:
    """Compare the selected development model across train, validation, and test."""
    _model_name, model_artifact = _find_model_name_and_artifact(
        best_model_config,
        baseline_model,
        optimized_model,
        xgboost_model,
    )
    _feature_columns, target_column = _get_feature_and_target_columns(columns)
    policy_name, policy_cfg = _active_deployment_policy(decision_policy_params)
    threshold = (
        float(selected_deployment_policy.get("decision_threshold"))
        if selected_deployment_policy
        and selected_deployment_policy.get("decision_threshold") is not None
        else float(policy_cfg.get("threshold", 0.5))
    )
    false_negative_cost = float(policy_cfg.get("false_negative_cost", 1.0))
    false_positive_cost = float(policy_cfg.get("false_positive_cost", 1.0))
    requested_splits = list(
        model_selection_params.get(
            "split_comparison_splits",
            ["train", "validation", "test"],
        )
    )
    rows: list[dict[str, Any]] = []

    for split_name in requested_splits:
        split_frame = master_table[master_table[SPLIT_COLUMN] == split_name].reset_index(
            drop=True
        )
        if split_frame.empty:
            continue
        y_true = model_artifact["target_encoder"].transform(split_frame[target_column])
        probabilities = _predict_scores(
            model_artifact["estimator"],
            split_frame[model_artifact["feature_columns"]],
        )
        payload = _merged_metric_payload(
            y_true,
            probabilities,
            threshold=threshold,
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
        rows.append(
            {
                "model_name": str(best_model_config.get("model_name", "selected")),
                "split": split_name,
                "n_samples": int(len(split_frame)),
                "policy_name": (
                    selected_deployment_policy.get("decision_policy_name")
                    if selected_deployment_policy
                    else policy_name
                ),
                "policy_threshold": threshold,
                "training_split_scope": ",".join(
                    model_artifact.get("train_splits", ["train"])
                ),
                **payload,
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        return report

    split_order = {name: index for index, name in enumerate(requested_splits)}
    report["split_order"] = report["split"].map(lambda name: split_order.get(name, 999))
    report = report.sort_values("split_order", kind="mergesort").reset_index(drop=True)
    report["roc_auc_gap_vs_train"] = report["roc_auc"] - float(
        report.loc[report["split"] == "train", "roc_auc"].iloc[0]
        if (report["split"] == "train").any()
        else np.nan
    )
    report["brier_gap_vs_train"] = report["brier"] - float(
        report.loc[report["split"] == "train", "brier"].iloc[0]
        if (report["split"] == "train").any()
        else np.nan
    )
    return report.drop(columns=["split_order"])


def build_nested_cv_audit(  # noqa: PLR0913, PLR0915
    master_table: pd.DataFrame,
    columns: dict[str, list[str]],
    ml_runtime: dict[str, Any],
    feature_selection_params: dict[str, Any],
    baseline_params: dict[str, Any],
    optimized_params: dict[str, Any],
    xgboost_params: dict[str, Any],
    evaluation_params: dict[str, Any],
    model_selection_params: dict[str, Any],
    decision_policy_params: dict[str, Any],
    nested_cv_params: dict[str, Any] | None,
    calibration_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Audit the full modelling workflow with an outer CV loop."""
    nested_cv_params = nested_cv_params or {}
    if not bool(nested_cv_params.get("enabled", False)):
        return pd.DataFrame()

    feature_columns, target_column = _get_feature_and_target_columns(columns)
    development_splits = nested_cv_params.get(
        "development_splits",
        decision_policy_params.get("development_splits", ["train", "validation"]),
    )
    development_frame = master_table[
        master_table[SPLIT_COLUMN].isin(development_splits)
    ].reset_index(drop=True)
    if development_frame.empty:
        return pd.DataFrame()

    target_encoder = _build_target_encoder(ml_runtime)
    target_encoder.fit(development_frame[target_column])
    y_development = target_encoder.transform(development_frame[target_column])
    desired_outer_folds = int(nested_cv_params.get("outer_folds", 3))
    outer_folds = _safe_n_splits(y_development, desired_outer_folds)
    if outer_folds < _MINIMUM_CV_SPLITS:
        logger.warning("build_nested_cv_audit: insufficient class support for outer CV")
        return pd.DataFrame()

    cv_cfg = ml_runtime["cross_validation"]
    cv_class = load_class(cv_cfg["class_path"])
    cv_kwargs = {**dict(cv_cfg.get("init_args") or {}), "n_splits": outer_folds}
    outer_cv = cv_class(**cv_kwargs)
    validation_fraction = float(nested_cv_params.get("validation_fraction", 0.25))
    base_random_state = int(nested_cv_params.get("random_state", 42))
    nested_selection_params = dict(model_selection_params)
    nested_selection_params["refit_train_splits"] = ["train", "validation"]
    rows: list[dict[str, Any]] = []

    for fold_id, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(development_frame[feature_columns], y_development),
        start=1,
    ):
        outer_train_frame = development_frame.iloc[outer_train_idx].reset_index(drop=True)
        outer_test_frame = development_frame.iloc[outer_test_idx].reset_index(drop=True)
        inner_train_frame, inner_valid_frame = _split_outer_train_for_nested_audit(
            outer_train_frame,
            target_column,
            validation_fraction=validation_fraction,
            random_state=base_random_state + fold_id,
        )
        working_table = _build_working_split_table(inner_train_frame, inner_valid_frame)
        baseline_selected_columns, _frontier, _stability, baseline_feature_manifest = select_feature_columns(
            working_table,
            columns,
            ml_runtime,
            feature_selection_params,
            baseline_params,
            "baseline",
        )
        baseline_model = optimize_model(
            working_table,
            baseline_selected_columns,
            _scaled_nested_optimization_params(baseline_params, nested_cv_params),
            ml_runtime,
            decision_policy_params,
            nested_selection_params,
            calibration_params,
        )
        baseline_metrics = evaluate_model(
            working_table,
            baseline_model,
            baseline_selected_columns,
            {**evaluation_params, "split": "validation"},
        )
        optimized_selected_columns, _frontier, _stability, optimized_feature_manifest = select_feature_columns(
            working_table,
            columns,
            ml_runtime,
            feature_selection_params,
            optimized_params,
            "optimized",
        )
        optimized_model = optimize_model(
            working_table,
            optimized_selected_columns,
            _scaled_nested_optimization_params(optimized_params, nested_cv_params),
            ml_runtime,
            decision_policy_params,
            nested_selection_params,
            calibration_params,
        )
        optimized_metrics = evaluate_model(
            working_table,
            optimized_model,
            optimized_selected_columns,
            {**evaluation_params, "split": "validation"},
        )
        xgboost_selected_columns, _frontier, _stability, xgboost_feature_manifest = select_feature_columns(
            working_table,
            columns,
            ml_runtime,
            feature_selection_params,
            xgboost_params,
            "xgboost",
        )
        xgboost_model = optimize_model(
            working_table,
            xgboost_selected_columns,
            _scaled_nested_optimization_params(xgboost_params, nested_cv_params),
            ml_runtime,
            decision_policy_params,
            nested_selection_params,
            calibration_params,
        )
        xgboost_metrics = evaluate_model(
            working_table,
            xgboost_model,
            xgboost_selected_columns,
            {**evaluation_params, "split": "validation"},
        )
        scorecard = build_model_selection_scorecard(
            working_table,
            baseline_model,
            baseline_metrics,
            optimized_model,
            optimized_metrics,
            xgboost_model,
            xgboost_metrics,
            columns,
            decision_policy_params,
            nested_selection_params,
        )
        best_model_config = select_best_model(
            baseline_model,
            baseline_metrics,
            optimized_model,
            optimized_metrics,
            xgboost_model,
            xgboost_metrics,
            nested_selection_params,
            scorecard,
        )
        threshold_metrics = build_threshold_metrics(
            best_model_config,
            baseline_model,
            optimized_model,
            xgboost_model,
            working_table,
            columns,
            decision_policy_params,
        )
        selected_policy = select_deployment_policy(
            best_model_config,
            threshold_metrics,
            decision_policy_params,
        )
        refit_config = enrich_best_model_config_with_policy(
            best_model_config,
            selected_policy,
        )
        selected_feature_lookup = {
            "baseline": (
                baseline_selected_columns,
                baseline_feature_manifest,
            ),
            "optimized": (
                optimized_selected_columns,
                optimized_feature_manifest,
            ),
            "xgboost": (
                xgboost_selected_columns,
                xgboost_feature_manifest,
            ),
        }
        selected_columns, feature_manifest = selected_feature_lookup.get(
            str(best_model_config.get("model_name", "baseline")),
            selected_feature_lookup["baseline"],
        )
        refit_artifact = train_model(
            working_table,
            selected_columns,
            refit_config,
            ml_runtime,
        )
        refit_artifact = calibrate_model(
            working_table,
            refit_artifact,
            columns,
            calibration_params,
        )
        selected_policy_name = str(selected_policy.get("decision_policy_name", "default_050"))
        selected_policy_cfg = dict(
            (decision_policy_params.get("policies") or {}).get(selected_policy_name, {})
        )
        y_outer_test = refit_artifact["target_encoder"].transform(
            outer_test_frame[target_column]
        )
        outer_probabilities = _predict_scores(
            refit_artifact["estimator"],
            outer_test_frame[refit_artifact["feature_columns"]],
        )
        outer_payload = _merged_metric_payload(
            y_outer_test,
            outer_probabilities,
            threshold=float(selected_policy.get("decision_threshold", 0.5)),
            false_negative_cost=float(
                selected_policy_cfg.get("false_negative_cost", 1.0)
            ),
            false_positive_cost=float(
                selected_policy_cfg.get("false_positive_cost", 1.0)
            ),
        )
        selected_model_name = str(best_model_config.get("model_name", "unknown"))
        selected_score_row = scorecard[scorecard["model_name"] == selected_model_name]
        selected_score = (
            float(selected_score_row.iloc[0]["selection_composite_score"])
            if not selected_score_row.empty
            else float("nan")
        )
        rows.append(
            {
                "fold_id": int(fold_id),
                "outer_train_rows": int(len(outer_train_frame)),
                "outer_test_rows": int(len(outer_test_frame)),
                "selected_model_name": selected_model_name,
                "selected_class_path": refit_artifact["class_path"],
                "selection_metric": str(
                    best_model_config.get(
                        "selection_metric",
                        "composite_probability_policy_score",
                    )
                ),
                "selection_score": selected_score,
                "selected_policy_name": selected_policy_name,
                "selected_policy_threshold": float(
                    selected_policy.get("decision_threshold", 0.5)
                ),
                "selected_feature_count": int(
                    feature_manifest.get("selected_feature_count", 0)
                ),
                "selected_feature_names_text": feature_manifest.get(
                    "selected_feature_names_text",
                    "",
                ),
                "candidate_feature_sets_evaluated": int(
                    feature_manifest.get("candidate_count", 0)
                ),
                **outer_payload,
            }
        )

    return pd.DataFrame(rows)


def build_modelling_design_audit(  # noqa: PLR0913
    raw_columns: dict[str, list[str]],
    columns: dict[str, list[str]],
    selected_feature_columns: dict[str, list[str]],
    feature_selection_manifest: dict[str, Any] | None,
    model_selection_params: dict[str, Any],
    decision_policy_params: dict[str, Any],
    perturbation_sensitivity_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Materialize config-level checks for leakage, circularity, and robustness coverage."""
    raw_input_names = set(raw_columns.get("categorical", []) + raw_columns.get("numerical", []))
    selected_feature_names = set(
        selected_feature_columns.get("categorical", [])
        + selected_feature_columns.get("numerical", [])
    )
    target_names = set(columns.get("target", []))
    derived_feature_names = set(
        columns.get("categorical", []) + columns.get("numerical", [])
    ) - raw_input_names
    selection_splits = set((feature_selection_manifest or {}).get("selection_splits", []))
    candidate_count = int((feature_selection_manifest or {}).get("candidate_count", 0))
    refit_splits = set(model_selection_params.get("refit_train_splits", []))
    policy_selection_split = str(
        decision_policy_params.get("policy_selection_split", "validation")
    )
    selected_engineered_features = sorted(selected_feature_names & derived_feature_names)
    rows = [
        {
            "check_name": "selected_features_exclude_target",
            "passed_flag": int(selected_feature_names.isdisjoint(target_names)),
            "detail": "Selected modelling features do not include the target column.",
        },
        {
            "check_name": "selected_features_exclude_split_marker",
            "passed_flag": int(SPLIT_COLUMN not in selected_feature_names),
            "detail": "Selected modelling features do not include the split marker.",
        },
        {
            "check_name": "feature_selection_train_only",
            "passed_flag": int(selection_splits.issubset({"train"})),
            "detail": f"Feature selection splits={sorted(selection_splits) or ['train']}.",
        },
        {
            "check_name": "multiple_feature_combinations_tested",
            "passed_flag": int(candidate_count > 1),
            "detail": f"Feature-selection candidate_count={candidate_count}.",
        },
        {
            "check_name": "policy_selection_not_on_test",
            "passed_flag": int(policy_selection_split != "test"),
            "detail": f"Policy threshold selection split={policy_selection_split}.",
        },
        {
            "check_name": "test_reserved_until_refit",
            "passed_flag": int("test" in refit_splits and policy_selection_split != "test"),
            "detail": (
                "The test split is excluded from selection/policy choice and only enters the final production refit."
            ),
        },
        {
            "check_name": "derived_features_declared_explicitly",
            "passed_flag": int(set(selected_engineered_features).issubset(derived_feature_names)),
            "detail": (
                "Selected engineered features="
                + (", ".join(selected_engineered_features) if selected_engineered_features else "none")
            ),
        },
        {
            "check_name": "sensitivity_audit_materialized",
            "passed_flag": int(
                perturbation_sensitivity_summary is not None
                and not perturbation_sensitivity_summary.empty
            ),
            "detail": "Perturbation sensitivity summary exists for the selected model.",
        },
    ]
    audit = pd.DataFrame(rows)
    audit["status"] = audit["passed_flag"].map({1: "pass", 0: "review"})
    return audit


def build_threshold_metrics(  # noqa: PLR0913
    best_model_config: dict[str, Any],
    baseline_model: ModelArtifact,
    optimized_model: ModelArtifact,
    xgboost_model: ModelArtifact,
    master_table: pd.DataFrame,
    columns: dict[str, list[str]],
    decision_policy_params: dict[str, Any],
) -> pd.DataFrame:
    """Compara políticas de cutoff no split de validação do modelo selecionado."""
    model_name, model_artifact = _find_model_name_and_artifact(
        best_model_config,
        baseline_model,
        optimized_model,
        xgboost_model,
    )
    split_name = decision_policy_params.get("policy_selection_split", "validation")
    _feature_columns, target_column = _get_feature_and_target_columns(columns)
    split_frame = master_table[master_table[SPLIT_COLUMN] == split_name]

    if split_frame.empty:
        return pd.DataFrame()

    estimator = model_artifact["estimator"]
    target_encoder = model_artifact["target_encoder"]
    y_true = target_encoder.transform(split_frame[target_column])
    probabilities = _predict_scores(
        estimator,
        split_frame[model_artifact["feature_columns"]],
    )

    rows: list[dict[str, Any]] = []
    policies = decision_policy_params.get("policies") or {}
    for policy_name, policy_cfg in policies.items():
        selected = _select_policy_threshold(
            y_true,
            probabilities,
            policy_name,
            dict(policy_cfg),
            decision_policy_params,
        )
        rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                **selected,
            }
        )

    return pd.DataFrame(rows)


def select_deployment_policy(
    _best_model_config: dict[str, Any],
    threshold_metrics: pd.DataFrame,
    decision_policy_params: dict[str, Any],
) -> dict[str, Any]:
    """Seleciona a política de produção e anexa o catálogo completo de políticas."""
    if threshold_metrics.empty:
        return {
            "decision_policy_name": decision_policy_params.get(
                "deployment_policy",
                "default_050",
            ),
            "decision_threshold": 0.5,
            "decision_policy_description": "",
            "policy_catalog": [],
            "risk_bands": decision_policy_params.get("risk_bands", []),
        }

    deployment_policy_name = decision_policy_params.get(
        "deployment_policy",
        "default_050",
    )
    records = threshold_metrics.to_dict(orient="records")
    selected = next(
        (row for row in records if row["policy_name"] == deployment_policy_name),
        records[0],
    )

    return {
        "model_name": selected["model_name"],
        "split": selected["split"],
        "decision_policy_name": selected["policy_name"],
        "decision_policy_description": selected.get("policy_description", ""),
        "decision_threshold": float(selected["threshold"]),
        "policy_catalog": records,
        "risk_bands": decision_policy_params.get("risk_bands", []),
    }


def enrich_best_model_config_with_policy(
    best_model_config: dict[str, Any],
    selected_deployment_policy: dict[str, Any],
) -> dict[str, Any]:
    """Anexa threshold/política selecionada ao config usado no refit de produção."""
    return {
        **best_model_config,
        "decision_threshold": float(
            selected_deployment_policy.get("decision_threshold", 0.5)
        ),
        "decision_policy_name": selected_deployment_policy.get(
            "decision_policy_name",
            "default_050",
        ),
        "decision_policy_description": selected_deployment_policy.get(
            "decision_policy_description",
            "",
        ),
        "policy_catalog": selected_deployment_policy.get("policy_catalog", []),
        "risk_bands": selected_deployment_policy.get("risk_bands", []),
    }


def evaluate_all_on_test(  # noqa: PLR0913
    master_table: pd.DataFrame,
    baseline_model: ModelArtifact,
    optimized_model: ModelArtifact,
    xgboost_model: ModelArtifact,
    columns: dict[str, list[str]],
    evaluation_params: dict[str, Any],
) -> dict[str, Any]:
    """Relatório no teste — reusa o mesmo bloco `evaluation` com split=test."""
    test_eval = {**evaluation_params, "split": "test"}
    report: dict[str, Any] = {}

    for name, model in [
        ("baseline", baseline_model),
        ("optimized", optimized_model),
        ("xgboost", xgboost_model),
    ]:
        metrics = evaluate_model(master_table, model, columns, test_eval)
        report[name] = metrics
        logger.info(
            "test_report: %-10s roc_auc=%.4f f1=%.4f recall=%.4f",
            name,
            metrics.get("roc_auc", 0),
            metrics.get("f1", 0),
            metrics.get("recall", 0),
        )

    return report


def calibrate_model(
    production_master_table: pd.DataFrame,
    model_artifact: ModelArtifact,
    columns: dict[str, list[str]],
    calibration_params: dict[str, Any] | None,
) -> ModelArtifact:
    """Calibração — classe e init_args no YAML (`refit.calibration`)."""
    calibration_params = calibration_params or {"enabled": False}
    if not bool(calibration_params.get("enabled", True)):
        return model_artifact

    feature_columns = model_artifact["feature_columns"]
    target_encoder = model_artifact["target_encoder"]
    _, target_column = _get_feature_and_target_columns(columns)

    x = production_master_table[feature_columns]
    y = target_encoder.transform(production_master_table[target_column])

    cal_class_path = calibration_params.get(
        "class_path",
        "sklearn.calibration.CalibratedClassifierCV",
    )
    cal_kwargs = dict(calibration_params.get("init_args") or {})
    cv_value = cal_kwargs.get("cv", 5)
    if isinstance(cv_value, str) and cv_value.lower() == "prefit":
        cal_class = load_class(cal_class_path)
        calibrated = cal_class(estimator=model_artifact["estimator"], **cal_kwargs)
        with _suppress_known_training_warnings():
            calibrated.fit(x, y)
        calibration_state = {
            "enabled": True,
            "class_path": cal_class_path,
            "init_args": cal_kwargs,
        }
        resolved_init_args = dict(model_artifact.get("init_args", {}))
    else:
        calibrated, resolved_init_args, calibration_state = (
            _fit_estimator_with_optional_calibration(
                model_artifact["class_path"],
                _merged_model_init_args(model_artifact),
                x,
                y,
                calibration_params,
            )
        )

    logger.info(
        "calibrate_model: %s init_args=%s",
        cal_class_path,
        cal_kwargs,
    )

    return {
        **model_artifact,
        "estimator": calibrated,
        "init_args": resolved_init_args,
        "calibration": calibration_state,
    }
