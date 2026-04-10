"""Data observability helpers: freshness, schema contract, and split drift."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from insper_deploy_kedro.constants import SPLIT_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[4]
_MINIMUM_BIN_EDGES = 2


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _safe_relative_delta(reference: float, current: float) -> float:
    if np.isclose(reference, 0.0):
        return 0.0 if np.isclose(current, 0.0) else 100.0
    return float(abs((current - reference) / reference) * 100.0)


def _population_stability_index(
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
    bins: int,
) -> float:
    if len(reference_values) == 0 or len(comparison_values) == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, int(bins) + 1)
    edges = np.quantile(reference_values, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    unique_edges = np.unique(edges)
    if len(unique_edges) < _MINIMUM_BIN_EDGES:
        return 0.0

    reference_hist, _ = np.histogram(reference_values, bins=unique_edges)
    comparison_hist, _ = np.histogram(comparison_values, bins=unique_edges)

    reference_ratio = reference_hist / max(reference_hist.sum(), 1)
    comparison_ratio = comparison_hist / max(comparison_hist.sum(), 1)

    epsilon = 1e-6
    reference_ratio = np.clip(reference_ratio, epsilon, None)
    comparison_ratio = np.clip(comparison_ratio, epsilon, None)
    return float(np.sum((comparison_ratio - reference_ratio) * np.log(comparison_ratio / reference_ratio)))


def build_data_freshness_report(data_quality: dict[str, Any]) -> pd.DataFrame:
    """Capture raw-source freshness against the configured SLA."""
    freshness_cfg = data_quality.get("freshness") or {}
    source_path = _resolve_project_path(
        freshness_cfg.get("source_path", "data/01_raw/diabetes-dataset-modelling.csv")
    )
    warning_age_hours = float(freshness_cfg.get("warning_age_hours", 24.0))
    max_age_hours = float(freshness_cfg.get("max_age_hours", warning_age_hours * 2))
    now = datetime.now(UTC)

    if not source_path.exists():
        return pd.DataFrame(
            [
                {
                    "source_path": str(source_path),
                    "exists": False,
                    "status": "missing",
                    "age_hours": None,
                    "warning_age_hours": warning_age_hours,
                    "max_age_hours": max_age_hours,
                }
            ]
        )

    modified_at = datetime.fromtimestamp(source_path.stat().st_mtime, tz=UTC)
    age_hours = float((now - modified_at).total_seconds() / 3600.0)
    status = "fresh"
    if age_hours > max_age_hours:
        status = "stale"
    elif age_hours > warning_age_hours:
        status = "warning"

    return pd.DataFrame(
        [
            {
                "source_path": str(source_path),
                "exists": True,
                "status": status,
                "age_hours": age_hours,
                "warning_age_hours": warning_age_hours,
                "max_age_hours": max_age_hours,
                "modified_at": modified_at.isoformat(),
                "checked_at": now.isoformat(),
            }
        ]
    )


def build_data_contract_report(
    cleaned_data: pd.DataFrame,
    raw_columns: dict[str, list[str]],
) -> pd.DataFrame:
    """Materialize a simple schema contract report for the cleaned dataset."""
    expected_columns = {
        column_name: group_name
        for group_name, group_columns in raw_columns.items()
        for column_name in group_columns
    }

    rows: list[dict[str, Any]] = []
    for column_name, group_name in expected_columns.items():
        present = column_name in cleaned_data.columns
        series = cleaned_data[column_name] if present else pd.Series(dtype="object")
        rows.append(
            {
                "column_name": column_name,
                "contract_group": group_name,
                "present": present,
                "observed_dtype": str(series.dtype) if present else None,
                "null_ratio": float(series.isna().mean()) if present and len(series) else 0.0,
                "contract_status": "ok" if present else "missing",
            }
        )

    for column_name in sorted(set(cleaned_data.columns) - set(expected_columns)):
        series = cleaned_data[column_name]
        rows.append(
            {
                "column_name": column_name,
                "contract_group": "unexpected",
                "present": True,
                "observed_dtype": str(series.dtype),
                "null_ratio": float(series.isna().mean()) if len(series) else 0.0,
                "contract_status": "unexpected",
            }
        )

    return pd.DataFrame(rows)


def build_data_drift_report(
    split_data: pd.DataFrame,
    columns: dict[str, list[str]],
    data_quality: dict[str, Any],
) -> pd.DataFrame:
    """Compare numerical feature behavior across splits with a PSI-based drift score."""
    drift_cfg = data_quality.get("drift") or {}
    reference_split = drift_cfg.get("reference_split", "train")
    compare_splits = list(drift_cfg.get("compare_splits", ["validation", "test"]))
    psi_bins = int(drift_cfg.get("psi_bins", 10))
    stable_max = float(drift_cfg.get("stable_max", 0.1))
    moderate_max = float(drift_cfg.get("moderate_max", 0.25))

    reference_frame = split_data[split_data[SPLIT_COLUMN] == reference_split]
    if reference_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for compare_split in compare_splits:
        comparison_frame = split_data[split_data[SPLIT_COLUMN] == compare_split]
        if comparison_frame.empty:
            continue

        for column_name in columns.get("numerical", []):
            if column_name not in reference_frame or column_name not in comparison_frame:
                continue

            reference_values = reference_frame[column_name].dropna().to_numpy(dtype=float)
            comparison_values = comparison_frame[column_name].dropna().to_numpy(dtype=float)
            if len(reference_values) == 0 or len(comparison_values) == 0:
                continue

            psi_value = _population_stability_index(
                reference_values,
                comparison_values,
                bins=psi_bins,
            )
            if psi_value <= stable_max:
                drift_label = "stable"
            elif psi_value <= moderate_max:
                drift_label = "monitor"
            else:
                drift_label = "drift"

            rows.append(
                {
                    "reference_split": reference_split,
                    "comparison_split": compare_split,
                    "feature_name": column_name,
                    "reference_mean": float(np.mean(reference_values)),
                    "comparison_mean": float(np.mean(comparison_values)),
                    "reference_std": float(np.std(reference_values, ddof=0)),
                    "comparison_std": float(np.std(comparison_values, ddof=0)),
                    "mean_delta_pct": _safe_relative_delta(
                        float(np.mean(reference_values)),
                        float(np.mean(comparison_values)),
                    ),
                    "psi": psi_value,
                    "drift_label": drift_label,
                }
            )

    return pd.DataFrame(rows)
