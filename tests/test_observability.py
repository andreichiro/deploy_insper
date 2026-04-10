"""Tests for data observability helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from insper_deploy_kedro.pipelines.data_engineering import observability


def test_build_data_contract_report_flags_missing_and_unexpected_columns():
    cleaned = pd.DataFrame(
        {
            "Pregnancies": [1, 2],
            "Glucose": [90, 100],
            "Unexpected": ["x", "y"],
        }
    )
    raw_columns = {
        "numerical": ["Pregnancies", "Glucose", "BMI"],
        "target": ["Outcome"],
        "categorical": [],
    }

    report = observability.build_data_contract_report(cleaned, raw_columns)

    status_by_column = dict(zip(report["column_name"], report["contract_status"]))
    assert status_by_column["Pregnancies"] == "ok"
    assert status_by_column["BMI"] == "missing"
    assert status_by_column["Outcome"] == "missing"
    assert status_by_column["Unexpected"] == "unexpected"


def test_build_data_freshness_report_marks_missing_source(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(observability, "PROJECT_ROOT", tmp_path)

    report = observability.build_data_freshness_report(
        {
            "freshness": {
                "source_path": "data/01_raw/missing.csv",
                "warning_age_hours": 24,
                "max_age_hours": 48,
            }
        }
    )

    assert report.iloc[0]["status"] == "missing"
    assert bool(report.iloc[0]["exists"]) is False


def test_build_data_freshness_report_marks_fresh_file(tmp_path: Path, monkeypatch):
    source = tmp_path / "data" / "01_raw" / "sample.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setattr(observability, "PROJECT_ROOT", tmp_path)

    report = observability.build_data_freshness_report(
        {
            "freshness": {
                "source_path": "data/01_raw/sample.csv",
                "warning_age_hours": 24,
                "max_age_hours": 48,
            }
        }
    )

    assert report.iloc[0]["status"] == "fresh"
    assert bool(report.iloc[0]["exists"]) is True


def test_build_data_drift_report_returns_drift_rows():
    split_data = pd.DataFrame(
        {
            "split": ["train"] * 4 + ["validation"] * 4 + ["test"] * 4,
            "Glucose": [80, 85, 90, 95, 160, 170, 180, 190, 88, 92, 95, 97],
            "BMI": [25, 26, 27, 28, 35, 36, 37, 38, 26, 27, 28, 29],
        }
    )
    columns = {"numerical": ["Glucose", "BMI"]}

    report = observability.build_data_drift_report(
        split_data,
        columns,
        {
            "drift": {
                "reference_split": "train",
                "compare_splits": ["validation", "test"],
                "psi_bins": 5,
                "stable_max": 0.1,
                "moderate_max": 0.25,
            }
        },
    )

    assert not report.empty
    assert {"feature_name", "comparison_split", "psi", "drift_label"}.issubset(
        report.columns
    )
    validation_glucose = report[
        (report["comparison_split"] == "validation")
        & (report["feature_name"] == "Glucose")
    ].iloc[0]
    assert validation_glucose["drift_label"] in {"monitor", "drift"}
