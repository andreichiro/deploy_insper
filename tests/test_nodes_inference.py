"""Testes dos nodes de inferência — to_dataframe, predict e risk report."""

from __future__ import annotations

import pandas as pd

from insper_deploy_kedro.pipelines.inference.nodes import (
    build_risk_report,
    predict,
    to_dataframe,
)


class TestToDataframe:
    def test_single_dict(self):
        result = to_dataframe({"a": 1, "b": 2})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert list(result.columns) == ["a", "b"]

    def test_list_of_dicts(self):
        result = to_dataframe([{"a": 1}, {"a": 2}])
        assert len(result) == 2

    def test_empty_list_returns_empty_dataframe(self):
        result = to_dataframe([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestPredict:
    def test_returns_prediction_column(self, master_table, trained_model):
        result = predict(master_table, trained_model)
        assert "prediction" in result.columns
        assert len(result) == len(master_table)

    def test_returns_proba_for_sklearn(self, master_table, trained_model):
        result = predict(master_table, trained_model)
        assert "prediction_proba" in result.columns
        assert result["prediction_proba"].between(0, 1).all()

    def test_predictions_are_valid_labels(
        self, sample_raw_data, master_table, trained_model
    ):
        result = predict(master_table, trained_model)
        valid_labels = set(sample_raw_data["Outcome"].unique())
        assert set(result["prediction"].unique()).issubset(valid_labels)

    def test_preserves_original_columns(self, master_table, trained_model):
        original_cols = set(master_table.columns)
        result = predict(master_table, trained_model)
        assert original_cols.issubset(set(result.columns))

    def test_uses_decision_threshold_when_present(self, master_table, trained_model):
        baseline = predict(master_table, trained_model)
        thresholded_artifact = {
            **trained_model,
            "decision_threshold": 0.99,
        }
        thresholded = predict(master_table, thresholded_artifact)

        assert "prediction_proba" in thresholded.columns
        assert thresholded["prediction"].sum() <= baseline["prediction"].sum()

    def test_adds_risk_score_and_risk_band_when_configured(
        self, master_table, trained_model
    ):
        artifact = {
            **trained_model,
            "risk_bands": [
                {
                    "name": "low",
                    "label": "Baixo risco",
                    "min_probability": 0.0,
                    "max_probability": 0.5,
                },
                {
                    "name": "high",
                    "label": "Alto risco",
                    "min_probability": 0.5,
                    "max_probability": 1.01,
                },
            ],
        }
        result = predict(master_table, artifact)

        assert "risk_score" in result.columns
        assert "risk_band" in result.columns


class TestBuildRiskReport:
    def test_builds_readable_ranked_report(self):
        source = pd.DataFrame(
            [
                {"Glucose": 148, "BMI": 33.6, "Age": 50},
                {"Glucose": 95, "BMI": 24.2, "Age": 31},
            ]
        )
        predictions = pd.DataFrame(
            [
                {
                    "prediction": 1,
                    "prediction_proba": 0.91,
                    "risk_score": 91.0,
                    "risk_band": "Alto risco",
                },
                {
                    "prediction": 0,
                    "prediction_proba": 0.20,
                    "risk_score": 20.0,
                    "risk_band": "Baixo risco",
                },
            ]
        )
        model_artifact = {
            "decision_policy_name": "prioritize_recall",
            "decision_threshold": 0.3,
        }

        report = build_risk_report(source, predictions, model_artifact)

        assert list(report["risk_priority_rank"]) == [1, 2]
        assert list(report["case_id"]) == [1, 2]
        assert list(report["prediction_label"]) == ["Positivo", "Negativo"]
        assert list(report["decision_policy_name"].unique()) == ["prioritize_recall"]
        assert "recommended_action" in report.columns
