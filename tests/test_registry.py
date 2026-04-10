"""Tests for experiment and model registry helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from insper_deploy_kedro import ops_store, registry


@pytest.fixture()
def isolated_registry_store(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ops_store, "OPS_DB_PATH", tmp_path / "ops.sqlite3")
    monkeypatch.setattr(ops_store, "_initialized", False)
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path / "models")
    registry.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    yield
    monkeypatch.setattr(ops_store, "_initialized", False)


def test_record_experiment_run_persists_selected_model(isolated_registry_store):
    record = registry.record_experiment_run(
        model_frontier=pd.DataFrame(
            [
                {
                    "model_name": "optimized",
                    "selected_for_refit": True,
                    "validation_roc_auc": 0.84,
                }
            ]
        ),
        threshold_metrics=pd.DataFrame(
            [{"policy_name": "prioritize_recall", "threshold": 0.25}]
        ),
        cv_metric_summary=pd.DataFrame(
            [{"model_name": "optimized", "metric_name": "roc_auc"}]
        ),
        selected_deployment_policy={
            "model_name": "optimized",
            "decision_policy_name": "prioritize_recall",
            "decision_threshold": 0.25,
        },
        best_model_config={"class_path": "catboost.CatBoostClassifier"},
        model_selection={"metric": "roc_auc"},
    )

    assert record["selected_model_name"] == "optimized"
    assert ops_store.list_experiment_runs(limit=1)[0]["selected_threshold"] == 0.25


def test_record_model_registry_entry_tracks_artifact_inventory(
    isolated_registry_store,
):
    for filename in (
        "production_encoders.pkl",
        "production_scalers.pkl",
        "raw_production_model.pkl",
        "production_model.pkl",
    ):
        (registry.MODELS_DIR / filename).write_bytes(b"artifact")

    record = registry.record_model_registry_entry(
        {
            "class_path": "catboost.CatBoostClassifier",
            "train_splits": ["train", "validation", "test"],
            "decision_policy_name": "prioritize_recall",
            "decision_threshold": 0.25,
            "risk_bands": [{"label": "Alto risco"}],
        }
    )

    assert record["stage"] == "production"
    assert len(record["artifact_inventory"]) == 4
    assert all(item["exists"] for item in record["artifact_inventory"])
    assert ops_store.list_model_registry_entries(limit=1)[0]["class_path"] == (
        "catboost.CatBoostClassifier"
    )


def test_build_training_run_manifest_includes_sensitivity_summary(
    isolated_registry_store,
):
    manifest = registry.build_training_run_manifest(
        latest_experiment_run={
            "experiment_id": "exp-1",
            "created_at": "2026-04-02T00:00:00+00:00",
            "selected_model_name": "optimized",
            "selected_class_path": "catboost.CatBoostClassifier",
            "selection_metric": "roc_auc",
            "selection_score": 0.84,
            "selected_policy_name": "prioritize_recall",
            "selected_threshold": 0.25,
            "best_model_config": {"class_path": "catboost.CatBoostClassifier"},
            "frontier": [],
            "threshold_metrics": [],
            "cv_metric_summary": [],
        },
        bootstrap_metric_intervals=pd.DataFrame(
            [
                {
                    "model_name": "optimized",
                    "metric_name": "roc_auc",
                    "ci_width": 0.04,
                }
            ]
        ),
        permutation_feature_importance=pd.DataFrame(
            [
                {
                    "feature_name": "Glucose",
                    "metric_name": "roc_auc",
                    "importance_mean": 0.08,
                }
            ]
        ),
        perturbation_sensitivity_summary=pd.DataFrame(
            [
                {
                    "feature_name": "Glucose",
                    "mean_sensitivity_ratio": 0.4,
                    "max_decision_flip_rate": 0.1,
                    "sensitivity_label": "proportional",
                }
            ]
        ),
        split_strategy_report=pd.DataFrame(
            [
                {
                    "requested_strategy_kind": "stratified_random",
                    "resolved_strategy_kind": "stratified_random",
                }
            ]
        ),
        feature_selection_frontier=pd.DataFrame(
            [
                {
                    "feature_names_text": "Glucose, BMI",
                    "selected_flag": 1,
                    "mean_brier": 0.18,
                }
            ]
        ),
        feature_selection_stability=pd.DataFrame(
            [
                {
                    "entity_type": "feature",
                    "entity_name": "Glucose",
                    "selection_frequency": 1.0,
                }
            ]
        ),
        feature_selection_manifest={
            "manifest_type": "feature_selection",
            "selected_feature_names": ["Glucose", "BMI"],
        },
    )

    assert manifest["manifest_type"] == "training_run"
    assert manifest["experiment_id"] == "exp-1"
    assert manifest["perturbation_sensitivity_summary"][0]["feature_name"] == (
        "Glucose"
    )
    assert manifest["bootstrap_metric_intervals"][0]["metric_name"] == "roc_auc"
    assert manifest["feature_selection_manifest"]["selected_feature_names"] == [
        "Glucose",
        "BMI",
    ]


def test_build_serving_manifest_and_inference_contract():
    production_model = {
        "class_path": "catboost.CatBoostClassifier",
        "feature_columns": ["Pregnancies", "Glucose", "glucose_bmi_interaction"],
        "train_splits": ["train", "validation", "test"],
        "decision_policy_name": "prioritize_recall",
        "decision_policy_description": "Reduce falsos negativos",
        "decision_threshold": 0.25,
        "risk_bands": [{"label": "Alto risco"}],
        "policy_catalog": [{"policy_name": "prioritize_recall"}],
    }
    serving_manifest = registry.build_serving_manifest(
        production_model,
        {
            "registry_id": "registry-1",
            "created_at": "2026-04-02T00:00:00+00:00",
            "artifact_inventory": [{"path": "data/06_models/production_model.pkl"}],
        },
    )
    inference_contract = registry.build_inference_contract(
        production_model,
        {
            "categorical": [],
            "numerical": ["Pregnancies", "Glucose", "BMI"],
        },
        {
            "target": ["Outcome"],
        },
    )

    assert serving_manifest["manifest_type"] == "serving_manifest"
    assert serving_manifest["decision_threshold"] == 0.25
    assert inference_contract["manifest_type"] == "inference_contract"
    assert inference_contract["derived_feature_columns"] == ["glucose_bmi_interaction"]
    assert inference_contract["outputs"][0]["name"] == "prediction"
