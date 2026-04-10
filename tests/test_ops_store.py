"""Tests for the persistent local ops store."""

from __future__ import annotations

from pathlib import Path

import pytest

from insper_deploy_kedro import ops_store


@pytest.fixture()
def isolated_ops_store(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ops_store, "OPS_DB_PATH", tmp_path / "ops.sqlite3")
    monkeypatch.setattr(ops_store, "_initialized", False)
    yield
    monkeypatch.setattr(ops_store, "_initialized", False)


def test_background_run_round_trip(isolated_ops_store):
    ops_store.upsert_background_run(
        "run-1",
        {
            "pipeline": "train",
            "status": "completed",
            "started_at": "2026-04-01T00:00:00+00:00",
            "finished_at": "2026-04-01T00:01:00+00:00",
            "error": None,
            "result": {"model_loaded": True},
            "updated_at": "2026-04-01T00:01:00+00:00",
        },
    )

    payload = ops_store.get_background_run("run-1")

    assert payload == {
        "pipeline": "train",
        "status": "completed",
        "started_at": "2026-04-01T00:00:00+00:00",
        "finished_at": "2026-04-01T00:01:00+00:00",
        "error": None,
        "result": {"model_loaded": True},
    }


def test_experiment_registry_round_trip(isolated_ops_store):
    record = ops_store.record_experiment_run(
        {
            "experiment_id": "exp-1",
            "created_at": "2026-04-01T00:00:00+00:00",
            "selected_model_name": "optimized",
            "selected_class_path": "catboost.CatBoostClassifier",
            "selection_metric": "roc_auc",
            "selection_score": 0.84,
            "selected_policy_name": "prioritize_recall",
            "selected_threshold": 0.25,
        }
    )

    rows = ops_store.list_experiment_runs()

    assert rows == [record]


def test_model_registry_round_trip(isolated_ops_store):
    record = ops_store.record_model_registry_entry(
        {
            "registry_id": "registry-1",
            "created_at": "2026-04-01T00:00:00+00:00",
            "stage": "production",
            "class_path": "catboost.CatBoostClassifier",
            "train_splits": ["train", "validation", "test"],
            "decision_policy_name": "prioritize_recall",
            "decision_threshold": 0.25,
        }
    )

    rows = ops_store.list_model_registry_entries()

    assert rows == [record]
