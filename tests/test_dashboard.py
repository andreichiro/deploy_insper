"""Focused unit tests for pure dashboard helpers."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from insper_deploy_kedro import dashboard


class _FakeBlock:
    def __init__(self, root):
        self.root = root

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, name):
        return getattr(self.root, name)


class _FakeStreamlit:
    def __init__(self, *, pressed_buttons=None, selectbox_value=None):
        self.calls: list[tuple[str, tuple, dict]] = []
        self.pressed_buttons = set(pressed_buttons or [])
        self.selectbox_value = selectbox_value
        self.sidebar = _FakeBlock(self)
        self.cache_data = mock.Mock()
        self.cache_resource = mock.Mock()

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def markdown(self, *args, **kwargs):
        self._record("markdown", *args, **kwargs)

    def write(self, *args, **kwargs):
        self._record("write", *args, **kwargs)

    def metric(self, *args, **kwargs):
        self._record("metric", *args, **kwargs)

    def subheader(self, *args, **kwargs):
        self._record("subheader", *args, **kwargs)

    def header(self, *args, **kwargs):
        self._record("header", *args, **kwargs)

    def caption(self, *args, **kwargs):
        self._record("caption", *args, **kwargs)

    def dataframe(self, *args, **kwargs):
        self._record("dataframe", *args, **kwargs)

    def bar_chart(self, *args, **kwargs):
        self._record("bar_chart", *args, **kwargs)

    def line_chart(self, *args, **kwargs):
        self._record("line_chart", *args, **kwargs)

    def info(self, *args, **kwargs):
        self._record("info", *args, **kwargs)

    def warning(self, *args, **kwargs):
        self._record("warning", *args, **kwargs)

    def error(self, *args, **kwargs):
        self._record("error", *args, **kwargs)

    def success(self, *args, **kwargs):
        self._record("success", *args, **kwargs)

    def progress(self, *args, **kwargs):
        self._record("progress", *args, **kwargs)

    def pyplot(self, *args, **kwargs):
        self._record("pyplot", *args, **kwargs)

    def json(self, *args, **kwargs):
        self._record("json", *args, **kwargs)

    def code(self, *args, **kwargs):
        self._record("code", *args, **kwargs)

    def download_button(self, *args, **kwargs):
        self._record("download_button", *args, **kwargs)

    def divider(self, *args, **kwargs):
        self._record("divider", *args, **kwargs)

    def title(self, *args, **kwargs):
        self._record("title", *args, **kwargs)

    def set_page_config(self, *args, **kwargs):
        self._record("set_page_config", *args, **kwargs)

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        self._record("columns", spec)
        return [_FakeBlock(self) for _ in range(count)]

    def tabs(self, labels):
        self._record("tabs", labels)
        return [_FakeBlock(self) for _ in labels]

    def empty(self):
        self._record("empty")
        return _FakeBlock(self)

    def selectbox(self, _label, options, *args, **kwargs):
        self._record("selectbox", options, *args, **kwargs)
        return self.selectbox_value or options[0]

    def button(self, label, *args, **kwargs):
        self._record("button", label, *args, **kwargs)
        return label in self.pressed_buttons

    def number_input(self, _label, *, value, **kwargs):
        self._record("number_input", value, **kwargs)
        return value

    def rerun(self):
        self._record("rerun")

    def exception(self, *args, **kwargs):
        self._record("exception", *args, **kwargs)


def _patch_streamlit(monkeypatch, *, pressed_buttons=None, selectbox_value=None):
    fake = _FakeStreamlit(
        pressed_buttons=pressed_buttons,
        selectbox_value=selectbox_value,
    )
    monkeypatch.setattr(dashboard, "st", fake)
    return fake


def test_find_artifact_returns_file_when_path_points_to_file(tmp_path: Path):
    file_path = tmp_path / "artifact.pkl"
    file_path.write_bytes(b"artifact")

    assert dashboard._find_artifact(file_path) == file_path


def test_find_artifact_resolves_latest_versioned_layout(tmp_path: Path):
    base_dir = tmp_path / "model_frontier.parquet"
    base_dir.mkdir()
    older = base_dir / "2026-03-01T10.00.00.000Z"
    newer = base_dir / "2026-03-02T10.00.00.000Z"
    older.mkdir()
    newer.mkdir()
    (older / "model_frontier.parquet").write_bytes(b"old")
    (newer / "model_frontier.parquet").write_bytes(b"new")

    resolved = dashboard._find_artifact(base_dir)

    assert resolved == newer / "model_frontier.parquet"


def test_artifact_signature_tracks_size_and_timestamp(tmp_path: Path):
    file_path = tmp_path / "artifact.parquet"
    file_path.write_bytes(b"abc")

    signature = dashboard._artifact_signature([file_path])

    assert len(signature) == 1
    assert signature[0][0] == str(file_path)
    assert signature[0][2] == 3


def test_selected_policy_from_artifacts_reads_model_metadata():
    policy = dashboard._selected_policy_from_artifacts(
        {
            "model": {
                "decision_policy_name": "prioritize_recall",
                "decision_policy_description": "Preferir recall na triagem clinica.",
                "decision_threshold": 0.25,
                "policy_catalog": [{"policy_name": "prioritize_recall"}],
                "risk_bands": [{"label": "Alto risco"}],
            }
        }
    )

    assert policy is not None
    assert policy["model_name"] == "production"
    assert policy["decision_policy_name"] == "prioritize_recall"
    assert policy["decision_policy_description"] == "Preferir recall na triagem clinica."
    assert policy["decision_threshold"] == 0.25
    assert policy["policy_catalog"] == [{"policy_name": "prioritize_recall"}]
    assert policy["risk_bands"] == [{"label": "Alto risco"}]


def test_selected_model_name_prefers_selected_policy():
    frontier = pd.DataFrame(
        [
            {"model_name": "baseline", "selected_for_refit": False},
            {"model_name": "optimized", "selected_for_refit": True},
        ]
    )

    selected = dashboard._selected_model_name(
        frontier,
        {"model_name": "baseline"},
    )

    assert selected == "baseline"


def test_selected_model_name_maps_production_class_back_to_frontier_winner():
    frontier = pd.DataFrame(
        [
            {
                "model_name": "baseline",
                "selected_for_refit": True,
                "class_path": "sklearn.linear_model.LogisticRegression",
            },
            {
                "model_name": "optimized",
                "selected_for_refit": False,
                "class_path": "catboost.CatBoostClassifier",
            },
        ]
    )

    selected = dashboard._selected_model_name(
        frontier,
        {
            "model_name": "production",
            "class_path": "sklearn.linear_model.LogisticRegression",
        },
    )

    assert selected == "baseline"


def test_selected_model_name_falls_back_to_frontier_flag():
    frontier = pd.DataFrame(
        [
            {"model_name": "baseline", "selected_for_refit": False},
            {"model_name": "optimized", "selected_for_refit": True},
        ]
    )

    selected = dashboard._selected_model_name(frontier, None)

    assert selected == "optimized"


def test_build_overview_metrics_formats_policy_and_refit_state():
    frontier = pd.DataFrame(
        [{"model_name": "optimized", "selected_for_refit": True}]
    )

    metrics = dashboard.build_overview_metrics(
        frontier,
        {
            "decision_policy_name": "prioritize_recall",
            "decision_threshold": 0.25,
        },
        {"model": {"train_splits": ["train", "validation", "test"]}},
    )

    assert metrics == {
        "model_name": "CatBoost (Optuna)",
        "policy_name": "prioritize_recall",
        "threshold": "0.25",
        "refit_all_data": "Sim",
    }


def test_prepare_frontier_view_adds_validation_minus_cv_gap():
    frontier = pd.DataFrame(
        [
            {
                "model_name": "baseline",
                "best_cv_score": 0.7,
                "validation_roc_auc": 0.8,
            }
        ]
    )

    view = dashboard.prepare_frontier_view(frontier)

    assert view is not None
    assert view.iloc[0]["model_name"] == "Logistic Regression"
    assert view.iloc[0]["validation_minus_cv_gap"] == pytest.approx(0.1)


def test_prepare_model_selection_scorecard_view_formats_ranked_table():
    view = dashboard.prepare_model_selection_scorecard_view(
        pd.DataFrame(
            [
                {
                    "selection_rank": 1,
                    "model_name": "baseline",
                    "selection_composite_score": 0.81,
                    "selection_policy_label": "Priorizar recall",
                    "selection_policy_threshold": 0.25,
                    "validation_roc_auc": 0.8,
                    "validation_brier": 0.18,
                    "validation_log_loss": 0.52,
                    "validation_calibration_slope_error": 0.1,
                    "policy_recall": 0.9,
                    "policy_false_negative_rate": 0.1,
                    "policy_precision": 0.5,
                    "policy_expected_cost_per_sample": 0.6,
                }
            ]
        )
    )

    assert view is not None
    assert view.iloc[0]["model_name"] == "Logistic Regression"
    assert "selection_composite_score" in view.columns


def test_prepare_split_comparison_view_orders_splits():
    view = dashboard.prepare_split_comparison_view(
        pd.DataFrame(
            [
                {"split": "test", "roc_auc": 0.82, "brier": 0.17},
                {"split": "train", "roc_auc": 0.84, "brier": 0.15},
                {"split": "validation", "roc_auc": 0.78, "brier": 0.19},
            ]
        )
    )

    assert view is not None
    assert list(view["split"]) == ["train", "validation", "test"]


def test_build_risk_report_preview_returns_counts_and_ranked_view():
    predictions = pd.DataFrame(
        [
            {
                "risk_priority_rank": 1,
                "prediction": 1,
                "prediction_label": "Positivo",
                "prediction_proba": 0.91,
                "risk_score": 91.0,
                "risk_band": "Alto risco",
                "recommended_action": "Priorizar revisão clínica",
            },
            {
                "risk_priority_rank": 2,
                "prediction": 0,
                "prediction_label": "Negativo",
                "prediction_proba": 0.20,
                "risk_score": 20.0,
                "risk_band": "Baixo risco",
                "recommended_action": "Manter monitoramento de rotina",
            },
        ]
    )

    risk_counts, ranked = dashboard.build_risk_report_preview(predictions)

    assert risk_counts is not None
    assert ranked is not None
    assert ranked.iloc[0]["risk_priority_rank"] == 1
    assert ranked.iloc[0]["prediction_proba"] == 0.91
    assert set(risk_counts["Faixa"]) == {"Alto risco", "Baixo risco"}


def test_build_test_report_summary_formats_model_names():
    summary = dashboard.build_test_report_summary(
        {
            "baseline": {"roc_auc": 0.7, "recall": 0.6, "precision": 0.5, "f1": 0.55},
            "optimized": {"roc_auc": 0.8, "recall": 0.7, "precision": 0.65, "f1": 0.67},
        },
        pd.DataFrame([{"model_name": "baseline", "selected_for_refit": True}]),
    )

    assert list(summary["model_name"]) == [
        "Logistic Regression",
        "CatBoost (Optuna)",
    ]


def test_build_robustness_chart_frame_pivots_models():
    chart = dashboard.build_robustness_chart_frame(
        pd.DataFrame(
            [
                {"fold_id": 1, "model_name": "baseline", "roc_auc": 0.7},
                {"fold_id": 1, "model_name": "optimized", "roc_auc": 0.8},
            ]
        ),
        "roc_auc",
    )

    assert list(chart.columns) == ["Logistic Regression", "CatBoost (Optuna)"]
    assert float(chart.iloc[0]["CatBoost (Optuna)"]) == 0.8


def test_prepare_threshold_metrics_view_and_active_policy_selection():
    threshold_metrics = pd.DataFrame(
        [
            {
                "policy_name": "default_050",
                "policy_label": "Threshold 0.50",
                "threshold": 0.5,
                "precision": 0.5,
                "recall": 0.8,
                "f1": 0.61,
                "false_negative_rate": 0.2,
                "false_positive_rate": 0.4,
                "expected_cost": 12.0,
                "tp": 8,
                "fp": 4,
                "tn": 6,
                "fn": 2,
            },
            {
                "policy_name": "prioritize_recall",
                "policy_label": "Priorizar recall",
                "threshold": 0.25,
                "precision": 0.3,
                "recall": 1.0,
                "f1": 0.46,
                "false_negative_rate": 0.0,
                "false_positive_rate": 0.7,
                "expected_cost": 7.0,
                "tp": 10,
                "fp": 7,
                "tn": 3,
                "fn": 0,
            },
        ]
    )

    view = dashboard.prepare_threshold_metrics_view(threshold_metrics)
    active = dashboard.select_active_policy_row(
        threshold_metrics,
        {"decision_policy_name": "prioritize_recall"},
    )

    assert view is not None
    assert list(view.columns) == [
        "policy_label",
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_negative_rate",
        "false_positive_rate",
        "expected_cost",
    ]
    assert active is not None
    assert active["policy_name"] == "prioritize_recall"


def test_prepare_perturbation_sensitivity_summary_view_sorts_riskiest_rows_first():
    view = dashboard.prepare_perturbation_sensitivity_summary_view(
        pd.DataFrame(
            [
                {
                    "feature_name": "BMI",
                    "max_decision_flip_rate": 0.05,
                    "max_sensitivity_ratio": 0.4,
                },
                {
                    "feature_name": "Glucose",
                    "max_decision_flip_rate": 0.20,
                    "max_sensitivity_ratio": 0.3,
                },
            ]
        )
    )

    assert view is not None
    assert list(view["feature_name"]) == ["Glucose", "BMI"]


def test_prepare_bootstrap_metric_intervals_view_maps_model_names():
    view = dashboard.prepare_bootstrap_metric_intervals_view(
        pd.DataFrame(
            [
                {
                    "model_name": "baseline",
                    "metric_name": "roc_auc",
                    "ci_width": 0.05,
                }
            ]
        )
    )

    assert view is not None
    assert view.iloc[0]["model_name"] == "Logistic Regression"


def test_prepare_permutation_feature_importance_view_sorts_descending():
    view = dashboard.prepare_permutation_feature_importance_view(
        pd.DataFrame(
            [
                {"feature_name": "BMI", "metric_name": "roc_auc", "importance_mean": 0.02},
                {"feature_name": "Glucose", "metric_name": "roc_auc", "importance_mean": 0.08},
            ]
        )
    )

    assert view is not None
    assert list(view["feature_name"]) == ["Glucose", "BMI"]


def test_prepare_feature_selection_frontier_view_prioritizes_selected_subset():
    view = dashboard.prepare_feature_selection_frontier_view(
        pd.DataFrame(
            [
                {
                    "selected_flag": 0,
                    "within_one_se": True,
                    "feature_count": 3,
                    "block_count": 3,
                    "feature_names_text": "Glucose, BMI, Age",
                    "mean_brier": 0.19,
                    "sem_brier": 0.02,
                    "mean_roc_auc": 0.72,
                    "mean_log_loss": 0.51,
                    "mean_calibration_slope_error": 0.12,
                    "mean_calibration_intercept_abs": 0.18,
                },
                {
                    "selected_flag": 1,
                    "within_one_se": True,
                    "feature_count": 2,
                    "block_count": 2,
                    "feature_names_text": "Glucose, BMI",
                    "mean_brier": 0.18,
                    "sem_brier": 0.01,
                    "mean_roc_auc": 0.74,
                    "mean_log_loss": 0.49,
                    "mean_calibration_slope_error": 0.08,
                    "mean_calibration_intercept_abs": 0.12,
                },
            ]
        )
    )

    assert view is not None
    assert view.iloc[0]["selected_flag"] == 1
    assert view.iloc[0]["feature_names_text"] == "Glucose, BMI"


def test_build_sidebar_snapshot_includes_model_metadata():
    snapshot = dashboard.build_sidebar_snapshot(
        {"model_loaded": True, "model_version": "catboost.CatBoostClassifier"},
        {
            "model": {
                "class_path": "catboost.CatBoostClassifier",
                "train_splits": ["train", "validation", "test"],
                "decision_threshold": 0.25,
            }
        },
    )

    assert snapshot["class_path"] == "catboost.CatBoostClassifier"
    assert snapshot["train_splits"] == ["train", "validation", "test"]
    assert snapshot["decision_threshold"] == 0.25


def test_load_registry_state_returns_error_payload_on_failure(monkeypatch):
    monkeypatch.setattr(
        dashboard.ops_store,
        "list_experiment_runs",
        mock.Mock(side_effect=OSError("registry unavailable")),
    )
    monkeypatch.setattr(
        dashboard.ops_store,
        "list_model_registry_entries",
        mock.Mock(),
    )

    payload = dashboard.load_registry_state()

    assert payload["experiment_runs"] == []
    assert payload["model_registry"] == []
    assert payload["error"] == "OSError: registry unavailable"


def test_load_registry_state_refreshes_when_ops_store_changes(monkeypatch):
    experiment_runs = mock.Mock(
        side_effect=[
            [{"created_at": "2026-04-01T00:00:00+00:00"}],
            [{"created_at": "2026-04-02T00:00:00+00:00"}],
        ]
    )
    model_registry = mock.Mock(return_value=[])
    monkeypatch.setattr(
        dashboard.ops_store,
        "list_experiment_runs",
        experiment_runs,
    )
    monkeypatch.setattr(
        dashboard.ops_store,
        "list_model_registry_entries",
        model_registry,
    )

    first = dashboard.load_registry_state()
    second = dashboard.load_registry_state()

    assert first["experiment_runs"][0]["created_at"] == "2026-04-01T00:00:00+00:00"
    assert second["experiment_runs"][0]["created_at"] == "2026-04-02T00:00:00+00:00"
    assert experiment_runs.call_count == 2


def test_policy_decisions_translates_thresholds_into_actions():
    view = dashboard._policy_decisions(
        0.62,
        [
            {
                "policy_name": "default_050",
                "policy_label": "Threshold 0.50",
                "threshold": 0.5,
                "expected_cost": 12.0,
                "recall": 0.8,
                "precision": 0.5,
            },
            {
                "policy_name": "prioritize_precision",
                "policy_label": "Priorizar precision",
                "threshold": 0.7,
                "expected_cost": 8.0,
                "recall": 0.5,
                "precision": 0.9,
            },
        ],
    )

    assert list(view["decision"]) == ["Positivo", "Negativo"]
    assert list(view["policy"]) == ["Threshold 0.50", "Priorizar precision"]


def test_fallback_model_frontier_marks_selected_model():
    report = {
        "baseline": {"roc_auc": 0.7, "recall": 0.6, "precision": 0.5, "f1": 0.55},
        "optimized": {"roc_auc": 0.8, "recall": 0.7, "precision": 0.65, "f1": 0.67},
    }

    frontier = dashboard._fallback_model_frontier(
        report,
        {
            "model_name": "optimized",
            "decision_policy_name": "prioritize_recall",
            "decision_threshold": 0.25,
        },
    )

    assert frontier is not None
    selected_row = frontier.loc[frontier["model_name"] == "optimized"].iloc[0]
    assert bool(selected_row["selected_for_refit"]) is True
    assert selected_row["deployment_policy_name"] == "prioritize_recall"
    assert selected_row["deployment_threshold"] == 0.25


def test_render_overview_tab_behaves_with_predictions(monkeypatch):
    fake = _patch_streamlit(monkeypatch)
    state = {
        "selected_deployment_policy": {
            "model_name": "optimized",
            "decision_policy_name": "prioritize_recall",
            "decision_threshold": 0.25,
        },
        "model_frontier": pd.DataFrame(
            [
                {
                    "model_name": "optimized",
                    "selected_for_refit": True,
                    "best_cv_score": 0.7,
                    "validation_roc_auc": 0.8,
                }
            ]
        ),
        "risk_report": pd.DataFrame(
            [
                {
                    "risk_priority_rank": 1,
                    "prediction": 1,
                    "prediction_label": "Positivo",
                    "prediction_proba": 0.91,
                    "risk_score": 91.0,
                    "risk_band": "Alto risco",
                    "recommended_action": "Priorizar revisão clínica",
                }
            ]
        ),
    }

    dashboard.render_overview_tab(
        state,
        {"model": {"train_splits": ["train", "validation", "test"]}},
    )

    recorded = [name for name, _, _ in fake.calls]
    assert "metric" in recorded
    assert "dataframe" in recorded
    assert "bar_chart" in recorded
    assert "download_button" in recorded


def test_render_model_comparison_tab_renders_table_and_chart(monkeypatch):
    fake = _patch_streamlit(monkeypatch)

    dashboard.render_model_comparison_tab(
        pd.DataFrame(
            [
                {
                    "model_name": "baseline",
                    "selected_for_refit": True,
                    "best_cv_score": 0.7,
                    "validation_roc_auc": 0.71,
                    "validation_recall": 0.6,
                    "validation_precision": 0.5,
                    "validation_f1": 0.55,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "selection_rank": 1,
                    "model_name": "baseline",
                    "selection_composite_score": 0.8,
                    "selection_policy_label": "Priorizar recall",
                    "selection_policy_threshold": 0.25,
                    "validation_roc_auc": 0.71,
                    "validation_brier": 0.19,
                    "validation_log_loss": 0.55,
                    "validation_calibration_slope_error": 0.1,
                    "policy_recall": 0.9,
                    "policy_false_negative_rate": 0.1,
                    "policy_precision": 0.5,
                    "policy_expected_cost_per_sample": 0.6,
                }
            ]
        ),
        pd.DataFrame(
            [
                {"split": "train", "roc_auc": 0.75, "brier": 0.18},
                {"split": "validation", "roc_auc": 0.71, "brier": 0.19},
                {"split": "test", "roc_auc": 0.73, "brier": 0.17},
            ]
        ),
        pd.DataFrame(
            [
                {"fold_id": 1, "selected_model_name": "baseline", "roc_auc": 0.72}
            ]
        ),
        {"baseline": {"roc_auc": 0.7, "recall": 0.6, "precision": 0.5, "f1": 0.55}},
    )

    recorded = [name for name, _, _ in fake.calls]
    assert "dataframe" in recorded
    assert "bar_chart" in recorded
    assert "line_chart" in recorded


@pytest.mark.parametrize(
    ("label", "expected_command"),
    [
        (dashboard.PIPELINE_ACTIONS[0][0], dashboard.PIPELINE_ACTIONS[0][1]),
        (dashboard.PIPELINE_ACTIONS[1][0], dashboard.PIPELINE_ACTIONS[1][1]),
        (dashboard.PIPELINE_ACTIONS[2][0], dashboard.PIPELINE_ACTIONS[2][1]),
    ],
)
def test_render_actions_tab_triggers_cli_for_pressed_button(
    monkeypatch,
    label: str,
    expected_command: list[str],
):
    _patch_streamlit(monkeypatch, pressed_buttons={label})
    run_cli = mock.Mock(return_value=(0, ["ok"]))
    monkeypatch.setattr(dashboard, "_run_cli", run_cli)

    dashboard.render_actions_tab()

    run_cli.assert_called_once()
    assert run_cli.call_args.args[0] == expected_command


def test_run_cli_clears_caches_after_success(monkeypatch, tmp_path: Path):
    fake = _patch_streamlit(monkeypatch)

    class _Proc:
        def __init__(self):
            self.stdout = ["line 1\n", "line 2\n"]

        def wait(self):
            return 0

    monkeypatch.setattr(dashboard.subprocess, "Popen", lambda *args, **kwargs: _Proc())

    return_code, lines = dashboard._run_cli(
        [dashboard.sys.executable, "-m", "kedro", "run"],
        tmp_path,
        "caption",
    )

    assert return_code == 0
    assert lines == ["line 1", "line 2"]
    fake.cache_data.clear.assert_called_once()
    fake.cache_resource.clear.assert_called_once()
    recorded = [name for name, _, _ in fake.calls]
    assert "success" in recorded
    assert "code" in recorded


def test_run_cli_reports_launch_failure(monkeypatch, tmp_path: Path):
    fake = _patch_streamlit(monkeypatch)
    monkeypatch.setattr(
        dashboard.subprocess,
        "Popen",
        mock.Mock(side_effect=OSError("no shell")),
    )
    mocked_logger = mock.Mock()
    monkeypatch.setattr(dashboard, "logger", mocked_logger)

    return_code, lines = dashboard._run_cli(
        [dashboard.sys.executable, "-m", "kedro", "run"],
        tmp_path,
        "caption",
    )

    assert return_code == 1
    assert lines == ["OSError: no shell"]
    recorded = [name for name, _, _ in fake.calls]
    assert "error" in recorded
    mocked_logger.exception.assert_called_once()


def test_render_robustness_tab_renders_chart_and_summary(monkeypatch):
    fake = _patch_streamlit(monkeypatch, selectbox_value="roc_auc")

    dashboard.render_robustness_tab(
        pd.DataFrame(
            [
                {
                    "selected_flag": 1,
                    "within_one_se": True,
                    "feature_count": 2,
                    "block_count": 2,
                    "feature_names_text": "Glucose, BMI",
                    "mean_brier": 0.18,
                    "sem_brier": 0.01,
                    "mean_roc_auc": 0.74,
                    "mean_log_loss": 0.49,
                    "mean_calibration_slope_error": 0.08,
                    "mean_calibration_intercept_abs": 0.12,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "entity_type": "feature",
                    "entity_name": "Glucose",
                    "selection_frequency": 1.0,
                    "winner_folds": 2,
                }
            ]
        ),
        {"selected_feature_names_text": "Glucose, BMI"},
        pd.DataFrame(
            [
                {"fold_id": 1, "model_name": "baseline", "roc_auc": 0.7, "recall": 0.6, "precision": 0.5, "f1": 0.55, "false_negative_rate": 0.4, "false_positive_rate": 0.5, "mean_risk_score": 0.42},
                {"fold_id": 2, "model_name": "baseline", "roc_auc": 0.72, "recall": 0.62, "precision": 0.52, "f1": 0.57, "false_negative_rate": 0.38, "false_positive_rate": 0.48, "mean_risk_score": 0.44},
            ]
        ),
        pd.DataFrame(
            [
                {
                    "model_name": "baseline",
                    "metric_name": "roc_auc",
                    "variation_pct_of_mean": 2.0,
                    "robustness_label": "stable",
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "model_name": "baseline",
                    "metric_name": "roc_auc",
                    "ci_width": 0.05,
                    "ci_low": 0.68,
                    "ci_high": 0.73,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "feature_name": "Glucose",
                    "feature_group": "clinical_basics",
                    "metric_name": "roc_auc",
                    "importance_mean": 0.08,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "feature_name": "Glucose",
                    "direction": "up",
                    "mean_sensitivity_ratio": 0.3,
                    "decision_flip_rate": 0.1,
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "feature_name": "Glucose",
                    "max_decision_flip_rate": 0.1,
                    "max_sensitivity_ratio": 0.4,
                    "mean_sensitivity_ratio": 0.3,
                    "mean_abs_probability_delta": 0.05,
                    "max_risk_band_change_rate": 0.2,
                    "sensitivity_label": "proportional",
                }
            ]
        ),
    )

    recorded = [name for name, _, _ in fake.calls]
    assert "line_chart" in recorded
    assert "dataframe" in recorded


def test_render_policy_tab_renders_metrics_and_confusion_matrix(monkeypatch):
    fake = _patch_streamlit(monkeypatch)

    dashboard.render_policy_tab(
        pd.DataFrame(
            [
                {
                    "policy_name": "prioritize_recall",
                    "policy_label": "Priorizar recall",
                    "threshold": 0.25,
                    "precision": 0.3,
                    "recall": 1.0,
                    "f1": 0.46,
                    "false_negative_rate": 0.0,
                    "false_positive_rate": 0.7,
                    "expected_cost": 7.0,
                    "tp": 10,
                    "fp": 7,
                    "tn": 3,
                    "fn": 0,
                }
            ]
        ),
        {"decision_policy_name": "prioritize_recall"},
    )

    recorded = [name for name, _, _ in fake.calls]
    assert "metric" in recorded
    assert "pyplot" in recorded


def test_render_inference_tab_runs_live_prediction(monkeypatch):
    fake = _patch_streamlit(monkeypatch, pressed_buttons={"Gerar score clínico"})
    run_inference = mock.Mock(
        return_value=pd.DataFrame(
            [
                {
                    "case_id": 1,
                    "prediction": 1,
                    "prediction_label": "Positivo",
                    "prediction_proba": 0.91,
                    "risk_score": 91.0,
                    "risk_band": "Alto risco",
                    "decision_policy_name": "prioritize_recall",
                    "decision_threshold": 0.25,
                    "recommended_action": "Priorizar revisão clínica",
                }
            ]
        )
    )
    monkeypatch.setattr(
        dashboard.serving_runtime,
        "run_online_inference",
        run_inference,
    )

    dashboard.render_inference_tab(
        {
            "model": {
                "decision_policy_name": "prioritize_recall",
                "decision_threshold": 0.25,
                "policy_catalog": [
                    {
                        "policy_name": "prioritize_recall",
                        "policy_label": "Priorizar recall",
                        "threshold": 0.25,
                        "expected_cost": 7.0,
                        "recall": 1.0,
                        "precision": 0.3,
                    }
                ]
            }
        },
        None,
    )

    recorded = [name for name, _, _ in fake.calls]
    assert "number_input" in recorded
    assert "progress" in recorded
    assert "dataframe" in recorded
    assert "download_button" in recorded
    assert run_inference.call_args.kwargs == {"output_dataset": "risk_report"}


def test_render_inference_tab_logs_and_surfaces_failures(monkeypatch):
    fake = _patch_streamlit(monkeypatch, pressed_buttons={"Gerar score clínico"})
    monkeypatch.setattr(
        dashboard.serving_runtime,
        "run_online_inference",
        mock.Mock(side_effect=RuntimeError("pipeline exploded")),
    )
    mocked_logger = mock.Mock()
    monkeypatch.setattr(dashboard, "logger", mocked_logger)

    dashboard.render_inference_tab({"model": {}}, None)

    recorded = [name for name, _, _ in fake.calls]
    assert "error" in recorded
    assert "exception" in recorded
    mocked_logger.exception.assert_called_once()


def test_render_manifests_tab_shows_json_payloads(monkeypatch):
    fake = _patch_streamlit(monkeypatch)

    dashboard.render_manifests_tab(
        {
            "training_run_manifest": {"manifest_type": "training_run"},
            "model_selection_scorecard": pd.DataFrame(
                [
                    {
                        "selection_rank": 1,
                        "model_name": "baseline",
                        "selection_composite_score": 0.8,
                    }
                ]
            ),
            "split_comparison_report": pd.DataFrame(
                [{"split": "train", "roc_auc": 0.8, "brier": 0.18}]
            ),
            "nested_cv_audit": pd.DataFrame(
                [{"fold_id": 1, "selected_model_name": "baseline"}]
            ),
            "modelling_design_audit": pd.DataFrame(
                [{"check_name": "feature_selection_train_only", "status": "pass"}]
            ),
            "split_strategy_report": pd.DataFrame(
                [
                    {
                        "split_name": "train",
                        "split_order": 1,
                        "requested_strategy_kind": "stratified_random",
                        "resolved_strategy_kind": "stratified_random",
                        "fallback_applied": False,
                        "observed_ratio": 0.7,
                        "rows": 456,
                        "stratified_flag": True,
                    }
                ]
            ),
            "serving_manifest": {"manifest_type": "serving_manifest"},
            "inference_contract": {
                "manifest_type": "inference_contract",
                "input_fields": [{"name": "Glucose"}],
                "model_feature_columns": ["Glucose", "BMI"],
                "derived_feature_columns": [],
                "outputs": [{"name": "prediction"}],
            },
        }
    )

    json_calls = [call for call in fake.calls if call[0] == "json"]
    assert len(json_calls) == 3


def test_main_renders_when_artifacts_exist(monkeypatch):
    fake = _patch_streamlit(monkeypatch)
    monkeypatch.setattr(
        dashboard,
        "load_production_artifacts",
        lambda: {"model": {"train_splits": ["train", "validation", "test"]}},
    )
    monkeypatch.setattr(
        dashboard,
        "load_dashboard_state",
        lambda _production: {
            "test_report": {},
            "model_frontier": pd.DataFrame(),
            "feature_selection_frontier": pd.DataFrame(),
            "feature_selection_stability": pd.DataFrame(),
            "feature_selection_manifest": None,
            "cv_fold_metrics": pd.DataFrame(),
            "cv_metric_summary": pd.DataFrame(),
            "bootstrap_metric_intervals": pd.DataFrame(),
            "permutation_feature_importance": pd.DataFrame(),
            "perturbation_sensitivity_audit": pd.DataFrame(),
            "perturbation_sensitivity_summary": pd.DataFrame(),
            "model_selection_scorecard": pd.DataFrame(),
            "split_comparison_report": pd.DataFrame(),
            "nested_cv_audit": pd.DataFrame(),
            "modelling_design_audit": pd.DataFrame(),
            "threshold_metrics": pd.DataFrame(),
            "selected_deployment_policy": None,
            "predictions": pd.DataFrame(),
            "risk_report": pd.DataFrame(),
            "training_run_manifest": None,
            "split_strategy_report": pd.DataFrame(),
            "serving_manifest": None,
            "inference_contract": None,
        },
    )
    monkeypatch.setattr(
        dashboard.serving_runtime,
        "get_production_status",
        lambda: {"model_loaded": True, "model_version": "catboost"},
    )
    monkeypatch.setattr(
        dashboard,
        "load_registry_state",
        lambda: {"experiment_runs": [], "model_registry": [], "error": None},
    )

    dashboard.main()

    recorded = [name for name, _, _ in fake.calls]
    assert "set_page_config" in recorded
    assert "tabs" in recorded


def test_main_warns_when_registry_state_is_unavailable(monkeypatch):
    fake = _patch_streamlit(monkeypatch)
    monkeypatch.setattr(
        dashboard,
        "load_production_artifacts",
        lambda: {"model": {"train_splits": ["train", "validation", "test"]}},
    )
    monkeypatch.setattr(
        dashboard,
        "load_dashboard_state",
        lambda _production: {
            "test_report": {},
            "model_frontier": pd.DataFrame(),
            "feature_selection_frontier": pd.DataFrame(),
            "feature_selection_stability": pd.DataFrame(),
            "feature_selection_manifest": None,
            "cv_fold_metrics": pd.DataFrame(),
            "cv_metric_summary": pd.DataFrame(),
            "bootstrap_metric_intervals": pd.DataFrame(),
            "permutation_feature_importance": pd.DataFrame(),
            "perturbation_sensitivity_audit": pd.DataFrame(),
            "perturbation_sensitivity_summary": pd.DataFrame(),
            "model_selection_scorecard": pd.DataFrame(),
            "split_comparison_report": pd.DataFrame(),
            "nested_cv_audit": pd.DataFrame(),
            "modelling_design_audit": pd.DataFrame(),
            "threshold_metrics": pd.DataFrame(),
            "selected_deployment_policy": None,
            "predictions": pd.DataFrame(),
            "risk_report": pd.DataFrame(),
            "training_run_manifest": None,
            "split_strategy_report": pd.DataFrame(),
            "serving_manifest": None,
            "inference_contract": None,
        },
    )
    monkeypatch.setattr(
        dashboard.serving_runtime,
        "get_production_status",
        lambda: {"model_loaded": True, "model_version": "catboost"},
    )
    monkeypatch.setattr(
        dashboard,
        "load_registry_state",
        lambda: {
            "experiment_runs": [],
            "model_registry": [],
            "error": "OSError: registry unavailable",
        },
    )

    dashboard.main()

    recorded = [name for name, _, _ in fake.calls]
    assert "warning" in recorded


def test_main_sidebar_clear_cache_button_triggers_rerun(monkeypatch):
    fake = _patch_streamlit(
        monkeypatch,
        pressed_buttons={"Limpar cache e recarregar"},
    )
    monkeypatch.setattr(
        dashboard,
        "load_production_artifacts",
        lambda: {"model": {"train_splits": ["train", "validation", "test"]}},
    )
    monkeypatch.setattr(
        dashboard,
        "load_dashboard_state",
        lambda _production: {
            "test_report": {},
            "model_frontier": pd.DataFrame(),
            "feature_selection_frontier": pd.DataFrame(),
            "feature_selection_stability": pd.DataFrame(),
            "feature_selection_manifest": None,
            "cv_fold_metrics": pd.DataFrame(),
            "cv_metric_summary": pd.DataFrame(),
            "bootstrap_metric_intervals": pd.DataFrame(),
            "permutation_feature_importance": pd.DataFrame(),
            "perturbation_sensitivity_audit": pd.DataFrame(),
            "perturbation_sensitivity_summary": pd.DataFrame(),
            "model_selection_scorecard": pd.DataFrame(),
            "split_comparison_report": pd.DataFrame(),
            "nested_cv_audit": pd.DataFrame(),
            "modelling_design_audit": pd.DataFrame(),
            "threshold_metrics": pd.DataFrame(),
            "selected_deployment_policy": None,
            "predictions": pd.DataFrame(),
            "risk_report": pd.DataFrame(),
            "training_run_manifest": None,
            "split_strategy_report": pd.DataFrame(),
            "serving_manifest": None,
            "inference_contract": None,
        },
    )
    monkeypatch.setattr(
        dashboard.serving_runtime,
        "get_production_status",
        lambda: {"model_loaded": True, "model_version": "catboost"},
    )
    monkeypatch.setattr(
        dashboard,
        "load_registry_state",
        lambda: {"experiment_runs": [], "model_registry": [], "error": None},
    )

    dashboard.main()

    fake.cache_data.clear.assert_called_once()
    fake.cache_resource.clear.assert_called_once()
    recorded = [name for name, _, _ in fake.calls]
    assert "rerun" in recorded
