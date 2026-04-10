"""Testes unitários dos nodes de modelagem"""

from __future__ import annotations

import warnings

import pandas as pd

from insper_deploy_kedro.pipelines.modelling.nodes import (
    build_bootstrap_metric_intervals,
    build_cv_fold_metrics,
    build_model_frontier,
    build_model_selection_scorecard,
    build_modelling_design_audit,
    build_nested_cv_audit,
    build_permutation_feature_importance,
    build_perturbation_sensitivity_audit,
    build_split_comparison_report,
    build_threshold_metrics,
    calibrate_model,
    enrich_best_model_config_with_policy,
    evaluate_all_on_test,
    evaluate_model,
    optimize_model,
    select_best_model,
    select_deployment_policy,
    select_feature_columns,
    summarize_cv_fold_metrics,
    summarize_perturbation_sensitivity_audit,
    train_model,
)


def build_sample_sensitivity_audit():
    return pd.DataFrame(
        [
            {
                "feature_name": "Glucose",
                "direction": "up",
                "mean_sensitivity_ratio": 0.4,
                "max_sensitivity_ratio": 0.8,
                "decision_flip_rate": 0.1,
                "risk_band_change_rate": 0.2,
                "mean_abs_probability_delta": 0.05,
                "sensitivity_label": "proportional",
            },
            {
                "feature_name": "Glucose",
                "direction": "down",
                "mean_sensitivity_ratio": 0.3,
                "max_sensitivity_ratio": 0.6,
                "decision_flip_rate": 0.05,
                "risk_band_change_rate": 0.1,
                "mean_abs_probability_delta": 0.04,
                "sensitivity_label": "proportional",
            },
        ]
    )


MODEL_SELECTION_CONFIG = {
    "metric": "roc_auc",
    "refit_train_splits": ["train", "validation", "test"],
    "split_comparison_splits": ["train", "validation", "test"],
    "score_components": [
        {
            "metric": "brier",
            "source": "validation",
            "direction": "minimize",
            "weight": 0.4,
        },
        {
            "metric": "roc_auc",
            "source": "validation",
            "direction": "maximize",
            "weight": 0.2,
        },
        {
            "metric": "recall",
            "source": "policy",
            "direction": "maximize",
            "weight": 0.2,
        },
        {
            "metric": "expected_cost_per_sample",
            "source": "policy",
            "direction": "minimize",
            "weight": 0.2,
        },
    ],
}

NESTED_CV_AUDIT_CONFIG = {
    "enabled": True,
    "development_splits": ["train", "validation"],
    "outer_folds": 2,
    "validation_fraction": 0.25,
    "random_state": 42,
    "trial_scaling": 0.1,
    "min_trials": 2,
}


class TestTrainModel:
    def test_returns_artifact_dict(
        self, master_table, columns_config, ml_runtime_config
    ):
        model_params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        }
        artifact = train_model(
            master_table, columns_config, model_params, ml_runtime_config
        )
        assert "estimator" in artifact
        assert "target_encoder" in artifact
        assert "feature_columns" in artifact
        assert "init_args" in artifact

    def test_feature_columns_include_derived(
        self, master_table, columns_config, ml_runtime_config
    ):
        model_params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        }
        artifact = train_model(
            master_table, columns_config, model_params, ml_runtime_config
        )
        assert "glucose_bmi_interaction" in artifact["feature_columns"]

    def test_catboost_defaults_to_non_writing_mode(
        self, master_table, columns_config, ml_runtime_config
    ):
        model_params = {
            "class_path": "catboost.CatBoostClassifier",
            "train_splits": ["train"],
            "init_args": {"iterations": 5, "verbose": 0},
        }
        artifact = train_model(
            master_table, columns_config, model_params, ml_runtime_config
        )

        assert artifact["init_args"]["allow_writing_files"] is False


class TestOptimizeModel:
    def test_returns_best_params(self, master_table, columns_config, ml_runtime_config):
        params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "init_args": {"max_iter": 1000},
            "search_space": {
                "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            },
            "n_trials": 3,
            "cv": 2,
            "scoring": "roc_auc",
        }
        artifact = optimize_model(
            master_table, columns_config, params, ml_runtime_config
        )
        assert "estimator" in artifact
        assert "best_params" in artifact or "class_path" in artifact

    def test_falls_back_to_train_without_search_space(
        self, master_table, columns_config, ml_runtime_config
    ):
        params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "train_splits": ["train"],
            "init_args": {"max_iter": 1000},
        }
        artifact = optimize_model(
            master_table, columns_config, params, ml_runtime_config
        )
        assert "estimator" in artifact
        assert "best_params" not in artifact

    def test_trained_model_can_predict(
        self, master_table, columns_config, ml_runtime_config
    ):
        params = {
            "class_path": "sklearn.linear_model.LogisticRegression",
            "init_args": {"max_iter": 1000},
            "search_space": {
                "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True},
            },
            "n_trials": 3,
            "cv": 2,
            "scoring": "roc_auc",
        }
        artifact = optimize_model(
            master_table, columns_config, params, ml_runtime_config
        )
        fc = artifact["feature_columns"]
        x = master_table[master_table["split"] == "train"][fc]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            preds = artifact["estimator"].predict(x)
        assert len(preds) == len(x)
        assert not any(
            "feature names" in str(warning.message).lower() for warning in caught
        )


class TestEvaluateModel:
    def test_returns_standard_metrics(
        self,
        master_table,
        columns_config,
        trained_model,
        evaluation_config,
    ):
        eval_params = {**evaluation_config, "split": "train"}
        metrics = evaluate_model(
            master_table, trained_model, columns_config, eval_params
        )

        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "brier", "log_loss"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_includes_r2_and_mape(
        self,
        master_table,
        columns_config,
        trained_model,
        evaluation_config,
    ):
        eval_params = {**evaluation_config, "split": "train"}
        metrics = evaluate_model(
            master_table, trained_model, columns_config, eval_params
        )
        assert "r2" in metrics
        assert "mape" in metrics
        assert "calibration_slope_error" in metrics
        assert "calibration_intercept_abs" in metrics

    def test_includes_confusion_matrix(
        self,
        master_table,
        columns_config,
        trained_model,
        evaluation_config,
    ):
        eval_params = {**evaluation_config, "split": "train"}
        metrics = evaluate_model(
            master_table, trained_model, columns_config, eval_params
        )
        assert "confusion_matrix" in metrics
        assert isinstance(metrics["confusion_matrix"], list)

    def test_handles_empty_split_gracefully(
        self,
        master_table,
        columns_config,
        trained_model,
        evaluation_config,
    ):
        eval_params = {**evaluation_config, "split": "nonexistent"}
        metrics = evaluate_model(
            master_table, trained_model, columns_config, eval_params
        )
        assert metrics["n_samples"] == 0
        assert metrics["f1"] == 0.0
        assert metrics["r2"] == 0.0
        assert metrics["mape"] == 0.0


class TestSelectBestModel:
    def test_returns_refit_config(
        self,
        master_table,
        columns_config,
        trained_model,
        evaluation_config,
    ):
        eval_params = {**evaluation_config, "split": "train"}
        metrics = evaluate_model(
            master_table, trained_model, columns_config, eval_params
        )

        config = select_best_model(
            trained_model,
            metrics,
            trained_model,
            metrics,
            trained_model,
            metrics,
            {
                "metric": "roc_auc",
                "refit_train_splits": ["train", "validation", "test"],
            },
        )
        assert "class_path" in config
        assert "train_splits" in config
        assert "init_args" in config

    def test_uses_composite_scorecard_when_available(
        self,
        master_table,
        columns_config,
        trained_model,
        evaluation_config,
        decision_policy_config,
    ):
        validation_metrics = evaluate_model(
            master_table,
            trained_model,
            columns_config,
            {**evaluation_config, "split": "validation"},
        )
        scorecard = build_model_selection_scorecard(
            master_table,
            trained_model,
            validation_metrics,
            trained_model,
            validation_metrics,
            trained_model,
            validation_metrics,
            columns_config,
            decision_policy_config,
            MODEL_SELECTION_CONFIG,
        )

        config = select_best_model(
            trained_model,
            validation_metrics,
            trained_model,
            validation_metrics,
            trained_model,
            validation_metrics,
            MODEL_SELECTION_CONFIG,
            scorecard,
        )

        assert config["selection_metric"] == "composite_probability_policy_score"
        assert "selection_score" in config


class TestFeatureSelection:
    def test_select_feature_columns_returns_manifest_and_filtered_columns(
        self,
        master_table,
        columns_config,
        ml_runtime_config,
        feature_selection_config,
    ):
        selected_columns, frontier, stability, manifest = select_feature_columns(
            master_table,
            columns_config,
            ml_runtime_config,
            feature_selection_config,
        )

        assert selected_columns["target"] == ["Outcome"]
        assert selected_columns["numerical"]
        assert set(selected_columns["numerical"]).issubset(
            set(columns_config["numerical"])
        )
        assert not frontier.empty
        assert not stability.empty
        assert manifest["enabled"] is True
        assert manifest["selected_feature_count"] == len(selected_columns["numerical"])

    def test_select_feature_columns_uses_train_only_selection_splits(
        self,
        master_table,
        columns_config,
        ml_runtime_config,
        feature_selection_config,
    ):
        selected_columns, frontier, _stability, manifest = select_feature_columns(
            master_table,
            columns_config,
            ml_runtime_config,
            feature_selection_config,
        )

        assert manifest["selection_splits"] == ["train"]
        assert frontier["selected_flag"].sum() == 1
        assert set(selected_columns["numerical"]).issubset(
            set(feature_selection_config["feature_blocks"]["glucose_axis"])
            | set(feature_selection_config["feature_blocks"]["bmi_axis"])
            | set(feature_selection_config["feature_blocks"]["age_axis"])
            | set(feature_selection_config["feature_blocks"]["interaction_axis"])
        )

    def test_select_feature_columns_enforces_required_parent_blocks(
        self,
        master_table,
        columns_config,
        ml_runtime_config,
        feature_selection_config,
    ):
        selected_columns, _frontier, _stability, _manifest = select_feature_columns(
            master_table,
            columns_config,
            ml_runtime_config,
            feature_selection_config,
        )

        if "glucose_bmi_interaction" in selected_columns["numerical"]:
            assert "Glucose" in selected_columns["numerical"]
            assert "BMI" in selected_columns["numerical"]


class TestModelDiagnostics:
    def test_build_model_selection_scorecard_returns_ranked_rows(
        self,
        master_table,
        columns_config,
        trained_model,
        evaluation_config,
        decision_policy_config,
    ):
        validation_metrics = evaluate_model(
            master_table,
            trained_model,
            columns_config,
            {**evaluation_config, "split": "validation"},
        )
        scorecard = build_model_selection_scorecard(
            master_table,
            trained_model,
            validation_metrics,
            trained_model,
            validation_metrics,
            trained_model,
            validation_metrics,
            columns_config,
            decision_policy_config,
            MODEL_SELECTION_CONFIG,
        )

        assert not scorecard.empty
        assert {
            "selection_composite_score",
            "selection_rank",
            "policy_expected_cost_per_sample",
            "selection_policy_threshold",
        }.issubset(scorecard.columns)

    def test_build_cv_fold_metrics_returns_fold_rows(
        self,
        master_table,
        columns_config,
        trained_model,
        ml_runtime_config,
        decision_policy_config,
    ):
        selected_policy = {
            "decision_policy_name": "prioritize_recall",
            "decision_threshold": 0.2,
        }
        cv_metrics = build_cv_fold_metrics(
            master_table,
            trained_model,
            trained_model,
            trained_model,
            columns_config,
            ml_runtime_config,
            decision_policy_config,
            selected_policy,
        )

        assert not cv_metrics.empty
        assert {
            "model_name",
            "fold_id",
            "roc_auc",
            "brier",
            "log_loss",
            "recall",
            "fn",
            "fp",
            "policy_threshold",
            "tp_share",
            "fp_share",
        }.issubset(cv_metrics.columns)
        assert set(cv_metrics["policy_threshold"]) == {0.2}

    def test_summarize_cv_fold_metrics_returns_relative_variation(
        self,
        master_table,
        columns_config,
        trained_model,
        ml_runtime_config,
        decision_policy_config,
    ):
        cv_metrics = build_cv_fold_metrics(
            master_table,
            trained_model,
            trained_model,
            trained_model,
            columns_config,
            ml_runtime_config,
            decision_policy_config,
            {
                "decision_policy_name": "prioritize_recall",
                "decision_threshold": 0.2,
            },
        )
        summary = summarize_cv_fold_metrics(cv_metrics)

        assert not summary.empty
        assert "variation_pct_of_mean" in summary.columns
        assert "robustness_label" in summary.columns
        assert {"expected_cost", "brier", "tp_share", "fn_share"}.issubset(
            set(summary["metric_name"])
        )

    def test_build_bootstrap_metric_intervals_returns_ci_rows(
        self,
        master_table,
        columns_config,
        trained_model,
        evaluation_config,
        decision_policy_config,
    ):
        bootstrap = build_bootstrap_metric_intervals(
            master_table,
            trained_model,
            trained_model,
            trained_model,
            columns_config,
            evaluation_config,
            decision_policy_config,
            {
                "decision_policy_name": "prioritize_recall",
                "decision_threshold": 0.2,
            },
            {
                "split": "train",
                "iterations": 20,
                "random_state": 42,
                "confidence_level": 0.90,
                "metrics": ["roc_auc", "brier", "expected_cost"],
            },
        )

        assert not bootstrap.empty
        assert {"metric_name", "ci_low", "ci_high", "ci_width", "bootstrap_samples"}.issubset(
            bootstrap.columns
        )
        assert set(bootstrap["metric_name"]) == {"roc_auc", "brier", "expected_cost"}
        assert set(bootstrap["policy_threshold"]) == {0.2}

    def test_build_permutation_feature_importance_returns_rankable_rows(
        self,
        master_table,
        columns_config,
        trained_model,
        decision_policy_config,
    ):
        importance = build_permutation_feature_importance(
            {
                "class_path": trained_model["class_path"],
                "init_args": trained_model["init_args"],
            },
            trained_model,
            trained_model,
            trained_model,
            master_table,
            columns_config,
            decision_policy_config,
            {
                "decision_policy_name": "prioritize_recall",
                "decision_threshold": 0.2,
            },
            {
                "split": "train",
                "repeats": 4,
                "random_state": 42,
                "metrics": ["roc_auc", "brier"],
                "feature_groups": {
                    "clinical_basics": ["Glucose", "BMI"],
                    "engineered_signals": ["glucose_bmi_interaction"],
                },
            },
        )

        assert not importance.empty
        assert {
            "feature_name",
            "feature_group",
            "metric_name",
            "importance_mean",
            "positive_means_feature_helps_flag",
        }.issubset(importance.columns)
        assert set(importance["policy_threshold"]) == {0.2}

    def test_build_permutation_feature_importance_uses_selected_policy_threshold(
        self,
        master_table,
        columns_config,
        trained_model,
        decision_policy_config,
    ):
        importance = build_permutation_feature_importance(
            {
                "class_path": trained_model["class_path"],
                "init_args": trained_model["init_args"],
            },
            trained_model,
            trained_model,
            trained_model,
            master_table,
            columns_config,
            decision_policy_config,
            {
                "decision_policy_name": "default_050",
                "decision_threshold": 0.2,
            },
            {
                "split": "train",
                "repeats": 2,
                "random_state": 42,
                "metrics": ["roc_auc"],
            },
        )

        assert not importance.empty
        assert set(importance["policy_name"]) == {"default_050"}
        assert set(importance["policy_threshold"]) == {0.2}

    def test_build_perturbation_sensitivity_audit_returns_rows(
        self,
        master_table,
        columns_config,
        trained_model,
    ):
        best_model_config = {
            "class_path": trained_model["class_path"],
            "init_args": trained_model["init_args"],
        }
        audit = build_perturbation_sensitivity_audit(
            best_model_config,
            trained_model,
            trained_model,
            trained_model,
            master_table,
            columns_config,
            {
                "decision_policy_name": "prioritize_recall",
                "decision_threshold": 0.2,
                "risk_bands": [
                    {
                        "label": "Baixo risco",
                        "min_probability": 0.0,
                        "max_probability": 0.5,
                    },
                    {
                        "label": "Alto risco",
                        "min_probability": 0.5,
                        "max_probability": 1.01,
                    },
                ],
            },
            {
                "split": "train",
                "sample_size": 8,
                "random_state": 42,
                "perturbation_percents": [0.1],
                "min_baseline_abs": 1.0,
                "insensitive_max_ratio": 0.05,
                "overreactive_min_ratio": 1.5,
            },
        )

        assert not audit.empty
        assert {
            "feature_name",
            "mean_sensitivity_ratio",
            "decision_flip_rate",
            "risk_band_change_rate",
            "sensitivity_label",
        }.issubset(audit.columns)

    def test_summarize_perturbation_sensitivity_audit_returns_feature_view(self):
        audit = summarize_perturbation_sensitivity_audit(
            build_sample_sensitivity_audit()
        )

        assert not audit.empty
        assert {
            "feature_name",
            "mean_sensitivity_ratio",
            "max_decision_flip_rate",
            "sensitivity_label",
        }.issubset(audit.columns)

    def test_build_threshold_metrics_returns_policy_tradeoffs(
        self,
        master_table,
        columns_config,
        trained_model,
        decision_policy_config,
    ):
        best_model_config = {
            "class_path": trained_model["class_path"],
            "init_args": trained_model["init_args"],
            "train_splits": ["train", "validation", "test"],
        }
        threshold_metrics = build_threshold_metrics(
            best_model_config,
            trained_model,
            trained_model,
            trained_model,
            master_table,
            columns_config,
            decision_policy_config,
        )

        assert not threshold_metrics.empty
        assert {"policy_name", "threshold", "expected_cost", "recall"}.issubset(
            threshold_metrics.columns
        )

    def test_selected_policy_is_attached_to_best_model_config(
        self,
        master_table,
        columns_config,
        trained_model,
        evaluation_config,
        decision_policy_config,
    ):
        metrics = evaluate_model(
            master_table,
            trained_model,
            columns_config,
            {**evaluation_config, "split": "train"},
        )
        best_model_config = select_best_model(
            trained_model,
            metrics,
            trained_model,
            metrics,
            trained_model,
            metrics,
            {
                "metric": "roc_auc",
                "refit_train_splits": ["train", "validation", "test"],
            },
        )
        threshold_metrics = build_threshold_metrics(
            best_model_config,
            trained_model,
            trained_model,
            trained_model,
            master_table,
            columns_config,
            decision_policy_config,
        )
        selected_policy = select_deployment_policy(
            best_model_config,
            threshold_metrics,
            decision_policy_config,
        )
        enriched = enrich_best_model_config_with_policy(
            best_model_config,
            selected_policy,
        )

        assert "decision_threshold" in enriched
        assert enriched["decision_policy_name"] == "prioritize_recall"
        assert isinstance(enriched["policy_catalog"], list)

    def test_build_model_frontier_marks_selected_model(
        self,
        master_table,
        columns_config,
        trained_model,
        evaluation_config,
    ):
        metrics = evaluate_model(
            master_table,
            trained_model,
            columns_config,
            {**evaluation_config, "split": "train"},
        )
        best_model_config = select_best_model(
            trained_model,
            metrics,
            trained_model,
            metrics,
            trained_model,
            metrics,
            {
                "metric": "roc_auc",
                "refit_train_splits": ["train", "validation", "test"],
            },
        )
        frontier = build_model_frontier(
            trained_model,
            metrics,
            trained_model,
            metrics,
            trained_model,
            metrics,
            best_model_config,
            None,
        )

        assert "selected_for_refit" in frontier.columns
        assert frontier["selected_for_refit"].sum() >= 1

    def test_build_split_comparison_report_returns_train_validation_test_rows(
        self,
        master_table,
        columns_config,
        trained_model,
        evaluation_config,
        decision_policy_config,
    ):
        validation_metrics = evaluate_model(
            master_table,
            trained_model,
            columns_config,
            {**evaluation_config, "split": "validation"},
        )
        best_model_config = select_best_model(
            trained_model,
            validation_metrics,
            trained_model,
            validation_metrics,
            trained_model,
            validation_metrics,
            MODEL_SELECTION_CONFIG,
        )
        selected_policy = {
            "decision_policy_name": "prioritize_recall",
            "decision_threshold": 0.2,
        }

        comparison = build_split_comparison_report(
            best_model_config,
            trained_model,
            trained_model,
            trained_model,
            master_table,
            columns_config,
            decision_policy_config,
            selected_policy,
            MODEL_SELECTION_CONFIG,
        )

        assert not comparison.empty
        assert list(comparison["split"]) == ["train", "validation", "test"]
        assert {"roc_auc", "brier", "expected_cost", "roc_auc_gap_vs_train"}.issubset(
            comparison.columns
        )

    def test_build_nested_cv_audit_returns_outer_fold_rows(
        self,
        master_table,
        columns_config,
        ml_runtime_config,
        feature_selection_config,
        evaluation_config,
        decision_policy_config,
    ):
        expanded_master_table = pd.concat(
            [master_table.copy() for _ in range(8)],
            ignore_index=True,
        )
        nested_audit = build_nested_cv_audit(
            expanded_master_table,
            columns_config,
            ml_runtime_config,
            feature_selection_config,
            {
                "class_path": "sklearn.linear_model.LogisticRegression",
                "init_args": {"max_iter": 500},
                "search_space": {
                    "C": {"type": "float", "low": 0.1, "high": 1.0, "log": True},
                },
                "n_trials": 3,
                "cv": 2,
                "scoring": "roc_auc",
            },
            {
                "class_path": "sklearn.linear_model.LogisticRegression",
                "init_args": {"max_iter": 500},
                "search_space": {
                    "C": {"type": "float", "low": 0.1, "high": 1.0, "log": True},
                },
                "n_trials": 3,
                "cv": 2,
                "scoring": "roc_auc",
            },
            {
                "class_path": "sklearn.linear_model.LogisticRegression",
                "init_args": {"max_iter": 500},
                "search_space": {
                    "C": {"type": "float", "low": 0.1, "high": 1.0, "log": True},
                },
                "n_trials": 3,
                "cv": 2,
                "scoring": "roc_auc",
            },
            evaluation_config,
            MODEL_SELECTION_CONFIG,
            decision_policy_config,
            NESTED_CV_AUDIT_CONFIG,
        )

        assert not nested_audit.empty
        assert {
            "fold_id",
            "selected_model_name",
            "selected_policy_threshold",
            "candidate_feature_sets_evaluated",
            "roc_auc",
        }.issubset(nested_audit.columns)

    def test_build_modelling_design_audit_reports_guardrails(
        self,
        columns_config,
        raw_columns_config,
    ):
        audit = build_modelling_design_audit(
            raw_columns_config,
            columns_config,
            {
                "target": ["Outcome"],
                "categorical": [],
                "numerical": ["Glucose", "BMI", "glucose_bmi_interaction"],
            },
            {
                "selection_splits": ["train"],
                "candidate_count": 4,
            },
            MODEL_SELECTION_CONFIG,
            {"policy_selection_split": "validation"},
            summarize_perturbation_sensitivity_audit(build_sample_sensitivity_audit()),
        )

        assert not audit.empty
        assert set(audit["status"]) == {"pass"}


class TestEvaluateAllOnTest:
    def test_returns_report_for_all_models(
        self,
        master_table,
        columns_config,
        trained_model,
        evaluation_config,
    ):
        report = evaluate_all_on_test(
            master_table,
            trained_model,
            trained_model,
            trained_model,
            columns_config,
            evaluation_config,
        )
        for name in ("baseline", "optimized", "xgboost"):
            assert name in report
            assert "f1" in report[name]
            assert "confusion_matrix" in report[name]


CALIBRATION_PARAMS = {
    "class_path": "sklearn.calibration.CalibratedClassifierCV",
    "init_args": {"method": "sigmoid", "cv": "prefit"},
}


class TestCalibrateModel:
    def test_returns_calibrated_estimator(
        self, master_table, columns_config, trained_model
    ):
        calibrated = calibrate_model(
            master_table,
            trained_model,
            columns_config,
            CALIBRATION_PARAMS,
        )
        assert "estimator" in calibrated
        assert hasattr(calibrated["estimator"], "predict_proba")

    def test_calibrated_model_can_predict_proba(
        self, master_table, columns_config, trained_model
    ):
        calibrated = calibrate_model(
            master_table,
            trained_model,
            columns_config,
            CALIBRATION_PARAMS,
        )
        fc = calibrated["feature_columns"]
        x = master_table[master_table["split"] == "train"][fc]
        proba = calibrated["estimator"].predict_proba(x)
        assert proba.shape[1] == 2

    def test_preserves_metadata(self, master_table, columns_config, trained_model):
        calibrated = calibrate_model(
            master_table,
            trained_model,
            columns_config,
            CALIBRATION_PARAMS,
        )
        assert calibrated["class_path"] == trained_model["class_path"]
        assert calibrated["feature_columns"] == trained_model["feature_columns"]
