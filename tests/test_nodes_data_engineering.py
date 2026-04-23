"""Testes unitários dos nodes de DE"""

from __future__ import annotations

import pandas as pd
import pytest

from insper_deploy_kedro.pipelines.data_engineering.nodes import (
    add_features,
    add_split_column,
    clean_data,
    fit_encoders,
    fit_outlier_cappers,
    fit_scalers,
    fit_zero_imputers,
    transform_encoders,
    transform_outlier_cappers,
    transform_scalers,
    transform_zero_imputers,
)
from insper_deploy_kedro.pipelines.data_engineering.validations import (
    validate_clean_data,
    validate_split_data,
)


def _prepare_featured_split(
    sample_raw_data,
    raw_columns_config,
    split_ratio,
    fit_transform_config,
    preprocessing_config,
    *,
    stratify_column=None,
) -> pd.DataFrame:
    cleaned = clean_data(sample_raw_data, raw_columns_config)
    split, _report = add_split_column(
        cleaned,
        split_ratio,
        random_state=42,
        stratify_column=stratify_column,
        preprocessing=preprocessing_config,
    )
    imputers = fit_zero_imputers(split, fit_transform_config, preprocessing_config)
    imputed = transform_zero_imputers(split, imputers)
    outlier_cappers = fit_outlier_cappers(
        imputed,
        raw_columns_config,
        fit_transform_config,
        preprocessing_config,
    )
    capped = transform_outlier_cappers(imputed, outlier_cappers)
    return add_features(capped)


class TestCleanData:
    def test_selects_configured_columns(self, sample_raw_data, raw_columns_config):
        result = clean_data(sample_raw_data, raw_columns_config)
        expected_cols = (
            raw_columns_config["target"]
            + raw_columns_config["categorical"]
            + raw_columns_config["numerical"]
        )
        assert set(result.columns) == set(expected_cols)

    def test_coerces_numerical_to_float(self, sample_raw_data, raw_columns_config):
        result = clean_data(sample_raw_data, raw_columns_config)
        for col in raw_columns_config["numerical"]:
            assert pd.api.types.is_numeric_dtype(result[col])

    def test_raises_on_missing_column(self, raw_columns_config):
        df = pd.DataFrame({"Pregnancies": [6]})
        with pytest.raises(KeyError, match="columns not found"):
            clean_data(df, raw_columns_config)


class TestAddFeatures:
    def test_adds_derived_column(self, sample_raw_data, raw_columns_config):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        expected_columns = {
            "glucose_bmi_interaction",
            "glucose_insulin_interaction",
            "glucose_pregnancies_interaction",
            "age_category",
            "bmi_category",
            "glucose_category",
            "age_bmi_category",
            "age_glucose_category",
            "insulin_category",
        }
        assert expected_columns.issubset(featured.columns)

    def test_derived_column_is_numeric(self, sample_raw_data, raw_columns_config):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        assert pd.api.types.is_numeric_dtype(featured["glucose_bmi_interaction"])

    def test_derived_column_formula(self, sample_raw_data, raw_columns_config):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        expected = cleaned["Glucose"] * cleaned["BMI"] / 1000
        pd.testing.assert_series_equal(
            featured["glucose_bmi_interaction"], expected, check_names=False
        )

    def test_clinical_categories_follow_configured_rules(
        self, sample_raw_data, raw_columns_config
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        assert featured.loc[0, "age_category"] == "senior"
        assert featured.loc[1, "bmi_category"] == "overweight"
        assert featured.loc[8, "glucose_category"] == "prediabetes"
        assert featured.loc[0, "age_bmi_category"] == "obese_senior"
        assert featured.loc[1, "age_glucose_category"] == "normal_mature"
        assert featured.loc[3, "insulin_category"] == "normal"


class TestAddSplitColumn:
    def test_adds_split_column(
        self, sample_raw_data, raw_columns_config, split_ratio, preprocessing_config
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        result, _report = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        assert "split" in result.columns

    def test_returns_split_strategy_report(
        self, sample_raw_data, raw_columns_config, split_ratio, preprocessing_config
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        result, report = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column="Outcome",
            preprocessing=preprocessing_config,
        )
        assert "split" in result.columns
        assert not report.empty
        assert {
            "requested_strategy_kind",
            "resolved_strategy_kind",
            "split_name",
        }.issubset(report.columns)

    def test_split_names_match_config(
        self,
        sample_raw_data,
        raw_columns_config,
        split_ratio,
        preprocessing_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        result, _report = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        assert set(result["split"].unique()).issubset(set(split_ratio.keys()))

    def test_reproducible_with_seed(
        self,
        sample_raw_data,
        raw_columns_config,
        split_ratio,
        preprocessing_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        r1, _report1 = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        r2, _report2 = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        pd.testing.assert_frame_equal(r1, r2)

    def test_stratified_split(
        self, sample_raw_data, raw_columns_config, split_ratio, preprocessing_config
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        bigger = pd.concat([featured] * 5, ignore_index=True)
        result, _report = add_split_column(
            bigger,
            split_ratio,
            random_state=42,
            stratify_column="Outcome",
            preprocessing=preprocessing_config,
        )
        assert "split" in result.columns
        assert set(result["split"].unique()) == set(split_ratio.keys())

    def test_unsupported_temporal_split_strategy_raises(
        self, sample_raw_data, raw_columns_config, split_ratio, preprocessing_config
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        unsupported_cfg = {
            **preprocessing_config,
            "split_strategy": {
                "kind": "temporal_holdout",
                "label": "unsupported_temporal",
            },
        }

        with pytest.raises(
            ValueError, match=r"unsupported_split_strategy_kind:temporal_holdout"
        ):
            add_split_column(
                featured,
                split_ratio,
                random_state=42,
                stratify_column=None,
                preprocessing=unsupported_cfg,
            )


class TestFitTransformEncoders:
    def test_fit_returns_one_encoder_per_engineered_categorical(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
        preprocessing_config,
    ):
        split = _prepare_featured_split(
            sample_raw_data,
            raw_columns_config,
            split_ratio,
            fit_transform_config,
            preprocessing_config,
        )
        encoders = fit_encoders(
            split, columns_config, fit_transform_config, preprocessing_config
        )
        assert set(encoders) == set(columns_config["categorical"])
        assert encoders["age_category"]["output_columns"] == ["age_category_senior"]

    def test_transform_creates_one_hot_columns(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
        preprocessing_config,
    ):
        split = _prepare_featured_split(
            sample_raw_data,
            raw_columns_config,
            split_ratio,
            fit_transform_config,
            preprocessing_config,
        )
        encoders = fit_encoders(
            split, columns_config, fit_transform_config, preprocessing_config
        )
        encoded = transform_encoders(split, encoders)
        assert set(columns_config["categorical"]).isdisjoint(encoded.columns)
        assert set(columns_config["encoded_categorical"]).issubset(encoded.columns)
        assert (
            encoded[columns_config["encoded_categorical"]].isin([0.0, 1.0]).all().all()
        )

    def test_transform_is_noop_without_encoders(
        self,
        sample_raw_data,
        raw_columns_config,
        split_ratio,
        fit_transform_config,
        preprocessing_config,
    ):
        split = _prepare_featured_split(
            sample_raw_data,
            raw_columns_config,
            split_ratio,
            fit_transform_config,
            preprocessing_config,
        )
        encoded = transform_encoders(split, {})
        pd.testing.assert_frame_equal(encoded, split)


class TestFitTransformImputers:
    def test_zero_as_missing_imputer_replaces_configured_zeros(
        self,
        sample_raw_data,
        raw_columns_config,
        split_ratio,
        fit_transform_config,
        preprocessing_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        split, _report = add_split_column(
            cleaned,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        imputers = fit_zero_imputers(split, fit_transform_config, preprocessing_config)
        imputed = transform_zero_imputers(split, imputers)
        for column_name in preprocessing_config["zero_as_missing"]["columns"]:
            assert not (imputed[column_name] == 0).any()


class TestFitTransformOutlierCappers:
    def test_outlier_capping_clips_using_fitted_thresholds(
        self,
        sample_raw_data,
        raw_columns_config,
        split_ratio,
        fit_transform_config,
        preprocessing_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        split, _report = add_split_column(
            cleaned,
            split_ratio,
            random_state=42,
            stratify_column=None,
            preprocessing=preprocessing_config,
        )
        imputers = fit_zero_imputers(split, fit_transform_config, preprocessing_config)
        imputed = transform_zero_imputers(split, imputers)
        cappers = fit_outlier_cappers(
            imputed,
            raw_columns_config,
            fit_transform_config,
            preprocessing_config,
        )
        capped = transform_outlier_cappers(imputed, cappers)
        lower_limit, upper_limit = cappers["thresholds"]["Insulin"]
        assert capped["Insulin"].max() <= upper_limit
        assert capped["Insulin"].min() >= lower_limit


class TestFitTransformScalers:
    def test_fit_returns_one_scaler_per_numerical(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
        preprocessing_config,
    ):
        split = _prepare_featured_split(
            sample_raw_data,
            raw_columns_config,
            split_ratio,
            fit_transform_config,
            preprocessing_config,
        )
        encoders = fit_encoders(
            split, columns_config, fit_transform_config, preprocessing_config
        )
        encoded = transform_encoders(split, encoders)
        scalers = fit_scalers(
            encoded, columns_config, fit_transform_config, preprocessing_config
        )
        assert set(scalers.keys()) == set(columns_config["numerical"])

    def test_transform_changes_scale(
        self,
        sample_raw_data,
        raw_columns_config,
        columns_config,
        split_ratio,
        fit_transform_config,
        preprocessing_config,
    ):
        split = _prepare_featured_split(
            sample_raw_data,
            raw_columns_config,
            split_ratio,
            fit_transform_config,
            preprocessing_config,
        )
        encoders = fit_encoders(
            split, columns_config, fit_transform_config, preprocessing_config
        )
        encoded = transform_encoders(split, encoders)
        scalers = fit_scalers(
            encoded, columns_config, fit_transform_config, preprocessing_config
        )
        scaled = transform_scalers(encoded, scalers)
        for col in columns_config["numerical"]:
            assert scaled[col].std() != encoded[col].std() or len(scaled) == 1


class TestValidations:
    def test_validate_clean_data_can_run_twice_in_same_process(
        self,
        sample_raw_data,
        raw_columns_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        data_quality = {
            "cleaned": {
                "min_rows": 5,
                "classes": {
                    "column_to_exist": "ExpectColumnToExist",
                    "column_not_null": "ExpectColumnValuesToNotBeNull",
                    "column_between": "ExpectColumnValuesToBeBetween",
                    "table_min_rows": "ExpectTableRowCountToBeBetween",
                    "target_distinct_in_set": "ExpectColumnDistinctValuesToBeInSet",
                },
                "not_null_severity": "critical",
                "between_severity": "warning",
                "table_min_rows_severity": "critical",
                "numerical_ranges": {
                    "Pregnancies": [0, 20],
                    "Glucose": [0, 300],
                    "BloodPressure": [0, 200],
                    "SkinThickness": [0, 100],
                    "Insulin": [0, 900],
                    "BMI": [0, 80],
                    "DiabetesPedigreeFunction": [0, 3],
                    "Age": [0, 120],
                },
                "target_allowed_values": [0, 1],
                "extra_expectations": [],
            }
        }

        first = validate_clean_data(cleaned, raw_columns_config, data_quality)
        second = validate_clean_data(cleaned, raw_columns_config, data_quality)

        pd.testing.assert_frame_equal(first, second)

    def test_validate_split_data_uses_gx_for_split_stage(
        self,
        sample_raw_data,
        raw_columns_config,
        split_ratio,
        preprocessing_config,
    ):
        cleaned = clean_data(sample_raw_data, raw_columns_config)
        featured = add_features(cleaned)
        split, split_report = add_split_column(
            featured,
            split_ratio,
            random_state=42,
            stratify_column="Outcome",
            preprocessing=preprocessing_config,
        )
        data_quality = {
            "split": {
                "classes": {
                    "column_to_exist": "ExpectColumnToExist",
                    "column_values_in_set": "ExpectColumnValuesToBeInSet",
                    "table_min_rows": "ExpectTableRowCountToBeBetween",
                },
                "split_column_severity": "critical",
                "split_values_severity": "critical",
                "table_min_rows_critical_severity": "critical",
                "table_min_rows_warning_severity": "warning",
                "min_rows_per_split": 1,
                "warn_when_split_rows_below": 1,
                "min_minority_ratio": 0.01,
            }
        }

        validated = validate_split_data(
            split,
            split_report,
            split_ratio,
            "Outcome",
            data_quality,
        )

        pd.testing.assert_frame_equal(validated, split)

    def test_validate_split_data_raises_when_required_split_is_missing(self):
        split = pd.DataFrame(
            {
                "Pregnancies": [1, 2],
                "Outcome": [0, 1],
                "split": ["train", "train"],
            }
        )
        data_quality = {
            "split": {
                "classes": {
                    "column_to_exist": "ExpectColumnToExist",
                    "column_values_in_set": "ExpectColumnValuesToBeInSet",
                    "table_min_rows": "ExpectTableRowCountToBeBetween",
                },
                "split_column_severity": "critical",
                "split_values_severity": "critical",
                "table_min_rows_critical_severity": "critical",
                "table_min_rows_warning_severity": "warning",
                "min_rows_per_split": 1,
                "warn_when_split_rows_below": 1,
                "min_minority_ratio": 0.01,
            }
        }

        with pytest.raises(ValueError, match="crítica"):
            validate_split_data(
                split,
                None,
                {"train": 0.5, "validation": 0.5},
                "Outcome",
                data_quality,
            )
