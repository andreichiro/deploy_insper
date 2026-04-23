"""Fixtures compartilhadas pros testes do projeto diabetes."""

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
from insper_deploy_kedro.pipelines.modelling.nodes import train_model

# Espelha conf/base/parameters/*.yml — testes ficam alinhados ao config declarativo.
PREPROCESSING_FIXTURE: dict = {
    "train_test_split_function": "sklearn.model_selection.train_test_split",
    "min_rows_for_stratify": 20,
    "split_strategy": {
        "kind": "stratified_random",
        "label": "stratified_random_baseline",
    },
    "categorical_encoder": {
        "class_path": "sklearn.preprocessing.OneHotEncoder",
        "init_args": {
            "handle_unknown": "ignore",
            "drop": "first",
            "sparse_output": False,
            "dtype": float,
        },
        "category_orders": {
            "age_category": ["mature", "senior"],
            "bmi_category": ["underweight", "healthy", "overweight", "obese"],
            "glucose_category": ["normal", "prediabetes", "diabetes"],
            "age_bmi_category": [
                "underweight_mature",
                "underweight_senior",
                "healthy_mature",
                "healthy_senior",
                "overweight_mature",
                "overweight_senior",
                "obese_mature",
                "obese_senior",
            ],
            "age_glucose_category": [
                "low_mature",
                "low_senior",
                "normal_mature",
                "normal_senior",
                "hidden_mature",
                "hidden_senior",
                "high_mature",
                "high_senior",
            ],
            "insulin_category": ["normal", "abnormal"],
        },
    },
    "zero_as_missing": {
        "columns": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "scaler": {"class_path": "sklearn.preprocessing.RobustScaler", "init_args": {}},
        "imputer": {
            "class_path": "sklearn.impute.KNNImputer",
            "init_args": {"n_neighbors": 2, "keep_empty_features": True},
        },
    },
    "outlier_capping": {
        "enabled": True,
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
        "iqr_multiplier": 1.5,
        "columns": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
    },
    "numerical_scaler": {
        "class_path": "sklearn.preprocessing.StandardScaler",
        "init_args": {},
    },
}

ML_RUNTIME_FIXTURE: dict = {
    "target_encoder": {
        "class_path": "sklearn.preprocessing.LabelEncoder",
        "init_args": {},
    },
    "cross_validation": {
        "class_path": "sklearn.model_selection.StratifiedKFold",
        "init_args": {"shuffle": True, "random_state": 42},
    },
    "optuna_study": {"direction": "maximize"},
    "optuna_sampler": {
        "class_path": "optuna.samplers.TPESampler",
        "init_args": {"seed": 42},
    },
}

DECISION_POLICIES_FIXTURE: dict = {
    "development_splits": ["train", "validation"],
    "policy_selection_split": "validation",
    "cv_folds": 2,
    "candidate_thresholds": {"start": 0.2, "stop": 0.8, "step": 0.2},
    "deployment_policy": "prioritize_recall",
    "risk_bands": [
        {
            "name": "low",
            "label": "Baixo risco",
            "min_probability": 0.0,
            "max_probability": 0.3,
        },
        {
            "name": "moderate",
            "label": "Risco moderado",
            "min_probability": 0.3,
            "max_probability": 0.6,
        },
        {
            "name": "high",
            "label": "Alto risco",
            "min_probability": 0.6,
            "max_probability": 1.01,
        },
    ],
    "policies": {
        "default_050": {
            "strategy": "fixed_threshold",
            "threshold": 0.5,
            "label": "Threshold 0.50",
            "description": "Cutoff padrão",
        },
        "prioritize_recall": {
            "strategy": "min_expected_cost",
            "false_negative_cost": 8.0,
            "false_positive_cost": 1.0,
            "min_recall": 0.5,
            "label": "Priorizar recall",
            "description": "Reduz falsos negativos",
        },
        "prioritize_precision": {
            "strategy": "min_expected_cost",
            "false_negative_cost": 1.0,
            "false_positive_cost": 6.0,
            "min_precision": 0.3,
            "label": "Priorizar precision",
            "description": "Reduz falsos positivos",
        },
    },
}

EVALUATION_FIXTURE: dict = {
    "confusion_matrix": {
        "function_path": "sklearn.metrics.confusion_matrix",
        "kwargs": {},
    },
    "metrics": [
        {
            "key": "accuracy",
            "function_path": "sklearn.metrics.accuracy_score",
            "prediction_input": "y_pred",
            "kwargs": {},
        },
        {
            "key": "precision",
            "function_path": "sklearn.metrics.precision_score",
            "prediction_input": "y_pred",
            "kwargs": {"zero_division": 0},
        },
        {
            "key": "recall",
            "function_path": "sklearn.metrics.recall_score",
            "prediction_input": "y_pred",
            "kwargs": {"zero_division": 0},
        },
        {
            "key": "f1",
            "function_path": "sklearn.metrics.f1_score",
            "prediction_input": "y_pred",
            "kwargs": {"zero_division": 0},
        },
        {
            "key": "roc_auc",
            "function_path": "sklearn.metrics.roc_auc_score",
            "prediction_input": "y_proba",
            "kwargs": {},
        },
        {
            "key": "brier",
            "function_path": "sklearn.metrics.brier_score_loss",
            "prediction_input": "y_proba",
            "kwargs": {},
        },
        {
            "key": "log_loss",
            "function_path": "sklearn.metrics.log_loss",
            "prediction_input": "y_proba",
            "kwargs": {"labels": [0, 1]},
        },
    ],
    "derived": {
        "r2": {
            "function_path": "sklearn.metrics.r2_score",
            "prediction_input": "y_proba",
            "kwargs": {},
        },
        "mape": {"type": "mae_as_percent_of_mean_label"},
        "calibration": {"enabled": True},
    },
}

FEATURE_SELECTION_FIXTURE: dict = {
    "enabled": True,
    "selection_splits": ["train"],
    "selector_model": {
        "class_path": "sklearn.linear_model.LogisticRegression",
        "init_args": {
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "liblinear",
        },
    },
    "cv": {"n_splits": 2},
    "primary_metric": "brier",
    "secondary_metrics": [
        "roc_auc",
        "log_loss",
        "calibration_slope_error",
        "calibration_intercept_abs",
    ],
    "prefer_fewer_features": True,
    "min_blocks": 1,
    "max_blocks": 3,
    "max_candidates": 32,
    "feature_blocks": {
        "glucose_axis": ["Glucose"],
        "bmi_axis": ["BMI"],
        "age_axis": ["Age"],
        "interaction_axis": ["glucose_bmi_interaction"],
    },
    "required_blocks": {
        "interaction_axis": ["glucose_axis", "bmi_axis"],
    },
}


@pytest.fixture()
def preprocessing_config() -> dict:
    return PREPROCESSING_FIXTURE


@pytest.fixture()
def ml_runtime_config() -> dict:
    return ML_RUNTIME_FIXTURE


@pytest.fixture()
def evaluation_config() -> dict:
    return EVALUATION_FIXTURE


@pytest.fixture()
def decision_policy_config() -> dict:
    return DECISION_POLICIES_FIXTURE


@pytest.fixture()
def feature_selection_config() -> dict:
    return FEATURE_SELECTION_FIXTURE


@pytest.fixture()
def raw_columns_config() -> dict[str, list[str]]:
    """Config de colunas brutas, o que clean_data seleciona do arquivo"""
    return {
        "target": ["Outcome"],
        "categorical": [],
        "numerical": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
    }


@pytest.fixture()
def columns_config() -> dict[str, list[str]]:
    """Config completa de colunas, incluindo features derivadas"""
    numerical = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "glucose_bmi_interaction",
        "glucose_insulin_interaction",
        "glucose_pregnancies_interaction",
    ]
    categorical = [
        "age_category",
        "bmi_category",
        "glucose_category",
        "age_bmi_category",
        "age_glucose_category",
        "insulin_category",
    ]
    encoded_categorical = [
        "age_category_senior",
        "bmi_category_healthy",
        "bmi_category_overweight",
        "bmi_category_obese",
        "glucose_category_prediabetes",
        "glucose_category_diabetes",
        "age_bmi_category_underweight_senior",
        "age_bmi_category_healthy_mature",
        "age_bmi_category_healthy_senior",
        "age_bmi_category_overweight_mature",
        "age_bmi_category_overweight_senior",
        "age_bmi_category_obese_mature",
        "age_bmi_category_obese_senior",
        "age_glucose_category_low_senior",
        "age_glucose_category_normal_mature",
        "age_glucose_category_normal_senior",
        "age_glucose_category_hidden_mature",
        "age_glucose_category_hidden_senior",
        "age_glucose_category_high_mature",
        "age_glucose_category_high_senior",
        "insulin_category_abnormal",
    ]
    return {
        "target": ["Outcome"],
        "categorical": categorical,
        "numerical": numerical,
        "encoded_categorical": encoded_categorical,
        "model_features": numerical + encoded_categorical,
    }


@pytest.fixture()
def inference_raw_columns(raw_columns_config: dict) -> dict[str, list[str]]:
    """Colunas brutas de inferência (sem target)"""
    return {k: v for k, v in raw_columns_config.items() if k != "target"}


@pytest.fixture()
def sample_raw_data() -> pd.DataFrame:
    """Sample hardcoded do dataframe diabetes Pima (10 linhas)"""
    return pd.DataFrame(
        {
            "Pregnancies": [6, 1, 8, 1, 0, 5, 3, 10, 2, 8],
            "Glucose": [148, 85, 183, 89, 137, 116, 78, 115, 197, 125],
            "BloodPressure": [72, 66, 64, 66, 40, 74, 50, 0, 70, 96],
            "SkinThickness": [35, 29, 0, 23, 35, 0, 32, 0, 45, 0],
            "Insulin": [0, 0, 0, 94, 168, 0, 88, 0, 543, 0],
            "BMI": [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0.0],
            "DiabetesPedigreeFunction": [
                0.627,
                0.351,
                0.672,
                0.167,
                2.288,
                0.201,
                0.248,
                0.134,
                0.158,
                0.232,
            ],
            "Age": [50, 31, 32, 21, 33, 30, 26, 29, 53, 54],
            "Outcome": [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        }
    )


@pytest.fixture()
def split_ratio() -> dict[str, float]:
    return {"train": 0.7, "validation": 0.15, "test": 0.15}


@pytest.fixture()
def fit_transform_config() -> dict[str, list[str]]:
    return {"split_to_fit": ["train"]}


@pytest.fixture()
def master_table(
    sample_raw_data,
    raw_columns_config,
    columns_config,
    split_ratio,
    fit_transform_config,
    preprocessing_config,
) -> pd.DataFrame:
    """Master table completa passando por todo o pipeline DE"""
    cleaned = clean_data(sample_raw_data, raw_columns_config)
    split = add_split_column(
        cleaned,
        split_ratio,
        random_state=42,
        stratify_column="Outcome",
        preprocessing=preprocessing_config,
    )[0]
    imputers = fit_zero_imputers(split, fit_transform_config, preprocessing_config)
    imputed = transform_zero_imputers(split, imputers)
    outlier_cappers = fit_outlier_cappers(
        imputed,
        raw_columns_config,
        fit_transform_config,
        preprocessing_config,
    )
    capped = transform_outlier_cappers(imputed, outlier_cappers)
    featured = add_features(capped)
    encoders = fit_encoders(
        featured, columns_config, fit_transform_config, preprocessing_config
    )
    encoded = transform_encoders(featured, encoders)
    scalers = fit_scalers(
        encoded, columns_config, fit_transform_config, preprocessing_config
    )
    return transform_scalers(encoded, scalers)


@pytest.fixture()
def trained_model(master_table, columns_config, ml_runtime_config) -> dict:
    """Artefato de modelo treinado pra reuso nos testes"""
    params = {
        "class_path": "sklearn.linear_model.LogisticRegression",
        "train_splits": ["train"],
        "init_args": {"max_iter": 1000},
    }
    return train_model(master_table, columns_config, params, ml_runtime_config)
