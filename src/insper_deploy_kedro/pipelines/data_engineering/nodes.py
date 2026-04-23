"""Nodes de data engineering — fit/transform com classes sklearn declaradas no YAML."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from insper_deploy_kedro.class_loading import load_class
from insper_deploy_kedro.constants import SPLIT_COLUMN
from insper_deploy_kedro.pipelines.data_engineering.splitting import (
    split_dataframe_with_report,
)

logger = logging.getLogger(__name__)

SENIOR_AGE_THRESHOLD = 50
LOW_GLUCOSE_THRESHOLD = 70
NORMAL_GLUCOSE_UPPER = 100
HIDDEN_GLUCOSE_UPPER = 125


def _flatten_column_groups(
    column_groups: dict[str, list[str]],
) -> list[str]:
    """Achata {grupo: [colunas]} numa lista única."""
    return [
        column_name
        for group_columns in column_groups.values()
        for column_name in group_columns
    ]


def _validate_columns_exist(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    *,
    caller: str,
) -> None:
    """Estoura erro cedo se tiver colunas faltando."""
    missing = set(required_columns) - set(dataframe.columns)
    if missing:
        raise KeyError(f"{caller}: columns not found in dataframe: {sorted(missing)}")


def _training_mask(
    dataframe: pd.DataFrame,
    fit_transform_params: dict[str, list[str]],
) -> pd.Series:
    splits_to_fit: list[str] = fit_transform_params["split_to_fit"]
    return dataframe[SPLIT_COLUMN].isin(splits_to_fit)


def _replace_configured_zeros_with_na(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    transformed = dataframe.copy()
    for column_name in columns:
        transformed[column_name] = transformed[column_name].mask(
            transformed[column_name] == 0,
            np.nan,
        )
    return transformed


def _outlier_limits(
    series: pd.Series,
    *,
    lower_quantile: float,
    upper_quantile: float,
    multiplier: float,
) -> tuple[float, float]:
    lower_value = float(series.quantile(lower_quantile))
    upper_value = float(series.quantile(upper_quantile))
    interquantile_range = upper_value - lower_value
    return (
        lower_value - multiplier * interquantile_range,
        upper_value + multiplier * interquantile_range,
    )


def fit_zero_imputers(
    split_dataframe: pd.DataFrame,
    fit_transform_params: dict[str, list[str]],
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    """Fita imputador KNN para zeros clinicamente inválidos usando só splits de fit."""
    cfg = preprocessing.get("zero_as_missing") or {}
    columns = list(cfg.get("columns") or [])
    _validate_columns_exist(split_dataframe, columns, caller="fit_zero_imputers")
    if not columns:
        return {"columns": [], "imputer": None, "scaler": None}

    training_values = split_dataframe.loc[
        _training_mask(split_dataframe, fit_transform_params),
        columns,
    ].copy()
    training_values = _replace_configured_zeros_with_na(training_values, columns)

    scaler_cfg = cfg.get("scaler") or {}
    scaler = None
    imputer_input = training_values
    if scaler_cfg.get("class_path"):
        scaler_class = load_class(scaler_cfg["class_path"])
        scaler = scaler_class(**dict(scaler_cfg.get("init_args") or {}))
        imputer_input = pd.DataFrame(
            scaler.fit_transform(training_values),
            columns=columns,
            index=training_values.index,
        )

    imputer_cfg = cfg["imputer"]
    imputer_class = load_class(imputer_cfg["class_path"])
    imputer = imputer_class(**dict(imputer_cfg.get("init_args") or {}))
    imputer.fit(imputer_input)

    logger.info(
        "fit_zero_imputers: fitted %s for %d columns",
        imputer_cfg["class_path"],
        len(columns),
    )
    return {"columns": columns, "imputer": imputer, "scaler": scaler}


def transform_zero_imputers(
    dataframe: pd.DataFrame,
    imputers: dict[str, Any],
) -> pd.DataFrame:
    """Transforma zeros clinicamente inválidos em NA e aplica imputador fitado."""
    columns = list(imputers.get("columns") or [])
    _validate_columns_exist(dataframe, columns, caller="transform_zero_imputers")
    transformed = dataframe.copy()
    if not columns:
        return transformed
    for column_name in columns:
        transformed[column_name] = transformed[column_name].astype(float)

    imputer_input = _replace_configured_zeros_with_na(transformed[columns], columns)
    scaler = imputers.get("scaler")
    if scaler is not None:
        imputer_input = pd.DataFrame(
            scaler.transform(imputer_input),
            columns=columns,
            index=transformed.index,
        )

    imputed_values = imputers["imputer"].transform(imputer_input)
    if scaler is not None:
        imputed_values = scaler.inverse_transform(imputed_values)

    imputed_frame = pd.DataFrame(
        imputed_values,
        columns=columns,
        index=transformed.index,
    )
    for column_name in columns:
        transformed[column_name] = imputed_frame[column_name]
    return transformed


def fit_outlier_cappers(
    dataframe: pd.DataFrame,
    columns: dict[str, list[str]],
    fit_transform_params: dict[str, list[str]],
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    """Calcula limites de winsorização/capping em splits de fit."""
    cfg = preprocessing.get("outlier_capping") or {}
    cap_columns = list(cfg.get("columns") or columns.get("numerical", []))
    _validate_columns_exist(dataframe, cap_columns, caller="fit_outlier_cappers")
    if not bool(cfg.get("enabled", True)) or not cap_columns:
        return {"enabled": False, "thresholds": {}}

    lower_quantile = float(cfg.get("lower_quantile", 0.05))
    upper_quantile = float(cfg.get("upper_quantile", 0.95))
    multiplier = float(cfg.get("iqr_multiplier", 1.5))
    training_frame = dataframe.loc[_training_mask(dataframe, fit_transform_params)]

    thresholds = {
        column_name: _outlier_limits(
            training_frame[column_name],
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            multiplier=multiplier,
        )
        for column_name in cap_columns
    }
    logger.info("fit_outlier_cappers: fitted caps for %d columns", len(thresholds))
    return {"enabled": True, "thresholds": thresholds}


def transform_outlier_cappers(
    dataframe: pd.DataFrame,
    outlier_cappers: dict[str, Any],
) -> pd.DataFrame:
    """Aplica limites fitados para reduzir influência de outliers extremos."""
    transformed = dataframe.copy()
    thresholds: dict[str, tuple[float, float]] = outlier_cappers.get("thresholds") or {}
    if not bool(outlier_cappers.get("enabled", True)) or not thresholds:
        return transformed

    _validate_columns_exist(
        transformed,
        list(thresholds),
        caller="transform_outlier_cappers",
    )
    for column_name, (lower_limit, upper_limit) in thresholds.items():
        transformed[column_name] = transformed[column_name].clip(
            lower=lower_limit,
            upper=upper_limit,
        )
    return transformed


def add_features(cleaned_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Cria sinais clínicos derivados a partir das colunas limpas e imputadas."""
    featured = cleaned_dataframe.copy()
    featured["glucose_bmi_interaction"] = featured["Glucose"] * featured["BMI"] / 1000
    featured["glucose_insulin_interaction"] = featured["Glucose"] * featured["Insulin"]
    featured["glucose_pregnancies_interaction"] = (
        featured["Glucose"] * featured["Pregnancies"]
    )

    featured["age_category"] = np.where(
        featured["Age"] >= SENIOR_AGE_THRESHOLD,
        "senior",
        "mature",
    )
    featured["bmi_category"] = pd.cut(
        featured["BMI"],
        bins=[0, 18.5, 24.9, 29.9, np.inf],
        labels=["underweight", "healthy", "overweight", "obese"],
        include_lowest=True,
    ).astype(str)
    featured["glucose_category"] = pd.cut(
        featured["Glucose"],
        bins=[0, 140, 200, 300],
        labels=["normal", "prediabetes", "diabetes"],
        include_lowest=True,
    ).astype(str)
    featured["age_bmi_category"] = (
        featured["bmi_category"] + "_" + featured["age_category"]
    )

    glucose_age_band = pd.Series("high", index=featured.index)
    glucose_age_band = glucose_age_band.mask(
        featured["Glucose"] < LOW_GLUCOSE_THRESHOLD,
        "low",
    )
    glucose_age_band = glucose_age_band.mask(
        (featured["Glucose"] >= LOW_GLUCOSE_THRESHOLD)
        & (featured["Glucose"] < NORMAL_GLUCOSE_UPPER),
        "normal",
    )
    glucose_age_band = glucose_age_band.mask(
        (featured["Glucose"] >= NORMAL_GLUCOSE_UPPER)
        & (featured["Glucose"] <= HIDDEN_GLUCOSE_UPPER),
        "hidden",
    )
    featured["age_glucose_category"] = glucose_age_band + "_" + featured["age_category"]
    featured["insulin_category"] = np.where(
        featured["Insulin"].between(16, 166, inclusive="both"),
        "normal",
        "abnormal",
    )

    logger.info("add_features: added clinical derived features")
    return featured


def clean_data(
    raw_data: pd.DataFrame,
    columns: dict[str, list[str]],
) -> pd.DataFrame:
    """Seleciona colunas do config e converte numéricas — valores inválidos viram 0."""
    selected_columns = _flatten_column_groups(columns)
    _validate_columns_exist(raw_data, selected_columns, caller="clean_data")

    cleaned_dataframe = raw_data[selected_columns].copy()

    for numerical_column in columns["numerical"]:
        original_nulls = (
            pd.to_numeric(cleaned_dataframe[numerical_column], errors="coerce")
            .isna()
            .sum()
        )
        cleaned_dataframe[numerical_column] = pd.to_numeric(
            cleaned_dataframe[numerical_column], errors="coerce"
        ).fillna(0)
        if original_nulls > 0:
            logger.warning(
                "clean_data: %s had %d non-numeric values → replaced with 0",
                numerical_column,
                original_nulls,
            )

    logger.info(
        "clean_data: kept %d columns, %d rows",
        len(selected_columns),
        len(cleaned_dataframe),
    )
    return cleaned_dataframe


def add_split_column(
    cleaned_dataframe: pd.DataFrame,
    split_ratio: dict[str, float],
    random_state: int,
    stratify_column: str | None,
    preprocessing: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/val/test — `train_test_split` e limiares vêm do YAML."""
    split_dataframe, split_strategy_report = split_dataframe_with_report(
        cleaned_dataframe,
        split_ratio,
        random_state=random_state,
        stratify_column=stratify_column,
        preprocessing=preprocessing,
    )

    for split_name in split_ratio:
        row_count = (split_dataframe[SPLIT_COLUMN] == split_name).sum()
        logger.info("add_split_column: %s -> %d rows", split_name, row_count)

    return split_dataframe, split_strategy_report


def fit_encoders(
    split_dataframe: pd.DataFrame,
    columns: dict[str, list[str]],
    fit_transform_params: dict[str, list[str]],
    preprocessing: dict[str, Any],
) -> dict[str, Any]:
    """Fita encoder categórico por coluna (classe + init_args no YAML)."""
    categorical_columns = columns["categorical"]
    _validate_columns_exist(split_dataframe, categorical_columns, caller="fit_encoders")

    enc_cfg = preprocessing["categorical_encoder"]
    encoder_class = load_class(enc_cfg["class_path"])
    encoder_init = dict(enc_cfg.get("init_args") or {})
    category_orders = dict(enc_cfg.get("category_orders") or {})

    splits_to_fit: list[str] = fit_transform_params["split_to_fit"]
    training_mask = split_dataframe[SPLIT_COLUMN].isin(splits_to_fit)
    fitted_encoders: dict[str, BaseEstimator] = {}

    for column_name in categorical_columns:
        column_init = dict(encoder_init)
        if column_name in category_orders:
            column_init["categories"] = [list(category_orders[column_name])]
        encoder = encoder_class(**column_init)
        training_values = split_dataframe.loc[training_mask, [column_name]].astype(str)
        encoder.fit(training_values)
        output_columns = (
            list(encoder.get_feature_names_out([column_name]))
            if hasattr(encoder, "get_feature_names_out")
            else [column_name]
        )
        fitted_encoders[column_name] = {
            "estimator": encoder,
            "output_columns": output_columns,
        }

        n_cat = len(encoder.categories_[0]) if hasattr(encoder, "categories_") else 0
        logger.info(
            "fit_encoders: %s -> %d categories (%s)",
            column_name,
            n_cat,
            enc_cfg["class_path"],
        )

    return fitted_encoders


def transform_encoders(
    split_dataframe: pd.DataFrame,
    encoders: dict[str, Any],
) -> pd.DataFrame:
    """Aplica encoders fitados em todos os splits."""
    encoded_dataframe = split_dataframe.copy()

    for column_name, encoder_payload in encoders.items():
        if isinstance(encoder_payload, dict):
            encoder = encoder_payload["estimator"]
            output_columns = list(
                encoder_payload.get("output_columns") or [column_name]
            )
        else:
            encoder = encoder_payload
            output_columns = [column_name]

        transformed_values = encoder.transform(
            encoded_dataframe[[column_name]].astype(str)
        )
        if hasattr(transformed_values, "toarray"):
            transformed_values = transformed_values.toarray()
        transformed_frame = pd.DataFrame(
            transformed_values,
            columns=output_columns,
            index=encoded_dataframe.index,
        )
        if len(output_columns) == 1 and output_columns[0] == column_name:
            encoded_dataframe[column_name] = transformed_frame[column_name].astype(int)
        else:
            encoded_dataframe = encoded_dataframe.drop(columns=[column_name])
            encoded_dataframe = pd.concat(
                [encoded_dataframe, transformed_frame], axis=1
            )

    return encoded_dataframe


def fit_scalers(
    encoded_dataframe: pd.DataFrame,
    columns: dict[str, list[str]],
    fit_transform_params: dict[str, list[str]],
    preprocessing: dict[str, Any],
) -> dict[str, BaseEstimator]:
    """Fita scaler numérico por coluna (classe + init_args no YAML)."""
    numerical_columns = columns["numerical"]
    _validate_columns_exist(encoded_dataframe, numerical_columns, caller="fit_scalers")

    sc_cfg = preprocessing["numerical_scaler"]
    scaler_class = load_class(sc_cfg["class_path"])
    scaler_init = dict(sc_cfg.get("init_args") or {})

    splits_to_fit: list[str] = fit_transform_params["split_to_fit"]
    training_mask = encoded_dataframe[SPLIT_COLUMN].isin(splits_to_fit)
    fitted_scalers: dict[str, BaseEstimator] = {}

    for column_name in numerical_columns:
        scaler = scaler_class(**scaler_init)
        training_values = encoded_dataframe.loc[training_mask, [column_name]]
        scaler.fit(training_values)
        fitted_scalers[column_name] = scaler

        mean_v = float(scaler.mean_[0]) if hasattr(scaler, "mean_") else 0.0
        scale_v = float(scaler.scale_[0]) if hasattr(scaler, "scale_") else 1.0
        logger.info(
            "fit_scalers: %s (%s) -> mean=%.4f, std=%.4f",
            column_name,
            sc_cfg["class_path"],
            mean_v,
            scale_v,
        )

    return fitted_scalers


def transform_scalers(
    encoded_dataframe: pd.DataFrame,
    scalers: dict[str, BaseEstimator],
) -> pd.DataFrame:
    """Aplica scalers fitados em todos os splits."""
    scaled_dataframe = encoded_dataframe.copy()

    for column_name, scaler in scalers.items():
        scaled_dataframe[column_name] = scaler.transform(
            scaled_dataframe[[column_name]]
        )

    return scaled_dataframe
