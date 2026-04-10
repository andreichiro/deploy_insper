"""Nodes de data engineering — fit/transform com classes sklearn declaradas no YAML."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator

from insper_deploy_kedro.class_loading import load_class
from insper_deploy_kedro.constants import SPLIT_COLUMN
from insper_deploy_kedro.pipelines.data_engineering.splitting import (
    split_dataframe_with_report,
)

logger = logging.getLogger(__name__)


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


def add_features(cleaned_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas a partir das colunas limpas."""
    featured = cleaned_dataframe.copy()
    featured["glucose_bmi_interaction"] = (
        featured["Glucose"] * featured["BMI"] / 1000
    )
    logger.info("add_features: added glucose_bmi_interaction")
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
) -> dict[str, BaseEstimator]:
    """Fita encoder categórico por coluna (classe + init_args no YAML)."""
    categorical_columns = columns["categorical"]
    _validate_columns_exist(split_dataframe, categorical_columns, caller="fit_encoders")

    enc_cfg = preprocessing["categorical_encoder"]
    encoder_class = load_class(enc_cfg["class_path"])
    encoder_init = dict(enc_cfg.get("init_args") or {})

    splits_to_fit: list[str] = fit_transform_params["split_to_fit"]
    training_mask = split_dataframe[SPLIT_COLUMN].isin(splits_to_fit)
    fitted_encoders: dict[str, BaseEstimator] = {}

    for column_name in categorical_columns:
        encoder = encoder_class(**encoder_init)
        training_values = split_dataframe.loc[training_mask, [column_name]].astype(str)
        encoder.fit(training_values)
        fitted_encoders[column_name] = encoder

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
    encoders: dict[str, BaseEstimator],
) -> pd.DataFrame:
    """Aplica encoders fitados em todos os splits."""
    encoded_dataframe = split_dataframe.copy()

    for column_name, encoder in encoders.items():
        encoded_dataframe[column_name] = (
            encoder.transform(encoded_dataframe[[column_name]].astype(str))
            .ravel()
            .astype(int)
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
