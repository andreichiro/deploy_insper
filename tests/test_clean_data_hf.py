"""Testes unitários de clean_data usando dados reais do HuggingFace

Usa o dataset Genius-Society/Pima (Pima Indians Diabetes) pra validar
contra uma distribuição realista, não só linhas fabricadas na mão
"""

from __future__ import annotations

import pandas as pd
import pytest

datasets = pytest.importorskip("datasets", reason="pip install datasets")

from insper_deploy_kedro.pipelines.data_engineering.nodes import (
    add_features,
    clean_data,
)

HF_DATASET = "Genius-Society/Pima"

RAW_COLUMNS: dict[str, list[str]] = {
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


@pytest.fixture(scope="module")
def hf_dataframe() -> pd.DataFrame:
    """Carrega as primeiras 200 linhas do HF (c/ cache)."""
    ds = datasets.load_dataset(HF_DATASET, split="train[:200]")
    return ds.to_pandas()


class TestCleanDataHuggingFace:
    def test_output_has_expected_columns(self, hf_dataframe):
        result = clean_data(hf_dataframe, RAW_COLUMNS)
        expected = set(
            RAW_COLUMNS["target"]
            + RAW_COLUMNS["categorical"]
            + RAW_COLUMNS["numerical"]
        )
        assert set(result.columns) == expected

    def test_no_rows_lost(self, hf_dataframe):
        result = clean_data(hf_dataframe, RAW_COLUMNS)
        assert len(result) == len(hf_dataframe)

    def test_numerical_columns_are_numeric(self, hf_dataframe):
        result = clean_data(hf_dataframe, RAW_COLUMNS)
        for col in RAW_COLUMNS["numerical"]:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is not numeric"

    def test_no_nulls_in_numerical(self, hf_dataframe):
        result = clean_data(hf_dataframe, RAW_COLUMNS)
        for col in RAW_COLUMNS["numerical"]:
            assert result[col].isna().sum() == 0, f"{col} has NaN values"

    def test_add_features_on_hf_data(self, hf_dataframe):
        cleaned = clean_data(hf_dataframe, RAW_COLUMNS)
        featured = add_features(cleaned)
        assert "glucose_bmi_interaction" in featured.columns
        assert featured["glucose_bmi_interaction"].isna().sum() == 0
