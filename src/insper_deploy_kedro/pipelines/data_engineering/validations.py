"""Validações de qualidade — Great Expectations guiado por YAML (classes e limites declarativos)."""

from __future__ import annotations

import logging
from typing import Any

import great_expectations as gx
import pandas as pd

from insper_deploy_kedro.constants import SPLIT_COLUMN

logger = logging.getLogger(__name__)


def _normalize_severity(value: Any) -> str:
    return str(value).split(".")[-1].lower()


def _ge_expectation_class(class_name: str) -> type:
    """Resolve nome de classe em gx.expectations (mesmo espírito do class_path da modelagem)."""
    try:
        return getattr(gx.expectations, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"Expectation '{class_name}' não existe em great_expectations.expectations"
        ) from exc


def _instantiate_expectation(class_name: str, kwargs: dict[str, Any]) -> Any:
    """Instancia uma expectation GX a partir do nome declarado no YAML."""
    cls = _ge_expectation_class(class_name)
    return cls(**kwargs)


def _make_batch(df: pd.DataFrame, asset_name: str) -> Any:
    """Cria um batch GX efêmero a partir de um DataFrame."""
    context = gx.get_context()
    ds_name = f"pandas_{asset_name}"
    data_source = context.data_sources.add_or_update_pandas(ds_name)
    return data_source.read_dataframe(df, asset_name=asset_name)


def _run_expectations(batch: Any, expectations: list[Any]) -> dict[str, Any]:
    """Roda lista de expectations e retorna resumo."""
    results: list[dict[str, Any]] = []
    all_passed = True

    for exp in expectations:
        result = batch.validate(exp)
        passed = result["success"]
        if not passed:
            all_passed = False
        kwargs = {
            k: v
            for k, v in result["expectation_config"]["kwargs"].items()
            if k != "batch_id"
        }
        severity = getattr(exp, "severity", None)
        if severity is not None and "severity" not in kwargs:
            kwargs["severity"] = _normalize_severity(severity)
        results.append({
            "expectation": type(exp).__name__,
            "kwargs": kwargs,
            "success": passed,
            "result_detail": result.get("result", {}),
        })

    return {"success": all_passed, "results": results}


def _flatten_column_groups(column_groups: dict[str, list[str]]) -> list[str]:
    return [
        column_name
        for group_columns in column_groups.values()
        for column_name in group_columns
    ]


def validate_clean_data(
    cleaned_data: pd.DataFrame,
    raw_columns: dict[str, list[str]],
    data_quality: dict[str, Any],
) -> pd.DataFrame:
    """Roda GE conforme `data_quality` no YAML — classes e ranges sem hardcode."""
    cfg = data_quality["cleaned"]
    classes = cfg["classes"]

    logger.info(
        "validate_clean_data: %d linhas, expectations declaradas no YAML",
        len(cleaned_data),
    )

    batch = _make_batch(cleaned_data, "cleaned_data")
    expectations: list[Any] = []

    exist_cls = classes["column_to_exist"]
    for col in _flatten_column_groups(raw_columns):
        expectations.append(
            _instantiate_expectation(exist_cls, {"column": col}),
        )

    not_null_cls = classes["column_not_null"]
    not_null_sev = cfg.get("not_null_severity", "critical")
    for col in raw_columns.get("numerical", []) + raw_columns.get("target", []):
        expectations.append(
            _instantiate_expectation(
                not_null_cls,
                {"column": col, "severity": not_null_sev},
            ),
        )

    between_cls = classes["column_between"]
    between_sev = cfg.get("between_severity", "warning")
    ranges = cfg.get("numerical_ranges", {})
    for col, bounds in ranges.items():
        if col not in cleaned_data.columns:
            continue
        lo, hi = bounds[0], bounds[1]
        expectations.append(
            _instantiate_expectation(
                between_cls,
                {
                    "column": col,
                    "min_value": lo,
                    "max_value": hi,
                    "severity": between_sev,
                },
            ),
        )

    min_rows = int(cfg["min_rows"])
    table_cls = classes["table_min_rows"]
    table_sev = cfg.get("table_min_rows_severity", "critical")
    table_kwargs: dict[str, Any] = {"severity": table_sev}
    if table_cls == "ExpectTableRowCountToBeBetween":
        table_kwargs["min_value"] = min_rows
    else:
        table_kwargs["value"] = min_rows
    expectations.append(
        _instantiate_expectation(
            table_cls,
            table_kwargs,
        ),
    )

    target_cols = raw_columns.get("target", [])
    if target_cols:
        target_cls = classes["target_distinct_in_set"]
        allowed = cfg.get("target_allowed_values", [0, 1])
        expectations.append(
            _instantiate_expectation(
                target_cls,
                {"column": target_cols[0], "value_set": list(allowed)},
            ),
        )

    for spec in cfg.get("extra_expectations", []) or []:
        exp_class = spec["expectation_class"]
        kwargs = dict(spec.get("kwargs", {}))
        expectations.append(_instantiate_expectation(exp_class, kwargs))

    report = _run_expectations(batch, expectations)

    critical_failures = [
        r
        for r in report["results"]
        if not r["success"]
        and _normalize_severity(r.get("kwargs", {}).get("severity")) == "critical"
    ]
    warning_failures = [
        r for r in report["results"] if not r["success"] and r not in critical_failures
    ]

    for w in warning_failures:
        logger.warning(
            "validate_clean_data WARNING: %s %s",
            w["expectation"],
            w["kwargs"],
        )

    if critical_failures:
        msgs = [f"{f['expectation']} {f['kwargs']}" for f in critical_failures]
        raise ValueError(
            "validate_clean_data: validação(ões) crítica(s) falharam: "
            + "; ".join(msgs)
        )

    logger.info(
        "validate_clean_data: %d expectativas OK, %d warnings",
        sum(1 for r in report["results"] if r["success"]),
        len(warning_failures),
    )
    return cleaned_data


def validate_split_data(  # noqa: PLR0912
    split_data: pd.DataFrame,
    split_strategy_report: pd.DataFrame | None,
    split_ratio: dict[str, float],
    stratify_column: str | None,
    data_quality: dict[str, Any],
) -> pd.DataFrame:
    """Checagens pós-split — GX declarativo mais warnings estatísticos leves."""
    cfg = data_quality["split"]
    classes = cfg.get("classes", {})
    min_minority = float(cfg["min_minority_ratio"])
    min_rows = int(cfg.get("min_rows_per_split", 1))
    warn_below = int(cfg["warn_when_split_rows_below"])
    split_names = list(split_ratio.keys())

    logger.info(
        "validate_split_data: %d linhas, %d splits",
        len(split_data),
        len(split_ratio),
    )

    expectation_results: list[dict[str, Any]] = []

    full_batch = _make_batch(split_data, "split_data")
    full_expectations = [
        _instantiate_expectation(
            classes.get("column_to_exist", "ExpectColumnToExist"),
            {
                "column": SPLIT_COLUMN,
                "severity": cfg.get("split_column_severity", "critical"),
            },
        ),
        _instantiate_expectation(
            classes.get("column_values_in_set", "ExpectColumnValuesToBeInSet"),
            {
                "column": SPLIT_COLUMN,
                "value_set": split_names,
                "severity": cfg.get("split_values_severity", "critical"),
            },
        ),
    ]
    full_report = _run_expectations(full_batch, full_expectations)
    for result in full_report["results"]:
        result["scope"] = "all_splits"
    expectation_results.extend(full_report["results"])

    if SPLIT_COLUMN not in split_data.columns:
        critical_failures = [
            result
            for result in expectation_results
            if not result["success"]
            and _normalize_severity(result.get("kwargs", {}).get("severity"))
            == "critical"
        ]
        msgs = [
            f"{failure['scope']}: {failure['expectation']} {failure['kwargs']}"
            for failure in critical_failures
        ]
        raise ValueError(
            "validate_split_data: validação(ões) crítica(s) falharam: "
            + "; ".join(msgs)
        )

    for split_name in split_names:
        split_df = split_data[split_data[SPLIT_COLUMN] == split_name]
        split_batch = _make_batch(split_df, f"split_{split_name}")
        split_expectations = [
            _instantiate_expectation(
                classes.get("table_min_rows", "ExpectTableRowCountToBeBetween"),
                {
                    "min_value": min_rows,
                    "severity": cfg.get("table_min_rows_critical_severity", "critical"),
                },
            ),
            _instantiate_expectation(
                classes.get("table_min_rows", "ExpectTableRowCountToBeBetween"),
                {
                    "min_value": warn_below,
                    "severity": cfg.get("table_min_rows_warning_severity", "warning"),
                },
            ),
            _instantiate_expectation(
                classes.get("column_values_in_set", "ExpectColumnValuesToBeInSet"),
                {
                    "column": SPLIT_COLUMN,
                    "value_set": [split_name],
                    "severity": cfg.get("split_values_severity", "critical"),
                },
            ),
        ]
        split_report = _run_expectations(split_batch, split_expectations)
        for result in split_report["results"]:
            result["scope"] = split_name
        expectation_results.extend(split_report["results"])

        if stratify_column and stratify_column in split_df.columns:
            value_counts = split_df[stratify_column].value_counts(normalize=True)
            if not value_counts.empty and float(value_counts.min()) < min_minority:
                logger.warning(
                    "validate_split_data: split '%s' classe minoritária %.1f%% (< %.0f%%)",
                    split_name,
                    float(value_counts.min()) * 100,
                    min_minority * 100,
                )

    critical_failures = [
        result
        for result in expectation_results
        if not result["success"]
        and _normalize_severity(result.get("kwargs", {}).get("severity")) == "critical"
    ]
    warning_failures = [
        result
        for result in expectation_results
        if not result["success"] and result not in critical_failures
    ]

    for warning in warning_failures:
        logger.warning(
            "validate_split_data WARNING [%s]: %s %s",
            warning.get("scope", "?"),
            warning["expectation"],
            warning["kwargs"],
        )

    if critical_failures:
        msgs = [
            f"{failure['scope']}: {failure['expectation']} {failure['kwargs']}"
            for failure in critical_failures
        ]
        raise ValueError(
            "validate_split_data: validação(ões) crítica(s) falharam: "
            + "; ".join(msgs),
        )

    logger.info(
        "validate_split_data: %d criticals, %d warnings, splits OK",
        len(critical_failures),
        len(warning_failures),
    )
    return split_data
