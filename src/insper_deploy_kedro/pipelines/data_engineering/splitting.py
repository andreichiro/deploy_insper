"""Split helpers with explicit strategy reporting for random/stratified holdouts."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from insper_deploy_kedro.class_loading import load_callable
from insper_deploy_kedro.constants import SPLIT_COLUMN

logger = logging.getLogger(__name__)
_TWO_SPLITS = 2


def _normalize_split_ratio(split_ratio: dict[str, float]) -> dict[str, float]:
    ratios = {name: float(value) for name, value in split_ratio.items()}
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("split_ratio must sum to a positive value")
    return {name: value / total for name, value in ratios.items()}


def _build_split_strategy_report(  # noqa: PLR0913
    dataframe: pd.DataFrame,
    split_names: list[str],
    split_ratio: dict[str, float],
    requested_kind: str,
    resolved_kind: str,
    strategy_cfg: dict[str, Any],
    stratify_column: str | None,
    stratified_flag: bool,
    fallback_reason: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fallback_applied = requested_kind != resolved_kind

    for split_order, split_name in enumerate(split_names, start=1):
        split_frame = dataframe[dataframe[SPLIT_COLUMN] == split_name]
        rows.append(
            {
                "strategy_label": strategy_cfg.get("label", requested_kind),
                "requested_strategy_kind": requested_kind,
                "resolved_strategy_kind": resolved_kind,
                "fallback_applied": fallback_applied,
                "fallback_reason": fallback_reason or "",
                "split_name": split_name,
                "split_order": split_order,
                "requested_ratio": float(
                    _normalize_split_ratio(split_ratio).get(split_name, 0.0)
                ),
                "observed_ratio": (
                    float(len(split_frame) / len(dataframe)) if len(dataframe) else 0.0
                ),
                "rows": int(len(split_frame)),
                "stratify_column": stratify_column,
                "stratified_flag": bool(stratified_flag),
            }
        )

    return pd.DataFrame(rows)


def _random_or_stratified_split(
    dataframe: pd.DataFrame,
    split_ratio: dict[str, float],
    random_state: int,
    stratify_column: str | None,
    preprocessing: dict[str, Any],
) -> tuple[pd.DataFrame, bool, str]:
    split_dataframe = dataframe.copy()
    split_names = list(split_ratio.keys())
    min_strat = int(preprocessing.get("min_rows_for_stratify", 20))
    tts_path = preprocessing.get(
        "train_test_split_function",
        "sklearn.model_selection.train_test_split",
    )
    train_test_split = load_callable(tts_path)

    use_stratified = (
        stratify_column is not None
        and stratify_column in split_dataframe.columns
        and len(split_dataframe) >= min_strat
    )

    split_dataframe[SPLIT_COLUMN] = ""
    fallback_reason = ""
    if stratify_column is not None and not use_stratified:
        fallback_reason = (
            f"stratification_disabled:{stratify_column}:missing_column"
            if stratify_column not in split_dataframe.columns
            else f"stratification_disabled:{len(split_dataframe)}_rows_below_min_{min_strat}"
        )
        logger.warning(
            "add_split_column: estratificação solicitada mas desativada — %s. Usando split aleatório.",
            fallback_reason,
        )

    normalized_ratio = _normalize_split_ratio(split_ratio)
    if use_stratified:
        stratify_values = split_dataframe[stratify_column]
        train_name = split_names[0]
        rest_ratio = 1.0 - normalized_ratio[train_name]

        train_idx, rest_idx = train_test_split(
            split_dataframe.index,
            test_size=rest_ratio,
            random_state=random_state,
            stratify=stratify_values,
        )
        split_dataframe.loc[train_idx, SPLIT_COLUMN] = train_name

        if len(split_names) == _TWO_SPLITS:
            split_dataframe.loc[rest_idx, SPLIT_COLUMN] = split_names[1]
        else:
            remaining_names = split_names[1:]
            remaining_ratio_total = sum(
                normalized_ratio[name] for name in remaining_names
            )
            current_idx = np.asarray(rest_idx)
            for position, split_name in enumerate(remaining_names, start=1):
                if position == len(remaining_names):
                    split_dataframe.loc[current_idx, SPLIT_COLUMN] = split_name
                    break
                target_ratio = normalized_ratio[split_name] / remaining_ratio_total
                rest_stratify = stratify_values.loc[current_idx]
                selected_idx, current_idx = train_test_split(
                    current_idx,
                    test_size=1.0 - target_ratio,
                    random_state=random_state,
                    stratify=rest_stratify,
                )
                split_dataframe.loc[selected_idx, SPLIT_COLUMN] = split_name
                remaining_ratio_total -= normalized_ratio[split_name]
        logger.info("add_split_column: using STRATIFIED split on '%s'", stratify_column)
        return split_dataframe, True, fallback_reason

    random_generator = np.random.default_rng(seed=random_state)
    probabilities = np.array(
        [normalized_ratio[name] for name in split_names], dtype=float
    )
    split_dataframe[SPLIT_COLUMN] = random_generator.choice(
        split_names,
        size=len(split_dataframe),
        p=probabilities,
    )
    return split_dataframe, False, fallback_reason


def split_dataframe_with_report(
    dataframe: pd.DataFrame,
    split_ratio: dict[str, float],
    random_state: int,
    stratify_column: str | None,
    preprocessing: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign split labels and materialize an explicit strategy report."""
    strategy_cfg = dict(preprocessing.get("split_strategy") or {})
    requested_kind = str(strategy_cfg.get("kind", "stratified_random"))
    if requested_kind not in {"stratified_random", "random"}:
        raise ValueError(f"unsupported_split_strategy_kind:{requested_kind}")

    effective_stratify_column = (
        stratify_column if requested_kind == "stratified_random" else None
    )
    split_dataframe, stratified_flag, fallback_reason = _random_or_stratified_split(
        dataframe,
        split_ratio,
        random_state=random_state,
        stratify_column=effective_stratify_column,
        preprocessing=preprocessing,
    )
    resolved_kind = "stratified_random" if stratified_flag else "random"

    split_names = list(split_ratio.keys())
    report = _build_split_strategy_report(
        split_dataframe,
        split_names=split_names,
        split_ratio=split_ratio,
        requested_kind=requested_kind,
        resolved_kind=resolved_kind,
        strategy_cfg=strategy_cfg,
        stratify_column=stratify_column,
        stratified_flag=stratified_flag,
        fallback_reason=fallback_reason,
    )
    return split_dataframe, report
