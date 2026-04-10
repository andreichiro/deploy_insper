"""Constantes e types compartilhados"""

from __future__ import annotations

from typing import Any, TypedDict

SPLIT_COLUMN = "split"


class ModelArtifact(TypedDict, total=False):
    """Estrutura do dict de artefatos retornado do train_model/optimize_model/calibrate_model.

    Campos obrigatórios; opcionais aparecem só dps da otimização
    """

    estimator: Any
    target_encoder: Any
    feature_columns: list[str]
    class_path: str
    init_args: dict[str, Any]
    best_params: dict[str, Any]
    best_cv_score: float
    cv_folds: int
    train_splits: list[str]
    decision_threshold: float
    decision_policy_name: str
    decision_policy_description: str
    policy_catalog: list[dict[str, Any]]
    risk_bands: list[dict[str, Any]]
