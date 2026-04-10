"""Carrega classes e callables por caminho pontilhado (mesmo padrão do class_path da modelagem)."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def load_class(class_path: str) -> type:
    """Ex.: sklearn.linear_model.LogisticRegression"""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_callable(qualified_path: str) -> Callable[..., Any]:
    """Ex.: sklearn.model_selection.train_test_split"""
    module_path, name = qualified_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    obj = getattr(module, name)
    if not callable(obj):
        raise TypeError(f"{qualified_path} não é callable")
    return obj
