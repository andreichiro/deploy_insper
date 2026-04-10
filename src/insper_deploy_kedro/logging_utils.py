"""Helpers for consistent project-side logging across API, Kedro, and UI."""

from __future__ import annotations

import logging
import logging.config
import os
import threading
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGGING_CONFIG_PATH = PROJECT_ROOT / "conf" / "logging.yml"

_logging_lock = threading.Lock()
_logging_configured = False


def _load_logging_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Logging configuration must be a mapping")
    return payload


def configure_project_logging() -> None:
    """Apply the project logging config once, with a safe fallback."""
    global _logging_configured  # noqa: PLW0603

    if _logging_configured:
        return

    with _logging_lock:
        if _logging_configured:
            return

        config_path = Path(
            os.getenv("KEDRO_LOGGING_CONFIG", str(DEFAULT_LOGGING_CONFIG_PATH))
        )
        if config_path.exists():
            try:
                logging.config.dictConfig(_load_logging_config(config_path))
            except Exception:
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                )
                logging.getLogger(__name__).exception(
                    "Failed to apply logging config from %s",
                    config_path,
                )
            else:
                _logging_configured = True
                return
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            logging.getLogger(__name__).warning(
                "Logging config file not found at %s; using basicConfig fallback",
                config_path,
            )

        _logging_configured = True
