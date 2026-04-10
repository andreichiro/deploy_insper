"""Persistent local operations store for runs, experiments, and model registry."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPS_DB_PATH = PROJECT_ROOT / "data" / "09_ops" / "ops.sqlite3"

_schema_lock = threading.Lock()
_initialized = False


def _connect() -> sqlite3.Connection:
    OPS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(OPS_DB_PATH, timeout=30, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    _ensure_schema(connection)
    return connection


@contextmanager
def _managed_connection():
    connection = _connect()
    try:
        yield connection
    finally:
        connection.close()


def _ensure_schema(connection: sqlite3.Connection) -> None:
    global _initialized  # noqa: PLW0603

    if _initialized:
        return

    with _schema_lock:
        if _initialized:
            return

        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS background_runs (
                run_id TEXT PRIMARY KEY,
                pipeline TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                error TEXT,
                result_json TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS experiment_registry (
                experiment_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                selected_model_name TEXT,
                selected_class_path TEXT,
                selection_metric TEXT,
                selection_score REAL,
                selected_policy_name TEXT,
                selected_threshold REAL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_registry (
                registry_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                stage TEXT NOT NULL,
                class_path TEXT,
                train_splits_json TEXT,
                decision_policy_name TEXT,
                decision_threshold REAL,
                payload_json TEXT NOT NULL
            );
            """
        )
        connection.commit()
        _initialized = True


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _from_json(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def upsert_background_run(run_id: str, record: dict[str, Any]) -> None:
    payload = dict(record)
    with _managed_connection() as connection:
        connection.execute(
            """
            INSERT INTO background_runs (
                run_id,
                pipeline,
                status,
                started_at,
                finished_at,
                error,
                result_json,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                pipeline = excluded.pipeline,
                status = excluded.status,
                started_at = excluded.started_at,
                finished_at = excluded.finished_at,
                error = excluded.error,
                result_json = excluded.result_json,
                updated_at = excluded.updated_at
            """,
            (
                run_id,
                str(payload["pipeline"]),
                str(payload["status"]),
                payload.get("started_at"),
                payload.get("finished_at"),
                payload.get("error"),
                _to_json(payload.get("result")),
                str(payload.get("updated_at") or payload.get("finished_at") or ""),
            ),
        )
        connection.commit()


def get_background_run(run_id: str) -> dict[str, Any] | None:
    with _managed_connection() as connection:
        row = connection.execute(
            """
            SELECT
                run_id,
                pipeline,
                status,
                started_at,
                finished_at,
                error,
                result_json
            FROM background_runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "pipeline": row["pipeline"],
        "status": row["status"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "error": row["error"],
        "result": _from_json(row["result_json"]),
    }


def record_experiment_run(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    with _managed_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO experiment_registry (
                experiment_id,
                created_at,
                selected_model_name,
                selected_class_path,
                selection_metric,
                selection_score,
                selected_policy_name,
                selected_threshold,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(payload["experiment_id"]),
                str(payload["created_at"]),
                payload.get("selected_model_name"),
                payload.get("selected_class_path"),
                payload.get("selection_metric"),
                payload.get("selection_score"),
                payload.get("selected_policy_name"),
                payload.get("selected_threshold"),
                _to_json(payload),
            ),
        )
        connection.commit()
    return payload


def record_model_registry_entry(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    with _managed_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO model_registry (
                registry_id,
                created_at,
                stage,
                class_path,
                train_splits_json,
                decision_policy_name,
                decision_threshold,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(payload["registry_id"]),
                str(payload["created_at"]),
                str(payload.get("stage", "production")),
                payload.get("class_path"),
                _to_json(payload.get("train_splits", [])),
                payload.get("decision_policy_name"),
                payload.get("decision_threshold"),
                _to_json(payload),
            ),
        )
        connection.commit()
    return payload


def list_experiment_runs(limit: int = 20) -> list[dict[str, Any]]:
    with _managed_connection() as connection:
        rows = connection.execute(
            """
            SELECT payload_json
            FROM experiment_registry
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [_from_json(row["payload_json"]) for row in rows]


def list_model_registry_entries(limit: int = 20) -> list[dict[str, Any]]:
    with _managed_connection() as connection:
        rows = connection.execute(
            """
            SELECT payload_json
            FROM model_registry
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [_from_json(row["payload_json"]) for row in rows]
