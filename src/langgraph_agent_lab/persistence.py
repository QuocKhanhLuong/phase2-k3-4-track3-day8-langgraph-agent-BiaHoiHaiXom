"""Checkpointer adapters used by the LangGraph builder."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return the configured LangGraph checkpointer.

    ``memory`` is the core-lab default. ``sqlite`` is a durable extension that
    survives process restarts when the optional sqlite extra is installed.
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                'SQLite checkpointer requires: pip install -e ".[sqlite]"'
            ) from exc

        raw_path = database_url or "checkpoints.db"
        if raw_path.startswith("sqlite:///"):
            raw_path = raw_path.removeprefix("sqlite:///")
        elif raw_path.startswith("sqlite://"):
            raw_path = raw_path.removeprefix("sqlite://")

        if raw_path != ":memory:":
            path = Path(raw_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            raw_path = str(path)

        connection = sqlite3.connect(raw_path, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(connection)

    if kind == "postgres":
        raise NotImplementedError(
            "Postgres is an optional extension; configure a lifecycle-managed "
            "PostgresSaver before selecting checkpointer=postgres."
        )

    raise ValueError(f"Unknown checkpointer kind: {kind}")
