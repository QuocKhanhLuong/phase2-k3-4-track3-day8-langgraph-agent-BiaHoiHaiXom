"""Metrics schema and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field


class ScenarioMetric(BaseModel):
    scenario_id: str
    success: bool
    expected_route: str
    actual_route: str | None = None
    nodes_visited: int = 0
    retry_count: int = 0
    interrupt_count: int = 0
    approval_required: bool = False
    approval_observed: bool = False
    latency_ms: int = 0
    errors: list[str] = Field(default_factory=list)


class MetricsReport(BaseModel):
    total_scenarios: int
    success_rate: float
    avg_nodes_visited: float
    total_retries: int
    total_interrupts: int
    resume_success: bool = False
    scenario_metrics: list[ScenarioMetric]


def metric_from_state(
    state: dict[str, Any],
    expected_route: str,
    approval_required: bool,
    latency_ms: int = 0,
) -> ScenarioMetric:
    """Convert final graph state into an auditable per-scenario metric."""
    events = state.get("events", []) or []
    errors = state.get("errors", []) or []
    actual_route = state.get("route")
    approval = state.get("approval")
    nodes = [event.get("node", "unknown") for event in events]
    retry_count = sum(1 for node in nodes if node == "retry")
    interrupt_count = sum(1 for node in nodes if node == "approval")
    critical_failure = any(
        event.get("node") in {"classify", "answer"} and event.get("event_type") == "failed"
        for event in events
    )
    success = (
        actual_route == expected_route
        and bool(state.get("final_answer") or state.get("pending_question"))
        and not critical_failure
    )
    if approval_required:
        success = success and approval is not None

    return ScenarioMetric(
        scenario_id=str(state.get("scenario_id", "unknown")),
        success=success,
        expected_route=expected_route,
        actual_route=actual_route,
        nodes_visited=len(nodes),
        retry_count=retry_count,
        interrupt_count=interrupt_count,
        approval_required=approval_required,
        approval_observed=approval is not None,
        latency_ms=max(0, latency_ms),
        errors=list(errors),
    )


def summarize_metrics(items: list[ScenarioMetric]) -> MetricsReport:
    """Aggregate per-scenario metrics without overstating resume evidence."""
    if not items:
        raise ValueError("No scenario metrics to summarize")
    return MetricsReport(
        total_scenarios=len(items),
        success_rate=sum(1 for item in items if item.success) / len(items),
        avg_nodes_visited=mean(item.nodes_visited for item in items),
        total_retries=sum(item.retry_count for item in items),
        total_interrupts=sum(item.interrupt_count for item in items),
        resume_success=False,
        scenario_metrics=items,
    )


def write_metrics(report: MetricsReport, output_path: str | Path) -> None:
    """Write metrics JSON using the grading schema."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
