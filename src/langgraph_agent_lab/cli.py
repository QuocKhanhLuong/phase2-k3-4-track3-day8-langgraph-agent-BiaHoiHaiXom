"""CLI for running scenarios, metrics validation, and persistence evidence."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


def _history_evidence(graph: Any, run_config: dict[str, Any], thread_id: str) -> dict[str, Any]:
    """Collect same-thread state-history evidence."""
    try:
        history = list(graph.get_state_history(run_config))
        return {
            "thread_id": thread_id,
            "checkpoint_count": len(history),
            "history_available": bool(history),
        }
    except Exception as exc:
        return {
            "thread_id": thread_id,
            "checkpoint_count": 0,
            "history_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _interrupt_items(result: dict[str, Any]) -> list[Any]:
    """Return LangGraph interrupt objects from an invoke result."""
    raw = result.get("__interrupt__", [])
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def _invoke_with_human_resume(
    graph: Any,
    state: dict[str, Any],
    run_config: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Invoke a graph and interactively resume every real LangGraph interrupt."""
    from langgraph.types import Command

    result = graph.invoke(state, config=run_config)
    interrupt_count = 0

    while True:
        interrupts = _interrupt_items(result)
        if not interrupts:
            return result, interrupt_count

        interrupt_count += len(interrupts)
        interrupt_obj = interrupts[0]
        payload = getattr(interrupt_obj, "value", interrupt_obj)

        typer.echo("\n--- LangGraph HITL approval required ---")
        if isinstance(payload, dict):
            proposed = payload.get("proposed_action")
            if proposed:
                typer.echo(f"Proposed action: {proposed}")
        else:
            typer.echo(f"Interrupt payload: {payload}")

        approved = typer.confirm("Approve this action?", default=False)
        reviewer = typer.prompt("Reviewer", default="human-reviewer")
        default_comment = "approved by human reviewer" if approved else "rejected by human reviewer"
        comment = typer.prompt("Comment", default=default_comment)

        decision = {
            "approved": approved,
            "reviewer": reviewer,
            "comment": comment,
        }
        result = graph.invoke(Command(resume=decision), config=run_config)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all sample scenarios and write metrics/report/evidence files."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)

    metrics = []
    persistence_evidence: list[dict[str, Any]] = []

    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}

        started = perf_counter()
        final_state, real_interrupts = _invoke_with_human_resume(graph, state, run_config)
        latency_ms = int((perf_counter() - started) * 1000)

        metrics.append(
            metric_from_state(
                final_state,
                scenario.expected_route.value,
                scenario.requires_approval,
                latency_ms=latency_ms,
                observed_interrupts=real_interrupts,
            )
        )
        evidence = _history_evidence(graph, run_config, state["thread_id"])
        evidence["scenario_id"] = scenario.id
        evidence["real_interrupt_count"] = real_interrupts
        evidence["resume_observed"] = real_interrupts > 0
        evidence["finalize_observed"] = any(
            event.get("node") == "finalize" for event in final_state.get("events", [])
        )
        persistence_evidence.append(evidence)

    report = summarize_metrics(metrics)
    approval_metrics = [item for item in metrics if item.approval_required]
    report.resume_success = bool(
        approval_metrics
        and all(item.success and item.interrupt_count > 0 for item in approval_metrics)
    )
    write_metrics(report, output)

    evidence_path = output.with_name("persistence.json")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(persistence_evidence, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])

    typer.echo(f"Wrote metrics to {output}")
    typer.echo(f"Wrote persistence evidence to {evidence_path}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for local grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


if __name__ == "__main__":
    app()
