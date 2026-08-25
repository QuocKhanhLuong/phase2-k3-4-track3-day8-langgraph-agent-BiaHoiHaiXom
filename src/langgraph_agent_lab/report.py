"""Deterministic Markdown report rendering from runtime metrics."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from .metrics import MetricsReport, ScenarioMetric


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _scenario_row(item: ScenarioMetric) -> str:
    return (
        f"| {_cell(item.scenario_id)} | {_cell(item.expected_route)} | "
        f"{_cell(item.actual_route or '-')} | {'yes' if item.success else 'no'} | "
        f"{item.retry_count} | {item.interrupt_count} | {item.latency_ms} |"
    )


def render_report(metrics: MetricsReport) -> str:
    """Render a complete evidence-oriented lab report."""
    student = os.getenv("LAB_STUDENT_NAME", "Lương Quốc Khánh")
    repo = os.getenv(
        "LAB_REPO",
        "QuocKhanhLuong/phase2-k3-4-track3-day8-langgraph-agent-BiaHoiHaiXom",
    )
    commit = os.getenv("GITHUB_SHA", "current checkout; record `git rev-parse HEAD` at submission")
    scenario_rows = "\n".join(_scenario_row(item) for item in metrics.scenario_metrics)

    return f"""# Day 08 Lab Report

## 1. Team / student

- Name: {student}
- Repo/commit: `{repo}` / {commit}
- Date: {date.today().isoformat()}
- Secrets: API keys are read from local environment only and are not written to metrics/report.

## 2. Architecture

The workflow is an 11-node LangGraph StateGraph:

`START → intake → classify → conditional route`

- `simple → answer → finalize → END`
- `tool → tool → evaluate → answer|retry`
- `missing_info → clarify → finalize → END`
- `risky → risky_action → approval → tool|clarify`
- `error → retry → tool|dead_letter`

There are four pure conditional routing functions after `classify`, `evaluate`, `retry`,
and `approval`. Every terminal path reaches `finalize`. The retry node owns the counter:
it increments `attempt` exactly once, and routing sends `attempt >= max_attempts` to
`dead_letter`.

## 3. State schema

| Field | Update mode | Why |
|---|---|---|
| `thread_id` | overwrite/ stable | checkpoint execution identity |
| `scenario_id` | overwrite/ stable | metrics identity only, never routing |
| `query` | overwrite | normalized ticket |
| `route` | overwrite | classifier decision |
| `risk_level` | overwrite | audit context |
| `attempt` | overwrite | bounded retry counter |
| `max_attempts` | overwrite/ stable | retry limit |
| `evaluation_result` | overwrite | evaluate → retry/answer gate |
| `pending_question` | overwrite | clarification output |
| `proposed_action` | overwrite | risky action awaiting review |
| `approval` | overwrite | serializable approval decision |
| `final_answer` | overwrite | terminal user-facing output |
| `messages` | append (`operator.add`) | conversation/audit trace |
| `tool_results` | append (`operator.add`) | ordered tool history |
| `errors` | append (`operator.add`) | ordered failure/retry history |
| `events` | append (`operator.add`) | normalized node audit trail |

Nodes return partial updates and never mutate append-only input lists in place.

## 4. Scenario results

### Summary

| Metric | Value |
|---|---:|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.2%} |
| Average events/nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| Approval-node visits | {metrics.total_interrupts} |
| Verified resume/replay | {'yes' if metrics.resume_success else 'no'} |

`total_interrupts` follows the scaffold convention and counts visits to the approval node;
it is not claimed as a real pause/resume interrupt unless the optional interrupt mode is used.

| Scenario | Expected route | Actual route | Success | Retries | Approval visits | Latency ms |
|---|---|---|---:|---:|---:|---:|
{scenario_rows}

## 5. Failure analysis

### Failure mode 1 — transient tool failure / exhausted retry budget

An `error` ticket enters `retry` first. The retry node increments `attempt`, then
`route_after_retry` either permits the tool (`attempt < max_attempts`) or fails closed to
`dead_letter`. Tool results containing `ERROR` are evaluated as `needs_retry`, which loops
back through the same bounded counter. Detection evidence is the ordered `tool`, `evaluate`,
`retry`, and optionally `dead_letter` events plus the append-only `errors` history.
Termination is guaranteed because only the retry node increments the counter and
`attempt >= max_attempts` cannot return to the tool.

Residual risk: the core evaluator is deterministic rather than semantic. A production
version should use a structured LLM judge with timeout, fallback, and cost controls.

### Failure mode 2 — consequential action executed without approval

A `risky` ticket must follow `risky_action → approval`. Only an explicit
`approval.approved == true` routes to `tool`; rejection routes to `clarify`.
The tool also checks approval defensively and returns an error instead of executing when
approval is absent/rejected. Audit evidence is the event ordering: `approval` must appear
before `tool`; on rejection, no tool event should appear after the approval event.

Residual risk: mock approval is the CI default. Real human pause/resume requires
`LANGGRAPH_INTERRUPT=true` and a reviewer resume command.

## 6. Persistence / recovery evidence

The compiled graph receives the checkpointer created by configuration, and every invocation
uses `configurable.thread_id`. The default lab config uses `MemorySaver`, which demonstrates
same-process checkpoint/state-history behavior but is not durable across process restart.

`run-scenarios` also writes `outputs/persistence.json` (next to the metrics output) by reading
`graph.get_state_history()` for every thread. This gives concrete checkpoint-count and
finalize evidence. `resume_success` remains `false` because this core submission does not
claim a verified crash-resume/replay.

## 7. Extension work

- SQLite checkpointer support is implemented behind the optional `sqlite` extra.
- Optional real HITL uses LangGraph `interrupt()` when `LANGGRAPH_INTERRUPT=true`.
- The core config remains memory + mock approval so CI does not depend on an interactive
  reviewer or external database.

These extensions do not change core routing, bounded retry, approval ordering, or
termination behavior.

## 8. Improvement plan

The next productionization priority is a structured LLM-as-judge evaluator with strict
timeout/fallback/cost guards, followed by a durable SQLite/Postgres recovery test that
proves process restart and resume using the same thread ID.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
