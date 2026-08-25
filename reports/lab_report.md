# Day 08 Lab Report

## 1. Team / student

- Name: Lương Quốc Khánh
- Repo/commit: `QuocKhanhLuong/phase2-k3-4-track3-day8-langgraph-agent-BiaHoiHaiXom` / current checkout; record `git rev-parse HEAD` at submission
- Date: 2026-08-25
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
| Total scenarios | 7 |
| Success rate | 100.00% |
| Average events/nodes visited | 6.14 |
| Total retries | 3 |
| Real HITL interrupts | 2 |
| Verified HITL resume | yes |

Real LangGraph `interrupt()`/`Command(resume=...)` was observed for every approval-required scenario.

| Scenario | Expected route | Actual route | Success | Retries | Real interrupts | Latency ms |
|---|---|---|---:|---:|---:|---:|
| S01_simple | simple | simple | yes | 0 | 0 | 3276 |
| S02_tool | tool | tool | yes | 0 | 0 | 2230 |
| S03_missing | missing_info | missing_info | yes | 0 | 0 | 978 |
| S04_risky | risky | risky | yes | 0 | 1 | 28998 |
| S05_error | error | error | yes | 2 | 0 | 2317 |
| S06_delete | risky | risky | yes | 0 | 1 | 45426 |
| S07_dead_letter | error | error | yes | 1 | 0 | 1450 |

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

A `risky` ticket must follow `risky_action → approval`. With real HITL enabled, the
approval node calls LangGraph `interrupt()`, suspending execution for the current
`thread_id`. The CLI obtains a human decision and resumes the same thread with
`Command(resume=decision)`. Only `approval.approved == true` routes to `tool`; rejection
routes to `clarify`.

The tool also checks approval defensively and returns an error instead of executing when
approval is absent/rejected. Audit evidence is the event ordering: `approval` must appear
before `tool`; on rejection, no tool event should appear after the approval event.

## 6. Persistence / recovery evidence

The compiled graph receives the checkpointer created by configuration, and every invocation
uses `configurable.thread_id`. The default lab config uses `MemorySaver`, which is sufficient
to pause and resume a real HITL interrupt in the same process and to inspect state history.
It is not durable across process restart.

`run-scenarios` writes `outputs/persistence.json` by reading `graph.get_state_history()` for
every thread. For HITL scenarios it also records `real_interrupt_count` and
`resume_observed`. The report sets `resume_success=true` only when every approval-required
scenario both completes successfully and has a real observed interrupt/resume.

## 7. Extension work

- OpenAI is the selected LLM provider for structured classification and grounded answers.
- Real HITL is implemented with LangGraph `interrupt()` and `Command(resume=...)`.
- SQLite checkpointer support remains available behind the optional `sqlite` extra for
  durable restart/recovery evidence.
- CI can still run with real HITL disabled so it does not block on an interactive reviewer.

These extensions do not change core routing, bounded retry, approval ordering, or
termination behavior.

## 8. Improvement plan

The next productionization priority is a structured LLM-as-judge evaluator with strict
timeout/fallback/cost guards, followed by a durable SQLite/Postgres recovery test that
proves process restart and resume using the same thread ID.
