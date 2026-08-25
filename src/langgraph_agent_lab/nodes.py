"""Node implementations for the support-ticket LangGraph workflow."""

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


class ClassificationDecision(BaseModel):
    """Validated output contract for the LLM classifier."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    risk_level: Literal["low", "medium", "high"] = "low"
    reason: str = Field(min_length=1)


def _content_to_text(content: object) -> str:
    """Normalize common LangChain message content shapes into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def intake_node(state: AgentState) -> dict[str, Any]:
    """Normalize the incoming support-ticket query."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict[str, Any]:
    """Classify ticket intent with a real LLM structured-output call."""
    query = state.get("query", "")
    prompt = f"""
You are the routing classifier for a support-ticket workflow.
Return exactly one route using the structured schema.

Decision policy, in priority order:
1. risky: a request that would create a consequential side effect such as refunding,
   deleting, changing an account, sending an external confirmation, or another
   irreversible/privileged action that requires approval.
2. tool: a safe request that needs a lookup or tool/data access to answer.
3. missing_info: the user asks for an action/help but has not supplied enough
   information to identify the target or problem.
4. error: the ticket itself reports a transient/system failure, timeout, unavailable
   dependency, or processing failure that should enter the bounded-retry path.
5. simple: an informational support question that can be answered directly.

Do not infer a route from scenario IDs or hidden labels. Use only the ticket meaning.
Set risk_level=high for risky, medium for operational errors, and low otherwise.

Ticket:
{query}
""".strip()

    try:
        structured = get_llm().with_structured_output(ClassificationDecision)
        raw_decision = structured.invoke(prompt)
        if isinstance(raw_decision, ClassificationDecision):
            decision = raw_decision
        else:
            decision = ClassificationDecision.model_validate(raw_decision)
    except Exception as exc:
        error = f"classification failed: {type(exc).__name__}: {exc}"
        return {
            "route": "simple",
            "risk_level": "unknown",
            "errors": [error],
            "events": [
                make_event(
                    "classify",
                    "failed",
                    "structured LLM classification failed; fail-safe route selected",
                    error_type=type(exc).__name__,
                )
            ],
        }

    risk_level = "high" if decision.route == "risky" else decision.risk_level
    return {
        "route": decision.route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                "structured LLM route selected",
                route=decision.route,
                risk_level=risk_level,
                reason=decision.reason[:160],
            )
        ],
    }


def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute a deterministic mock tool and simulate transient error-route failures."""
    route = state.get("route", "")
    attempt = int(state.get("attempt", 0))
    query = state.get("query", "")

    if route == "risky":
        approval = state.get("approval") or {}
        if approval.get("approved") is not True:
            result = "ERROR: risky action blocked because approval is missing or rejected"
            return {
                "tool_results": [result],
                "errors": [result],
                "events": [
                    make_event(
                        "tool",
                        "failed",
                        "risky tool blocked before side effect",
                        attempt=attempt,
                    )
                ],
            }

    if route == "error" and attempt < 2:
        result = f"ERROR: transient tool failure on attempt {attempt}"
        return {
            "tool_results": [result],
            "events": [
                make_event(
                    "tool",
                    "failed",
                    "simulated transient failure",
                    attempt=attempt,
                )
            ],
        }

    result = f"OK: support tool completed for '{query[:80]}'"
    return {
        "tool_results": [result],
        "events": [
            make_event(
                "tool",
                "completed",
                "mock tool completed",
                attempt=attempt,
            )
        ],
    }


def evaluate_node(state: AgentState) -> dict[str, Any]:
    """Evaluate the latest tool result and gate the bounded retry loop."""
    results = state.get("tool_results", [])
    latest = results[-1] if results else ""
    verdict = "needs_retry" if not latest or "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": verdict,
        "events": [
            make_event(
                "evaluate",
                "completed",
                "tool result evaluated",
                verdict=verdict,
            )
        ],
    }


def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate a grounded final support response with a real LLM call."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    context = "\n".join(tool_results[-3:]) if tool_results else "(no tool result)"
    prompt = f"""
You are a concise support agent. Answer the user's ticket using only the supplied
ticket and workflow context. Do not invent tool results, approval decisions, IDs,
or actions that are not present. If a tool result exists, treat the latest result
as authoritative. If an approval exists, accurately reflect whether it was approved.

Ticket:
{query}

Tool context:
{context}

Approval:
{approval if approval is not None else "(not required)"}

Write the final helpful response in plain text.
""".strip()

    try:
        response = get_llm(temperature=0.2).invoke(prompt)
        text = _content_to_text(getattr(response, "content", response))
        if not text:
            raise ValueError("LLM returned empty content")
    except Exception as exc:
        error = f"answer generation failed: {type(exc).__name__}: {exc}"
        return {
            "final_answer": "Unable to generate the final support response because the LLM call failed.",
            "errors": [error],
            "events": [
                make_event(
                    "answer",
                    "failed",
                    "grounded LLM answer generation failed",
                    error_type=type(exc).__name__,
                )
            ],
        }

    return {
        "final_answer": text,
        "messages": [f"answer:{text[:80]}"],
        "events": [make_event("answer", "completed", "grounded LLM answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict[str, Any]:
    """Ask for missing information or a safer alternative after rejection."""
    approval = state.get("approval")
    if approval is not None and approval.get("approved") is False:
        question = (
            "The proposed risky action was not approved. What safer alternative "
            "would you like me to take?"
        )
    else:
        question = (
            "Please provide the specific account, order, resource, or error details "
            "needed to identify what should be fixed."
        )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict[str, Any]:
    """Prepare, but do not execute, a consequential action for approval."""
    query = state.get("query", "")
    proposed = f"Execute the requested consequential support action: {query}"
    return {
        "proposed_action": proposed,
        "events": [
            make_event(
                "risky_action",
                "prepared",
                "risky action prepared for review; no side effect executed",
            )
        ],
    }


def approval_node(state: AgentState) -> dict[str, Any]:
    """Apply mock approval by default, with optional LangGraph interrupt/resume."""
    proposed_action = state.get("proposed_action") or state.get("query", "")

    if os.getenv("LANGGRAPH_INTERRUPT", "").strip().lower() in {"1", "true", "yes"}:
        from langgraph.types import interrupt

        resumed = interrupt(
            {
                "type": "approval_required",
                "proposed_action": proposed_action,
                "instruction": "Resume with {'approved': bool, 'reviewer': str, 'comment': str}.",
            }
        )
        if isinstance(resumed, dict):
            decision = ApprovalDecision.model_validate(resumed)
        else:
            decision = ApprovalDecision(
                approved=bool(resumed),
                reviewer="human-reviewer",
                comment="decision supplied through interrupt resume",
            )
        source = "human_interrupt"
    else:
        mock_value = os.getenv("MOCK_APPROVAL", "true").strip().lower()
        approved = mock_value not in {"0", "false", "no", "reject", "rejected"}
        decision = ApprovalDecision(
            approved=approved,
            reviewer="mock-reviewer",
            comment="mock approval for offline tests/CI",
        )
        source = "mock"

    event_type = "approved" if decision.approved else "rejected"
    return {
        "approval": decision.model_dump(),
        "events": [
            make_event(
                "approval",
                event_type,
                "approval decision recorded before tool execution",
                source=source,
                reviewer=decision.reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict[str, Any]:
    """Increment the bounded retry counter exactly once."""
    next_attempt = int(state.get("attempt", 0)) + 1
    note = f"retry attempt {next_attempt} scheduled"
    return {
        "attempt": next_attempt,
        "errors": [note],
        "events": [
            make_event(
                "retry",
                "scheduled",
                "retry counter incremented",
                attempt=next_attempt,
                max_attempts=int(state.get("max_attempts", 0)),
            )
        ],
    }


def dead_letter_node(state: AgentState) -> dict[str, Any]:
    """Terminate safely after the retry budget is exhausted."""
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 0))
    answer = (
        "The request could not be completed after the allowed retry attempts. "
        "It has been escalated for manual review."
    )
    return {
        "final_answer": answer,
        "events": [
            make_event(
                "dead_letter",
                "completed",
                "retry budget exhausted; request escalated",
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Emit the terminal audit event shared by every route."""
    _ = state
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
