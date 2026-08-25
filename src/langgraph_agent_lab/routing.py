"""Pure routing functions for LangGraph conditional edges."""

from __future__ import annotations

from .state import AgentState


def route_after_classify(state: AgentState) -> str:
    """Map the classified route to the next registered graph node."""
    mapping = {
        "simple": "answer",
        "tool": "tool",
        "missing_info": "clarify",
        "risky": "risky_action",
        "error": "retry",
    }
    return mapping.get(state.get("route", ""), "answer")


def route_after_evaluate(state: AgentState) -> str:
    """Retry only when the evaluator explicitly requests it."""
    return "retry" if state.get("evaluation_result") == "needs_retry" else "answer"


def route_after_retry(state: AgentState) -> str:
    """Keep retrying only while the bounded attempt budget remains."""
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 0))
    return "tool" if attempt < max_attempts else "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Proceed with a risky tool only after an explicit approval."""
    approval = state.get("approval") or {}
    return "tool" if approval.get("approved") is True else "clarify"
