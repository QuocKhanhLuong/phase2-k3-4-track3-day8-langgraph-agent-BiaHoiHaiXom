"""Routing function tests.

These tests verify correct routing logic. They will fail with NotImplementedError
until you implement the routing functions in routing.py.
"""

from langgraph_agent_lab.routing import (
    route_after_approval,
    route_after_classify,
    route_after_evaluate,
    route_after_retry,
)
from langgraph_agent_lab.state import Route


def test_route_after_classify_simple():
    assert route_after_classify({"route": Route.SIMPLE.value}) == "answer"


def test_route_after_classify_tool():
    assert route_after_classify({"route": Route.TOOL.value}) == "tool"


def test_route_after_classify_risky():
    assert route_after_classify({"route": Route.RISKY.value}) == "risky_action"


def test_route_after_classify_missing():
    assert route_after_classify({"route": Route.MISSING_INFO.value}) == "clarify"


def test_route_after_classify_error():
    assert route_after_classify({"route": Route.ERROR.value}) == "retry"


def test_route_after_classify_unknown_defaults():
    assert route_after_classify({"route": "unknown_route"}) == "answer"


def test_route_after_approval_approved():
    assert route_after_approval({"approval": {"approved": True}}) == "tool"


def test_route_after_approval_rejected():
    assert route_after_approval({"approval": {"approved": False}}) == "clarify"


def test_route_after_retry_within_limit():
    assert route_after_retry({"attempt": 0, "max_attempts": 3}) == "tool"
    assert route_after_retry({"attempt": 1, "max_attempts": 3}) == "tool"
    assert route_after_retry({"attempt": 2, "max_attempts": 3}) == "tool"


def test_route_after_retry_at_limit():
    assert route_after_retry({"attempt": 3, "max_attempts": 3}) == "dead_letter"


def test_route_after_retry_over_limit():
    assert route_after_retry({"attempt": 5, "max_attempts": 3}) == "dead_letter"


def test_route_after_evaluate_success():
    assert route_after_evaluate({"evaluation_result": "success"}) == "answer"


def test_route_after_evaluate_retry():
    assert route_after_evaluate({"evaluation_result": "needs_retry"}) == "retry"
