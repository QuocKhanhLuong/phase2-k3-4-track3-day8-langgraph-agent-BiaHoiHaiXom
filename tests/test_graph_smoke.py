"""Graph smoke tests.

These tests verify end-to-end graph execution. They will fail with NotImplementedError
until you implement nodes, routing, and graph wiring.

Note: These tests require a configured LLM (OPENAI_API_KEY or ANTHROPIC_API_KEY)
because classify_node and answer_node use real LLM calls.
"""

import importlib.util
import os

import pytest

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("langgraph") is None,
        reason="langgraph not installed",
    ),
    pytest.mark.skipif(
        not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"),
        reason="No LLM API key configured (set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)",
    ),
]


@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("How do I reset my password?", Route.SIMPLE.value),
        ("Please lookup order status for order 123", Route.TOOL.value),
        ("Refund this customer", Route.RISKY.value),
        ("Can you fix it?", Route.MISSING_INFO.value),
        ("Timeout failure while processing", Route.ERROR.value),
    ],
)
def test_graph_runs_and_routes_correctly(query, expected_route):
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="smoke", query=query, expected_route=Route(expected_route))
    state = initial_state(scenario)
    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    assert result["route"] == expected_route
    assert result.get("final_answer") or result.get("pending_question")


def test_graph_terminates_all_routes():
    """Verify every route reaches finalize node."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    queries = [
        ("simple query about help", Route.SIMPLE),
        ("lookup order status 999", Route.TOOL),
        ("fix it", Route.MISSING_INFO),
        ("delete user account now", Route.RISKY),
        ("timeout error in system", Route.ERROR),
    ]
    for query, route in queries:
        scenario = Scenario(id=f"term-{route.value}", query=query, expected_route=route)
        state = initial_state(scenario)
        result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
        events = result.get("events", [])
        finalize_events = [e for e in events if e.get("node") == "finalize"]
        assert finalize_events, f"Route {route.value} did not reach finalize node"
