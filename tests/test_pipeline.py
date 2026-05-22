import pytest
from task_a.evaluation.amazon_fixtures import EVAL_USER_ID, UNKNOWN_PRODUCT_CASES
from task_a.inference.task_a import simulate_review


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    UNKNOWN_PRODUCT_CASES,
    ids=[c.label for c in UNKNOWN_PRODUCT_CASES],
)
async def test_unknown_product_pipeline(case):
    events = []

    async def collect(ev):
        events.append(ev)

    result = await simulate_review(case.to_simulate_request(), collect)

    types = [e["type"] for e in events]
    assert "step" in types and "token" in types and "final" in types
    assert "error" not in types, f"errors: {[e for e in events if e['type'] == 'error']}"
    assert result is not None
    assert result["evaluation_context"]["item_resolved_in_kg"] is False
    assert len(result["review_text"]) > 10
    assert 1 <= result["predicted_rating"] <= 5
    step_ids = [e.get("id") for e in events if e["type"] == "step"]
    assert "retrieval" in step_ids


@pytest.mark.asyncio
async def test_unknown_product_emits_graph_data():
    case = UNKNOWN_PRODUCT_CASES[0]
    events = []

    async def collect(ev):
        events.append(ev)

    await simulate_review(case.to_simulate_request(), collect)
    graph_events = [e for e in events if e["type"] == "graph_data"]
    assert graph_events
    assert "nodes" in graph_events[0] and "edges" in graph_events[0]
