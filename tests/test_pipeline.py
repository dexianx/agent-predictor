import pytest
from inference.task_a import SimulateRequest, simulate_review


@pytest.mark.asyncio
async def test_full_pipeline_nl_message():
    events = []

    async def collect(ev):
        events.append(ev)

    req = SimulateRequest(
        user_id="NG_KONGA_2ae66f90",
        raw_message="Write a review for Sony WH-1000XM4 headphones",
    )
    result = await simulate_review(req, collect)

    types = [e["type"] for e in events]
    assert "step" in types and "token" in types and "final" in types
    assert "error" not in types, f"errors: {[e for e in events if e['type'] == 'error']}"
    assert result is not None
    assert len(result["review_text"]) > 10
    assert 1 <= result["predicted_rating"] <= 5
    step_ids = [e.get("id") for e in events if e["type"] == "step"]
    assert "nl_extract" not in step_ids
    assert "retrieval" in step_ids


@pytest.mark.asyncio
async def test_pipeline_emits_graph_data():
    events = []

    async def collect(ev):
        events.append(ev)

    req = SimulateRequest(user_id="NG_KONGA_2ae66f90", raw_message="Review the iPhone 15 Pro")
    await simulate_review(req, collect)
    graph_events = [e for e in events if e["type"] == "graph_data"]
    assert graph_events
    assert "nodes" in graph_events[0] and "edges" in graph_events[0]
