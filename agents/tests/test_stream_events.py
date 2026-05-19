from dataclasses import dataclass

from data_junior_agent.agent import _format_mcp_tool_error, _summarize_stream_event


@dataclass
class FakeData:
    type: str
    delta: str


@dataclass
class FakeEvent:
    type: str
    data: FakeData | None = None
    name: str = ""
    item: object | None = None


def test_summarize_stream_event_allows_only_output_text_delta():
    event = FakeEvent("raw_response_event", FakeData("response.output_text.delta", "hello"))
    summary = _summarize_stream_event(event)
    assert summary is not None
    assert summary.kind == "delta"
    assert summary.message == "hello"


def test_summarize_stream_event_suppresses_reasoning_delta():
    event = FakeEvent("raw_response_event", FakeData("response.reasoning_text.delta", "private"))
    summary = _summarize_stream_event(event)
    assert summary is not None
    assert summary.kind == "reasoning_delta"
    assert summary.message == "private"


def test_summarize_stream_event_suppresses_reasoning_item():
    event = FakeEvent("run_item_stream_event", name="reasoning_item_created")
    summary = _summarize_stream_event(event)
    assert summary is not None
    assert summary.kind == "summary"


def test_format_mcp_tool_error_handles_timeout():
    message = _format_mcp_tool_error(None, RuntimeError("Timed out while waiting for response"))
    assert "Do not retry" in message
