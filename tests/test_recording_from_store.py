"""Recording.from_trace_store: a captured proxy trace becomes a replayable recording."""

from agentsense.model.spans import Span
from agentsense.replay import ModelResponse, Recording, ScriptedAdapter, ToolCall, replay
from agentsense.store.sqlite import SpanStore


def _write_tool_call(store: SpanStore, trace_id: str, name: str, args: dict, result: dict):
    span = Span(
        trace_id=trace_id,
        method="tools/call",
        tool_name=name,
        request={"params": {"name": name, "arguments": args}},
        response={"result": result},
    )
    span.close()
    store.write(span)


def test_recording_reconstructed_from_captured_spans(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    _write_tool_call(store, "tr1", "get_weather", {"city": "Paris"}, {"temp_c": 18})
    _write_tool_call(store, "tr1", "search", {"q": "hotels"}, {"hits": 5})

    rec = Recording.from_trace_store(store, "tr1", question="Plan a Paris trip")
    found, result = rec.lookup("get_weather", {"city": "Paris"})
    assert found and result["temp_c"] == 18

    # And it drives a replay end-to-end using the store-sourced result.
    script = [
        ModelResponse(tool_calls=[ToolCall("t1", "get_weather", {"city": "Paris"})]),
        ModelResponse(text="18C in Paris"),
    ]
    traj = replay(rec, ScriptedAdapter(script))
    assert traj.final_text == "18C in Paris"
    assert traj.stopped_reason is None
    store.close()


def test_store_sourced_results_are_already_redacted(tmp_path):
    # The proxy redacts on write, so results pulled for replay carry tokens, not PII.
    store = SpanStore(tmp_path / "t.db")
    _write_tool_call(store, "tr2", "read_contact", {"id": 1},
                     {"email": "alice@example.com"})
    rec = Recording.from_trace_store(store, "tr2", question="who?")
    _, result = rec.lookup("read_contact", {"id": 1})
    assert "@" not in result["email"] and result["email"].startswith("<EMAIL_")
    store.close()
