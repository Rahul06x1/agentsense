"""Capture -> replay bridge: an SDK-captured run becomes replayable, and the
captured trajectory can be diffed against a replay with a different model."""

from agentsense.replay import (
    ModelResponse,
    Recording,
    ScriptedAdapter,
    ToolCall,
    captured_trajectory,
    diff_trajectories,
    replay,
)
from agentsense.sdk import Tracer
from agentsense.store.sqlite import SpanStore

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "current weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _capture_a_run(store: SpanStore, trace_id: str):
    """Simulate an instrumented agent: it asked the weather, called the tool, answered."""
    with Tracer(store).session("weather-agent", trace_id=trace_id) as s:
        s.llm_call(
            "claude-haiku-4-5",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=[WEATHER_TOOL],
            response={"text": ""},
            finish_reason="tool_use",
        )
        s.tool_call("get_weather", args={"city": "Paris"}, result={"temp_c": 18})
        s.llm_call(
            "claude-haiku-4-5",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            response={"text": "It's 18C in Paris."},
            finish_reason="end_turn",
        )


def test_recording_reconstructed_from_sdk_trace(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    _capture_a_run(store, "run1")

    rec = Recording.from_sdk_trace(store, "run1")
    assert rec.question == "What's the weather in Paris?"
    assert rec.model_id == "claude-haiku-4-5"
    assert [t.name for t in rec.tools] == ["get_weather"]
    found, result = rec.lookup("get_weather", {"city": "Paris"})
    assert found and result == {"temp_c": 18}
    store.close()


def test_replay_the_captured_run_with_a_different_model(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    _capture_a_run(store, "run2")
    rec = Recording.from_sdk_trace(store, "run2")

    # A different model that reaches the same decision (calls get_weather, then answers).
    same = ScriptedAdapter(
        [
            ModelResponse(tool_calls=[ToolCall("t1", "get_weather", {"city": "Paris"})]),
            ModelResponse(text="18C, mild."),
        ],
        model_id="opus-4.8",
    )
    replayed = replay(rec, same)
    original = captured_trajectory(store, "run2")

    d = diff_trajectories(original, replayed)
    assert d.aligned  # same tool decision + both terminate
    store.close()


def test_captured_vs_replay_divergence_is_detected(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    _capture_a_run(store, "run3")
    rec = Recording.from_sdk_trace(store, "run3")

    # A model that answers without calling the tool -> diverges at decision 0.
    lazy = ScriptedAdapter([ModelResponse(text="Probably mild.")], model_id="haiku-lazy")
    replayed = replay(rec, lazy)
    original = captured_trajectory(store, "run3")

    d = diff_trajectories(original, replayed)
    assert not d.aligned
    assert d.first_divergence == 0
    assert d.a_step.tool_name == "get_weather"  # original called the tool
    store.close()


def test_bedrock_tool_shape_is_parsed(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    with Tracer(store).session("a", trace_id="run4") as s:
        s.llm_call(
            "claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"toolSpec": {"name": "search", "description": "d",
                                 "inputSchema": {"json": {"type": "object"}}}}],
            response={"text": "ok"},
        )
    rec = Recording.from_sdk_trace(store, "run4")
    assert [t.name for t in rec.tools] == ["search"]
    assert rec.tools[0].input_schema == {"type": "object"}
    store.close()
