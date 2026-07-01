"""Replay engine: injects recorded results, never calls a live tool, completes.

Uses the scripted adapter so it runs offline (no Bedrock/SSO)."""

from tracekit.replay import (
    ModelResponse,
    Recording,
    ScriptedAdapter,
    ToolCall,
    ToolSpec,
    replay,
)
from tracekit.replay.trajectory import FINAL, TOOL_CALL, TOOL_RESULT

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Weather for a city",
    input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
)


def _recording() -> Recording:
    rec = Recording(question="Weather in Paris?", tools=[WEATHER_TOOL])
    rec.record("get_weather", {"city": "Paris"}, {"temp_c": 18, "sky": "sunny"})
    return rec


def test_replay_injects_recorded_result_and_completes():
    script = [
        ModelResponse(tool_calls=[ToolCall(id="t1", name="get_weather",
                                           input={"city": "Paris"})]),
        ModelResponse(text="It is 18C and sunny in Paris."),
    ]
    adapter = ScriptedAdapter(script, model_id="haiku")
    traj = replay(_recording(), adapter)

    kinds = [s.kind for s in traj.steps]
    assert TOOL_CALL in kinds and TOOL_RESULT in kinds
    assert traj.steps[-1].kind == FINAL
    assert "sunny" in traj.final_text
    assert traj.stopped_reason is None
    # The injected result reached the model on the second turn (as a tool result).
    second_call_turns = adapter.calls[1]
    assert any(t.tool_results for t in second_call_turns)


def test_engine_never_calls_a_live_tool():
    # The recording holds a canned result; the engine has no live-tool hook at all.
    # If it ever fetched live data, this sentinel result could not appear verbatim.
    rec = Recording(question="Weather in Paris?", tools=[WEATHER_TOOL])
    rec.record("get_weather", {"city": "Paris"}, {"sentinel": "recorded-only"})
    script = [
        ModelResponse(tool_calls=[ToolCall(id="t1", name="get_weather",
                                           input={"city": "Paris"})]),
        ModelResponse(text="done"),
    ]
    traj = replay(rec, ScriptedAdapter(script))
    tool_results = [s for s in traj.steps if s.kind == TOOL_RESULT]
    assert tool_results[0].result == {"sentinel": "recorded-only"}


def test_unrecorded_tool_call_stops_as_divergence():
    rec = _recording()  # only knows get_weather(Paris)
    script = [
        ModelResponse(tool_calls=[ToolCall(id="t1", name="get_weather",
                                           input={"city": "Berlin"})]),  # not recorded
    ]
    traj = replay(rec, ScriptedAdapter(script))
    assert traj.stopped_reason == "unrecorded_tool_call"
    assert traj.steps[-1].kind == TOOL_RESULT and traj.steps[-1].missing


def test_no_tool_path_returns_final_immediately():
    rec = Recording(question="Say hi", tools=[])
    traj = replay(rec, ScriptedAdapter([ModelResponse(text="hi")]))
    assert [s.kind for s in traj.decisions] == [FINAL]
    assert traj.final_text == "hi"


def test_usage_incl_cache_tokens_captured():
    from tracekit.replay.types import Usage

    script = [ModelResponse(text="ok", usage=Usage(input_tokens=10, output_tokens=3,
                                                   cache_read_input_tokens=4))]
    traj = replay(Recording(question="q"), ScriptedAdapter(script))
    assert traj.total_input_tokens == 10
    assert traj.usages[0].cache_read_input_tokens == 4
