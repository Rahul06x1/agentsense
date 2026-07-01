"""Trajectory diff: same recorded results, two models, find first divergence."""

from tracekit.replay import (
    ModelResponse,
    Recording,
    ScriptedAdapter,
    ToolCall,
    ToolSpec,
    diff_trajectories,
    replay,
)

SEARCH = ToolSpec("search", "search", {"type": "object"})
WEATHER = ToolSpec("get_weather", "weather", {"type": "object"})
FORECAST = ToolSpec("get_forecast", "forecast", {"type": "object"})


def _recording() -> Recording:
    rec = Recording(question="Paris weather?", tools=[SEARCH, WEATHER, FORECAST])
    rec.record("search", {"q": "paris weather"}, {"hits": 3})
    rec.record("get_weather", {"city": "Paris"}, {"temp_c": 18})
    rec.record("get_forecast", {"city": "Paris", "days": 3}, {"days": 3})
    return rec


def test_identical_trajectories_align():
    script = [
        ModelResponse(tool_calls=[ToolCall("a", "search", {"q": "paris weather"})]),
        ModelResponse(tool_calls=[ToolCall("b", "get_weather", {"city": "Paris"})]),
        ModelResponse(text="18C"),
    ]
    a = replay(_recording(), ScriptedAdapter(list(script), model_id="haiku"))
    b = replay(_recording(), ScriptedAdapter(list(script), model_id="opus"))
    d = diff_trajectories(a, b)
    assert d.aligned and d.first_divergence is None


def test_divergence_at_second_decision():
    common_first = ModelResponse(tool_calls=[ToolCall("a", "search", {"q": "paris weather"})])
    a_script = [
        common_first,
        ModelResponse(tool_calls=[ToolCall("b", "get_weather", {"city": "Paris"})]),
        ModelResponse(text="done"),
    ]
    b_script = [
        common_first,
        # Opus picks a richer tool -> diverges here (decision index 1).
        ModelResponse(tool_calls=[ToolCall("b", "get_forecast",
                                           {"city": "Paris", "days": 3})]),
        ModelResponse(text="done"),
    ]
    a = replay(_recording(), ScriptedAdapter(a_script, model_id="haiku"))
    b = replay(_recording(), ScriptedAdapter(b_script, model_id="opus"))
    d = diff_trajectories(a, b)
    assert not d.aligned
    assert d.first_divergence == 1
    assert d.a_step.tool_name == "get_weather"
    assert d.b_step.tool_name == "get_forecast"
    assert "get_forecast" in d.summary


def test_early_stop_diverges_against_longer_run():
    a_script = [ModelResponse(text="answered directly")]  # stops immediately
    b_script = [
        ModelResponse(tool_calls=[ToolCall("a", "search", {"q": "paris weather"})]),
        ModelResponse(text="answered after searching"),
    ]
    a = replay(_recording(), ScriptedAdapter(a_script, model_id="haiku"))
    b = replay(_recording(), ScriptedAdapter(b_script, model_id="opus"))
    d = diff_trajectories(a, b)
    assert not d.aligned and d.first_divergence == 0
