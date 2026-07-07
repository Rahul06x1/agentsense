"""Trajectory diff: same recorded results, two models, find first divergence."""

from agentsense.replay import (
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


# ---- A1: honest diff output (kind + comparable_until) -------------------------

from agentsense.replay.diff import ALIGNED, DIVERGED, UNRESOLVABLE_FORK  # noqa: E402
from agentsense.replay.trajectory import (  # noqa: E402
    FINAL,
    TOOL_CALL,
    TOOL_RESULT,
    Step,
    Trajectory,
)


def _traj(model, decisions, stopped_reason=None):
    t = Trajectory(model_id=model, stopped_reason=stopped_reason)
    for kind, name, inp, result in decisions:
        if kind == TOOL_CALL:
            t.add(Step(kind=TOOL_CALL, tool_name=name, tool_input=inp))
            t.add(Step(kind=TOOL_RESULT, tool_name=name, result=result,
                       missing=result is None))
        else:
            t.add(Step(kind=FINAL, text="done"))
    return t


def test_aligned_kind():
    a = _traj("m1", [(TOOL_CALL, "w", {"c": "P"}, {"t": 1}), (FINAL, None, None, None)])
    b = _traj("m2", [(TOOL_CALL, "w", {"c": "P"}, {"t": 1}), (FINAL, None, None, None)])
    d = diff_trajectories(a, b)
    assert d.kind == ALIGNED and d.comparable_until == 2 and not d.redaction_suspect


def test_clean_divergence_kind():
    a = _traj("m1", [(TOOL_CALL, "get_weather", {"c": "P"}, {"t": 1})])
    b = _traj("m2", [(TOOL_CALL, "get_forecast", {"c": "P"}, {"d": 3})])
    d = diff_trajectories(a, b)
    assert d.kind == DIVERGED and d.first_divergence == 0 and d.comparable_until == 0


def test_unresolvable_fork_kind():
    # b requested an unrecorded tool and stopped — downstream is uncomparable.
    a = _traj("m1", [(TOOL_CALL, "get_weather", {"c": "Paris"}, {"t": 1}),
                     (FINAL, None, None, None)])
    b = _traj("m2", [(TOOL_CALL, "get_weather", {"c": "Berlin"}, None)],
              stopped_reason="unrecorded_tool_call")
    d = diff_trajectories(a, b)
    assert d.kind == UNRESOLVABLE_FORK
    assert d.first_divergence == 0
    assert "unresolvable fork" in d.summary


def test_redaction_suspect_flag():
    # A redacted token in the inputs before the divergence -> flagged suspect.
    a = _traj("m1", [(TOOL_CALL, "send", {"to": "<EMAIL_1a2b3c4d>"}, {"ok": 1}),
                     (TOOL_CALL, "get_weather", {"c": "P"}, {"t": 1})])
    b = _traj("m2", [(TOOL_CALL, "send", {"to": "<EMAIL_1a2b3c4d>"}, {"ok": 1}),
                     (TOOL_CALL, "get_forecast", {"c": "P"}, {"d": 3})])
    d = diff_trajectories(a, b)
    assert d.kind == DIVERGED and d.first_divergence == 1
    assert d.redaction_suspect is True
    assert "redaction" in d.summary


def test_no_redaction_no_suspect():
    a = _traj("m1", [(TOOL_CALL, "get_weather", {"c": "P"}, {"t": 1})])
    b = _traj("m2", [(TOOL_CALL, "get_forecast", {"c": "P"}, {"d": 3})])
    assert diff_trajectories(a, b).redaction_suspect is False
