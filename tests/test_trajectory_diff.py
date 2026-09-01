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

from agentsense.replay.diff import (  # noqa: E402
    ALIGNED,
    DIVERGED,
    UNKNOWN_TERMINAL,
    UNRESOLVABLE_FORK,
)
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


# ---- Unknown endings: never claim a run finished when we didn't see it ---------


def test_missing_terminal_is_not_reported_as_aligned():
    """Two runs matching on every observed decision, but neither ending captured.

    Calling this "identical trajectory" claims more than the trace supports —
    either run could have gone on to do something different.
    """
    a = _traj("m", [(TOOL_CALL, "search", {"q": "x"}, {"hits": 1})])
    b = _traj("m", [(TOOL_CALL, "search", {"q": "x"}, {"hits": 1})])
    d = diff_trajectories(a, b)
    assert d.kind == UNKNOWN_TERMINAL
    assert d.aligned is False
    assert d.first_divergence is None  # matched everywhere observed
    assert d.comparable_until == 1


def test_run_that_stops_early_is_not_credited_with_answering():
    """A truncated run must not be described as having given a final answer."""
    a = _traj("short", [(TOOL_CALL, "search", {"q": "x"}, {"hits": 1})])
    b = _traj("long", [(TOOL_CALL, "search", {"q": "x"}, {"hits": 1}),
                       (TOOL_CALL, "get_weather", {"city": "Paris"}, {"t": 18}),
                       (FINAL, None, None, None)])
    d = diff_trajectories(a, b)
    assert d.kind == UNKNOWN_TERMINAL
    assert d.comparable_until == 1
    assert "final answer" not in d.summary  # the invention this guards against
    assert "unknown" in d.summary


def test_interrupted_run_does_not_align_with_a_completed_one():
    """The worst case: one answered, one was cut off, previously reported ALIGNED."""
    cut = _traj("m", [(TOOL_CALL, "search", {"q": "x"}, {"hits": 1})])
    done = _traj("m", [(TOOL_CALL, "search", {"q": "x"}, {"hits": 1}),
                       (FINAL, None, None, None)])
    d = diff_trajectories(cut, done)
    assert d.aligned is False
    assert d.kind == UNKNOWN_TERMINAL


def test_both_runs_finished_still_align():
    """Guard against overcorrecting: real terminals on both sides is still aligned."""
    a = _traj("m", [(TOOL_CALL, "search", {"q": "x"}, {"hits": 1}),
                    (FINAL, None, None, None)])
    b = _traj("m", [(TOOL_CALL, "search", {"q": "x"}, {"hits": 1}),
                    (FINAL, None, None, None)])
    d = diff_trajectories(a, b)
    assert d.aligned is True
    assert d.kind == ALIGNED


def test_real_divergence_still_outranks_a_missing_terminal():
    """An actual conflicting decision is a divergence, not an unknown ending."""
    a = _traj("m", [(TOOL_CALL, "search", {"q": "x"}, {"hits": 1})])
    b = _traj("m", [(TOOL_CALL, "get_weather", {"city": "Paris"}, {"t": 18})])
    d = diff_trajectories(a, b)
    assert d.kind == DIVERGED
    assert d.first_divergence == 0


def test_unresolvable_fork_still_outranks_a_missing_terminal():
    """A fork is the more specific explanation; it must not be masked."""
    a = _traj("m", [(TOOL_CALL, "search", {"q": "x"}, {"hits": 1})],
              stopped_reason="unrecorded_tool_call")
    b = _traj("m", [(TOOL_CALL, "get_weather", {"city": "Paris"}, {"t": 18})])
    d = diff_trajectories(a, b)
    assert d.kind == UNRESOLVABLE_FORK


def test_no_decisions_at_all_says_nothing_to_compare():
    d = diff_trajectories(_traj("m", []), _traj("m", []))
    assert d.kind == UNKNOWN_TERMINAL
    assert "nothing to compare" in d.summary


def test_unknown_terminal_reports_no_first_divergence():
    """kind and first_divergence must not contradict each other.

    The UI paints the `first_divergence` row red; setting it alongside a
    non-divergent kind showed a false conflict under an "unknown" banner.
    """
    a = _traj("short", [(TOOL_CALL, "search", {"q": "x"}, {"hits": 1})])
    b = _traj("long", [(TOOL_CALL, "search", {"q": "x"}, {"hits": 1}),
                       (TOOL_CALL, "get_weather", {"city": "Paris"}, {"t": 18}),
                       (FINAL, None, None, None)])
    d = diff_trajectories(a, b)
    assert d.kind == UNKNOWN_TERMINAL
    assert d.first_divergence is None
    assert d.comparable_until == 1  # this is what bounds the claim


def test_unknown_terminal_does_not_claim_redaction_caused_it():
    """No conflicting decision means nothing for redaction to be blamed for."""
    a = _traj("short", [(TOOL_CALL, "send", {"to": "<EMAIL_1a2b3c4d>"}, {"ok": 1})])
    b = _traj("long", [(TOOL_CALL, "send", {"to": "<EMAIL_1a2b3c4d>"}, {"ok": 1}),
                       (TOOL_CALL, "search", {"q": "x"}, {"hits": 1}),
                       (FINAL, None, None, None)])
    d = diff_trajectories(a, b)
    assert d.kind == UNKNOWN_TERMINAL
    assert d.redaction_suspect is False
    assert "redaction" not in d.summary
