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


# ---- How a run ENDED: read finish_reason, not the presence of text ------------

from agentsense.replay.diff import ALIGNED, UNKNOWN_TERMINAL  # noqa: E402
from agentsense.replay.trajectory import FINAL, TOOL_CALL  # noqa: E402


def _run(store, trace_id, steps):
    """Capture a run from a compact script of llm/tool steps."""
    with Tracer(store).session("agent", trace_id=trace_id) as s:
        for step in steps:
            if step[0] == "llm":
                _, text, finish = step
                s.llm_call("m", messages=[{"role": "user", "content": "weather?"}],
                           response={"text": text}, finish_reason=finish)
            else:
                s.tool_call("get_weather", args={"city": "Paris"}, result={"t": 18})


def test_mid_turn_narration_is_not_an_ending(tmp_path):
    """"Let me check the weather." before a tool call is narration, not an answer.

    Regression: any text at all used to produce a `final`, so a run cut off after
    a narrated tool call reported as an identical trajectory to one that finished
    — and the fabricated step carried mid-run text, placed after the tool call.
    """
    store = SpanStore(tmp_path / "t.db")
    _run(store, "cut", [("llm", "Let me check the weather.", "tool_use"), ("tool",)])
    _run(store, "done", [("llm", "", "tool_use"), ("tool",), ("llm", "18C.", "end_turn")])

    cut = captured_trajectory(store, "cut")
    assert [d.kind for d in cut.decisions] == [TOOL_CALL]  # no invented ending
    assert cut.final_text == ""

    d = diff_trajectories(cut, captured_trajectory(store, "done"))
    assert d.kind == UNKNOWN_TERMINAL
    assert d.aligned is False
    store.close()


def test_run_that_ends_without_saying_anything_still_ended(tmp_path):
    """finish_reason is the signal, so an empty answer is still a real ending."""
    store = SpanStore(tmp_path / "t.db")
    _run(store, "silent", [("llm", "", "tool_use"), ("tool",), ("llm", "", "end_turn")])
    _run(store, "spoken", [("llm", "", "tool_use"), ("tool",), ("llm", "18C.", "end_turn")])

    silent = captured_trajectory(store, "silent")
    assert [d.kind for d in silent.decisions] == [TOOL_CALL, FINAL]
    # Same tool decision, both genuinely ended -> aligned, despite different text.
    d = diff_trajectories(silent, captured_trajectory(store, "spoken"))
    assert d.kind == ALIGNED
    store.close()


def test_final_text_comes_from_the_terminal_call_not_earlier_chatter(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    _run(store, "t", [("llm", "Let me check.", "tool_use"), ("tool",),
                      ("llm", "It is 18C.", "end_turn")])

    traj = captured_trajectory(store, "t")
    assert traj.final_text == "It is 18C."
    assert traj.decisions[-1].text == "It is 18C."
    store.close()


def test_missing_finish_reason_leaves_the_ending_unknown(tmp_path):
    """Without a recorded reason we don't know; falling back to text is the bug."""
    store = SpanStore(tmp_path / "t.db")
    _run(store, "unknown", [("llm", "some text", None), ("tool",)])
    _run(store, "done", [("llm", "", "tool_use"), ("tool",), ("llm", "18C.", "end_turn")])

    assert [d.kind for d in captured_trajectory(store, "unknown").decisions] == [TOOL_CALL]
    d = diff_trajectories(captured_trajectory(store, "unknown"),
                          captured_trajectory(store, "done"))
    assert d.kind == UNKNOWN_TERMINAL
    store.close()


def test_stopping_for_a_non_tool_reason_counts_as_ended(tmp_path):
    """max_tokens/stop/length: the agent didn't choose to stop, but it did stop."""
    store = SpanStore(tmp_path / "t.db")
    for trace_id, reason in (("maxtok", "max_tokens"), ("stop", "stop"),
                             ("length", "length"), ("seq", "stop_sequence")):
        _run(store, trace_id, [("llm", "", "tool_use"), ("tool",),
                               ("llm", "trunc", reason)])
        traj = captured_trajectory(store, trace_id)
        assert [d.kind for d in traj.decisions] == [TOOL_CALL, FINAL], reason
    store.close()


def test_tool_calls_finish_reason_is_not_an_ending(tmp_path):
    """OpenAI spells the continuing case `tool_calls`, Anthropic `tool_use`."""
    store = SpanStore(tmp_path / "t.db")
    _run(store, "openai", [("llm", "", "tool_calls"), ("tool",)])
    assert [d.kind for d in captured_trajectory(store, "openai").decisions] == [TOOL_CALL]
    store.close()
