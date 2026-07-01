"""Reconstruct the trajectory an agent ACTUALLY took, from capture-SDK spans.

This is the other half of the bridge: `Recording.from_sdk_trace` gives you a
replayable recording, and `captured_trajectory` gives you the original decision
trajectory. Diff the two to answer "would a different model have decided
differently than my agent actually did?".

Decisions are the ordered `tool_call` spans (tool name + arguments) followed by a
terminal `final` step carrying the last model answer. Because both the captured
spans and a replay run pass through the same deterministic redaction, the two
trajectories align and the diff attributes differences to the model.
"""

from __future__ import annotations

from tracekit.model.spans import LLM_CALL
from tracekit.model.spans import TOOL_CALL as SPAN_TOOL_CALL
from tracekit.replay.trajectory import FINAL, MODEL_TEXT, TOOL_CALL, Step, Trajectory


def captured_trajectory(store, trace_id: str) -> Trajectory:
    spans = store.spans_for_trace(trace_id)  # ordered by ts_start
    model_id = None
    for s in spans:
        if s.kind == LLM_CALL:
            model_id = s.attributes.get("gen_ai.request.model")
            break

    traj = Trajectory(model_id=model_id)
    last_answer = ""
    for s in spans:
        if s.kind == SPAN_TOOL_CALL:
            traj.add(
                Step(kind=TOOL_CALL, tool_name=s.tool_name,
                     tool_input=(s.request or {}).get("arguments", {}) or {})
            )
        elif s.kind == LLM_CALL:
            text = ((s.response or {}).get("response") or {}).get("text", "")
            if isinstance(text, str) and text:
                last_answer = text
                traj.add(Step(kind=MODEL_TEXT, text=text))

    traj.final_text = last_answer
    traj.add(Step(kind=FINAL, text=last_answer))
    return traj
