"""Mocked replay engine.

Re-drives the agent loop against a (possibly different) model, INJECTING recorded
tool results instead of calling live tools. The engine has no ability to call a
real tool — the "0 live tool calls" guarantee holds by construction, not by
convention.

Redaction-aware: recorded results from the proxy store are already redacted, and
the engine applies the same deterministic redaction to model output and tool
inputs it records. Because redaction is deterministic, two replays (or a replay
vs the original) stay aligned — differences are attributable to the model, not to
redaction noise.
"""

from __future__ import annotations

from typing import Any

from tracekit.redaction.redactor import redact_object, redact_text
from tracekit.replay.adapters.base import ModelAdapter
from tracekit.replay.recording import Recording
from tracekit.replay.trajectory import (
    FINAL,
    MODEL_TEXT,
    TOOL_CALL,
    TOOL_RESULT,
    Step,
    Trajectory,
)
from tracekit.replay.types import ToolResult, Turn


def replay(
    recording: Recording,
    adapter: ModelAdapter,
    max_steps: int = 12,
    redact: bool = True,
) -> Trajectory:
    """Replay `recording` against `adapter`. Returns the resulting Trajectory."""
    turns: list[Turn] = [Turn(role="user", text=recording.question)]
    traj = Trajectory(model_id=adapter.model_id)

    for _ in range(max_steps):
        resp = adapter.converse(recording.system, turns, recording.tools)
        if resp.usage is not None:
            traj.usages.append(resp.usage)

        if resp.text:
            traj.add(Step(kind=MODEL_TEXT, text=_r_text(resp.text, redact)))

        if not resp.tool_calls:
            traj.final_text = _r_text(resp.text, redact)
            traj.add(Step(kind=FINAL, text=traj.final_text))
            return traj

        # Assistant turn that requested tools (fed back to the model next round).
        turns.append(Turn(role="assistant", text=resp.text, tool_calls=resp.tool_calls))

        result_turn: list[ToolResult] = []
        for tc in resp.tool_calls:
            traj.add(
                Step(kind=TOOL_CALL, tool_name=tc.name,
                     tool_input=_r_obj(tc.input, redact))
            )
            found, result = recording.lookup(tc.name, tc.input)
            if not found:
                # Model went somewhere the recording doesn't cover — a divergence.
                traj.add(Step(kind=TOOL_RESULT, tool_name=tc.name, missing=True))
                traj.stopped_reason = "unrecorded_tool_call"
                return traj
            traj.add(
                Step(kind=TOOL_RESULT, tool_name=tc.name, result=_r_obj(result, redact))
            )
            result_turn.append(ToolResult(id=tc.id, name=tc.name, result=result))

        turns.append(Turn(role="user", tool_results=result_turn))

    traj.stopped_reason = "max_steps"
    return traj


def _r_text(text: str, redact: bool) -> str:
    return redact_text(text)[0] if redact else text


def _r_obj(obj: Any, redact: bool) -> Any:
    return redact_object(obj)[0] if redact else obj
