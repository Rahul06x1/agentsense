"""Reconstruct the trajectory an agent ACTUALLY took, from captured spans.

This is the other half of the bridge: `Recording.from_sdk_trace` gives you a
replayable recording, and `captured_trajectory` gives you the original decision
trajectory. Diff the two to answer "would a different model have decided
differently than my agent actually did?".

Decisions are the ordered tool calls (tool name + arguments) followed by a
terminal `final` step carrying the last model answer. Because both the captured
spans and a replay run pass through the same deterministic redaction, the two
trajectories align and the diff attributes differences to the model.

Both capture paths are understood, and they nest the same call differently:
the SDK emits `tool_call` spans with flat `request.arguments`, while the proxy
emits `mcp_call` spans holding the raw JSON-RPC envelope, so its arguments sit
under `params.arguments`. Reading only the SDK shape silently reduced every
proxy trace to a lone empty `final` step — which made any two of them compare
as "identical trajectory" no matter what the agent did.
"""

from __future__ import annotations

from agentsense.model.spans import LLM_CALL, MCP_CALL
from agentsense.model.spans import TOOL_CALL as SPAN_TOOL_CALL
from agentsense.replay.trajectory import FINAL, MODEL_TEXT, TOOL_CALL, Step, Trajectory


def _client_label(spans) -> str | None:
    """Name the MCP client from the handshake, for traces with no model to name.

    The proxy sees the `initialize` request go past, which is the only place the
    client identifies itself.
    """
    for s in spans:
        if s.kind == MCP_CALL and s.method == "initialize":
            info = ((s.request or {}).get("params") or {}).get("clientInfo") or {}
            name = info.get("name")
            return name if isinstance(name, str) and name else None
    return None


def captured_trajectory(store, trace_id: str) -> Trajectory:
    spans = store.spans_for_trace(trace_id)  # ordered by ts_start
    model_id = None
    for s in spans:
        if s.kind == LLM_CALL:
            model_id = s.attributes.get("gen_ai.request.model")
            break

    # Only for traces with no model of their own — an SDK run names its model.
    label = _client_label(spans) if model_id is None else None
    traj = Trajectory(model_id=model_id, label=label)
    last_answer = ""
    for s in spans:
        if s.kind == SPAN_TOOL_CALL:
            traj.add(
                Step(kind=TOOL_CALL, tool_name=s.tool_name,
                     tool_input=(s.request or {}).get("arguments", {}) or {})
            )
        elif s.kind == MCP_CALL and s.method == "tools/call":
            # A tool call the agent made is a decision whether or not it succeeded,
            # so this deliberately does not skip error/unanswered spans the way
            # Recording.from_trace_store (which needs a *result*) has to.
            params = (s.request or {}).get("params") or {}
            traj.add(
                Step(kind=TOOL_CALL,
                     tool_name=s.tool_name or params.get("name"),
                     tool_input=params.get("arguments") or {},
                     result=(s.response or {}).get("result"))
            )
        elif s.kind == LLM_CALL:
            text = ((s.response or {}).get("response") or {}).get("text", "")
            if isinstance(text, str) and text:
                last_answer = text
                traj.add(Step(kind=MODEL_TEXT, text=text))

    traj.final_text = last_answer
    traj.add(Step(kind=FINAL, text=last_answer))
    return traj
