"""A Recording — everything the engine needs to re-drive an agent run.

Tool results are looked up by (tool_name, canonicalised input). Recorded results
can be supplied directly or pulled from the proxy's SQLite store: the proxy
already captured every tools/call request+result at the protocol level (and
redacted them on write), so `from_trace_store` turns a captured trace into a
replayable recording.

The proxy captures the MCP side (tool I/O); `from_trace_store` needs the question,
tools, and model supplied. The capture SDK records the model conversation too, so
`from_sdk_trace` reconstructs a full recording end-to-end from a captured run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tracekit.model.spans import LLM_CALL, TOOL_CALL
from tracekit.replay.types import ToolSpec


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


@dataclass
class Recording:
    question: str
    tools: list[ToolSpec] = field(default_factory=list)
    system: str | None = None
    model_id: str | None = None
    #: (tool_name, canonical_input) -> recorded result
    tool_results: dict[tuple[str, str], Any] = field(default_factory=dict)

    def record(self, name: str, tool_input: dict[str, Any], result: Any) -> None:
        self.tool_results[(name, _canonical(tool_input))] = result

    def lookup(self, name: str, tool_input: dict[str, Any]) -> tuple[bool, Any]:
        """Return (found, result). Match is strict on (name, canonical input).

        A miss means the model called a tool with arguments the recording doesn't
        cover — a genuine trajectory divergence we want surfaced, not smoothed
        over. (Canonicalisation already absorbs key-ordering differences.)
        """
        key = (name, _canonical(tool_input))
        if key in self.tool_results:
            return True, self.tool_results[key]
        return False, None

    @classmethod
    def from_trace_store(
        cls,
        store,
        trace_id: str,
        question: str,
        tools: list[ToolSpec] | None = None,
        system: str | None = None,
        model_id: str | None = None,
    ) -> Recording:
        """Build a recording from the proxy's captured tools/call spans."""
        rec = cls(question=question, tools=tools or [], system=system, model_id=model_id)
        for span in store.spans_for_trace(trace_id):
            if span.method != "tools/call" or span.response is None:
                continue
            args = (span.request or {}).get("params", {}).get("arguments", {})
            result = span.response.get("result")
            if span.tool_name is not None and result is not None:
                rec.record(span.tool_name, args, result)
        return rec

    @classmethod
    def from_sdk_trace(
        cls,
        store,
        trace_id: str,
        question: str | None = None,
        tools: list[ToolSpec] | None = None,
        system: str | None = None,
        model_id: str | None = None,
    ) -> Recording:
        """Reconstruct a replayable recording from a capture-SDK trace.

        Tool results come from `tool_call` spans; the question, tool declarations,
        system prompt, and model id are read from the first `llm_call` span (each
        overridable by the matching argument). Tool declarations are parsed from
        the normalized, OpenAI, or Bedrock shapes.
        """
        spans = store.spans_for_trace(trace_id)
        llm_calls = [s for s in spans if s.kind == LLM_CALL]
        first_llm = llm_calls[0] if llm_calls else None
        req = (first_llm.request if first_llm else None) or {}

        if question is None:
            question = _extract_question(req.get("messages"))
        if question is None:
            raise ValueError(
                "could not extract a question from the trace; pass question=..."
            )
        if tools is None:
            tools = _parse_tools(req.get("tools"))
        if system is None:
            system = req.get("system")
        if model_id is None and first_llm is not None:
            model_id = first_llm.attributes.get("gen_ai.request.model") or req.get("model")

        rec = cls(question=question, tools=tools, system=system, model_id=model_id)
        for span in spans:
            if span.kind != TOOL_CALL:
                continue
            args = (span.request or {}).get("arguments", {}) or {}
            result = (span.response or {}).get("result")
            if span.tool_name is not None and result is not None:
                rec.record(span.tool_name, args, result)
        return rec


def _extract_question(messages: Any) -> str | None:
    """Best-effort first user-message text from common message shapes."""
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and "text" in b
            ]
            if parts:
                return "".join(parts)
    return None


def _parse_tools(raw: Any) -> list[ToolSpec]:
    """Parse tool declarations from normalized / OpenAI / Bedrock shapes."""
    if not isinstance(raw, list):
        return []
    specs: list[ToolSpec] = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        if "toolSpec" in t:  # Bedrock Converse
            ts = t["toolSpec"]
            specs.append(ToolSpec(ts.get("name", ""), ts.get("description", ""),
                                  (ts.get("inputSchema") or {}).get("json", {})))
        elif t.get("type") == "function" and "function" in t:  # OpenAI
            fn = t["function"]
            specs.append(ToolSpec(fn.get("name", ""), fn.get("description", ""),
                                  fn.get("parameters", {})))
        elif "input_schema" in t or "name" in t:  # normalized ToolSpec-like
            specs.append(ToolSpec(t.get("name", ""), t.get("description", ""),
                                  t.get("input_schema", {})))
    return specs
