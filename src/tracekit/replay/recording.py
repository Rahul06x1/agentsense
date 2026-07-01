"""A Recording — everything the engine needs to re-drive an agent run.

Tool results are looked up by (tool_name, canonicalised input). Recorded results
can be supplied directly or pulled from the proxy's SQLite store: the proxy
already captured every tools/call request+result at the protocol level (and
redacted them on write), so `from_trace_store` turns a captured trace into a
replayable recording.

Note: the proxy captures the MCP side (tool I/O). The user question, tool
declarations, and model id are replay inputs today; the capture SDK (fast-follow)
will record the model conversation so a full trace replays end-to-end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

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
