"""Span/trace models.

A span is the logical unit of a trace. For the MCP proxy, a span is a correlated
request/response pair (matched by JSON-RPC id) plus timing. Whole message objects
are retained verbatim — no field whitelist — so unknown/vendor fields survive
(Phase 0 row 8).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex


# Span kinds. mcp_call comes from the proxy; the rest come from the capture SDK.
MCP_CALL = "mcp_call"
SESSION = "session"
REASONING = "reasoning"
TOOL_CALL = "tool_call"
LLM_CALL = "llm_call"


class RedactionEvent(BaseModel):
    """One redaction action, for the audit trail (what type, where)."""

    type: str  # detector name, e.g. "email"
    path: str  # dotted path into the object, e.g. "params.arguments.to"
    token: str  # the deterministic token that replaced the value


class Span(BaseModel):
    """A single traced operation.

    `request` / `response` hold the WHOLE parsed message objects (post-redaction
    at persist time). `kind` is "mcp_call" for the proxy; other kinds
    (llm_call, reasoning) arrive with the SDK.
    """

    span_id: str = Field(default_factory=_new_id)
    trace_id: str
    parent_span_id: str | None = None
    kind: str = MCP_CALL
    name: str | None = None  # human label: session/step/tool/model-op name

    # MCP / JSON-RPC identity
    jsonrpc_id: str | None = None
    method: str | None = None
    tool_name: str | None = None

    # Free-form attributes (SDK uses OTel GenAI conventions, e.g. gen_ai.request.model)
    attributes: dict[str, Any] = Field(default_factory=dict)

    # Timing (epoch seconds; latency in ms)
    ts_start: float = Field(default_factory=time.time)
    ts_end: float | None = None
    latency_ms: float | None = None

    # Whole objects, preserved verbatim (no whitelist)
    request: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    # Redaction audit
    redactions: list[RedactionEvent] = Field(default_factory=list)

    def close(self, ts_end: float | None = None) -> None:
        self.ts_end = ts_end if ts_end is not None else time.time()
        self.latency_ms = (self.ts_end - self.ts_start) * 1000.0

    def set_latency(self, latency_ms: float) -> None:
        """Close the span with a known latency (e.g. reported by the model API)."""
        self.latency_ms = latency_ms
        self.ts_end = self.ts_start + latency_ms / 1000.0
