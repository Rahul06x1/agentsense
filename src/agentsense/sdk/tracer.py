"""Capture SDK — instrument an agent's reasoning-level steps.

Framework-agnostic: wrap an agent run in a session and record reasoning steps,
tool calls, and LLM calls. Every span is written through the SAME SpanStore as
the proxy, so redaction runs on one audited code path.

    tracer = Tracer(SpanStore("traces.db"))
    with tracer.session("booking-agent") as s:
        s.step("plan", reasoning="search flights first", model="claude-opus-4-8")
        s.tool_call("search_flights", args={...}, result={...}, cost=0.002)
        s.llm_call("claude-opus-4-8", messages=[...], response={...},
                   usage={"input_tokens": 1200, "output_tokens": 80})

Spans form a shallow tree: the session is the root; steps/tool_calls/llm_calls are
its children. LLM calls record model + messages + tools + response so a captured
run can later be reconstructed into a replayable Recording.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from typing import Any

from agentsense.model.spans import (
    LLM_CALL,
    REASONING,
    SESSION,
    TOOL_CALL,
    Span,
)
from agentsense.sdk.conventions import llm_attributes
from agentsense.store.sqlite import SpanStore


class Session:
    """A single traced agent run. Create via `Tracer.session(...)`."""

    def __init__(self, store: SpanStore, trace_id: str, root: Span) -> None:
        self.store = store
        self.trace_id = trace_id
        self.root = root

    def _child(self, kind: str, name: str, **fields: Any) -> Span:
        return Span(
            trace_id=self.trace_id,
            parent_span_id=self.root.span_id,
            kind=kind,
            name=name,
            **fields,
        )

    def _finish(self, span: Span, latency_ms: float | None) -> Span:
        if latency_ms is not None:
            span.set_latency(latency_ms)
        else:
            span.close()
        return self.store.write(span)

    def step(
        self,
        name: str,
        reasoning: str | None = None,
        model: str | None = None,
        latency_ms: float | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Record a reasoning/decision step."""
        attrs = dict(attributes or {})
        if model:
            attrs["gen_ai.request.model"] = model
        request = {"reasoning": reasoning} if reasoning is not None else None
        span = self._child(REASONING, name, request=request, attributes=attrs)
        return self._finish(span, latency_ms)

    def tool_call(
        self,
        name: str,
        args: Any = None,
        result: Any = None,
        error: Any = None,
        cost: float | None = None,
        latency_ms: float | None = None,
    ) -> Span:
        """Record a tool invocation (reasoning-level; the proxy captures MCP tools)."""
        attrs = {"agentsense.cost_usd": cost} if cost is not None else {}
        span = self._child(
            TOOL_CALL,
            name,
            tool_name=name,
            request={"arguments": args} if args is not None else None,
            response={"result": result} if result is not None else None,
            error={"error": error} if error is not None else None,
            attributes=attrs,
        )
        return self._finish(span, latency_ms)

    def llm_call(
        self,
        model: str,
        messages: Any = None,
        tools: Any = None,
        response: Any = None,
        usage: dict[str, Any] | None = None,
        system: str | None = None,
        finish_reason: str | None = None,
        cost: float | None = None,
        latency_ms: float | None = None,
    ) -> Span:
        """Record a model call with OTel GenAI attributes.

        `messages`/`tools`/`response` are stored whole so the run stays replayable.
        """
        attrs = llm_attributes(
            model, usage=usage, finish_reason=finish_reason, system=system, cost=cost
        )
        span = self._child(
            LLM_CALL,
            f"chat {model}",
            request={"model": model, "system": system, "messages": messages, "tools": tools},
            response={"response": response, "usage": usage},
            attributes=attrs,
        )
        return self._finish(span, latency_ms)


class Tracer:
    """Entry point for the capture SDK. Bind to a SpanStore, open sessions."""

    def __init__(self, store: SpanStore) -> None:
        self.store = store

    @contextlib.contextmanager
    def session(
        self,
        name: str,
        trace_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[Session]:
        tid = trace_id or uuid.uuid4().hex
        root = Span(
            trace_id=tid, kind=SESSION, name=name, attributes=dict(attributes or {})
        )
        session = Session(self.store, tid, root)
        try:
            yield session
        finally:
            # Root closes last so its duration spans the whole session.
            root.close()
            self.store.write(root)
