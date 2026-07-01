"""Trace tap — turns a stream of framed messages into correlated spans.

The transport calls `observe(direction, raw_bytes)` for every line it forwards.
The tap parses a COPY (never the forwarded bytes), pairs each request with its
response by JSON-RPC id, and writes one span per pair to the store.

Correlation is direction-aware: client->server requests are answered
server->client and vice-versa. The two id spaces are kept separate so ids that
collide across directions don't get mismatched.

All work here is best-effort: a parse/store failure must never break forwarding,
so `observe` swallows and logs its own errors.
"""

from __future__ import annotations

import logging
import time

from tracekit.model.jsonrpc import parse_line
from tracekit.model.spans import Span
from tracekit.store.sqlite import SpanStore

log = logging.getLogger("tracekit.tap")

C2S = "c2s"  # client -> server
S2C = "s2c"  # server -> client


class TraceTap:
    def __init__(self, store: SpanStore, trace_id: str) -> None:
        self.store = store
        self.trace_id = trace_id
        # Pending requests keyed by id, split by the direction they travelled.
        self._pending: dict[str, dict[str, Span]] = {C2S: {}, S2C: {}}

    def observe(self, direction: str, raw: bytes) -> None:
        try:
            self._observe(direction, raw)
        except Exception:  # noqa: BLE001 - transparency: never disturb the wire
            log.exception("tap failed to process a message; forwarding is unaffected")

    def _observe(self, direction: str, raw: bytes) -> None:
        msg = parse_line(raw)
        if msg is None:
            return

        if msg.is_request:
            span = Span(
                trace_id=self.trace_id,
                jsonrpc_id=msg.id,
                method=msg.method,
                tool_name=_tool_name(msg),
                request=msg.raw,
                ts_start=time.time(),
            )
            self._pending[direction][msg.id] = span
            return

        if msg.is_notification:
            # No response to wait for — persist immediately as a closed span.
            span = Span(
                trace_id=self.trace_id,
                method=msg.method,
                tool_name=_tool_name(msg),
                request=msg.raw,
            )
            span.close()
            self.store.write(span)
            return

        if msg.is_response:
            # A response travelling `direction` answers a request that went the
            # other way.
            origin = C2S if direction == S2C else S2C
            span = self._pending[origin].pop(msg.id, None)
            if span is None:
                # Unmatched response (e.g. request predates the proxy). Record it
                # standalone rather than drop it.
                span = Span(trace_id=self.trace_id, jsonrpc_id=msg.id)
            if msg.has_error:
                span.error = msg.raw
            else:
                span.response = msg.raw
            span.close()
            self.store.write(span)

    def flush(self) -> None:
        """Persist any requests that never got a response (e.g. session ended)."""
        for pending in self._pending.values():
            for span in pending.values():
                span.close()
                self.store.write(span)
            pending.clear()


def _tool_name(msg) -> str | None:
    if msg.method == "tools/call":
        return (msg.raw.get("params") or {}).get("name")
    return None
