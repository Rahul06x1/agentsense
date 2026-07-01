"""SQLite trace store — the write path.

Persistence rule (Phase 0): store the WHOLE message object as a JSON blob (no
field whitelist) so unknown/vendor fields survive; index only a few columns for
querying. Every object passes through deterministic redaction BEFORE it is
written — nothing sensitive is ever persisted raw.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tracekit.model.spans import Span
from tracekit.redaction.redactor import redact_object

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    span_id        TEXT PRIMARY KEY,
    trace_id       TEXT NOT NULL,
    parent_span_id TEXT,
    kind           TEXT,
    name           TEXT,
    jsonrpc_id     TEXT,
    method         TEXT,
    tool_name      TEXT,
    ts_start       REAL,
    ts_end         REAL,
    latency_ms     REAL,
    request_json   TEXT,
    response_json  TEXT,
    error_json     TEXT,
    attributes_json TEXT,
    redactions_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_tool  ON spans(tool_name);
"""


class SpanStore:
    """Thin SQLite writer/reader for spans. Redacts on write."""

    def __init__(self, db_path: str | Path = "tracekit.db") -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def write(self, span: Span) -> Span:
        """Redact request/response/error in place, then persist the whole objects.

        Redaction events from all three payloads are merged into the span's audit
        log so the trace itself proves redaction happened.
        """
        audit = list(span.redactions)
        if span.request is not None:
            span.request, hits = redact_object(span.request)
            audit.extend(hits)
        if span.response is not None:
            span.response, hits = redact_object(span.response)
            audit.extend(hits)
        if span.error is not None:
            span.error, hits = redact_object(span.error)
            audit.extend(hits)
        if span.attributes:
            span.attributes, hits = redact_object(span.attributes)
            audit.extend(hits)
        span.redactions = audit

        self._conn.execute(
            """INSERT OR REPLACE INTO spans (
                span_id, trace_id, parent_span_id, kind, name, jsonrpc_id, method,
                tool_name, ts_start, ts_end, latency_ms,
                request_json, response_json, error_json, attributes_json, redactions_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                span.span_id,
                span.trace_id,
                span.parent_span_id,
                span.kind,
                span.name,
                span.jsonrpc_id,
                span.method,
                span.tool_name,
                span.ts_start,
                span.ts_end,
                span.latency_ms,
                _dumps(span.request),
                _dumps(span.response),
                _dumps(span.error),
                _dumps(span.attributes),
                json.dumps([e.model_dump() for e in span.redactions]),
            ),
        )
        self._conn.commit()
        return span

    def get(self, span_id: str) -> Span | None:
        cur = self._conn.execute("SELECT * FROM spans WHERE span_id = ?", (span_id,))
        row = cur.fetchone()
        return _row_to_span(cur, row) if row else None

    def spans_for_trace(self, trace_id: str) -> list[Span]:
        cur = self._conn.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY ts_start", (trace_id,)
        )
        return [_row_to_span(cur, row) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


def _dumps(obj) -> str | None:
    # default=str keeps the write lossless even for non-JSON-native values.
    return None if obj is None else json.dumps(obj, default=str)


def _loads(text: str | None):
    return None if text is None else json.loads(text)


def _row_to_span(cur: sqlite3.Cursor, row: tuple) -> Span:
    cols = [c[0] for c in cur.description]
    data = dict(zip(cols, row, strict=True))
    return Span(
        span_id=data["span_id"],
        trace_id=data["trace_id"],
        parent_span_id=data["parent_span_id"],
        kind=data["kind"],
        name=data["name"],
        jsonrpc_id=data["jsonrpc_id"],
        method=data["method"],
        tool_name=data["tool_name"],
        ts_start=data["ts_start"],
        ts_end=data["ts_end"],
        latency_ms=data["latency_ms"],
        request=_loads(data["request_json"]),
        response=_loads(data["response_json"]),
        error=_loads(data["error_json"]),
        attributes=_loads(data["attributes_json"]) or {},
        redactions=_loads(data["redactions_json"]) or [],
    )
