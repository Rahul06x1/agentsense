"""Minimal JSON-RPC 2.0 message classification — operates on a COPY only.

The proxy forwards raw bytes unchanged; this module parses a *copy* of a framed
line purely to build trace spans. It never re-serializes anything on the wire path.

MCP over stdio frames one JSON object per line. A message is one of:
  - request:      has "method" and "id"        (expects a response)
  - notification: has "method", no "id"         (fire-and-forget)
  - response:     has "id" and "result"/"error" (no "method")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedMessage:
    """A parsed copy of one framed JSON-RPC message."""

    raw: dict[str, Any]
    id: str | None
    method: str | None

    @property
    def is_request(self) -> bool:
        return self.method is not None and self.id is not None

    @property
    def is_notification(self) -> bool:
        return self.method is not None and self.id is None

    @property
    def is_response(self) -> bool:
        return self.method is None and self.id is not None

    @property
    def has_error(self) -> bool:
        return "error" in self.raw


def parse_line(raw: bytes) -> ParsedMessage | None:
    """Parse one framed line into a ParsedMessage, or None if it is not JSON.

    Never raises on malformed input — the wire path must not be disturbed by a
    parse failure, so callers get None and forward the bytes regardless.
    """
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return None
    try:
        msg = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(msg, dict):
        return None
    raw_id = msg.get("id")
    return ParsedMessage(
        raw=msg,
        id=str(raw_id) if raw_id is not None else None,
        method=msg.get("method"),
    )
