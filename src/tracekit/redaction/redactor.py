"""Deterministic PII redaction — the shared write-path module.

Both the MCP proxy and the (fast-follow) SDK call this before anything hits the
trace store. Determinism is a hard requirement (Phase 0 row 6): the same input
must redact to the same token in the original run and in replay, so a redacted
trace still aligns with its replay.

Token = "<TYPE_" + sha256(value)[:8] + ">". Irreversible by default; the plaintext
is never persisted. Keyed/reversible tokenization is a documented fast-follow.
"""

from __future__ import annotations

import hashlib
from typing import Any

from tracekit.model.spans import RedactionEvent
from tracekit.redaction.detectors import DETECTORS


def _token(kind: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"<{kind.upper()}_{digest}>"


def redact_text(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Redact one string. Returns (redacted_text, [(detector, token), ...])."""
    events: list[tuple[str, str]] = []

    def make_sub(kind: str):
        def _sub(m) -> str:
            tok = _token(kind, m.group(0))
            events.append((kind, tok))
            return tok

        return _sub

    for name, pattern in DETECTORS:
        text = pattern.sub(make_sub(name), text)
    return text, events


def redact_object(obj: Any, _path: str = "") -> tuple[Any, list[RedactionEvent]]:
    """Recursively redact strings in a JSON-like object.

    Returns a NEW redacted object plus a flat audit log. Structure, keys, and
    non-string values (incl. unknown/vendor fields) are preserved exactly — only
    string leaves are rewritten, so round-trip fidelity (row 8) is maintained.
    """
    audit: list[RedactionEvent] = []

    if isinstance(obj, str):
        redacted, hits = redact_text(obj)
        for kind, tok in hits:
            audit.append(RedactionEvent(type=kind, path=_path or "<root>", token=tok))
        return redacted, audit

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            child_path = f"{_path}.{key}" if _path else str(key)
            out[key], child_audit = redact_object(value, child_path)
            audit.extend(child_audit)
        return out, audit

    if isinstance(obj, list):
        out_list: list[Any] = []
        for i, value in enumerate(obj):
            child_path = f"{_path}[{i}]"
            red, child_audit = redact_object(value, child_path)
            out_list.append(red)
            audit.extend(child_audit)
        return out_list, audit

    # int, float, bool, None — nothing to redact, preserved as-is.
    return obj, audit
