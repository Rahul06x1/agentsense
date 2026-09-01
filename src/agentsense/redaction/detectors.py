"""Structured-PII detectors — regex/heuristic, stdlib only.

Each detector is (name, compiled_regex). Free-text NER (Presidio) is an optional
fast-follow; v0 ships the deterministic regex layer that the compliance story needs.
Order matters: earlier detectors win on overlapping matches.
"""

from __future__ import annotations

import re

# ISO-8601 dates are digit-and-hyphen runs the phone pattern would otherwise
# swallow — and every MCP `initialize` carries one in `protocolVersion`, so an
# unguarded phone detector puts a bogus hit in literally every trace.
_ISO_DATE = r"\d{4}-\d{2}-\d{2}"

# Credit cards first (before phone) so 13-16 digit runs aren't mislabelled.
DETECTORS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    # E.164-ish / common phone forms: optional +, 7-15 digits with separators.
    # The two guards are load-bearing together: the lookahead refuses to start a
    # match on a date, and the lookbehind stops the engine from sidestepping it
    # by restarting one digit in ("2024-11-05" -> "024-11-05").
    (
        "phone",
        re.compile(
            r"(?<![-\d])"  # not mid-way through a longer digit/hyphen run
            rf"(?!{_ISO_DATE})"  # and never starting on an ISO-8601 date
            r"\+?\d[\d\s().-]{6,}\d"
        ),
    ),
]
