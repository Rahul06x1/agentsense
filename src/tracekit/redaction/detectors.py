"""Structured-PII detectors — regex/heuristic, stdlib only.

Each detector is (name, compiled_regex). Free-text NER (Presidio) is an optional
fast-follow; v0 ships the deterministic regex layer that the compliance story needs.
Order matters: earlier detectors win on overlapping matches.
"""

from __future__ import annotations

import re

# Credit cards first (before phone) so 13-16 digit runs aren't mislabelled.
DETECTORS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    # E.164-ish / common phone forms: optional +, 7-15 digits with separators.
    ("phone", re.compile(r"\+?\d[\d\s().-]{6,}\d")),
]
