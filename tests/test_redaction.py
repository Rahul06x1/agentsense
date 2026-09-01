"""Redaction: determinism (Phase 0 row 6) + no plaintext leak + structure fidelity."""

from agentsense.redaction.redactor import redact_object, redact_text


def test_email_deterministic_and_no_leak():
    a, _ = redact_text("alice@example.com")
    a2, _ = redact_text("alice@example.com")
    b, _ = redact_text("bob@corp.io")
    assert a == a2  # same input -> same token (aligns across replay)
    assert a != b  # different input -> different token
    assert "@" not in a  # no plaintext leak


def test_redaction_events_carry_path():
    obj = {"params": {"arguments": {"to": "alice@example.com", "n": 3}}}
    redacted, audit = redact_object(obj)
    assert redacted["params"]["arguments"]["to"].startswith("<EMAIL_")
    assert redacted["params"]["arguments"]["n"] == 3  # non-string preserved
    assert len(audit) == 1
    assert audit[0].type == "email"
    assert audit[0].path == "params.arguments.to"


def test_structure_and_vendor_fields_preserved():
    obj = {
        "known": "email me at a@b.io",
        "vendor_field": {"thought_signature": "xyz", "nested": [1, 2, {"k": "v"}]},
        "count": 7,
        "flag": True,
        "nothing": None,
    }
    redacted, _ = redact_object(obj)
    # Only the string leaf with PII changed; everything else is byte-identical.
    assert redacted["vendor_field"] == obj["vendor_field"]
    assert redacted["count"] == 7 and redacted["flag"] is True
    assert redacted["nothing"] is None
    assert "@" not in redacted["known"]


def test_same_object_redacts_identically_across_runs():
    obj = {"content": [{"text": "reach bob@corp.io or +1 415 555 0100"}]}
    r1, _ = redact_object(obj)
    r2, _ = redact_object(obj)
    assert r1 == r2  # deterministic at the object level -> replay aligns


def test_iso_dates_are_not_mistaken_for_phone_numbers():
    """Dates are digit/hyphen runs the phone detector used to swallow.

    Regression: found by tracing a real MCP session. Left unguarded this fires on
    every trace ever captured (`initialize` always carries a protocolVersion date)
    and mangles timestamps inside tool results, which in turn trips the replay
    diff's `redaction_suspect` flag on runs where nothing was actually redacted.
    """
    for value in ("2024-11-05", "2025-06-11", "2024-11-05T14:30:00Z", "0.2.0"):
        redacted, audit = redact_text(value)
        assert redacted == value, f"{value!r} was rewritten to {redacted!r}"
        assert audit == []


def test_initialize_response_redacts_to_nothing():
    """The exact payload that exposed the bug: a clean handshake must stay clean."""
    obj = {
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": True}},
            "serverInfo": {"name": "secure-filesystem-server", "version": "0.2.0"},
        },
        "jsonrpc": "2.0",
        "id": 0,
    }
    redacted, audit = redact_object(obj)
    assert redacted == obj
    assert audit == []


def test_real_phone_numbers_still_redacted():
    """The date guard must not cost us the detections the guard exists to protect."""
    for value in ("+1 415 555 0100", "+44 7700 900123", "415-555-0100"):
        redacted, audit = redact_text(value)
        assert "<PHONE_" in redacted
        assert [kind for kind, _ in audit] == ["phone"]

    # A date sitting next to a phone number must not shield it.
    redacted, audit = redact_text("on 2024-11-05 call +44 7700 900123")
    assert "2024-11-05" in redacted
    assert "900123" not in redacted
    assert [kind for kind, _ in audit] == ["phone"]
