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
