"""Capture SDK shares the proxy's redaction code path — PII never persists raw."""

from agentsense.model.spans import LLM_CALL, REASONING, TOOL_CALL
from agentsense.sdk import Tracer
from agentsense.store.sqlite import SpanStore


def test_pii_in_sdk_spans_is_redacted(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    with Tracer(store).session("a", trace_id="s1") as s:
        s.step("plan", reasoning="email the customer at alice@example.com")
        s.tool_call("send_email", args={"to": "bob@corp.io"}, result={"ok": True})
        s.llm_call("claude-haiku-4-5",
                   messages=[{"role": "user", "content": "reach me at carol@x.io"}],
                   response={"text": "will contact carol@x.io"})

    spans = {sp.kind: sp for sp in store.spans_for_trace("s1")}

    reasoning = spans[REASONING].request["reasoning"]
    assert "@" not in reasoning and "<EMAIL_" in reasoning

    tool = spans[TOOL_CALL]
    assert "@" not in tool.request["arguments"]["to"]

    llm = spans[LLM_CALL]
    assert "@" not in llm.request["messages"][0]["content"]
    assert "@" not in llm.response["response"]["text"]

    # Redaction was audited on each span that contained PII.
    assert any(e.type == "email" for e in spans[REASONING].redactions)
    assert any(e.type == "email" for e in llm.redactions)
    store.close()


def test_same_email_same_token_across_spans(tmp_path):
    # Determinism across spans (and thus across replay) — the shared module guarantees it.
    store = SpanStore(tmp_path / "t.db")
    with Tracer(store).session("a", trace_id="s2") as s:
        s.step("one", reasoning="mail alice@example.com")
        s.step("two", reasoning="again alice@example.com")
    spans = [sp for sp in store.spans_for_trace("s2") if sp.kind == REASONING]
    tokens = {sp.request["reasoning"].split()[-1] for sp in spans}
    assert len(tokens) == 1  # identical token in both spans
    store.close()
