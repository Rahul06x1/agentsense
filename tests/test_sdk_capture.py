"""Capture SDK: session tree, span kinds, parent links, timing."""

from tracekit.model.spans import LLM_CALL, REASONING, SESSION, TOOL_CALL
from tracekit.sdk import Tracer
from tracekit.store.sqlite import SpanStore


def test_session_produces_rooted_tree(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    tracer = Tracer(store)

    with tracer.session("booking-agent", trace_id="s1") as s:
        s.step("plan", reasoning="search flights first", model="claude-opus-4-8")
        s.tool_call("search_flights", args={"from": "LHR"}, result={"n": 3},
                    cost=0.002, latency_ms=42.0)
        s.llm_call("claude-opus-4-8", messages=[{"role": "user"}],
                   response={"text": "done"},
                   usage={"input_tokens": 1200, "output_tokens": 80})

    spans = store.spans_for_trace("s1")
    by_kind = {sp.kind: sp for sp in spans}
    assert set(by_kind) == {SESSION, REASONING, TOOL_CALL, LLM_CALL}

    root = by_kind[SESSION]
    assert root.parent_span_id is None
    assert root.name == "booking-agent"
    # Every non-session span is a child of the session root.
    for sp in spans:
        if sp.kind != SESSION:
            assert sp.parent_span_id == root.span_id
    store.close()


def test_reported_latency_is_used(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    with Tracer(store).session("a", trace_id="s2") as s:
        s.tool_call("t", args={}, result={}, latency_ms=42.0)
    tool = next(sp for sp in store.spans_for_trace("s2") if sp.kind == TOOL_CALL)
    assert tool.latency_ms == 42.0
    assert tool.ts_end == tool.ts_start + 0.042
    store.close()


def test_tool_error_captured(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    with Tracer(store).session("a", trace_id="s3") as s:
        s.tool_call("t", args={"x": 1}, error={"message": "boom"})
    tool = next(sp for sp in store.spans_for_trace("s3") if sp.kind == TOOL_CALL)
    assert tool.error["error"]["message"] == "boom"
    assert tool.response is None
    store.close()


def test_root_duration_spans_children(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    with Tracer(store).session("a", trace_id="s4") as s:
        s.step("one")
        s.step("two")
    root = next(sp for sp in store.spans_for_trace("s4") if sp.kind == SESSION)
    assert root.latency_ms is not None and root.ts_end is not None
    store.close()
