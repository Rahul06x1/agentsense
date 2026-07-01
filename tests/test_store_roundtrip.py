"""Store: whole objects (incl. unknown/vendor fields) survive persistence (row 8),
and redaction runs on the write path."""

from tracekit.model.spans import Span
from tracekit.store.sqlite import SpanStore


def _bedrock_like_response() -> dict:
    # Mirrors a Converse response's shape, incl. fields a whitelist would drop.
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "ok"},
                    {"toolUse": {"toolUseId": "tu_1", "name": "get_weather",
                                 "input": {"city": "Paris"}, "type": "undocumented"}},
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 5,
                  "cacheReadInputTokens": 2, "cacheWriteInputTokens": 0},
        "metrics": {"latencyMs": 321},
        "additionalModelResponseFields": {"vendor": "anthropic"},
    }


def test_whole_object_roundtrips(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    resp = _bedrock_like_response()
    span = Span(trace_id="trace1", method="tools/call", tool_name="get_weather",
                response=resp)
    span.close()
    store.write(span)

    got = store.get(span.span_id)
    assert got is not None
    # Nothing dropped: usage (incl. cache tokens), metrics, vendor fields, nested type.
    assert got.response["usage"]["cacheReadInputTokens"] == 2
    assert got.response["metrics"]["latencyMs"] == 321
    assert got.response["additionalModelResponseFields"]["vendor"] == "anthropic"
    tu = got.response["output"]["message"]["content"][1]["toolUse"]
    assert tu["toolUseId"] == "tu_1" and tu["type"] == "undocumented"
    store.close()


def test_write_path_redacts_and_audits(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    span = Span(
        trace_id="trace2",
        method="tools/call",
        tool_name="send_email",
        request={"params": {"arguments": {"to": "alice@example.com"}}},
    )
    span.close()
    store.write(span)

    got = store.get(span.span_id)
    assert "@" not in got.request["params"]["arguments"]["to"]  # redacted at rest
    assert any(e.type == "email" for e in got.redactions)  # audit recorded
    store.close()


def test_spans_for_trace_ordered(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    for i in range(3):
        s = Span(trace_id="trace3", method="ping", ts_start=100 + i)
        s.close()
        store.write(s)
    spans = store.spans_for_trace("trace3")
    assert [s.ts_start for s in spans] == [100, 101, 102]
    store.close()
