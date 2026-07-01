"""Tap: request/response correlation by JSON-RPC id -> one span with latency."""

import json

from tracekit.proxy.tap import TraceTap
from tracekit.store.sqlite import SpanStore


def _line(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def test_request_response_correlate_into_one_span(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    tap = TraceTap(store, "traceX")

    tap.observe("c2s", _line({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "read_file", "arguments": {"path": "/tmp/a"}},
    }))
    # No span yet — waiting on the response.
    assert store.spans_for_trace("traceX") == []

    tap.observe("s2c", _line({
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"type": "text", "text": "hello"}]},
    }))

    spans = store.spans_for_trace("traceX")
    assert len(spans) == 1
    span = spans[0]
    assert span.method == "tools/call"
    assert span.tool_name == "read_file"
    assert span.request["id"] == 1
    assert span.response["result"]["content"][0]["text"] == "hello"
    assert span.latency_ms is not None and span.latency_ms >= 0
    store.close()


def test_error_response_captured_as_error(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    tap = TraceTap(store, "traceE")
    tap.observe("c2s", _line({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                              "params": {"name": "denied"}}))
    tap.observe("s2c", _line({"jsonrpc": "2.0", "id": 7,
                              "error": {"code": -32000, "message": "access denied"}}))
    span = store.spans_for_trace("traceE")[0]
    assert span.error["error"]["message"] == "access denied"
    assert span.response is None
    store.close()


def test_notification_persisted_standalone(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    tap = TraceTap(store, "traceN")
    tap.observe("c2s", _line({"jsonrpc": "2.0",
                              "method": "notifications/initialized"}))
    spans = store.spans_for_trace("traceN")
    assert len(spans) == 1
    assert spans[0].method == "notifications/initialized"
    assert spans[0].latency_ms is not None
    store.close()


def test_colliding_ids_across_directions_dont_mismatch(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    tap = TraceTap(store, "traceC")
    # Both a client->server and a server->client request use id=1.
    tap.observe("c2s", _line({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))
    tap.observe("s2c", _line({"jsonrpc": "2.0", "id": 1,
                              "method": "sampling/createMessage"}))
    # Each response goes back the opposite way; they must match their own request.
    tap.observe("s2c", _line({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}))
    tap.observe("c2s", _line({"jsonrpc": "2.0", "id": 1, "result": {"model": "x"}}))

    spans = {s.method: s for s in store.spans_for_trace("traceC")}
    assert spans["tools/list"].response["result"] == {"tools": []}
    assert spans["sampling/createMessage"].response["result"] == {"model": "x"}
    store.close()


def test_unmatched_request_flushed_on_close(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    tap = TraceTap(store, "traceF")
    tap.observe("c2s", _line({"jsonrpc": "2.0", "id": 99, "method": "tools/list"}))
    tap.flush()
    spans = store.spans_for_trace("traceF")
    assert len(spans) == 1 and spans[0].response is None
    store.close()
