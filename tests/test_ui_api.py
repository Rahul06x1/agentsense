"""UI backend: read-only API over the trace store, offline via TestClient."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from tracekit.model.spans import Span  # noqa: E402
from tracekit.sdk import Tracer  # noqa: E402
from tracekit.store.sqlite import SpanStore  # noqa: E402
from tracekit.ui.app import create_app  # noqa: E402


def _seed(db_path):
    store = SpanStore(db_path)
    # A proxy-style mcp trace with a redacted email in a tool result.
    mcp = Span(trace_id="mcp1", method="tools/call", tool_name="read_contact",
               request={"params": {"name": "read_contact"}},
               response={"result": {"email": "alice@example.com"}})
    mcp.close()
    store.write(mcp)
    # An SDK session trace.
    with Tracer(store).session("booking-agent", trace_id="sdk1") as s:
        s.step("plan", reasoning="pick flights")
        s.tool_call("search", args={"q": "CDG"}, result={"n": 2}, latency_ms=12.0)
    store.close()


def _client(tmp_path):
    db = tmp_path / "t.db"
    _seed(db)
    return TestClient(create_app(str(db)))


def test_list_traces(tmp_path):
    r = _client(tmp_path).get("/api/traces")
    assert r.status_code == 200
    traces = {t["trace_id"]: t for t in r.json()}
    assert set(traces) == {"mcp1", "sdk1"}
    assert traces["sdk1"]["name"] == "booking-agent"
    assert "session" in traces["sdk1"]["kinds"]
    assert traces["sdk1"]["tool_calls"] == 1
    assert traces["mcp1"]["tool_calls"] == 1  # method='tools/call' counted


def test_trace_spans_and_redaction_count(tmp_path):
    client = _client(tmp_path)
    r = client.get("/api/traces/mcp1/spans")
    assert r.status_code == 200
    body = r.json()
    assert body["redaction_count"] >= 1  # the email was redacted on write
    span = body["spans"][0]
    assert "@" not in span["response"]["result"]["email"]


def test_missing_trace_404(tmp_path):
    r = _client(tmp_path).get("/api/traces/nope/spans")
    assert r.status_code == 404


def test_index_page_served(tmp_path):
    r = _client(tmp_path).get("/")
    assert r.status_code == 200
    assert "tracekit" in r.text and "trace explorer" in r.text
