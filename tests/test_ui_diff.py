"""UI diff endpoint: decision-level trajectory diff between two captured traces."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from agentsense.model.spans import Span  # noqa: E402
from agentsense.sdk import Tracer  # noqa: E402
from agentsense.store.sqlite import SpanStore  # noqa: E402
from agentsense.ui.app import create_app  # noqa: E402


def _run(store, trace_id, *, call_tool):
    """Capture a run that either calls a tool then answers, or answers directly."""
    with Tracer(store).session("agent", trace_id=trace_id) as s:
        s.llm_call("m", messages=[{"role": "user", "content": "weather in Paris?"}],
                   response={"text": ""}, finish_reason="tool_use" if call_tool else "end")
        if call_tool:
            s.tool_call("get_weather", args={"city": "Paris"}, result={"t": 18})
        s.llm_call("m", messages=[{"role": "user", "content": "..."}],
                   response={"text": "done"}, finish_reason="end_turn")


def _client(tmp_path):
    db = tmp_path / "t.db"
    store = SpanStore(db)
    _run(store, "with_tool", call_tool=True)
    _run(store, "with_tool_2", call_tool=True)   # identical decisions
    _run(store, "no_tool", call_tool=False)      # diverges: answers directly
    store.close()
    return TestClient(create_app(str(db)))


def test_identical_traces_align(tmp_path):
    r = _client(tmp_path).get("/api/diff", params={"a": "with_tool", "b": "with_tool_2"})
    assert r.status_code == 200
    d = r.json()
    assert d["aligned"] is True
    assert d["first_divergence"] is None
    assert d["a"]["decisions"][0]["tool_name"] == "get_weather"


def test_divergent_traces_report_first_divergence(tmp_path):
    r = _client(tmp_path).get("/api/diff", params={"a": "with_tool", "b": "no_tool"})
    d = r.json()
    assert d["aligned"] is False
    assert d["first_divergence"] == 0
    assert d["a"]["decisions"][0]["kind"] == "tool_call"
    assert d["b"]["decisions"][0]["kind"] == "final"
    assert "diverge" in d["summary"]


def test_diff_missing_trace_404(tmp_path):
    r = _client(tmp_path).get("/api/diff", params={"a": "with_tool", "b": "nope"})
    assert r.status_code == 404


def _proxy_run(store, trace_id, calls, client="claude-code"):
    """Capture a run the way the PROXY does: mcp_call spans, JSON-RPC envelopes."""
    init = Span(
        trace_id=trace_id,
        method="initialize",
        request={"jsonrpc": "2.0", "method": "initialize",
                 "params": {"clientInfo": {"name": client, "version": "1.0"}}},
        response={"result": {"protocolVersion": "2024-11-05"}},
    )
    init.close()
    store.write(init)
    for name, args in calls:
        span = Span(
            trace_id=trace_id,
            method="tools/call",
            tool_name=name,
            request={"jsonrpc": "2.0", "method": "tools/call",
                     "params": {"name": name, "arguments": args}},
            response={"result": {"ok": True}},
        )
        span.close()
        store.write(span)


def _proxy_client(tmp_path):
    db = tmp_path / "p.db"
    store = SpanStore(db)
    _proxy_run(store, "px_read", [("read_text_file", {"path": "README.md"})])
    _proxy_run(store, "px_read_2", [("read_text_file", {"path": "README.md"})])
    _proxy_run(store, "px_list", [("list_directory", {"path": "/src"}),
                                  ("get_file_info", {"path": "p"})])
    store.close()
    return TestClient(create_app(str(db)))


def test_proxy_traces_with_different_calls_report_divergence(tmp_path):
    """Regression: these compared as "identical trajectory" at HTTP 200.

    The proxy is the zero-code-change capture path, so these are the traces most
    users have — a silent false `aligned` here reads as "both models behaved the
    same" when they did entirely different things.
    """
    r = _proxy_client(tmp_path).get("/api/diff", params={"a": "px_read", "b": "px_list"})
    assert r.status_code == 200
    d = r.json()
    assert d["aligned"] is False
    assert d["first_divergence"] == 0
    assert d["a"]["decisions"][0]["tool_name"] == "read_text_file"
    assert d["b"]["decisions"][0]["tool_name"] == "list_directory"


def test_diff_labels_proxy_sides_with_their_client(tmp_path):
    """The frontend header reads `label || model_id || trace_id` — give it a label."""
    db = tmp_path / "l.db"
    store = SpanStore(db)
    _proxy_run(store, "px_a", [("read_text_file", {"path": "a"})], client="claude-code")
    _proxy_run(store, "px_b", [("list_directory", {"path": "b"})], client="cursor")
    store.close()

    d = TestClient(create_app(str(db))).get(
        "/api/diff", params={"a": "px_a", "b": "px_b"}
    ).json()
    assert d["a"]["label"] == "claude-code"
    assert d["b"]["label"] == "cursor"
    assert d["a"]["model_id"] is None  # the client is reported as a label, not a model
    assert "claude-code →" in d["summary"] and "cursor →" in d["summary"]


def test_sdk_diff_still_reports_model_id(tmp_path):
    d = _client(tmp_path).get("/api/diff", params={"a": "with_tool", "b": "no_tool"}).json()
    assert d["a"]["model_id"] == "m"
    assert d["a"]["label"] is None


def test_proxy_traces_with_identical_calls_match_without_claiming_aligned(tmp_path):
    """Same calls, but the proxy never sees an ending — so bounded, not "identical"."""
    r = _proxy_client(tmp_path).get("/api/diff", params={"a": "px_read", "b": "px_read_2"})
    d = r.json()
    assert d["kind"] == "unknown_terminal"
    assert d["aligned"] is False
    assert d["first_divergence"] is None  # nothing observed actually conflicted
    assert d["comparable_until"] == 1
    assert d["a"]["decisions"][0]["tool_name"] == "read_text_file"


def test_proxy_and_sdk_traces_compare_on_equal_footing(tmp_path):
    """The UI dropdowns list both capture paths, so mixed pairs must work."""
    db = tmp_path / "m.db"
    store = SpanStore(db)
    _run(store, "sdk_run", call_tool=True)  # calls get_weather
    _proxy_run(store, "proxy_run", [("get_weather", {"city": "Paris"})])
    store.close()

    d = TestClient(create_app(str(db))).get(
        "/api/diff", params={"a": "sdk_run", "b": "proxy_run"}
    ).json()
    # Same tool + same arguments from either capture path -> the same decision.
    assert d["a"]["decisions"][0]["tool_name"] == "get_weather"
    assert d["b"]["decisions"][0]["tool_name"] == "get_weather"
    # The SDK run answered; the proxy run's ending was never observed. Whether the
    # proxied agent would also have stopped there is unknowable, so the comparison
    # is bounded rather than declared aligned.
    assert d["kind"] == "unknown_terminal"
    assert d["comparable_until"] == 1
