"""UI diff endpoint: decision-level trajectory diff between two captured traces."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from tracekit.sdk import Tracer  # noqa: E402
from tracekit.store.sqlite import SpanStore  # noqa: E402
from tracekit.ui.app import create_app  # noqa: E402


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
