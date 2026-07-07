"""UI live-replay endpoint: POST /api/replay reruns a captured trace and diffs it.

The happy path monkeypatches the adapter so no real model call is made."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from agentsense.model.spans import Span  # noqa: E402
from agentsense.replay.types import ModelResponse  # noqa: E402
from agentsense.sdk import Tracer  # noqa: E402
from agentsense.store.sqlite import SpanStore  # noqa: E402
from agentsense.ui import app as ui_app  # noqa: E402
from agentsense.ui.app import create_app  # noqa: E402


class FakeAdapter:
    """Stands in for a model client: answers directly (no tool call)."""

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id

    def converse(self, system, turns, tools):
        return ModelResponse(text="answered without checking")


def _seed_sdk_trace(db):
    store = SpanStore(db)
    with Tracer(store).session("agent", trace_id="run1") as s:
        s.llm_call("claude-haiku-4-5",
                   messages=[{"role": "user", "content": "weather in Paris?"}],
                   tools=[{"type": "function", "function": {"name": "get_weather",
                           "description": "w", "parameters": {"type": "object"}}}],
                   response={"text": ""}, finish_reason="tool_use")
        s.tool_call("get_weather", args={"city": "Paris"}, result={"t": 18})
        s.llm_call("claude-haiku-4-5", messages=[{"role": "user", "content": "..."}],
                   response={"text": "18C"}, finish_reason="end_turn")
    # A proxy-only trace (no llm_call) — not replayable.
    m = Span(trace_id="proxy1", method="tools/call", tool_name="read_file",
             request={"params": {"arguments": {}}}, response={"result": {}})
    m.close()
    store.write(m)
    store.close()


def test_live_replay_returns_diff(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed_sdk_trace(db)
    monkeypatch.setattr(ui_app, "OpenAICompatAdapter", FakeAdapter)
    client = TestClient(create_app(str(db)))

    r = client.post("/api/replay", json={"trace_id": "run1", "model": "gpt-x"})
    assert r.status_code == 200
    d = r.json()
    # Captured agent called the tool; the replay model answered directly -> diverge at 0.
    assert d["aligned"] is False
    assert d["first_divergence"] == 0
    assert d["a"]["decisions"][0]["kind"] == "tool_call"
    assert d["b"]["decisions"][0]["kind"] == "final"
    assert d["replay"]["model"] == "gpt-x"
    assert "captured" in d["a"]["label"] and "replay" in d["b"]["label"]


def test_replay_missing_trace_404(tmp_path):
    db = tmp_path / "t.db"
    _seed_sdk_trace(db)
    r = TestClient(create_app(str(db))).post(
        "/api/replay", json={"trace_id": "nope", "model": "gpt-x"})
    assert r.status_code == 404


def test_replay_non_replayable_trace_400(tmp_path):
    db = tmp_path / "t.db"
    _seed_sdk_trace(db)
    # proxy1 has no llm_call, so no question can be reconstructed.
    r = TestClient(create_app(str(db))).post(
        "/api/replay", json={"trace_id": "proxy1", "model": "gpt-x"})
    assert r.status_code == 400
    assert "recording" in r.json()["detail"]


# ---- fork policy + honest diff fields exposed over the API --------------------

from agentsense.replay.types import ToolCall  # noqa: E402


class FakeForkAdapter:
    """Calls an unrecorded tool (get_weather/Berlin — recording has Paris), then answers."""

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self._i = 0

    def converse(self, system, turns, tools):
        self._i += 1
        if self._i == 1:
            return ModelResponse(tool_calls=[ToolCall(id="t1", name="get_weather",
                                                      input={"city": "Berlin"})])
        return ModelResponse(text="guessed")


def test_replay_stub_policy_continues_over_api(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed_sdk_trace(db)
    monkeypatch.setattr(ui_app, "OpenAICompatAdapter", FakeForkAdapter)
    client = TestClient(create_app(str(db)))

    r = client.post("/api/replay",
                    json={"trace_id": "run1", "model": "gpt-x", "on_unrecorded": "stub"})
    assert r.status_code == 200
    d = r.json()
    assert d["replay"]["on_unrecorded"] == "stub"
    assert d["replay"]["stubbed_calls"] == 1
    assert d["kind"] == "diverged"          # different args -> a real decision diff
    assert "comparable_until" in d and "redaction_suspect" in d


def test_replay_stop_policy_reports_unresolvable_fork(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed_sdk_trace(db)
    monkeypatch.setattr(ui_app, "OpenAICompatAdapter", FakeForkAdapter)
    client = TestClient(create_app(str(db)))

    r = client.post("/api/replay",
                    json={"trace_id": "run1", "model": "gpt-x", "on_unrecorded": "stop"})
    d = r.json()
    assert d["kind"] == "unresolvable_fork"
    assert d["replay"]["stopped_reason"] == "unrecorded_tool_call"


def test_replay_live_without_executor_is_400(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _seed_sdk_trace(db)
    monkeypatch.setattr(ui_app, "OpenAICompatAdapter", FakeForkAdapter)
    client = TestClient(create_app(str(db)))

    r = client.post("/api/replay",
                    json={"trace_id": "run1", "model": "gpt-x", "on_unrecorded": "live"})
    assert r.status_code == 400
    assert "live_tool" in r.json()["detail"]
