"""captured_trajectory over PROXY (`mcp_call`) spans.

Regression: `captured_trajectory` only understood the capture SDK's span shapes,
so a proxy trace reconstructed to a single empty `final` step. Any two proxy
traces then compared as "identical trajectory" — the Compare view reported
`aligned` for runs with nothing in common, and did so with HTTP 200 rather than
an error, so nothing surfaced the loss.
"""

from agentsense.model.spans import Span
from agentsense.replay import captured_trajectory, diff_trajectories
from agentsense.replay.trajectory import FINAL, TOOL_CALL
from agentsense.sdk import Tracer
from agentsense.store.sqlite import SpanStore


def _proxy_call(store: SpanStore, trace_id: str, name: str, args: dict, result=None):
    """One `tools/call` as the proxy tap records it: raw JSON-RPC envelope."""
    span = Span(
        trace_id=trace_id,
        method="tools/call",
        tool_name=name,
        request={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                 "params": {"name": name, "arguments": args}},
        response={"result": result} if result is not None else None,
    )
    span.close()
    store.write(span)


def _proxy_handshake(store: SpanStore, trace_id: str, client: str | None = "probe"):
    """The non-decision traffic every proxy trace carries around its tool calls."""
    init_params = {"protocolVersion": "2024-11-05"}
    if client is not None:
        init_params["clientInfo"] = {"name": client, "version": "1.0"}
    for method, params, response in (
        ("initialize", init_params, {"result": {"protocolVersion": "2024-11-05"}}),
        ("notifications/initialized", {}, None),
        ("tools/list", {}, {"result": {"tools": [{"name": "read_text_file"}]}}),
        ("roots/list", {}, {"result": {"roots": []}}),
    ):
        span = Span(trace_id=trace_id, method=method,
                    request={"jsonrpc": "2.0", "method": method, "params": params},
                    response=response)
        span.close()
        store.write(span)


def test_proxy_tool_calls_become_decisions(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    _proxy_handshake(store, "p1")
    _proxy_call(store, "p1", "read_text_file", {"path": "README.md"}, {"text": "hi"})
    _proxy_call(store, "p1", "get_file_info", {"path": "pyproject.toml"}, {"size": 12})

    traj = captured_trajectory(store, "p1")
    decisions = traj.decisions

    assert [d.kind for d in decisions] == [TOOL_CALL, TOOL_CALL, FINAL]
    assert [d.tool_name for d in decisions[:2]] == ["read_text_file", "get_file_info"]
    # Arguments come from params.arguments, not the SDK's flat request.arguments.
    assert decisions[0].tool_input == {"path": "README.md"}
    store.close()


def test_handshake_traffic_is_not_a_decision(tmp_path):
    """initialize / tools/list / roots/list are protocol noise, not agent choices."""
    store = SpanStore(tmp_path / "t.db")
    _proxy_handshake(store, "p2")

    traj = captured_trajectory(store, "p2")
    assert [d.kind for d in traj.decisions] == [FINAL]
    store.close()


def test_different_proxy_traces_do_not_report_aligned(tmp_path):
    """The exact false-alignment bug: different tool calls must diverge."""
    store = SpanStore(tmp_path / "t.db")
    _proxy_handshake(store, "a")
    _proxy_call(store, "a", "read_text_file", {"path": "README.md"}, {"text": "hi"})
    _proxy_handshake(store, "b")
    _proxy_call(store, "b", "list_directory", {"path": "/src"}, {"text": "x"})
    _proxy_call(store, "b", "get_file_info", {"path": "p"}, {"size": 1})

    d = diff_trajectories(captured_trajectory(store, "a"), captured_trajectory(store, "b"))
    assert not d.aligned
    assert d.first_divergence == 0
    assert d.a_step.tool_name == "read_text_file"
    assert d.b_step.tool_name == "list_directory"
    store.close()


def test_same_tool_calls_with_different_arguments_diverge(tmp_path):
    """Same tool, different arguments is still a different decision."""
    store = SpanStore(tmp_path / "t.db")
    _proxy_call(store, "a", "read_text_file", {"path": "README.md"}, {"text": "hi"})
    _proxy_call(store, "b", "read_text_file", {"path": "CLAUDE.md"}, {"text": "hi"})

    d = diff_trajectories(captured_trajectory(store, "a"), captured_trajectory(store, "b"))
    assert not d.aligned
    assert d.first_divergence == 0
    store.close()


def test_identical_proxy_traces_still_align(tmp_path):
    """The fix must not turn every comparison into a divergence."""
    store = SpanStore(tmp_path / "t.db")
    for trace_id in ("a", "b"):
        _proxy_handshake(store, trace_id)
        _proxy_call(store, trace_id, "read_text_file", {"path": "README.md"}, {"text": "hi"})

    d = diff_trajectories(captured_trajectory(store, "a"), captured_trajectory(store, "b"))
    assert d.aligned
    assert d.first_divergence is None
    store.close()


def test_failed_tool_call_is_still_a_decision(tmp_path):
    """The agent chose to make the call; whether it succeeded is a separate fact.

    Recording.from_trace_store must skip result-less spans (it needs a result to
    inject); a trajectory must not, or an erroring run looks like it never acted.
    """
    store = SpanStore(tmp_path / "t.db")
    _proxy_call(store, "ok", "read_text_file", {"path": "nope.md"}, {"text": "hi"})
    _proxy_call(store, "err", "read_text_file", {"path": "nope.md"}, result=None)

    traj = captured_trajectory(store, "err")
    assert [d.kind for d in traj.decisions] == [TOOL_CALL, FINAL]
    assert traj.decisions[0].tool_name == "read_text_file"
    # Same decision, different outcome -> the trajectories still align.
    d = diff_trajectories(captured_trajectory(store, "ok"), traj)
    assert d.aligned
    store.close()


def test_proxy_trace_is_labelled_with_its_mcp_client(tmp_path):
    """A proxy trace has no model to name, so it names the client that acted."""
    store = SpanStore(tmp_path / "t.db")
    _proxy_handshake(store, "p", client="claude-code")
    _proxy_call(store, "p", "read_text_file", {"path": "README.md"}, {"text": "hi"})

    traj = captured_trajectory(store, "p")
    assert traj.label == "claude-code"
    assert traj.model_id is None  # a client is not a model; never conflate them
    store.close()


def test_client_label_appears_in_the_divergence_summary(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    _proxy_handshake(store, "a", client="claude-code")
    _proxy_call(store, "a", "read_text_file", {"path": "README.md"}, {"text": "hi"})
    _proxy_handshake(store, "b", client="mcp-inspector")
    _proxy_call(store, "b", "list_directory", {"path": "/src"}, {"text": "x"})

    d = diff_trajectories(captured_trajectory(store, "a"), captured_trajectory(store, "b"))
    assert "claude-code → call read_text_file" in d.summary
    assert "mcp-inspector → call list_directory" in d.summary
    store.close()


def test_missing_client_info_falls_back_to_a_and_b(tmp_path):
    """Not every client sends clientInfo — an absent label must not crash or lie."""
    store = SpanStore(tmp_path / "t.db")
    _proxy_handshake(store, "a", client=None)
    _proxy_call(store, "a", "read_text_file", {"path": "README.md"}, {"text": "hi"})
    _proxy_handshake(store, "b", client=None)
    _proxy_call(store, "b", "list_directory", {"path": "/src"}, {"text": "x"})

    ta = captured_trajectory(store, "a")
    assert ta.label is None
    d = diff_trajectories(ta, captured_trajectory(store, "b"))
    assert "A → call read_text_file" in d.summary
    assert "B → call list_directory" in d.summary
    store.close()


def test_sdk_trace_keeps_naming_its_model(tmp_path):
    """The label must not displace model_id on traces that do have a model."""
    store = SpanStore(tmp_path / "t.db")
    with Tracer(store).session("agent", trace_id="sdk") as s:
        s.llm_call("claude-haiku-4-5", messages=[{"role": "user", "content": "hi"}],
                   response={"text": "done"})
        s.tool_call("get_weather", args={"city": "Paris"}, result={"t": 18})

    traj = captured_trajectory(store, "sdk")
    assert traj.model_id == "claude-haiku-4-5"
    assert traj.label is None
    store.close()


def test_redacted_value_in_a_proxy_divergence_is_flagged_suspect(tmp_path):
    """Redaction runs before storage, so a proxy diff can be redaction-driven too."""
    store = SpanStore(tmp_path / "t.db")
    _proxy_call(store, "a", "send_mail", {"to": "alice@example.com"}, {"ok": True})
    _proxy_call(store, "b", "send_mail", {"to": "bob@example.com"}, {"ok": True})

    d = diff_trajectories(captured_trajectory(store, "a"), captured_trajectory(store, "b"))
    assert not d.aligned
    assert d.redaction_suspect
    store.close()
