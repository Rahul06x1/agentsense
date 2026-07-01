"""Integration: a real MCP client drives server-filesystem THROUGH the proxy.

Proves Phase 0 rows 1-4 for the real build: the handshake and a tool call succeed
transparently, and the proxy captures the tool call at the protocol level.

Skipped automatically when npx is unavailable.
"""

import shutil
import sys

import pytest

pytestmark = pytest.mark.skipif(shutil.which("npx") is None, reason="npx not available")


@pytest.mark.asyncio
async def test_client_drives_filesystem_through_proxy(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from tracekit.store.sqlite import SpanStore

    allowed = tmp_path / "workspace"
    allowed.mkdir()
    (allowed / "hello.txt").write_text("hello from tracekit")
    db = tmp_path / "trace.db"

    # The client launches the PROXY; the proxy launches the real server.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m", "tracekit.cli", "proxy",
            "--db", str(db), "--trace-id", "itest",
            "--", "npx", "-y", "@modelcontextprotocol/server-filesystem", str(allowed),
        ],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # handshake succeeds through the proxy
            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            assert tool_names  # server advertised tools transparently

            # list_directory is stable across server-filesystem versions.
            result = await session.call_tool("list_directory", {"path": str(allowed)})
            assert result.content  # real tool result forwarded back
            text = "".join(getattr(c, "text", "") for c in result.content)
            assert "hello.txt" in text

    # Protocol-level capture: the tools/call was recorded to the trace store.
    store = SpanStore(db)
    spans = store.spans_for_trace("itest")
    tool_calls = [s for s in spans if s.method == "tools/call"]
    assert any(s.tool_name == "list_directory" for s in tool_calls)
    assert all(s.latency_ms is not None for s in tool_calls)
    store.close()
