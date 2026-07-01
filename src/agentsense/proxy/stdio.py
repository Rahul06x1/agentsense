"""Transparent stdio transport for the MCP proxy.

Launches the real MCP server as a subprocess and forwards raw JSON-RPC bytes
UNCHANGED in both directions. A copy of every framed line is handed to the tap
for tracing; the forwarded bytes are never modified or re-serialized.

Design rules proven in Phase 0:
  - Forward raw bytes unchanged; parse only a copy (do not round-trip).
  - Nothing but JSON-RPC goes to stdout; all logs go to stderr / file.
  - Forward first, observe second — tap work can never delay or corrupt the wire.

The `run_stdio_proxy` signature is the transport seam: an HTTP/SSE transport
(fast-follow) implements the same "forward + tap" contract without touching the
tap or store.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from agentsense.proxy.tap import C2S, S2C, TraceTap

log = logging.getLogger("agentsense.proxy.stdio")


async def _pump(
    direction: str,
    src: asyncio.StreamReader,
    dst: asyncio.StreamWriter,
    tap: TraceTap,
) -> None:
    """Forward src->dst line by line, unchanged, tapping a copy of each line."""
    while True:
        line = await src.readline()
        if not line:
            break
        dst.write(line)  # forward UNCHANGED, first
        await dst.drain()
        tap.observe(direction, line)  # trace a copy, second
    try:
        dst.write_eof()
    except (OSError, RuntimeError):
        pass


async def run_stdio_proxy(server_cmd: list[str], tap: TraceTap) -> int:
    """Proxy this process's stdin/stdout to `server_cmd`'s stdio. Returns exit code."""
    proc = await asyncio.create_subprocess_exec(
        *server_cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr,  # server's own logs pass through to our stderr
    )

    loop = asyncio.get_running_loop()

    client_reader = asyncio.StreamReader()
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(client_reader), sys.stdin
    )
    w_transport, w_proto = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    client_writer = asyncio.StreamWriter(w_transport, w_proto, None, loop)

    log.info("stdio proxy forwarding to: %s", " ".join(server_cmd))

    await asyncio.gather(
        _pump(C2S, client_reader, proc.stdin, tap),  # requests
        _pump(S2C, proc.stdout, client_writer, tap),  # responses
    )
    tap.flush()
    return await proc.wait()
