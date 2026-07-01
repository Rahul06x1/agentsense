"""tracekit CLI.

v0 exposes the proxy:

    tracekit proxy --db traces.db -- npx -y @modelcontextprotocol/server-filesystem /tmp

The client's MCP config points at this command instead of the real server; the
proxy forwards transparently and traces every message to the SQLite store.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

from tracekit.proxy.stdio import run_stdio_proxy
from tracekit.proxy.tap import TraceTap
from tracekit.store.sqlite import SpanStore


def _setup_logging(log_file: str | None) -> None:
    # NEVER log to stdout — stdout is the JSON-RPC channel.
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def _split_server_cmd(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv at the first `--`: (our args, server command)."""
    if "--" not in argv:
        return argv, []
    i = argv.index("--")
    return argv[:i], argv[i + 1 :]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    our_args, server_cmd = _split_server_cmd(argv)

    parser = argparse.ArgumentParser(prog="tracekit")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("proxy", help="Transparent MCP proxy that traces all traffic")
    p.add_argument("--db", default="tracekit.db", help="SQLite trace store path")
    p.add_argument("--trace-id", default=None, help="Trace id (default: random)")
    p.add_argument("--log-file", default=None, help="Mirror logs to this file")

    args = parser.parse_args(our_args)

    if args.command == "proxy":
        if not server_cmd:
            parser.error("proxy requires a server command after `--`")
        _setup_logging(args.log_file)
        trace_id = args.trace_id or uuid.uuid4().hex
        Path(args.db).parent.mkdir(parents=True, exist_ok=True)
        store = SpanStore(args.db)
        tap = TraceTap(store, trace_id)
        logging.getLogger("tracekit").info("trace_id=%s db=%s", trace_id, args.db)
        try:
            return asyncio.run(run_stdio_proxy(server_cmd, tap))
        finally:
            store.close()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
