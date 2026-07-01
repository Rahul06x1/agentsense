# Contributing to agentsense

Thanks for your interest! agentsense is an early-stage, local-first tool for making sense
of what AI agents did. Contributions of all kinds are welcome.

## Development setup

```bash
# Install uv: https://docs.astral.sh/uv/
uv sync --group dev --extra ui --extra replay
```

The core (proxy + store + redaction + SDK) has no model or network dependency. The `ui`
extra adds FastAPI/uvicorn; the `replay` extra adds boto3/openai for live model calls.

## Running checks

```bash
uv run pytest         # full suite, runs entirely offline
uv run ruff check .   # lint
uv run ruff format .  # format (if you use it)
```

`tests/test_proxy_transparency.py` drives a real MCP server through the proxy and
self-skips when `npx` is unavailable. Live model calls (Bedrock/OpenAI) are used only in
the `examples/`, never in the test suite.

## Design rules (please preserve)

These are load-bearing invariants proven during validation — keep them intact:

- **The proxy forwards raw bytes unchanged**; only a *copy* is parsed for tracing. Never
  re-serialize a message onto the wire.
- **Nothing but JSON-RPC goes to stdout.** All logs go to stderr / a file.
- **The trace store keeps whole objects** — no field whitelist, so vendor/unknown fields
  survive.
- **Redaction is deterministic** (hash-derived tokens) and runs on the write path, so a
  redacted trace still aligns with its replay. There is one shared redaction module.
- **The replay model client stays pluggable** behind `ModelAdapter`; the engine speaks
  only provider-neutral types.

## Pull requests

1. Add or update tests for your change (they should pass offline).
2. Run `ruff check .` and the test suite.
3. Keep changes focused; describe the "why" in the PR.

By contributing, you agree that your contributions are licensed under the Apache 2.0
License.
