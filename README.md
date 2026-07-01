# tracekit

MCP-native agent observability & debugging tool. **v0** ships the differentiator
first: a transparent MCP proxy that traces every protocol message with **zero code
change to the agent**, redacts PII deterministically on the write path, and stores
whole message objects (no field whitelist) in local SQLite.

> Working name — see `../mbo-architecture.md` for the design and locked v0 scope.

## What's here (v0)

| Component | Module | Status |
|---|---|---|
| MCP proxy (stdio, transparent) | `proxy/` | ✅ built first |
| Deterministic PII redaction | `redaction/` | ✅ write-path |
| SQLite trace store (whole objects) | `store/` | ✅ |
| Python capture SDK | `sdk/` | fast-follow |
| Mocked replay + trajectory diff | `replay/` | fast-follow |

## Design rules (proven in Phase 0, enforced here)

- Proxy forwards raw bytes **unchanged**; only a *copy* is parsed for tracing.
- Logs go to **stderr / file, never stdout** (stdout is the JSON-RPC channel).
- Trace store preserves **unknown/vendor fields** — whole objects, no whitelist.
- Redaction is **deterministic** (hash-derived tokens) so replay aligns.

## Quick start

```bash
uv sync --group dev

# Point your MCP client's server config at this command instead of the real server.
# The proxy forwards transparently and traces to SQLite.
uv run tracekit proxy --db traces.db -- \
    npx -y @modelcontextprotocol/server-filesystem /tmp
```

## Tests

```bash
uv run pytest            # unit tests run offline; the proxy has no model dependency
uv run ruff check .
```

`test_proxy_transparency.py` is an integration test that drives `server-filesystem`
through the proxy with a real MCP client; it self-skips if `npx` is unavailable.

## Model access (replay only — not needed for the proxy)

The replay engine (fast-follow) uses a **pluggable** model client: AWS Bedrock
(Converse API) for dev, OpenAI-compatible for the OSS release. Before running
replay: `aws sso login --profile coredev` (temporary SSO creds).
