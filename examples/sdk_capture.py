"""Capture SDK demo — instrument an agent run, then print the captured trace.

Offline, no creds:  uv run python examples/sdk_capture.py
"""

import tempfile
from pathlib import Path

from tracekit.sdk import Tracer
from tracekit.store import SpanStore

db = Path(tempfile.mkdtemp()) / "traces.db"
store = SpanStore(db)
tracer = Tracer(store)

with tracer.session("booking-agent", trace_id="demo") as s:
    s.step("plan", reasoning="Find flights for alice@example.com, then a hotel.",
           model="claude-opus-4-8")
    s.llm_call("claude-opus-4-8",
               messages=[{"role": "user", "content": "book me a trip to Paris"}],
               response={"text": "searching flights"},
               usage={"input_tokens": 1200, "output_tokens": 40,
                      "cache_read_input_tokens": 256},
               finish_reason="tool_use", cost=0.003, latency_ms=880.0)
    s.tool_call("search_flights", args={"to": "CDG", "pax": "alice@example.com"},
                result={"options": 3}, cost=0.0, latency_ms=120.0)

print(f"captured trace -> {db}\n")
for sp in store.spans_for_trace("demo"):
    indent = "" if sp.kind == "session" else "  "
    extra = ""
    if sp.kind == "llm_call":
        extra = f"  [{sp.attributes.get('gen_ai.usage.input_tokens')} in tokens]"
    if sp.kind == "tool_call":
        # Show that PII in the args was redacted at rest.
        extra = f"  args={sp.request['arguments']}"
    lat = f"{sp.latency_ms:.0f}ms" if sp.latency_ms else "-"
    print(f"{indent}{sp.kind:<9} {sp.name:<22} {lat:>7}{extra}")

store.close()
