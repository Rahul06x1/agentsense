"""Python capture SDK.

Wraps agent steps for reasoning-level capture using OpenTelemetry GenAI semantic
conventions (attribute names), writing through the same SpanStore — and therefore
the same redaction code path — as the proxy.

    from agentsense.sdk import Tracer
    from agentsense.store import SpanStore

    tracer = Tracer(SpanStore("traces.db"))
    with tracer.session("booking-agent") as s:
        s.step("plan", reasoning=..., model="claude-opus-4-8")
        s.tool_call("search_flights", args=..., result=..., cost=...)
"""

from agentsense.sdk.tracer import Session, Tracer

__all__ = ["Tracer", "Session"]
