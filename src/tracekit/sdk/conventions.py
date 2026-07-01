"""OpenTelemetry GenAI semantic-convention attribute names.

We adopt the OTel GenAI *naming* conventions for span attributes so traces
interoperate with Langfuse/Phoenix/etc., without taking a dependency on the
OpenTelemetry SDK in v0 (a real OTLP exporter is post-v0). Only the attribute
keys matter here; they live in `Span.attributes`.
"""

from __future__ import annotations

from typing import Any

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

# Non-standard extras (tracekit namespace) — no OTel convention exists yet.
CACHE_READ_INPUT_TOKENS = "tracekit.usage.cache_read_input_tokens"
CACHE_WRITE_INPUT_TOKENS = "tracekit.usage.cache_write_input_tokens"
COST_USD = "tracekit.cost_usd"


def infer_system(model: str) -> str | None:
    """Best-effort gen_ai.system from a model id (anthropic, openai, ...)."""
    m = model.lower()
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "gpt" in m or "openai" in m or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if "gemini" in m:
        return "gcp.gemini"
    return None


def llm_attributes(
    model: str,
    usage: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    system: str | None = None,
    cost: float | None = None,
) -> dict[str, Any]:
    """Build gen_ai.* attributes for an llm_call span.

    `usage` is a plain dict (e.g. from replay's Usage): input_tokens,
    output_tokens, cache_read_input_tokens, cache_write_input_tokens.
    """
    attrs: dict[str, Any] = {
        GEN_AI_OPERATION_NAME: "chat",
        GEN_AI_REQUEST_MODEL: model,
    }
    sys = system or infer_system(model)
    if sys:
        attrs[GEN_AI_SYSTEM] = sys
    if usage:
        if usage.get("input_tokens") is not None:
            attrs[GEN_AI_USAGE_INPUT_TOKENS] = usage["input_tokens"]
        if usage.get("output_tokens") is not None:
            attrs[GEN_AI_USAGE_OUTPUT_TOKENS] = usage["output_tokens"]
        if usage.get("cache_read_input_tokens") is not None:
            attrs[CACHE_READ_INPUT_TOKENS] = usage["cache_read_input_tokens"]
        if usage.get("cache_write_input_tokens") is not None:
            attrs[CACHE_WRITE_INPUT_TOKENS] = usage["cache_write_input_tokens"]
    if finish_reason is not None:
        attrs[GEN_AI_RESPONSE_FINISH_REASONS] = [finish_reason]
    if cost is not None:
        attrs[COST_USD] = cost
    return attrs
