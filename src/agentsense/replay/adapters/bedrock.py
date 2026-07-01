"""AWS Bedrock adapter (Converse API) — the dev model client.

boto3 is imported lazily so the package (and the proxy) works without it; install
with the `replay` extra. Auth is AWS SSO:

    aws sso login --profile <your-aws-profile>
    export AWS_PROFILE=<your-aws-profile> AWS_REGION=eu-west-1

Converse specifics handled here:
  - tools declared under toolConfig.tools[].toolSpec with inputSchema.json
  - a call is a `toolUse` content block; results go back as a `toolResult` block
  - usage (incl. cacheReadInputTokens / cacheWriteInputTokens) and
    metrics.latencyMs are captured into the normalized response
"""

from __future__ import annotations

from typing import Any

from agentsense.replay.adapters.base import ModelAdapter
from agentsense.replay.types import ModelResponse, ToolCall, ToolSpec, Turn, Usage

DEV_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"


class BedrockAdapter(ModelAdapter):
    def __init__(self, model_id: str = DEV_MODEL, region: str = "eu-west-1", client=None):
        self.model_id = model_id
        self.region = region
        self._client = client  # injectable for tests

    @property
    def client(self):
        if self._client is None:
            import boto3  # lazy: only needed when actually calling the model

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def converse(
        self, system: str | None, turns: list[Turn], tools: list[ToolSpec]
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [_turn_to_converse(t) for t in turns],
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if tools:
            kwargs["toolConfig"] = {"tools": [_toolspec_to_converse(t) for t in tools]}

        resp = self.client.converse(**kwargs)
        return _parse_converse(resp)


def _turn_to_converse(turn: Turn) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if turn.text:
        content.append({"text": turn.text})
    for tc in turn.tool_calls:
        content.append({"toolUse": {"toolUseId": tc.id, "name": tc.name, "input": tc.input}})
    for tr in turn.tool_results:
        content.append(
            {"toolResult": {"toolUseId": tr.id, "content": [_result_block(tr.result)]}}
        )
    return {"role": turn.role, "content": content}


def _result_block(result: Any) -> dict[str, Any]:
    # Converse accepts json or text result blocks; use json for structured data.
    if isinstance(result, str):
        return {"text": result}
    return {"json": result}


def _toolspec_to_converse(spec: ToolSpec) -> dict[str, Any]:
    return {
        "toolSpec": {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": {"json": spec.input_schema},
        }
    }


def _parse_converse(resp: dict[str, Any]) -> ModelResponse:
    resp = {k: v for k, v in resp.items() if k != "ResponseMetadata"}
    blocks = resp.get("output", {}).get("message", {}).get("content", [])
    text = "".join(b.get("text", "") for b in blocks if "text" in b)
    tool_calls = [
        ToolCall(id=b["toolUse"]["toolUseId"], name=b["toolUse"]["name"],
                 input=b["toolUse"].get("input", {}))
        for b in blocks
        if "toolUse" in b
    ]
    u = resp.get("usage", {})
    usage = Usage(
        input_tokens=u.get("inputTokens"),
        output_tokens=u.get("outputTokens"),
        cache_read_input_tokens=u.get("cacheReadInputTokens"),
        cache_write_input_tokens=u.get("cacheWriteInputTokens"),
        raw=u,
    )
    return ModelResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason=resp.get("stopReason"),
        usage=usage,
        latency_ms=resp.get("metrics", {}).get("latencyMs"),
        raw=resp,
    )
