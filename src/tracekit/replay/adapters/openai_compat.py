"""OpenAI-compatible adapter — the OSS-release model client.

Works against any OpenAI-compatible endpoint (OpenAI, Gemini's compat endpoint,
Ollama, ...). `openai` is imported lazily; install with the `replay` extra.

This adapter exists to prove the seam: the replay engine drives it with the exact
same normalized Turn/ToolSpec types as the Bedrock adapter — only the wire
translation differs.
"""

from __future__ import annotations

import json
from typing import Any

from tracekit.replay.adapters.base import ModelAdapter
from tracekit.replay.types import ModelResponse, ToolCall, ToolSpec, Turn, Usage


class OpenAICompatAdapter(ModelAdapter):
    def __init__(self, model_id: str, base_url: str | None = None,
                 api_key: str | None = None, client=None):
        self.model_id = model_id
        self._base_url = base_url
        self._api_key = api_key
        self._client = client  # injectable for tests

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI  # lazy

            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def converse(
        self, system: str | None, turns: list[Turn], tools: list[ToolSpec]
    ) -> ModelResponse:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        for turn in turns:
            messages.extend(_turn_to_openai(turn))

        kwargs: dict[str, Any] = {"model": self.model_id, "messages": messages}
        if tools:
            kwargs["tools"] = [_toolspec_to_openai(t) for t in tools]

        resp = self.client.chat.completions.create(**kwargs)
        return _parse_openai(resp)


def _turn_to_openai(turn: Turn) -> list[dict[str, Any]]:
    if turn.tool_results:  # tool results become one message each
        return [
            {"role": "tool", "tool_call_id": tr.id,
             "content": tr.result if isinstance(tr.result, str) else json.dumps(tr.result)}
            for tr in turn.tool_results
        ]
    msg: dict[str, Any] = {"role": turn.role, "content": turn.text or None}
    if turn.tool_calls:
        msg["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": json.dumps(tc.input)}}
            for tc in turn.tool_calls
        ]
    return [msg]


def _toolspec_to_openai(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def _parse_openai(resp: Any) -> ModelResponse:
    choice = resp.choices[0].message
    tool_calls = [
        ToolCall(id=tc.id, name=tc.function.name,
                 input=json.loads(tc.function.arguments or "{}"))
        for tc in (choice.tool_calls or [])
    ]
    u = getattr(resp, "usage", None)
    cached = None
    if u is not None and getattr(u, "prompt_tokens_details", None) is not None:
        cached = getattr(u.prompt_tokens_details, "cached_tokens", None)
    usage = Usage(
        input_tokens=getattr(u, "prompt_tokens", None),
        output_tokens=getattr(u, "completion_tokens", None),
        cache_read_input_tokens=cached,
        raw=u.model_dump() if hasattr(u, "model_dump") else {},
    ) if u is not None else None
    return ModelResponse(
        text=choice.content or "",
        tool_calls=tool_calls,
        stop_reason=resp.choices[0].finish_reason,
        usage=usage,
        raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
    )
