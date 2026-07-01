"""Adapters: normalized types <-> vendor wire formats, no network calls.

Fake clients capture the request kwargs and return canned responses, so we verify
translation + parsing (incl. Bedrock usage/metrics capture) without SSO/Bedrock."""

from types import SimpleNamespace

from agentsense.replay import BedrockAdapter, OpenAICompatAdapter, ToolSpec, Turn
from agentsense.replay.types import ToolCall, ToolResult

WEATHER = ToolSpec("get_weather", "weather", {"type": "object",
                                              "properties": {"city": {"type": "string"}}})


class FakeBedrockClient:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def converse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_bedrock_translation_and_usage_capture():
    canned = {
        "output": {"message": {"role": "assistant", "content": [
            {"text": "checking"},
            {"toolUse": {"toolUseId": "tu1", "name": "get_weather",
                         "input": {"city": "Paris"}}},
        ]}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": 12, "outputTokens": 6,
                  "cacheReadInputTokens": 4, "cacheWriteInputTokens": 1},
        "metrics": {"latencyMs": 321},
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }
    client = FakeBedrockClient(canned)
    adapter = BedrockAdapter(client=client)

    turns = [
        Turn(role="user", text="Weather in Paris?"),
        Turn(role="assistant", tool_calls=[ToolCall("tu0", "get_weather", {"city": "Paris"})]),
        Turn(role="user",
             tool_results=[ToolResult("tu0", "get_weather", {"temp_c": 18})]),
    ]
    resp = adapter.converse(system="be terse", turns=turns, tools=[WEATHER])

    # Request translation into Converse shape.
    sent = client.kwargs
    assert sent["system"] == [{"text": "be terse"}]
    assert sent["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_weather"
    assert sent["messages"][1]["content"][0]["toolUse"]["toolUseId"] == "tu0"
    assert sent["messages"][2]["content"][0]["toolResult"]["content"][0]["json"] == {"temp_c": 18}

    # Response parsing incl. usage (cache tokens) + latency; raw drops HTTP wrapper.
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.stop_reason == "tool_use"
    assert resp.usage.cache_read_input_tokens == 4
    assert resp.latency_ms == 321
    assert "ResponseMetadata" not in resp.raw


class FakeOpenAIClient:
    def __init__(self, response):
        self._response = response
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self._response


def test_openai_translation_and_parse():
    message = SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(
            id="c1", type="function",
            function=SimpleNamespace(name="get_weather", arguments='{"city": "Paris"}'),
        )],
    )
    usage = SimpleNamespace(prompt_tokens=9, completion_tokens=4,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=2),
                            model_dump=lambda: {"prompt_tokens": 9})
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=usage, model_dump=lambda: {"ok": True},
    )
    client = FakeOpenAIClient(response)
    adapter = OpenAICompatAdapter(model_id="gpt-x", client=client)

    turns = [Turn(role="user", text="Weather in Paris?")]
    resp = adapter.converse(system="sys", turns=turns, tools=[WEATHER])

    sent = client.kwargs
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert sent["tools"][0]["function"]["name"] == "get_weather"
    assert resp.tool_calls[0].input == {"city": "Paris"}
    assert resp.usage.cache_read_input_tokens == 2
    assert resp.stop_reason == "tool_calls"
