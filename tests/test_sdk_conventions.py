"""Capture SDK: OTel GenAI attributes on llm_call spans + usage capture."""

from tracekit.model.spans import LLM_CALL
from tracekit.sdk import Tracer
from tracekit.sdk.conventions import (
    CACHE_READ_INPUT_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    infer_system,
)
from tracekit.store.sqlite import SpanStore


def test_llm_call_carries_genai_attributes(tmp_path):
    store = SpanStore(tmp_path / "t.db")
    with Tracer(store).session("a", trace_id="s1") as s:
        s.llm_call(
            "claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "x"}],
            response={"text": "hello"},
            usage={"input_tokens": 10, "output_tokens": 3, "cache_read_input_tokens": 4},
            finish_reason="end_turn",
            cost=0.0001,
        )
    llm = next(sp for sp in store.spans_for_trace("s1") if sp.kind == LLM_CALL)
    a = llm.attributes
    assert a[GEN_AI_REQUEST_MODEL] == "claude-haiku-4-5"
    assert a[GEN_AI_SYSTEM] == "anthropic"
    assert a[GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert a[CACHE_READ_INPUT_TOKENS] == 4
    assert a["tracekit.cost_usd"] == 0.0001
    # Conversation stored whole (replayable): messages + tools + response survive.
    assert llm.request["messages"][0]["content"] == "hi"
    assert llm.request["tools"][0]["name"] == "x"
    assert llm.response["response"]["text"] == "hello"
    store.close()


def test_infer_system():
    assert infer_system("eu.anthropic.claude-opus-4-8") == "anthropic"
    assert infer_system("gpt-4o") == "openai"
    assert infer_system("gemini-3.5-flash") == "gcp.gemini"
    assert infer_system("mystery-model") is None
