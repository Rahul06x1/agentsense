"""End-to-end: instrument a run (SDK) -> reconstruct -> replay a different model -> diff.

Offline, no creds:  uv run python examples/capture_then_replay.py

This is the full loop: capture what an agent actually did, then ask "would a
different model have decided differently, given the same tool outputs?"
"""

import tempfile
from pathlib import Path

from agentsense.replay import (
    ModelResponse,
    Recording,
    ScriptedAdapter,
    captured_trajectory,
    diff_trajectories,
    replay,
)
from agentsense.sdk import Tracer
from agentsense.store import SpanStore

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "current weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}

store = SpanStore(Path(tempfile.mkdtemp()) / "traces.db")

# 1) An instrumented agent run: it checked the weather, then answered.
with Tracer(store).session("weather-agent", trace_id="run") as s:
    s.llm_call("claude-haiku-4-5",
               messages=[{"role": "user", "content": "What's the weather in Paris?"}],
               tools=[WEATHER], response={"text": ""}, finish_reason="tool_use")
    s.tool_call("get_weather", args={"city": "Paris"}, result={"temp_c": 18})
    s.llm_call("claude-haiku-4-5", messages=[{"role": "user", "content": "..."}],
               response={"text": "It's 18C in Paris."}, finish_reason="end_turn")

# 2) Reconstruct a replayable recording straight from the captured spans.
rec = Recording.from_sdk_trace(store, "run")
print(f"reconstructed: question={rec.question!r} model={rec.model_id} "
      f"tools={[t.name for t in rec.tools]}")

# 3) Replay against a 'different model' that skips the tool and guesses.
guesser = ScriptedAdapter([ModelResponse(text="Probably mild.")], model_id="rushed-model")
original = captured_trajectory(store, "run")
replayed = replay(rec, guesser)

# 4) Diff what the agent DID vs what the other model WOULD do.
print("\n" + diff_trajectories(original, replayed).summary)
store.close()
