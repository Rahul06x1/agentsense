"""Offline replay + trajectory-diff demo — no SSO, no network, no cost.

Two "models" (scripted) replay the same recorded tool results; the diff shows
where their decisions diverge. Run:  uv run python examples/replay_scripted.py
"""

from agentsense.replay import (
    ModelResponse,
    Recording,
    ScriptedAdapter,
    ToolCall,
    ToolSpec,
    diff_trajectories,
    replay,
)

rec = Recording(
    question="What should I pack for Paris this week?",
    tools=[
        ToolSpec("get_weather", "current weather", {"type": "object"}),
        ToolSpec("get_forecast", "multi-day forecast", {"type": "object"}),
    ],
)
rec.record("get_weather", {"city": "Paris"}, {"temp_c": 18, "sky": "sunny"})
rec.record("get_forecast", {"city": "Paris", "days": 7}, {"rain_days": 3})

# "Haiku" checks only current weather; "Opus" reasons it needs the week's forecast.
haiku = ScriptedAdapter(
    [
        ModelResponse(tool_calls=[ToolCall("1", "get_weather", {"city": "Paris"})]),
        ModelResponse(text="It's sunny and 18C — light layers."),
    ],
    model_id="haiku-4.5",
)
opus = ScriptedAdapter(
    [
        ModelResponse(tool_calls=[ToolCall("1", "get_forecast",
                                           {"city": "Paris", "days": 7})]),
        ModelResponse(text="3 rainy days ahead — pack a jacket and umbrella."),
    ],
    model_id="opus-4.8",
)

a = replay(rec, haiku)
b = replay(rec, opus)

for traj in (a, b):
    calls = [f"{s.tool_name}" for s in traj.decisions if s.tool_name]
    print(f"{traj.model_id:>10}: tools={calls}  answer={traj.final_text!r}")

print("\n" + diff_trajectories(a, b).summary)
