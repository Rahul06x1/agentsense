"""Live replay + model-swap trajectory diff against AWS Bedrock (Converse).

Records one tool result, then replays the SAME question against two models and
diffs their decision trajectories. No live tool ever fires during replay.

Prereqs (temporary SSO creds — re-run when they expire):
    aws sso login --profile coredev
    export AWS_PROFILE=coredev AWS_REGION=eu-west-1
Then:
    uv run --extra replay python examples/replay_bedrock.py
"""

from tracekit.replay import (
    BedrockAdapter,
    Recording,
    ToolSpec,
    diff_trajectories,
    replay,
)

DEV_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"
SWAP_MODEL = "eu.anthropic.claude-sonnet-4-6"  # the model-swap comparison

recording = Recording(
    question="What is the weather in Paris? Use the tool, then answer in one sentence.",
    tools=[
        ToolSpec(
            name="get_weather",
            description="Get current weather for a city",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    ],
)
# The one recorded tool result the engine will inject during replay.
recording.record("get_weather", {"city": "Paris"}, {"temp_c": 18, "sky": "sunny"})


def run(model_id: str):
    traj = replay(recording, BedrockAdapter(model_id=model_id))
    tools = [s.tool_name for s in traj.decisions if s.tool_name]
    print(f"\n=== {model_id} ===")
    print(f"tool decisions : {tools}")
    print(f"final answer   : {traj.final_text!r}")
    print(f"tokens in/out  : {traj.total_input_tokens}/{traj.total_output_tokens}")
    print(f"stopped_reason : {traj.stopped_reason}")
    return traj


if __name__ == "__main__":
    a = run(DEV_MODEL)
    b = run(SWAP_MODEL)
    print("\n--- trajectory diff ---")
    print(diff_trajectories(a, b).summary)
