"""Replay engine.

Mocked, deterministic-redaction-aware replay + trajectory diff. The model client
is pluggable behind one adapter: AWS Bedrock (Converse API, boto3) for dev;
OpenAI-compatible (Gemini/Ollama/OpenAI) for the OSS release. Bedrock usage
(incl. cache tokens) + metrics.latencyMs are captured into the normalized Usage.
"""

from agentsense.replay.adapters import (
    BedrockAdapter,
    ModelAdapter,
    OpenAICompatAdapter,
    ScriptedAdapter,
)
from agentsense.replay.capture import captured_trajectory
from agentsense.replay.diff import TrajectoryDiff, diff_trajectories
from agentsense.replay.engine import replay
from agentsense.replay.recording import Recording
from agentsense.replay.trajectory import Step, Trajectory
from agentsense.replay.types import (
    ModelResponse,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
    Usage,
)

__all__ = [
    "replay",
    "Recording",
    "captured_trajectory",
    "Trajectory",
    "Step",
    "diff_trajectories",
    "TrajectoryDiff",
    "ModelAdapter",
    "ScriptedAdapter",
    "BedrockAdapter",
    "OpenAICompatAdapter",
    "ModelResponse",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Turn",
    "Usage",
]
