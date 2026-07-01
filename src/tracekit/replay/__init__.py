"""Replay engine.

Mocked, deterministic-redaction-aware replay + trajectory diff. The model client
is pluggable behind one adapter: AWS Bedrock (Converse API, boto3) for dev;
OpenAI-compatible (Gemini/Ollama/OpenAI) for the OSS release. Bedrock usage
(incl. cache tokens) + metrics.latencyMs are captured into the normalized Usage.
"""

from tracekit.replay.adapters import (
    BedrockAdapter,
    ModelAdapter,
    OpenAICompatAdapter,
    ScriptedAdapter,
)
from tracekit.replay.diff import TrajectoryDiff, diff_trajectories
from tracekit.replay.engine import replay
from tracekit.replay.recording import Recording
from tracekit.replay.trajectory import Step, Trajectory
from tracekit.replay.types import (
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
