"""Provider-neutral conversation & response types for replay.

The replay engine speaks ONLY these types. Each model adapter translates them to
and from its own wire format (Bedrock Converse, OpenAI chat-completions, ...), so
the engine never sees a vendor-specific shape. This is the "pluggable model
client behind one adapter" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResult:
    """A recorded tool result fed back to the model during replay."""

    id: str
    name: str
    result: Any


@dataclass
class Turn:
    """One conversation turn in the normalized format.

    - user question:  role="user", text set
    - assistant turn: role="assistant", text and/or tool_calls set
    - tool results:   role="user", tool_results set (carried back to the model)
    """

    role: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class ToolSpec:
    """A tool declaration, provider-neutral (JSON Schema for the input)."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Usage:
    """Token usage, incl. Bedrock cache tokens (captured into span fields)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Normalized model response returned by every adapter."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    usage: Usage | None = None
    latency_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
