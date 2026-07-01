"""Model adapter interface — the one seam the replay engine talks through.

An adapter translates the normalized conversation (list[Turn] + list[ToolSpec])
into a provider's wire format, calls the model, and returns a normalized
ModelResponse. Swapping Bedrock for an OpenAI-compatible endpoint is a one-line
change at the call site; the engine is untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentsense.replay.types import ModelResponse, ToolSpec, Turn


class ModelAdapter(ABC):
    """Base class for pluggable model clients."""

    #: Human-readable id of the model this adapter is bound to.
    model_id: str

    @abstractmethod
    def converse(
        self,
        system: str | None,
        turns: list[Turn],
        tools: list[ToolSpec],
    ) -> ModelResponse:
        """Send one request and return the normalized response."""
        raise NotImplementedError
