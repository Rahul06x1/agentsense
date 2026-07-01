"""Scripted adapter — deterministic, offline, no network.

Replays a fixed list of ModelResponses in order. Used by tests and the offline
example to drive the engine without SSO/Bedrock, and to model a "different model"
(a second script) for the trajectory-diff demo.
"""

from __future__ import annotations

from tracekit.replay.adapters.base import ModelAdapter
from tracekit.replay.types import ModelResponse, ToolSpec, Turn


class ScriptedAdapter(ModelAdapter):
    def __init__(self, script: list[ModelResponse], model_id: str = "scripted"):
        self.model_id = model_id
        self._script = list(script)
        self._i = 0
        #: turns passed to each converse() call, for assertions.
        self.calls: list[list[Turn]] = []

    def converse(
        self, system: str | None, turns: list[Turn], tools: list[ToolSpec]
    ) -> ModelResponse:
        self.calls.append(list(turns))
        if self._i >= len(self._script):
            # Ran past the script — treat as a terminal empty answer.
            return ModelResponse(text="")
        resp = self._script[self._i]
        self._i += 1
        return resp
