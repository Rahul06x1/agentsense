"""Trajectory — the ordered record of what happened during a replay.

Steps are the unit the diff aligns on. `model_text` and `tool_result` are kept
for display; the *decision* steps (`tool_call`, `final`) are what the trajectory
diff compares, because that's where a model-swap actually changes behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentsense.replay.types import Usage

MODEL_TEXT = "model_text"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
FINAL = "final"


@dataclass
class Step:
    kind: str
    text: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    result: Any = None
    missing: bool = False  # tool_result the recording didn't cover (fork, policy=stop)
    stubbed: bool = False  # synthetic placeholder result (policy=stub)
    live: bool = False  # result from a real live tool call (policy=live)


@dataclass
class Trajectory:
    model_id: str | None = None
    #: Display name for the side that produced this run, when it isn't a model.
    #: A proxy trace has no llm_call span to name, so it labels itself with the
    #: MCP client that made the calls. Kept separate from `model_id` so the API
    #: never reports a client name as if it were a model.
    label: str | None = None
    steps: list[Step] = field(default_factory=list)
    usages: list[Usage] = field(default_factory=list)
    final_text: str = ""
    stopped_reason: str | None = None  # e.g. "unrecorded_tool_call", "max_steps"

    def add(self, step: Step) -> None:
        self.steps.append(step)

    @property
    def decisions(self) -> list[Step]:
        """The steps that represent a model decision (tool choice or stop)."""
        return [s for s in self.steps if s.kind in (TOOL_CALL, FINAL)]

    @property
    def total_input_tokens(self) -> int:
        return sum(u.input_tokens or 0 for u in self.usages)

    @property
    def total_output_tokens(self) -> int:
        return sum(u.output_tokens or 0 for u in self.usages)
