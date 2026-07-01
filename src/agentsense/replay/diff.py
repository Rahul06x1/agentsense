"""Trajectory diff — the screenshot that sells the project.

Aligns two trajectories on their *decision* steps (tool calls + final stop) and
reports the first point of divergence. This answers: given the same recorded tool
outputs, where does model B decide differently from model A?
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from agentsense.replay.trajectory import FINAL, TOOL_CALL, Step, Trajectory


@dataclass
class TrajectoryDiff:
    aligned: bool  # True if the decision sequences are identical
    first_divergence: int | None  # index into the decision sequence, or None
    a_step: Step | None
    b_step: Step | None
    summary: str


def _key(step: Step) -> tuple:
    if step.kind == TOOL_CALL:
        return (TOOL_CALL, step.tool_name, json.dumps(step.tool_input, sort_keys=True))
    if step.kind == FINAL:
        return (FINAL,)
    return (step.kind,)


def _describe(step: Step | None) -> str:
    if step is None:
        return "∅ (no step)"
    if step.kind == TOOL_CALL:
        return f"call {step.tool_name}({json.dumps(step.tool_input)})"
    if step.kind == FINAL:
        return "final answer"
    return step.kind


def diff_trajectories(a: Trajectory, b: Trajectory) -> TrajectoryDiff:
    da, db = a.decisions, b.decisions
    for i in range(max(len(da), len(db))):
        step_a = da[i] if i < len(da) else None
        step_b = db[i] if i < len(db) else None
        ka = _key(step_a) if step_a else None
        kb = _key(step_b) if step_b else None
        if ka != kb:
            return TrajectoryDiff(
                aligned=False,
                first_divergence=i,
                a_step=step_a,
                b_step=step_b,
                summary=(
                    f"diverge at decision {i}: "
                    f"{a.model_id or 'A'} → {_describe(step_a)} | "
                    f"{b.model_id or 'B'} → {_describe(step_b)}"
                ),
            )
    return TrajectoryDiff(
        aligned=True,
        first_divergence=None,
        a_step=None,
        b_step=None,
        summary=f"identical trajectory ({len(da)} decisions)",
    )
