"""Trajectory diff — the screenshot that sells the project.

Aligns two trajectories on their *decision* steps (tool calls + final stop) and
reports the first point of divergence. This answers: given the same recorded tool
outputs, where does model B decide differently from model A?

Two honesty guards, added after community feedback:
  - **Unresolvable forks.** If a run diverges because the replay requested a tool
    the recording doesn't cover (`stopped_reason == "unrecorded_tool_call"`), the
    steps after that point never actually happened — so `kind` is
    `"unresolvable_fork"` and `comparable_until` marks where comparison stops.
  - **Redaction-influenced divergences.** Redaction runs before storage, so the
    replay sees redacted inputs the original never did. If a redacted value is in
    play up to the divergence, `redaction_suspect` is set — the divergence may be
    caused by redaction, not the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agentsense.replay.trajectory import FINAL, TOOL_CALL, Step, Trajectory

# Redaction tokens look like <EMAIL_1a2b3c4d> / <PHONE_deadbeef> (see redaction.redactor).
_TOKEN_RE = re.compile(r"<[A-Z][A-Z0-9_]*_[0-9a-f]{8}>")

ALIGNED = "aligned"
DIVERGED = "diverged"
UNRESOLVABLE_FORK = "unresolvable_fork"


@dataclass
class TrajectoryDiff:
    aligned: bool  # True if the decision sequences are identical
    first_divergence: int | None  # index into the decision sequence, or None
    a_step: Step | None
    b_step: Step | None
    summary: str
    kind: str = ALIGNED  # aligned | diverged | unresolvable_fork
    comparable_until: int | None = None  # decisions are only comparable before this
    redaction_suspect: bool = False  # divergence may be caused by redaction


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


def _contains_token(obj) -> bool:
    return obj is not None and bool(_TOKEN_RE.search(json.dumps(obj, default=str)))


def _steps_through_decision(traj: Trajectory, decision_index: int) -> list[Step]:
    """All steps (incl. tool_results) up to and including the Nth decision step."""
    out: list[Step] = []
    seen = -1
    for s in traj.steps:
        out.append(s)
        if s.kind in (TOOL_CALL, FINAL):
            seen += 1
            if seen == decision_index:
                break
    return out


def _redaction_in_play(a: Trajectory, b: Trajectory, index: int) -> bool:
    """True if a redacted value appears in either run's inputs/results up to `index`."""
    steps = _steps_through_decision(a, index) + _steps_through_decision(b, index)
    return any(_contains_token(s.tool_input) or _contains_token(s.result) for s in steps)


def diff_trajectories(a: Trajectory, b: Trajectory) -> TrajectoryDiff:
    da, db = a.decisions, b.decisions
    fork = "unrecorded_tool_call" in (a.stopped_reason, b.stopped_reason)

    for i in range(max(len(da), len(db))):
        step_a = da[i] if i < len(da) else None
        step_b = db[i] if i < len(db) else None
        if (_key(step_a) if step_a else None) != (_key(step_b) if step_b else None):
            kind = UNRESOLVABLE_FORK if fork else DIVERGED
            suspect = _redaction_in_play(a, b, i)
            summary = (
                f"diverge at decision {i}: "
                f"{a.model_id or 'A'} → {_describe(step_a)} | "
                f"{b.model_id or 'B'} → {_describe(step_b)}"
            )
            if kind == UNRESOLVABLE_FORK:
                summary += (
                    f" · unresolvable fork — the replay requested an unrecorded tool, "
                    f"so decisions after {i} never happened and can't be compared"
                )
            if suspect:
                summary += (
                    " · ⚠ a redacted value is in play before this point — the "
                    "divergence may be caused by redaction, not the model"
                )
            return TrajectoryDiff(
                aligned=False,
                first_divergence=i,
                a_step=step_a,
                b_step=step_b,
                summary=summary,
                kind=kind,
                comparable_until=i,
                redaction_suspect=suspect,
            )

    return TrajectoryDiff(
        aligned=True,
        first_divergence=None,
        a_step=None,
        b_step=None,
        summary=f"identical trajectory ({len(da)} decisions)",
        kind=ALIGNED,
        comparable_until=len(da),
        redaction_suspect=False,
    )
