"""Trajectory diff — the screenshot that sells the project.

Aligns two trajectories on their *decision* steps (tool calls + final stop) and
reports the first point of divergence. This answers: given the same recorded tool
outputs, where does model B decide differently from model A?

Two honesty guards, added after community feedback:
  - **Unresolvable forks.** If a run diverges because the replay requested a tool
    the recording doesn't cover (`stopped_reason == "unrecorded_tool_call"`), the
    steps after that point never actually happened — so `kind` is
    `"unresolvable_fork"` and `comparable_until` marks where comparison stops.
  - **Unknown endings.** A `final` step is only recorded when an answer was
    actually observed. The proxy never sees one, and a truncated capture or a
    max_steps replay has none either. Comparing past a missing ending would
    claim knowledge the trace doesn't hold, so `kind` is `"unknown_terminal"`
    and `comparable_until` marks the last decision that was really compared.
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
UNKNOWN_TERMINAL = "unknown_terminal"


@dataclass
class TrajectoryDiff:
    aligned: bool  # True if the decision sequences are identical
    first_divergence: int | None  # index into the decision sequence, or None
    a_step: Step | None
    b_step: Step | None
    summary: str
    kind: str = ALIGNED  # aligned | diverged | unresolvable_fork | unknown_terminal
    comparable_until: int | None = None  # decisions are only comparable before this
    redaction_suspect: bool = False  # divergence may be caused by redaction


def _key(step: Step) -> tuple:
    if step.kind == TOOL_CALL:
        return (TOOL_CALL, step.tool_name, json.dumps(step.tool_input, sort_keys=True))
    if step.kind == FINAL:
        return (FINAL,)
    return (step.kind,)


def _has_terminal(traj: Trajectory) -> bool:
    """Did we actually observe this run end, or did the recording just stop?

    A `final` step is only emitted when a real answer was seen, so its absence
    means the ending is unknown: a proxy trace (which never sees the model), an
    interrupted capture, or a replay that hit max_steps.
    """
    decisions = traj.decisions
    return bool(decisions) and decisions[-1].kind == FINAL


def _side_name(traj: Trajectory, fallback: str) -> str:
    """What to call one side in the summary: its model, else its client, else A/B."""
    return traj.model_id or traj.label or fallback


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
    a_ends, b_ends = _has_terminal(a), _has_terminal(b)

    for i in range(max(len(da), len(db))):
        step_a = da[i] if i < len(da) else None
        step_b = db[i] if i < len(db) else None
        if (_key(step_a) if step_a else None) != (_key(step_b) if step_b else None):
            # One side simply ran out. If its ending was never observed, it may
            # have gone on to do exactly what the other side did -- we cannot
            # tell, so this is not evidence of a divergence.
            ran_out = (step_a is None and not a_ends) or (step_b is None and not b_ends)
            kind = UNRESOLVABLE_FORK if fork else (
                UNKNOWN_TERMINAL if ran_out else DIVERGED
            )
            # With no conflicting decision there is nothing for redaction to have
            # caused, so the suspect flag would only add a contradictory note.
            suspect = kind != UNKNOWN_TERMINAL and _redaction_in_play(a, b, i)
            if kind == UNKNOWN_TERMINAL:
                short = _side_name(a, "A") if step_a is None else _side_name(b, "B")
                other = _describe(step_b if step_a is None else step_a)
                summary = (
                    f"not comparable past decision {i}: {short}'s recording stops "
                    f"there without a captured ending, while the other side went on "
                    f"to {other} — whether {short} would have done the same is unknown"
                )
            else:
                summary = (
                    f"diverge at decision {i}: "
                    f"{_side_name(a, 'A')} → {_describe(step_a)} | "
                    f"{_side_name(b, 'B')} → {_describe(step_b)}"
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
                # An unknown ending is not a divergence: nothing observed
                # conflicted, we just ran out of trace. `comparable_until` is
                # what bounds the claim; reporting a first_divergence here would
                # contradict the kind and paint a false conflict in the UI.
                first_divergence=None if kind == UNKNOWN_TERMINAL else i,
                a_step=step_a,
                b_step=step_b,
                summary=summary,
                kind=kind,
                comparable_until=i,
                redaction_suspect=suspect,
            )

    if not (a_ends and b_ends):
        # Every observed decision matched, but at least one run's ending wasn't
        # captured, so "identical" would claim more than the trace supports.
        names = [name for name, ends in
                 ((_side_name(a, "A"), a_ends), (_side_name(b, "B"), b_ends)) if not ends]
        # Both sides often carry the same label (two runs of one client/model),
        # and "claude-code, claude-code" reads like a bug.
        missing = "either run" if len(names) == 2 and names[0] == names[1] else \
            " and ".join(dict.fromkeys(names))
        if not da:
            summary = (
                f"nothing to compare: no decisions were recorded, and no ending "
                f"was captured for {missing}"
            )
        else:
            summary = (
                f"matched on all {len(da)} observed decisions, but no ending was "
                f"captured for {missing} — nothing after decision {len(da)} is comparable"
            )
        return TrajectoryDiff(
            aligned=False,
            first_divergence=None,
            a_step=None,
            b_step=None,
            summary=summary,
            kind=UNKNOWN_TERMINAL,
            comparable_until=len(da),
            redaction_suspect=False,
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
