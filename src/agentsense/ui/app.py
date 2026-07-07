"""Read-only FastAPI app over the trace store — the local UI backend.

Decoupled from capture: it only reads. Each request opens its own short-lived
SpanStore connection, so the app is safe under uvicorn's threadpool and never
holds a write lock. FastAPI/uvicorn are an optional dependency (`ui` extra);
import this module only when serving the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agentsense.replay import (
    BedrockAdapter,
    OpenAICompatAdapter,
    Recording,
    Trajectory,
    captured_trajectory,
    diff_trajectories,
    replay,
)
from agentsense.replay.trajectory import TOOL_CALL
from agentsense.store.sqlite import SpanStore

STATIC_DIR = Path(__file__).parent / "static"


class ReplayRequest(BaseModel):
    trace_id: str
    model: str
    adapter: str = "openai"  # "openai" (OpenAI-compatible/Ollama) or "bedrock"
    base_url: str | None = None
    api_key: str | None = None
    region: str = "eu-west-1"
    on_unrecorded: str = "stop"  # fork policy: "stop" | "stub" (UI); "live" needs code


def create_app(db_path: str) -> FastAPI:
    app = FastAPI(title="agentsense", docs_url="/api/docs")

    def _store() -> SpanStore:
        return SpanStore(db_path)

    @app.get("/api/traces")
    def list_traces() -> list[dict[str, Any]]:
        store = _store()
        try:
            return store.list_traces()
        finally:
            store.close()

    @app.get("/api/traces/{trace_id}/spans")
    def trace_spans(trace_id: str) -> dict[str, Any]:
        store = _store()
        try:
            spans = store.spans_for_trace(trace_id)
            if not spans:
                raise HTTPException(status_code=404, detail="trace not found")
            payload = [s.model_dump() for s in spans]
            redaction_count = sum(len(s.redactions) for s in spans)
            return {"trace_id": trace_id, "redaction_count": redaction_count,
                    "spans": payload}
        finally:
            store.close()

    @app.get("/api/diff")
    def diff(a: str, b: str) -> dict[str, Any]:
        """Decision-level trajectory diff between two captured traces."""
        store = _store()
        try:
            if not store.spans_for_trace(a):
                raise HTTPException(status_code=404, detail=f"trace not found: {a}")
            if not store.spans_for_trace(b):
                raise HTTPException(status_code=404, detail=f"trace not found: {b}")
            ta = captured_trajectory(store, a)
            tb = captured_trajectory(store, b)
        finally:
            store.close()
        result = diff_trajectories(ta, tb)
        return {
            "a": {"trace_id": a, "model_id": ta.model_id, "decisions": _decisions(ta)},
            "b": {"trace_id": b, "model_id": tb.model_id, "decisions": _decisions(tb)},
            **_diff_fields(result),
        }

    @app.post("/api/replay")
    def replay_trace(req: ReplayRequest) -> dict[str, Any]:
        """Live-replay a captured trace against a chosen model, return the diff.

        Ephemeral: makes an outbound model call but does NOT write to the store.
        Only traces with llm_call spans can be rebuilt into a recording.
        """
        store = _store()
        try:
            if not store.spans_for_trace(req.trace_id):
                raise HTTPException(status_code=404, detail="trace not found")
            try:
                recording = Recording.from_sdk_trace(store, req.trace_id)
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"cannot rebuild a replayable recording from this trace: {e}",
                ) from e
            captured = captured_trajectory(store, req.trace_id)
        finally:
            store.close()

        if req.adapter == "bedrock":
            adapter = BedrockAdapter(model_id=req.model, region=req.region)
        else:
            adapter = OpenAICompatAdapter(
                model_id=req.model, base_url=req.base_url, api_key=req.api_key
            )

        try:
            replayed = replay(recording, adapter, on_unrecorded=req.on_unrecorded)
        except ModuleNotFoundError as e:  # adapter SDK not installed
            raise HTTPException(
                status_code=400,
                detail=f"adapter '{req.adapter}' needs the 'replay' extra ({e.name})",
            ) from e
        except ValueError as e:  # invalid policy, or "live" without an executor
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001 - surface model/credential errors to the UI
            raise HTTPException(status_code=502, detail=f"model call failed: {e}") from e

        result = diff_trajectories(captured, replayed)
        return {
            "a": {"label": f"captured · {captured.model_id or 'agent'}",
                  "decisions": _decisions(captured)},
            "b": {"label": f"replay · {adapter.model_id}",
                  "decisions": _decisions(replayed)},
            **_diff_fields(result),
            "replay": {
                "model": adapter.model_id,
                "on_unrecorded": req.on_unrecorded,
                "stopped_reason": replayed.stopped_reason,
                "stubbed_calls": sum(1 for s in replayed.steps if s.stubbed),
                "input_tokens": replayed.total_input_tokens,
                "output_tokens": replayed.total_output_tokens,
            },
        }

    # Serve the single-page frontend. Registered AFTER the API routes so /api/*
    # is matched first; the mount at "/" then serves index.html and assets.
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


def _diff_fields(result) -> dict[str, Any]:
    """Shared diff fields for /api/diff and /api/replay responses."""
    return {
        "aligned": result.aligned,
        "kind": result.kind,  # aligned | diverged | unresolvable_fork
        "first_divergence": result.first_divergence,
        "comparable_until": result.comparable_until,
        "redaction_suspect": result.redaction_suspect,
        "summary": result.summary,
    }


def _decisions(traj: Trajectory) -> list[dict[str, Any]]:
    """Serialize a trajectory's decision steps for the diff view."""
    out: list[dict[str, Any]] = []
    for s in traj.decisions:
        if s.kind == TOOL_CALL:
            out.append({"kind": "tool_call", "tool_name": s.tool_name,
                        "tool_input": s.tool_input})
        else:
            out.append({"kind": "final", "text": s.text})
    return out
