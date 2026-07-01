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

from tracekit.replay import Trajectory, captured_trajectory, diff_trajectories
from tracekit.replay.trajectory import TOOL_CALL
from tracekit.store.sqlite import SpanStore

STATIC_DIR = Path(__file__).parent / "static"


def create_app(db_path: str) -> FastAPI:
    app = FastAPI(title="tracekit", docs_url="/api/docs")

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
            "aligned": result.aligned,
            "first_divergence": result.first_divergence,
            "summary": result.summary,
        }

    # Serve the single-page frontend. Registered AFTER the API routes so /api/*
    # is matched first; the mount at "/" then serves index.html and assets.
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


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
