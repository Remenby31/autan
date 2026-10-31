import asyncio
import logging
import webbrowser
from collections.abc import AsyncIterator, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from watchfiles import Change, awatch

from autan_ui.registry import Registry, RunNotFoundError, read_history, read_run
from autan_ui.surface import synthetic_wing

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"
WATCH_DEBOUNCE_MS = 50
MAX_SYNTHETIC_TRIANGLES = 40_000_000


def _changes(root: Path, stop: asyncio.Event) -> AsyncIterator[set[tuple[Change, str]]]:
    return awatch(
        root,
        stop_event=stop,
        debounce=WATCH_DEBOUNCE_MS,
        step=WATCH_DEBOUNCE_MS,
        recursive=True,
        rust_timeout=1_000,
        yield_on_timeout=True,
    )


async def _until_disconnected(request: Request, stop: asyncio.Event) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(0.5)
    stop.set()


def _touched_runs(root: Path, changes: Iterable[tuple[Change, str]]) -> set[str]:
    touched = set()
    for _, raw in changes:
        parts = Path(raw).relative_to(root).parts
        if parts and not parts[0].startswith("."):
            touched.add(parts[0])
    return touched


def create_app(runs_root: Path, static: Path | None = STATIC) -> FastAPI:
    registry = Registry(runs_root)
    app = FastAPI(title="Autan", docs_url=None, redoc_url=None)

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        runs, rejected = registry.scan()
        return {"runs": runs, "rejected": [asdict(r) for r in rejected]}

    @app.get("/api/runs/events", response_class=EventSourceResponse)
    async def run_events(request: Request) -> AsyncIterator[ServerSentEvent]:
        """`runs` snapshot first, then `run` (upsert) and `removed` events."""
        runs, rejected = registry.scan()
        yield ServerSentEvent(
            event="runs", data={"runs": runs, "rejected": [asdict(r) for r in rejected]}
        )
        stop = asyncio.Event()
        watcher = asyncio.create_task(_until_disconnected(request, stop))
        try:
            async for changes in _changes(registry.root, stop):
                for name in sorted(_touched_runs(registry.root, changes)):
                    directory = registry.root / name
                    if not directory.exists():
                        yield ServerSentEvent(event="removed", data=name)
                        continue
                    try:
                        yield ServerSentEvent(event="run", data=read_run(directory))
                    except (OSError, ValueError):
                        continue
        finally:
            stop.set()
            watcher.cancel()

    @app.get("/api/runs/{run_id}")
    def run_details(run_id: str) -> dict[str, Any]:
        try:
            return registry.details(run_id)
        except RunNotFoundError as error:
            raise HTTPException(404, f"no run {run_id}") from error

    @app.get("/api/runs/{run_id}/history/events", response_class=EventSourceResponse)
    async def history_events(
        run_id: str,
        request: Request,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> AsyncIterator[ServerSentEvent]:
        """`run` with the current run.json, then `history` batches; the event id is the byte
        offset reached, so a reconnecting EventSource resumes where it stopped."""
        try:
            directory = registry.run_dir(run_id)
        except RunNotFoundError as error:
            raise HTTPException(404, f"no run {run_id}") from error
        offset = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
        run_json = directory / "run.json"
        run_mtime = run_json.stat().st_mtime_ns
        yield ServerSentEvent(event="run", data=read_run(directory))
        chunk = read_history(directory, offset)
        yield ServerSentEvent(event="history", data=chunk.records, id=str(chunk.offset))
        offset = chunk.offset
        stop = asyncio.Event()
        watcher = asyncio.create_task(_until_disconnected(request, stop))
        try:
            # Re-read on every wake-up, timeouts included: an append that lands before the
            # watcher starts is still picked up.
            async for _ in _changes(directory, stop):
                chunk = read_history(directory, offset)
                if chunk.records:
                    offset = chunk.offset
                    yield ServerSentEvent(event="history", data=chunk.records, id=str(offset))
                mtime = run_json.stat().st_mtime_ns
                if mtime != run_mtime:
                    try:
                        run = read_run(directory)
                    except (OSError, ValueError):
                        continue
                    run_mtime = mtime
                    yield ServerSentEvent(event="run", data=run)
        finally:
            stop.set()
            watcher.cancel()

    @app.get("/api/surfaces/synthetic")
    def synthetic_surface(
        triangles: Annotated[int, Query(ge=128, le=MAX_SYNTHETIC_TRIANGLES)] = 200_000,
    ) -> Response:
        return Response(synthetic_wing(triangles), media_type="application/octet-stream")

    if static is not None and (static / "index.html").is_file():
        index = static / "index.html"

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404)
            candidate = (static / path).resolve()
            if candidate.is_file() and candidate.is_relative_to(static.resolve()):
                return FileResponse(candidate)
            return FileResponse(index)
    else:
        log.warning("no built web app in %s: serving the API only (run `bun run build`)", static)

    return app


def serve(
    runs: Path, host: str = "127.0.0.1", port: int = 8765, *, open_browser: bool = True
) -> None:
    """Serve the UI for the run directories under `runs` until interrupted."""
    if not runs.is_dir():
        log.info("creating the runs directory %s", runs)
        runs.mkdir(parents=True)
    app = create_app(runs)
    if open_browser:
        webbrowser.open(f"http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port, log_level="info")
