import asyncio
import logging
import webbrowser
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from watchfiles import Change, awatch

from autan_ui.registry import (
    Registry,
    RunNotFoundError,
    RunRejectedError,
    read_history,
    read_run,
)
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


LOOPBACK = ("127.0.0.1", "localhost", "[::1]")


def create_app(
    runs_root: Path, static: Path | None = STATIC, *, hosts: tuple[str, ...] = LOOPBACK
) -> FastAPI:
    registry = Registry(runs_root)
    app = FastAPI(title="Autan", docs_url=None, redoc_url=None)
    # A page on another origin that rebinds its DNS name to 127.0.0.1 sends its own Host header.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(hosts))

    def existing_run(run_id: str) -> Path:
        try:
            return registry.run_dir(run_id)
        except RunNotFoundError as error:
            raise HTTPException(404, f"no run {run_id}") from error
        except RunRejectedError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        return (await asyncio.to_thread(registry.scan)).document()

    @app.get("/api/runs/events", response_class=EventSourceResponse)
    async def run_events(request: Request) -> AsyncIterator[ServerSentEvent]:
        """`runs` snapshot first, then `run` (upsert), `removed` (run id) and `rejected` (the
        whole list) as they change. Every wake-up rescans, timeouts included, so a change made
        before the watcher starts is still reported."""
        previous = await asyncio.to_thread(registry.scan)
        yield ServerSentEvent(event="runs", data=previous.document())
        stop = asyncio.Event()
        watcher = asyncio.create_task(_until_disconnected(request, stop))
        try:
            async for _ in _changes(registry.root, stop):
                current = await asyncio.to_thread(registry.scan)
                for run_id, run in current.runs.items():
                    if previous.runs.get(run_id) != run:
                        yield ServerSentEvent(event="run", data=run)
                for run_id in sorted(previous.runs.keys() - current.runs.keys()):
                    yield ServerSentEvent(event="removed", data=run_id)
                if current.rejected != previous.rejected:
                    yield ServerSentEvent(event="rejected", data=current.document()["rejected"])
                previous = current
        finally:
            stop.set()
            watcher.cancel()

    @app.get("/api/runs/{run_id}")
    def run_details(directory: Annotated[Path, Depends(existing_run)]) -> dict[str, Any]:
        return registry.details(directory)

    @app.get("/api/runs/{run_id}/history/events", response_class=EventSourceResponse)
    async def history_events(
        directory: Annotated[Path, Depends(existing_run)],
        request: Request,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> AsyncIterator[ServerSentEvent]:
        """`run` with the current run.json, then `history` batches; the event id is the byte
        offset reached, so a reconnecting EventSource resumes where it stopped. A history that
        breaks the contract ends the stream with `invalid` and its reason."""
        offset = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
        run_json = directory / "run.json"
        run_mtime = run_json.stat().st_mtime_ns
        run = read_run(directory)
        yield ServerSentEvent(event="run", data=run)
        stop = asyncio.Event()
        watcher = asyncio.create_task(_until_disconnected(request, stop))
        changes = _changes(directory, stop)
        try:
            while True:
                try:
                    chunk = read_history(directory, offset, live=run["status"] == "running")
                except ValueError as error:
                    yield ServerSentEvent(event="invalid", data=str(error))
                    return
                if chunk.records or offset == 0:
                    offset = chunk.offset
                    yield ServerSentEvent(event="history", data=chunk.records, id=str(offset))
                # Every wake-up re-reads, timeouts included: an append that lands before the
                # watcher starts is still picked up.
                if await anext(changes, None) is None:
                    return
                mtime = run_json.stat().st_mtime_ns
                if mtime != run_mtime:
                    try:
                        run_mtime, run = mtime, read_run(directory)
                    except ValueError as error:
                        yield ServerSentEvent(event="invalid", data=str(error))
                        return
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
    app = create_app(runs, hosts=tuple({*LOOPBACK, host}))
    if open_browser:
        webbrowser.open(f"http://{host}:{port}/")
    uvicorn.run(app, host=host, port=port, log_level="info")
