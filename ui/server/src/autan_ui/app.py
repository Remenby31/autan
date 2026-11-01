import asyncio
import logging
import mimetypes
import os
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from watchfiles import Change, awatch

from autan_ui.confine import Folder, is_link
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


def _mtime(folder: Folder) -> int | None:
    stat = folder.stat("run.json")
    return stat.st_mtime_ns if stat else None


def _open_build(static: Path | None) -> Folder | None:
    if static is None:
        return None
    try:
        build = Folder(static)
    except OSError:
        return None
    try:
        os.close(build.open("index.html"))
    except OSError:
        build.close()
        return None
    return build


def _list_build(build: Folder) -> set[str]:
    names = set()
    for folder, subfolders, files in os.walk(build.path):
        here = Path(folder)
        subfolders[:] = [name for name in subfolders if not is_link(here / name)]
        for name in files:
            if not is_link(here / name):
                names.add((here / name).relative_to(build.path).as_posix())
    return names


def create_app(
    runs_root: Path, static: Path | None = STATIC, *, hosts: tuple[str, ...] = LOOPBACK
) -> FastAPI:
    registry = Registry(runs_root)
    build = _open_build(static)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if build is not None:
                build.close()

    app = FastAPI(title="Autan", docs_url=None, redoc_url=None, lifespan=lifespan)
    # A page on another origin that rebinds its DNS name to 127.0.0.1 sends its own Host header.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(hosts))

    def find_run(run_id: str) -> Path:
        try:
            return registry.run_dir(run_id)
        except RunNotFoundError as error:
            raise HTTPException(404, f"no run {run_id}") from error
        except RunRejectedError as error:
            raise HTTPException(409, str(error)) from error

    def existing_run(run_id: str) -> str:
        """404 or 409 before a response starts. Handlers find the directory again by listing
        the runs directory: an injected path would count as request input."""
        find_run(run_id)
        return run_id

    @app.get("/api/runs")
    async def list_runs() -> dict[str, Any]:
        return (await asyncio.to_thread(registry.scan)).document()

    @app.get("/api/runs/events", response_class=EventSourceResponse)
    async def run_events(request: Request) -> AsyncIterator[ServerSentEvent]:
        """`runs` snapshot first, then `run` (upsert), `removed` (run id), `rejected` (the
        whole list) and `progress` (last iteration of each running run) as they change.
        Every wake-up rescans, timeouts included, so a change made before the watcher starts is
        still reported."""
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
                if current.progress != previous.progress:
                    yield ServerSentEvent(event="progress", data=current.progress)
                if current.rejected != previous.rejected:
                    yield ServerSentEvent(event="rejected", data=current.document()["rejected"])
                previous = current
        finally:
            stop.set()
            watcher.cancel()

    @app.get("/api/runs/{run_id}")
    def run_details(run_id: Annotated[str, Depends(existing_run)]) -> dict[str, Any]:
        return registry.details(find_run(run_id))

    @app.get("/api/runs/{run_id}/history/events", response_class=EventSourceResponse)
    async def history_events(
        run_id: Annotated[str, Depends(existing_run)],
        request: Request,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> AsyncIterator[ServerSentEvent]:
        """`run` with the current run.json, then `history` batches; the event id is the byte
        offset reached, so a reconnecting EventSource resumes where it stopped. A history that
        breaks the contract ends the stream with `invalid` and its reason."""
        offset = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
        # Pinned for the whole stream: replacing the directory later cannot redirect reads.
        try:
            directory = registry.run_dir(run_id)
            folder = Folder(directory)
        except (LookupError, ValueError, OSError) as error:
            yield ServerSentEvent(event="invalid", data=f"run {run_id} is gone: {error}")
            return
        stop = asyncio.Event()
        watcher = asyncio.create_task(_until_disconnected(request, stop))
        changes = _changes(directory, stop)
        try:
            run_mtime = _mtime(folder)
            run = read_run(folder)
            yield ServerSentEvent(event="run", data=run)
            while True:
                try:
                    chunk = read_history(folder, offset, live=run["status"] == "running")
                except (OSError, ValueError) as error:
                    yield ServerSentEvent(event="invalid", data=str(error))
                    return
                if chunk.records or offset == 0:
                    offset = chunk.offset
                    yield ServerSentEvent(event="history", data=chunk.records, id=str(offset))
                # Every wake-up re-reads, timeouts included: an append that lands before the
                # watcher starts is still picked up.
                if await anext(changes, None) is None:
                    return
                mtime = _mtime(folder)
                if mtime != run_mtime:
                    try:
                        run_mtime, run = mtime, read_run(folder)
                    except ValueError as error:
                        yield ServerSentEvent(event="invalid", data=str(error))
                        return
                    yield ServerSentEvent(event="run", data=run)
        finally:
            stop.set()
            watcher.cancel()
            folder.close()

    @app.get("/api/surfaces/synthetic")
    def synthetic_surface(
        triangles: Annotated[int, Query(ge=128, le=MAX_SYNTHETIC_TRIANGLES)] = 200_000,
    ) -> Response:
        return Response(synthetic_wing(triangles), media_type="application/octet-stream")

    if build is not None:
        # The request only selects an entry by name: it never becomes a path. The file is
        # opened beneath the pinned build directory without following any link, then served
        # from that same descriptor, so a change after start-up cannot redirect it.
        assets = _list_build(build)

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> Response:
            if path.startswith("api/"):
                raise HTTPException(404)
            for name in (path if path in assets else None, "index.html"):
                if name is None:
                    continue
                try:
                    content = build.read(name)
                except OSError:
                    continue
                media = mimetypes.guess_type(name)[0] or "application/octet-stream"
                return Response(content, media_type=media)
            raise HTTPException(404)

    else:
        log.warning(
            "no built web app in %s (or its index.html is a link): serving the API only", static
        )

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
