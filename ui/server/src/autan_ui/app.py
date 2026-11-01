import asyncio
import json
import logging
import mimetypes
import os
import re
import tempfile
import tomllib
import webbrowser
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

import autan
import uvicorn
from autan.errors import CaseError, doc_url
from autan.results import LogLine, log_line_problems
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import ValidationError
from watchfiles import Change, awatch

from autan_ui.confine import Folder, is_link
from autan_ui.history import read_lines
from autan_ui.registry import (
    Registry,
    RunNotFoundError,
    RunRejectedError,
    read_run,
)
from autan_ui.surface import synthetic_wing
from autan_ui.wall import wall_surface

SHA256 = re.compile(r"[0-9a-f]{64}")

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


MAX_FOLLOWED = 8


class Follower:
    """One run read for a stream: run.json when it changes, history.jsonl and log.jsonl from
    byte offsets, through a folder pinned at the start so that replacing the directory cannot
    redirect reads. `done` once a file breaks the contract (an `invalid` event was produced).
    Log lines carry the documentation page of their code as `doc`."""

    def __init__(self, directory: Path, offsets: dict[str, int]) -> None:
        self.directory = directory
        self.folder = Folder(directory)
        self.offsets = {name: offsets.get(name, 0) for name in STREAMED}
        self.read = dict(self.offsets)
        self.run: dict[str, Any] | None = None
        self.run_mtime: int | None = None
        self.sent = False
        self.done = False

    def poll(self) -> list[tuple[str, Any]]:
        """New events, in order. A `history`/`log` event's data is `(records, file, offset)`;
        the stream calls `delivered` once it has sent it, so that a reconnection resumes from
        what the client received, not from what was read."""
        events: list[tuple[str, Any]] = []
        mtime = _mtime(self.folder)
        if self.run is None or mtime != self.run_mtime:
            try:
                self.run_mtime, self.run = mtime, read_run(self.folder)
            except ValueError as error:
                self.done = True
                return [("invalid", str(error))]
            events.append(("run", self.run))
        live = self.run["status"] == "running"
        for name, event in STREAMED.items():
            try:
                chunk = read_lines(self.folder, name, self.read[name], live=live)
                records = (
                    [_log_line(line) for line in chunk.records] if event == "log" else chunk.records
                )
            except (OSError, ValueError) as error:
                self.done = True
                return [*events, ("invalid", str(error))]
            if chunk.records or not self.sent:
                self.read[name] = chunk.offset
                events.append((event, (records, name, chunk.offset)))
        self.sent = True
        return events

    def delivered(self, name: str, offset: int) -> None:
        self.offsets[name] = offset

    def close(self) -> None:
        self.folder.close()


STREAMED = {"history.jsonl": "history", "log.jsonl": "log"}


def _log_line(record: object) -> dict[str, Any]:
    """A log.jsonl record checked by the SDK's model and rules (results.md#log), with the
    documentation page of its code; ValueError otherwise."""
    try:
        line = LogLine.model_validate(record)
    except ValidationError as error:
        msg = f"log.jsonl: {error.errors(include_url=False)[0]['msg']}"
        raise ValueError(msg) from error
    problems = log_line_problems(line)
    if problems:
        msg = f"log.jsonl: {problems[0]}"
        raise ValueError(msg)
    document = line.model_dump(mode="json")
    return document | {"doc": doc_url(line.code) if line.code else None}


def _offsets(last_event_id: str | None) -> dict[str, Any]:
    """The offsets of a reconnecting stream, as its last event id carried them; anything
    malformed restarts from the start of the files."""
    try:
        parsed = json.loads(last_event_id) if last_event_id else {}
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _file_offsets(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if k in STREAMED and type(v) is int and v >= 0}


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


def checked_run(folder: Folder) -> dict[str, Any]:
    try:
        return read_run(folder)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


def create_app(
    runs_root: Path, static: Path | None = STATIC, *, hosts: tuple[str, ...] = LOOPBACK
) -> FastAPI:
    registry = Registry(runs_root)
    # The last surface built: reopening a run in the viewer needs no new extraction.
    surfaces: dict[tuple[str, int, int], bytes] = {}
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
        whole list) and `summaries` (iteration and residual drop of each run) as they change.
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
                if current.summaries != previous.summaries:
                    yield ServerSentEvent(event="summaries", data=current.summaries)
                if current.rejected != previous.rejected:
                    yield ServerSentEvent(event="rejected", data=current.document()["rejected"])
                previous = current
        finally:
            stop.set()
            watcher.cancel()

    def existing_runs(run: Annotated[list[str], Query()]) -> list[str]:
        """Every requested run exists and is accepted, before a stream starts (≤ 8)."""
        if not 1 <= len(run) <= MAX_FOLLOWED:
            raise HTTPException(422, f"between 1 and {MAX_FOLLOWED} runs")
        for run_id in run:
            find_run(run_id)
        return list(dict.fromkeys(run))

    @app.get("/api/runs/history/events", response_class=EventSourceResponse)
    async def histories_events(
        run_ids: Annotated[list[str], Depends(existing_runs)],
        request: Request,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> AsyncIterator[ServerSentEvent]:
        """Several runs over one connection (a browser opens at most six per host): the events
        of the single-run stream with `{run_id, ...}` payloads; the event id is the JSON map of
        byte offsets reached, for reconnection."""
        offsets = _offsets(last_event_id)
        followers: dict[str, Follower] = {}
        try:
            for run_id in run_ids:
                followers[run_id] = Follower(
                    registry.run_dir(run_id), _file_offsets(offsets.get(run_id))
                )
        except (LookupError, ValueError, OSError) as error:
            for follower in followers.values():
                follower.close()
            yield ServerSentEvent(event="invalid", data={"run_id": None, "data": str(error)})
            return
        stop = asyncio.Event()
        watcher = asyncio.create_task(_until_disconnected(request, stop))
        changes = _changes(registry.root, stop)
        try:
            while True:
                for run_id, follower in followers.items():
                    if follower.done:
                        continue
                    for event, data in follower.poll():
                        if event in STREAMED.values():
                            records, name, offset = data
                            follower.delivered(name, offset)
                            ids = {k: f.offsets for k, f in followers.items()}
                            payload = {"run_id": run_id, "data": records}
                            yield ServerSentEvent(event=event, data=payload, id=json.dumps(ids))
                        else:
                            payload = {"run_id": run_id, "data": data}
                            yield ServerSentEvent(event=event, data=payload)
                if all(f.done for f in followers.values()):
                    return
                if await anext(changes, None) is None:
                    return
        finally:
            stop.set()
            watcher.cancel()
            for follower in followers.values():
                follower.close()

    @app.get("/api/runs/{run_id}")
    def run_details(run_id: Annotated[str, Depends(existing_run)]) -> dict[str, Any]:
        try:
            return registry.details(find_run(run_id))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/runs/{run_id}/plan")
    def run_plan(run_id: Annotated[str, Depends(existing_run)]) -> dict[str, Any]:
        """`autan plan --json` of the run's stored case against its stored MeshInfo."""
        with Folder(find_run(run_id)) as folder:
            run = checked_run(folder)
            case_toml = folder.read("case.toml")
            mesh_info = folder.read("mesh-info.json")
            manufactured = (
                folder.read("manufactured.json") if folder.stat("manufactured.json") else None
            )
            # The stored case is planned from a scratch copy built from the pinned folder, so
            # the SDK reads no file of the run directory, nor any file the case names: the mesh
            # path is cut to its name (plan reads only the MeshInfo), a verification case points
            # at the archived manufactured solution and an oracle-state rule at its archived
            # face states (results.md, `states/<sha256>.json`).
            try:
                document = tomllib.loads(case_toml.decode())
            except (tomllib.TOMLDecodeError, UnicodeDecodeError) as error:
                raise HTTPException(422, f"case.toml: {error}") from error
            states: dict[str, bytes] = {}
            for rule in document.get("boundaries", []):
                condition = rule.get("condition") if isinstance(rule, dict) else None
                if not isinstance(condition, dict) or condition.get("kind") != "oracle-state":
                    continue
                sha256 = condition.get("sha256")
                valid = isinstance(sha256, str) and SHA256.fullmatch(sha256) is not None
                # Repointed first, whatever follows: the source the case names is never read.
                condition["path"] = f"{sha256}.json" if valid else "states.json"
                if not valid or sha256 in states:
                    continue  # An invalid case, or staged already: plan reports the former.
                try:
                    states[sha256] = folder.read(f"states/{sha256}.json")
                except OSError:
                    continue  # Missing or not a regular file: plan reports it cannot read it.
        document["mesh"] = PurePosixPath(str(document.get("mesh", ""))).name or "mesh"
        verification = document.get("verification")
        if isinstance(verification, dict) and isinstance(verification.get("manufactured"), dict):
            verification["manufactured"]["path"] = "manufactured.json"
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "case.json").write_text(json.dumps(document))
            (root / "mesh.meshinfo").write_bytes(mesh_info)
            if manufactured is not None:
                (root / "manufactured.json").write_bytes(manufactured)
            for sha256, data in states.items():
                (root / f"{sha256}.json").write_bytes(data)
            try:
                case = autan.load(root / "case.json")
                info = autan.MeshInfo.load(root / "mesh.meshinfo")
                plan = case.plan(info, gpus=run["devices"]["count"])
            except CaseError as error:
                detail = [d.model_dump(mode="json") for d in error.diagnostics]
                raise HTTPException(422, detail) from error
            return plan.document()

    @app.get("/api/runs/{run_id}/history/events", response_class=EventSourceResponse)
    async def history_events(
        run_id: Annotated[str, Depends(existing_run)],
        request: Request,
        last_event_id: Annotated[str | None, Header()] = None,
    ) -> AsyncIterator[ServerSentEvent]:
        """`run` with the current run.json, then `history` and `log` batches; the event id is
        the JSON map of byte offsets reached, so a reconnecting EventSource resumes where it
        stopped. A file that breaks the contract ends the stream with `invalid` and its reason."""
        try:
            follower = Follower(registry.run_dir(run_id), _file_offsets(_offsets(last_event_id)))
        except (LookupError, ValueError, OSError) as error:
            yield ServerSentEvent(event="invalid", data=f"run {run_id} is gone: {error}")
            return
        stop = asyncio.Event()
        watcher = asyncio.create_task(_until_disconnected(request, stop))
        changes = _changes(follower.directory, stop)
        try:
            while True:
                for event, data in follower.poll():
                    if event in STREAMED.values():
                        records, name, offset = data
                        follower.delivered(name, offset)
                        cursor = json.dumps(follower.offsets)
                        yield ServerSentEvent(event=event, data=records, id=cursor)
                    else:
                        yield ServerSentEvent(event=event, data=data)
                if follower.done or await anext(changes, None) is None:
                    return
        finally:
            stop.set()
            watcher.cancel()
            follower.close()

    @app.get("/api/runs/{run_id}/surface")
    def run_surface(run_id: Annotated[str, Depends(existing_run)]) -> Response:
        """The wall surface of the run's solution.cgns with Cp, Cf and Pressure per node."""
        directory = find_run(run_id)
        with Folder(directory) as folder:
            if checked_run(folder)["status"] == "running":
                # results.md: the solution is not a result while the run is running.
                raise HTTPException(409, f"run {run_id} is still running")
            stat = folder.stat("solution.cgns")
            if stat is None:
                raise HTTPException(404, f"run {run_id} has no solution.cgns")
            key = (run_id, stat.st_mtime_ns, stat.st_size)
            message = surfaces.get(key)
            if message is None:
                try:
                    message = wall_surface(folder)
                except (OSError, ValueError, KeyError) as error:
                    raise HTTPException(422, f"solution.cgns: {error}") from error
                surfaces.clear()
                surfaces[key] = message
        return Response(message, media_type="application/octet-stream")

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
