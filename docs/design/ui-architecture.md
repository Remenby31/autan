# Web UI architecture

The web UI of milestone M10 (`docs/PLAN.md`): a local application served by `autan ui`, never a cloud service. It shows every run and study whoever started it (UI, SDK, CLI, agent), prepares cases as `autan.case/1` code, follows runs live, renders 3D fields and compares runs. Visual and interaction direction: [`ui-inspiration/README.md`](ui-inspiration/README.md). Code: `ui/`.

## Shape

```
browser ── HTTP + SSE ──▶ autan_ui (FastAPI, Python) ──▶ <runs>/<run_id>/   (results.md)
  React app (static, served by the same process)        surface extraction (NumPy) → binary
```

- **One process, one port, loopback only.** Requests whose `Host` is not the served address are refused (DNS rebinding). `autan_ui.serve(runs, host, port)` serves the built app and the API. `autan ui [--runs DIR] [--port N] [--no-open]` in the SDK CLI is a thin call to it; the default runs directory is `./runs`, the same one `autan run` writes.
- **The registry is the runs directory.** Every directory under `<runs>` that the SDK's reader (`autan.Results.open`) accepts is a run, whoever wrote it; names starting with `.` are directories still being written. There is no database and no UI-side state: a run started from a notebook appears in the UI the moment its directory does. Directories that do not conform are listed as rejected with the reason, never hidden.
- **Heavy data stays on the server.** Volume fields (tens of millions of cells) never reach the browser. The server extracts what is displayed (wall surfaces, slices, iso-surfaces, probes) and sends triangles and node values. Slices and probes are server-side NumPy operations, so the browser renderer only needs to draw triangle surfaces fast.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/runs` | `{runs: [run.json…], rejected: [{dir, reason}], progress: {run_id: iteration}}`, newest first. `run.json` is rewritten only when the status changes, so `progress` gives the last complete `history.jsonl` iteration of each running run. |
| `GET /api/runs/events` | SSE: `runs` (snapshot), then `run` (upsert, the new run.json), `removed` (run id), `rejected` (the whole list) and `progress` as they change |
| `GET /api/runs/{id}` | `{run, case, mesh_info}`: run.json, case.toml as JSON, mesh-info.json of an accepted run |
| `GET /api/runs/{id}/history/events` | SSE: `run`, then `history` batches of history.jsonl records. The event id is the byte offset reached, so a reconnecting `EventSource` resumes without loss or duplicates. A line still being written is held until its newline while the run is running; after it ends, an unterminated line ends the stream with `invalid`. Unknown runs are 404 and rejected ones 409 with the reason, before any stream starts. |
| `GET /api/surfaces/synthetic?triangles=N` | A synthetic wing surface for the viewer and its benchmark, until `solution.cgns` surfaces are served |

Live updates come from file-system events (`watchfiles`, 50 ms debounce) and a 1 s timeout: every wake-up rescans, so a change made before the watcher started is still reported. A directory is checked by `autan.Results.open` again only when its `run.json`, `case.toml`, `mesh-info.json` or `solution.cgns` changes; history.jsonl is re-read from the last offset. SSE rather than WebSocket: traffic is server → browser only, `EventSource` reconnects by itself, and FastAPI ships SSE natively.

### Surface message `autan-ui.surface/1`

An internal format between `autan_ui` and the web app (not a project contract): a little-endian `u32` header length, a UTF-8 JSON header padded to 4 bytes, then the buffers.

```json
{"schema": "autan-ui.surface/1", "origin": [x, y, z], "nodes": N, "triangles": T,
 "zones": [{"name", "first_triangle", "triangles"}],
 "fields": [{"name", "location": "node", "min", "max"}],
 "buffers": [{"name": "positions" | "indices" | "field:<name>", "dtype": "f32" | "u32", "offset", "length"}]}
```

Positions are FP32 relative to `origin` (FP64), so a model far from the mesh origin keeps sub-micrometre resolution. Triangles are sorted by zone, so a zone is one contiguous range: one draw group, one pick id. The browser wraps the buffers in typed-array views without copying and hands them to the GPU. Field values are per node; face-centred wall data (`WallSolution`) is averaged to nodes with face-area weights when real surfaces land. Arrow IPC was considered and rejected: a columnar table format and its library for what are three flat arrays.

## Stack

Chosen on the project's criteria: fluid 3D on 5–20 M faces, live data, keyboard-first UX, local-only, permissive licences, a small number of durable dependencies.

| Concern | Choice | Licence | Why, and what was rejected |
|---|---|---|---|
| Server | FastAPI 0.162 + uvicorn + watchfiles, Python ≥ 3.12, own `uv.lock` | MIT, BSD-3, MIT | Same language as the SDK, so `autan ui` is a function call and the server checks run directories with `autan.Results.open`; native SSE. |
| Package manager | Bun 1.3, text `bun.lock` | MIT | Fast installs, one binary; Vite and Vitest run unchanged. |
| Build | Vite 8 (Rolldown) + TypeScript 7 (native compiler) | MIT, Apache-2.0 | Sub-second builds and type checks. |
| UI | React 19 | MIT | Largest ecosystem for accessible primitives, palette and charts. |
| Routing, server state | TanStack Router (typed search params) + TanStack Query | MIT | Filters, camera and displayed field live in the URL (design principle 7) with types; SSE events write straight into the Query cache. |
| Styling | Tailwind v4 over `tokens.css` (`@theme inline`) | MIT | Utilities read the design tokens; no second source of values. Geist via `@fontsource-variable` (OFL-1.1), self-hosted. |
| Primitives | Base UI 1.9 | MIT | Unstyled, accessible components (toggle groups, menus, dialogs, popovers) styled by data attributes. |
| Palette | cmdk | MIT | ⌘K with fuzzy ranking and grouped results. |
| Motion | Motion 14 | MIT | Layout and presence animations (rows entering the registry); follows the system's reduced-motion setting. |
| Charts | uPlot | MIT | Canvas time series built for large, appended data; log axes; 50 kB. |
| 3D | three.js 0.186 `WebGPURenderer`, wrapped in one imperative `SurfaceViewer` class | MIT | Measured below. No react-three-fiber: the scene is one mesh per surface, and a declarative scene graph adds a reconciler for nothing. vtk.js rejected (below). |
| Lint, format | Biome 2.6 (repository-wide, with the docs site) | MIT/Apache-2.0 | One tool for TS, JSON and CSS. |
| Tests | Vitest 5, Playwright 1.64 on headless Chrome | MIT, Apache-2.0 | Playwright drives the real server, real files and the real GPU. |

`WebGPURenderer` falls back to WebGL2 where WebGPU is missing; the fallback is logged and shown in the viewer status line.

### 3D renderer: three.js against vtk.js

Same synthetic surface (swept NACA 0012 wing, structured triangulation, one node field), same 1600 × 1000 canvas, camera orbiting, headless Chrome 156 on an Apple M4 Pro (Metal), vsync off. Frame time is measured to GPU completion (`queue.onSubmittedWorkDone` on WebGPU, a 1-pixel `readPixels` on WebGL2), median of 180 frames after 20 warm-up frames. First frame (ms) runs from the surface message received to the first frame completed on the GPU, including geometry set-up (one bounding-sphere pass) and shader compilation. three.js renders with 4× MSAA, vtk.js with its defaults.

| Triangles | three.js WebGPU: first / frame | three.js WebGL2: first / frame | vtk.js 38 WebGL: first / frame |
|---|---|---|---|
| 1 M | 287 / 1.0 ms | 172 / 1.9 ms | 376 / 1.2 ms |
| 5 M | 190 / 2.4 ms | 100 / 3.1 ms | 1158 / 2.1 ms |
| 10 M | 457 / 3.3 ms | 223 / 4.2 ms | 2246 / 3.1 ms |
| 20 M | 782 / 5.3 ms | 413 / 6.2 ms | 4550 / 5.3 ms |

Once loaded, all three draw 20 M triangles in about 5–6 ms, far under a 16.7 ms frame. vtk.js takes 6–11× longer to show the first frame at 20 M, because it rebuilds its own vertex buffers on the CPU, and that cost returns at every change of surface or snapshot during a live run. Its WebGPU backend did not initialise in this setup. vtk.js's strengths (cutters, contour filters, transfer functions) are server-side operations in this architecture.

Transfer of the 20 M-triangle message (400 MB, loopback): 1.0–1.5 s; at 5 M triangles the whole path from request to first frame is about 0.4 s. The measurement uses a structured triangulation with good vertex locality; an unstructured surface will be slower to draw and is re-measured when `solution.cgns` surfaces are served.

Reproduce: `cd ui && bun run bench:viewer` writes one JSON per case to `ui/bench/results/`. The vtk.js comparison was a one-off prototype and is not kept.

### Viewer

- Geometry: `position`, `index` and the displayed field as a `float` attribute; no normals (flat shading from screen-space derivatives), so a 20 M-triangle surface costs 400 MB of buffers and one CPU pass (the bounding sphere) before the upload.
- Colour: the field is normalised by uniforms and looked up in a 256-texel colormap texture; changing the range or the colormap is a uniform update, not a buffer rebuild. A field that changes sign gets a range symmetric about zero, so the neutral colour marks zero and the legend shows it. Cool–warm (Moreland 2009) for signed fields; viridis for magnitudes comes with the first real fields.
- Picking: each zone is drawn with its id as colour into a 1 × 1 render target under the cursor, then read back: one GPU pass whatever the triangle count.
- Rendering is on demand (camera or state change), not a continuous loop.
- Buffer limits: the WebGPU device is requested with the adapter's `maxBufferSize`, since 10 M+ triangles exceed the 256 MiB default.

## Develop

```sh
cd ui
bun install --frozen-lockfile
uv sync --locked --project server
scripts/sample-runs.sh runs --live &                               # fake-backend runs, one live
uv run --project server python -m autan_ui --runs runs --no-open    # API on :8765
bun run dev                                                          # app on :5173, proxies /api
```

Checks: `bun run check` (Biome, `tsc`), `bun run test` (Vitest), `bun run e2e` (Playwright, builds the app and starts the server on synthetic runs), `uv run --project server pytest`, `bun run bench:viewer`.

Sample and test runs come from the SDK's fake backend (`autan run`, `autan.FakeBackend`), the real producer of `autan.results/1`, so the tests send its output through this server.
