import { useQuery } from "@tanstack/react-query";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { Box } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { runsQuery } from "../api/runs";
import { fetchSurface, type Surface } from "../api/surface";
import { TopBar } from "../components/shell/TopBar";
import { useResolvedTheme } from "../components/shell/theme";
import { Button } from "../components/ui/Button";
import { EmptyState } from "../components/ui/EmptyState";
import { Note } from "../components/ui/Note";
import { RelativeTime } from "../components/ui/RelativeTime";
import { Status } from "../components/ui/Status";
import { TabList, TabsRoot } from "../components/ui/Tabs";
import { COOL_WARM, colormapFor, cssGradient, displayRange } from "../viewer/colormap";
import { type Backend, SurfaceViewer } from "../viewer/SurfaceViewer";

interface Search {
  run?: string;
  field?: string;
  triangles?: number;
  webgl?: boolean;
  bench?: boolean;
}

export const Route = createFileRoute("/viewer")({
  validateSearch: (raw: Record<string, unknown>): Search => {
    const search: Search = {};
    if (typeof raw.run === "string" && raw.run) search.run = raw.run;
    if (typeof raw.field === "string" && raw.field) search.field = raw.field;
    const triangles = Number(raw.triangles);
    if (Number.isFinite(triangles) && triangles > 0) search.triangles = triangles;
    if (raw.webgl === true || raw.webgl === 1 || raw.webgl === "1") search.webgl = true;
    if (raw.bench === true || raw.bench === 1 || raw.bench === "1") search.bench = true;
    return search;
  },
  component: Analysis,
});

// A run's solution, or the synthetic benchmark surface when asked for by `bench` or
// `triangles`; otherwise, the runs there is a solution to show.
function Analysis() {
  const { run, triangles, bench } = Route.useSearch();
  return run || bench || triangles ? <Viewer /> : <RunPicker />;
}

function RunPicker() {
  const registry = useQuery(runsQuery);
  const solutions = new Set(registry.data?.solutions);
  const solved = (registry.data?.runs ?? [])
    .filter((r) => solutions.has(r.run_id))
    .sort((a, b) => b.started.localeCompare(a.started));
  return (
    <>
      <TopBar crumbs={[{ label: "Analysis" }]} />
      <div className="flex min-h-0 flex-1 flex-col items-center overflow-auto px-8 py-10">
        {registry.isError && !registry.data ? (
          <Note tone="error" title="The runs could not be listed">
            {registry.error.message}{" "}
            <Button size="sm" onClick={() => void registry.refetch()}>
              Retry
            </Button>
          </Note>
        ) : registry.isSuccess && solved.length === 0 ? (
          <EmptyState
            icon={<Box className="size-5" />}
            title="No solution to analyse yet"
            description="Analysis shows the solution a run writes when it ends."
          >
            <Link to="/" className="text-accent">
              Go to Runs
            </Link>
          </EmptyState>
        ) : (
          <section className="flex w-full max-w-2xl flex-col gap-3" data-testid="run-picker">
            <h1 className="m-0 font-medium text-md">Choose a run to analyse</h1>
            <ul className="m-0 flex list-none flex-col p-0">
              {solved.map((r) => (
                <li key={r.run_id}>
                  <Link
                    to="/viewer"
                    search={{ run: r.run_id }}
                    className="flex items-center gap-4 rounded-md px-3 py-2 hover:bg-bg-2"
                  >
                    <Status status={r.status} />
                    <span className="min-w-0 flex-1 truncate">{r.case.name}</span>
                    <span className="font-mono text-fg-3 text-sm">{r.run_id.slice(-8)}</span>
                    <span className="w-28 text-right text-fg-3 text-sm">
                      <RelativeTime iso={r.started} card={false} />
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </>
  );
}

export interface BenchReport {
  backend: Backend;
  triangles: number;
  nodes: number;
  fetchMs: number;
  firstFrameMs: number;
  frameMs: number[];
}

declare global {
  interface Window {
    __bench?: BenchReport;
  }
}

const BENCH_FRAMES = 200;
const UNITS: Record<string, string> = { Pressure: "Pa" };
const format = (v: number) =>
  Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-2) ? v.toExponential(2) : v.toFixed(3);

function Viewer() {
  const {
    run,
    field: fieldName = "Cp",
    triangles = 200_000,
    webgl = false,
    bench = false,
  } = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const registry = useQuery(runsQuery);
  const runDoc = registry.data?.runs.find((r) => r.run_id === run);
  const canvas = useRef<HTMLCanvasElement>(null);
  const viewer = useRef<SurfaceViewer | null>(null);
  const [surface, setSurface] = useState<Surface | null>(null);
  const [backend, setBackend] = useState<Backend | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const abort = new AbortController();
    let disposed = false;
    // A new run starts from nothing: the previous surface's fields must not reach this viewer.
    setSurface(null);
    setSelected(null);
    setError(null);
    const surfaceRequest = fetchSurface(
      run ? `/api/runs/${run}/surface` : `/api/surfaces/synthetic?triangles=${triangles}`,
      abort.signal,
    );
    surfaceRequest.catch(() => {});
    (async () => {
      const t0 = performance.now();
      const v = await SurfaceViewer.create(el, { forceWebGL: webgl });
      if (disposed) {
        v.dispose();
        return;
      }
      viewer.current = v;
      setBackend(v.backend);
      const s = await surfaceRequest;
      const fetched = performance.now();
      setSurface(s);
      // The displayed field is applied by the effect below, so changing it reloads nothing.
      v.setSurface(s, s.fields[0]?.name ?? "Cp");
      if (!bench) return;
      await v.benchmark(1);
      const firstFrame = performance.now() - fetched;
      window.__bench = {
        backend: v.backend,
        triangles: s.indices.length / 3,
        nodes: s.positions.length / 3,
        fetchMs: fetched - t0,
        firstFrameMs: firstFrame,
        frameMs: await v.benchmark(BENCH_FRAMES),
      };
    })().catch((e: unknown) => {
      if (!disposed) setError(String(e));
    });
    return () => {
      disposed = true;
      abort.abort();
      viewer.current?.dispose();
      viewer.current = null;
    };
  }, [run, triangles, webgl, bench]);

  useEffect(() => {
    if (surface?.fields.some((f) => f.name === fieldName)) viewer.current?.setField(fieldName);
  }, [surface, fieldName]);

  useEffect(() => viewer.current?.setSelectedZone(selected), [selected]);

  const theme = useResolvedTheme();
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs when the displayed theme changes
  useEffect(() => {
    // After this commit the shell has set data-theme, so the tokens read below are current.
    const frame = requestAnimationFrame(() => viewer.current?.applyTheme());
    return () => cancelAnimationFrame(frame);
  }, [theme]);

  const onClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const zone = await viewer.current?.pick(e.clientX - rect.left, e.clientY - rect.top);
    setSelected(zone ?? null);
  };

  const field = surface?.fields.find((f) => f.name === fieldName) ?? surface?.fields[0];
  const range = field ? displayRange(field.min, field.max) : null;
  const map = field ? colormapFor(field.min, field.max) : COOL_WARM;
  return (
    <>
      <TopBar
        crumbs={
          run
            ? [
                { label: "Runs", to: "/" },
                { label: runDoc?.case.name ?? "…", to: `/runs/${run}` },
                { label: <span className="font-mono">{run.slice(-8)}</span>, to: `/runs/${run}` },
                { label: "Analysis" },
              ]
            : [{ label: "Analysis", to: "/viewer" }, { label: "Synthetic surface" }]
        }
        actions={
          surface && surface.fields.length > 1 ? (
            <TabsRoot
              value={field?.name ?? fieldName}
              onValueChange={(value) =>
                void navigate({
                  search: (s: Search) => ({ ...s, field: String(value) }),
                  replace: true,
                })
              }
            >
              <TabList tabs={surface.fields.map((f) => ({ value: f.name, label: f.name }))} />
            </TabsRoot>
          ) : !run ? (
            <span className="text-fg-3 text-sm">Synthetic surface · viewer benchmark</span>
          ) : null
        }
      />
      <div className="relative min-h-0 flex-1">
        <canvas
          ref={canvas}
          onClick={onClick}
          className="block h-full w-full touch-none"
          aria-label="3D surface"
          data-testid="viewport"
        />
        {error && <p className="absolute top-4 left-4 m-0 text-danger">{error}</p>}
        {surface && (
          <aside
            data-testid="zones"
            className="absolute top-4 left-4 flex w-60 flex-col gap-1 rounded-lg border border-line bg-glass p-2 backdrop-blur"
          >
            {surface.zones.map((zone, k) => (
              <button
                type="button"
                key={zone.name}
                aria-pressed={selected === k}
                onClick={() => setSelected(selected === k ? null : k)}
                className="flex items-center justify-between rounded-md px-2 py-1.5 text-left font-mono text-sm hover:bg-bg-2 aria-pressed:bg-accent-soft aria-pressed:text-accent"
              >
                {zone.name}
                <span className="text-fg-3 text-xs">{zone.triangles.toLocaleString("en")}</span>
              </button>
            ))}
          </aside>
        )}
        {field && range && (
          <div className="pointer-events-none absolute right-4 bottom-10 flex w-56 flex-col gap-1 font-mono text-fg-3 text-xs">
            <span data-testid="legend">
              {field.name}
              {UNITS[field.name] ? ` (${UNITS[field.name]})` : ""}
              {run ? "" : " · synthetic"}
            </span>
            <div className="h-2 rounded" style={{ background: cssGradient(map) }} />
            <div className="flex justify-between">
              {field.max - field.min <= 1e-9 * Math.max(1, Math.abs(field.max)) ? (
                <span className="w-full text-center">uniform {format(field.min)}</span>
              ) : (
                <>
                  <span>{format(range.min)}</span>
                  {range.min < 0 && range.max > 0 && <span>0</span>}
                  <span>{format(range.max)}</span>
                </>
              )}
            </div>
          </div>
        )}
        <div
          className="pointer-events-none absolute bottom-3 left-4 font-mono text-fg-3 text-xs"
          data-testid="viewer-status"
          data-backend={backend ?? ""}
        >
          {backend ?? "…"} · {surface ? (surface.indices.length / 3).toLocaleString("en") : "–"}{" "}
          triangles
        </div>
      </div>
    </>
  );
}
