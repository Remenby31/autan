import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { fetchSurface, type Surface } from "../api/surface";
import { TopBar } from "../components/shell/TopBar";
import { COOL_WARM, cssGradient, displayRange } from "../viewer/colormap";
import { type Backend, SurfaceViewer } from "../viewer/SurfaceViewer";

interface Search {
  triangles?: number;
  webgl?: boolean;
  bench?: boolean;
}

export const Route = createFileRoute("/viewer")({
  validateSearch: (raw: Record<string, unknown>): Search => {
    const search: Search = {};
    const triangles = Number(raw.triangles);
    if (Number.isFinite(triangles) && triangles > 0) search.triangles = triangles;
    if (raw.webgl === true || raw.webgl === "1") search.webgl = true;
    if (raw.bench === true || raw.bench === "1") search.bench = true;
    return search;
  },
  component: Viewer,
});

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

function Viewer() {
  const { triangles = 200_000, webgl = false, bench = false } = Route.useSearch();
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
    const surfaceRequest = fetchSurface(
      `/api/surfaces/synthetic?triangles=${triangles}`,
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
      v.setSurface(s, "Cp");
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
  }, [triangles, webgl, bench]);

  useEffect(() => viewer.current?.setSelectedZone(selected), [selected]);

  const onClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const zone = await viewer.current?.pick(e.clientX - rect.left, e.clientY - rect.top);
    setSelected(zone ?? null);
  };

  const field = surface?.fields[0];
  const range = field ? displayRange(field.min, field.max) : null;
  return (
    <>
      <TopBar
        crumbs={[{ label: "Analysis" }]}
        actions={
          <span className="text-fg-3 text-sm">
            Synthetic surface until solution.cgns surfaces are served
          </span>
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
          <aside data-testid="zones" className="absolute top-4 left-4 flex w-60 flex-col gap-1 rounded-lg border border-line bg-glass p-2 backdrop-blur">
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
            <span>{field.name} · synthetic</span>
            <div className="h-2 rounded" style={{ background: cssGradient(COOL_WARM) }} />
            <div className="flex justify-between">
              <span>{range.min.toFixed(2)}</span>
              {range.min < 0 && range.max > 0 && <span>0</span>}
              <span>{range.max.toFixed(2)}</span>
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
