import { useEffect, useRef, useState } from "react";
import { fetchSurface, type Surface } from "../../api/surface";
import { SurfaceViewer } from "../../viewer/SurfaceViewer";
import { useResolvedTheme } from "../shell/theme";

/** The boundary of a mesh in 3D, one zone per boundary zone; a click picks the zone under the
 * cursor. `selected` is a zone name, highlighted when the mesh has it. */
export function MeshView({
  mesh,
  selected,
  onPick,
  webgl,
}: {
  mesh: string;
  selected: string | null;
  onPick: (zone: string) => void;
  webgl: boolean;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const viewer = useRef<SurfaceViewer | null>(null);
  const [surface, setSurface] = useState<Surface | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const abort = new AbortController();
    let disposed = false;
    setSurface(null);
    setError(null);
    const request = fetchSurface(`/api/meshes/${encodeURIComponent(mesh)}/surface`, abort.signal);
    request.catch(() => {});
    (async () => {
      const v = await SurfaceViewer.create(el, { forceWebGL: webgl });
      if (disposed) {
        v.dispose();
        return;
      }
      viewer.current = v;
      const s = await request;
      setSurface(s);
      v.setSurface(s, "");
    })().catch((e: unknown) => {
      if (!disposed) setError(e instanceof Error ? e.message : String(e));
    });
    return () => {
      disposed = true;
      abort.abort();
      viewer.current?.dispose();
      viewer.current = null;
    };
  }, [mesh, webgl]);

  useEffect(() => {
    const index = surface?.zones.findIndex((z) => z.name === selected) ?? -1;
    viewer.current?.setSelectedZone(index >= 0 ? index : null);
  }, [surface, selected]);

  const theme = useResolvedTheme();
  // biome-ignore lint/correctness/useExhaustiveDependencies: runs when the displayed theme changes
  useEffect(() => {
    const frame = requestAnimationFrame(() => viewer.current?.applyTheme());
    return () => cancelAnimationFrame(frame);
  }, [theme]);

  return (
    <div className="relative h-72 flex-none overflow-hidden rounded-lg border border-line bg-bg-0">
      <canvas
        ref={canvas}
        className="block size-full"
        data-testid="mesh-view"
        data-ready={surface ? "true" : undefined}
        aria-label={`Boundary of ${mesh}; click a zone to set its condition`}
        onClick={async (e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const index = await viewer.current?.pick(e.clientX - rect.left, e.clientY - rect.top);
          const zone = index == null ? undefined : surface?.zones[index]?.name;
          if (zone) onPick(zone);
        }}
      />
      {error && (
        <p className="absolute inset-x-0 bottom-0 m-0 bg-bg-1/90 px-3 py-2 text-fg-2 text-sm">
          {error.includes("404")
            ? "No boundary surface for this mesh: write it with autan-mesh-info --surface."
            : error}
        </p>
      )}
      {selected && surface?.zones.some((z) => z.name === selected) && (
        <span className="absolute top-2 left-2 rounded-sm bg-bg-1/90 px-2 py-1 font-mono text-sm">
          {selected}
        </span>
      )}
    </div>
  );
}
