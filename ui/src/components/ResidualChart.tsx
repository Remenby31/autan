import { useEffect, useRef } from "react";
import uPlot from "uplot";
import type { HistoryColumns } from "../api/runs";
import { Tooltip } from "./ui/Tooltip";

const css = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const series = (k: number) => css(`--series-${(k % 8) + 1}`);

// An equation satisfied identically (the spanwise momentum of a one-cell-thick slab) leaves
// round-off residuals near 1e-33, or exactly zero. The axis spans whole decades, at most DECADES
// below its top; zero, which a log axis cannot place, is a gap.
const DECADES = 20;
const positive = (values: number[]) => values.map((v) => (v > 0 ? v : null));
const range = (
  _: uPlot,
  data: number | null,
  max: number | null,
  floors: number[],
): uPlot.Range.MinMax => {
  const min = Math.min(data ?? Number.POSITIVE_INFINITY, ...floors);
  if (!Number.isFinite(min) || max == null || !(max > 0)) return [data, max];
  const top = Math.ceil(Math.log10(max));
  const bottom = Math.max(Math.floor(Math.log10(Math.max(min, Number.MIN_VALUE))), top - DECADES);
  return [10 ** Math.min(bottom, top - 1), 10 ** top];
};

// Ticks, labels and grid lines on whole decades, every n-th past LABELS of them. uPlot's own
// log splits are not used: they overflow on decades far below 1.
const LABELS = 8;
const decades = (_: uPlot, __: number, min: number, max: number) => {
  if (!(min > 0 && max > min)) return [];
  const low = Math.ceil(Math.log10(min) - 1e-9);
  const high = Math.floor(Math.log10(max) + 1e-9);
  const step = Math.ceil((high - low + 1) / LABELS);
  const splits = [];
  for (let k = Math.ceil(low / step) * step; k <= high; k += step) splits.push(10 ** k);
  return splits;
};

/** Residuals on a log axis, with the round-off floor of each equation dashed in its colour
 * (run.json `convergence.floor`); redraws in place as the history grows. */
export function ResidualChart({
  history,
  floors,
}: {
  history: HistoryColumns;
  floors?: Record<string, number> | undefined;
}) {
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const floorsRef = useRef(floors);
  floorsRef.current = floors;
  const names = [...history.residuals.keys()];
  const key = names.join(",");

  // biome-ignore lint/correctness/useExhaustiveDependencies: the plot is rebuilt only when the equations change
  useEffect(() => {
    const el = host.current;
    if (!el || !names.length) return;
    const chart = new uPlot(
      {
        width: el.clientWidth,
        height: el.clientHeight,
        scales: {
          x: { time: false },
          y: {
            distr: 3,
            range: (u, min, max) =>
              range(
                u,
                min,
                max,
                Object.values(floorsRef.current ?? {}).filter((v) => v > 0),
              ),
          },
        },
        hooks: {
          draw: [
            (u) => {
              const ctx = u.ctx;
              ctx.save();
              ctx.setLineDash([4 * devicePixelRatio, 4 * devicePixelRatio]);
              ctx.lineWidth = devicePixelRatio;
              names.forEach((name, k) => {
                const floor = floorsRef.current?.[name];
                if (floor === undefined || !(floor > 0)) return;
                const y = Math.round(u.valToPos(floor, "y", true));
                ctx.strokeStyle = series(k);
                ctx.beginPath();
                ctx.moveTo(u.bbox.left, y);
                ctx.lineTo(u.bbox.left + u.bbox.width, y);
                ctx.stroke();
              });
              ctx.restore();
            },
          ],
        },
        axes: [
          { stroke: css("--fg-3"), grid: { stroke: css("--line") } },
          {
            stroke: css("--fg-3"),
            grid: { stroke: css("--line") },
            size: 60,
            splits: decades,
            values: (_, splits) => splits.map((v) => (v == null ? "" : v.toExponential(0))),
          },
        ],
        legend: { show: false },
        cursor: { drag: { x: true, y: false } },
        series: [
          { label: "iteration" },
          ...names.map((name, k) => ({
            label: name,
            stroke: series(k),
            points: { show: false },
            width: 1.5,
          })),
        ],
      },
      [[], ...names.map(() => [])],
      el,
    );
    plot.current = chart;
    const resize = new ResizeObserver(() =>
      chart.setSize({ width: el.clientWidth, height: el.clientHeight }),
    );
    resize.observe(el);
    return () => {
      resize.disconnect();
      chart.destroy();
      plot.current = null;
    };
  }, [key]);

  // A new floor rescales the axis: the data is set again, which recomputes the range.
  const floorsKey = JSON.stringify(floors ?? null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `version` changes on every append
  useEffect(() => {
    plot.current?.setData([
      history.iteration,
      ...names.map((name) => positive(history.residuals.get(name) ?? [])),
    ]);
  }, [history.version, key, floorsKey]);

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-fg-3 text-xs">
        {names.map((name, k) =>
          history.exempt.has(name) ? (
            <Tooltip
              key={name}
              content="Identically zero: exempt while below 1e-12 × the largest initial residual"
            >
              <span className="flex items-center gap-1.5 opacity-50" data-exempt>
                <Swatch k={k} />
                {name}
                <span className="font-sans">exempt</span>
              </span>
            </Tooltip>
          ) : (
            <span key={name} className="flex items-center gap-1.5">
              <Swatch k={k} />
              {name}
            </span>
          ),
        )}
      </div>
      <div ref={host} className="min-h-0 flex-1" data-testid="residual-chart" />
    </div>
  );
}

function Swatch({ k }: { k: number }) {
  return (
    <i className="inline-block h-0.5 w-3" style={{ background: `var(--series-${(k % 8) + 1})` }} />
  );
}
