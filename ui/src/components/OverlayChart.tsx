import { useEffect, useRef } from "react";
import uPlot from "uplot";

export interface OverlaySeries {
  label: string;
  color: string;
  x: number[];
  y: number[];
}

const number = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 4 });
const css = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Several runs on one chart, one series colour per run; cursors of charts with the same
 * `sync` key move together. `points` draws markers (polars), otherwise lines only. */
export function OverlayChart({
  series,
  xLabel,
  sync,
  points = false,
  version,
}: {
  series: OverlaySeries[];
  xLabel: string;
  sync?: string;
  points?: boolean;
  /** Changes whenever the data grows, since arrays are appended in place. */
  version: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const shape = series.map((s) => `${s.label}:${s.color}`).join("|");

  // biome-ignore lint/correctness/useExhaustiveDependencies: rebuilt only when the series set changes
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const axis = {
      stroke: css("--fg-3"),
      grid: { stroke: css("--line") },
      ticks: { stroke: css("--line") },
      // Numbers read the same everywhere in the app: point decimals, whatever the locale.
      values: (_: uPlot, splits: number[]) => splits.map((v) => number.format(v)),
    };
    const chart = new uPlot(
      {
        width: el.clientWidth,
        height: el.clientHeight,
        scales: { x: { time: false } },
        axes: [
          { ...axis, label: xLabel, labelSize: 20 },
          { ...axis, size: 56 },
        ],
        legend: { show: false },
        cursor: sync ? { sync: { key: sync } } : {},
        series: [
          { label: xLabel },
          ...series.map((s) => ({
            label: s.label,
            stroke: s.color,
            width: 1.75,
            spanGaps: true,
            points: { show: points, size: 6, fill: s.color },
          })),
        ],
      },
      [[]],
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
  }, [shape, xLabel, sync, points]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: `version` changes on every append
  useEffect(() => {
    if (!plot.current) return;
    const tables = series.map((s) => [s.x, s.y] as uPlot.AlignedData);
    plot.current.setData(tables.length ? uPlot.join(tables) : [[]]);
  }, [version, shape]);

  return <div ref={host} className="h-full w-full" />;
}
