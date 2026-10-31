import { useEffect, useRef } from "react";
import uPlot from "uplot";
import type { HistoryColumns } from "../api/runs";

const css = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const series = (k: number) => css(`--series-${(k % 8) + 1}`);

/** Residuals on a log axis; redraws in place as the history grows. */
export function ResidualChart({ history }: { history: HistoryColumns }) {
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
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
        scales: { x: { time: false }, y: { distr: 3 } },
        axes: [
          { stroke: css("--fg-3"), grid: { stroke: css("--line") } },
          {
            stroke: css("--fg-3"),
            grid: { stroke: css("--line") },
            size: 60,
            values: (_, splits) =>
              splits.map((v) => (Number.isInteger(Math.log10(v)) ? v.toExponential(0) : "")),
          },
        ],
        legend: { show: false },
        cursor: { drag: { x: true, y: false } },
        series: [
          { label: "iteration" },
          ...names.map((name, k) => ({
            label: name,
            stroke: series(k),
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

  // biome-ignore lint/correctness/useExhaustiveDependencies: `version` changes on every append
  useEffect(() => {
    plot.current?.setData([
      history.iteration,
      ...names.map((name) => history.residuals.get(name) ?? []),
    ]);
  }, [history.version, key]);

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-fg-3 text-xs">
        {names.map((name, k) => (
          <span key={name} className="flex items-center gap-1.5">
            <i
              className="inline-block h-0.5 w-3"
              style={{ background: `var(--series-${(k % 8) + 1})` }}
            />
            {name}
          </span>
        ))}
      </div>
      <div ref={host} className="min-h-0 flex-1" data-testid="residual-chart" />
    </div>
  );
}
