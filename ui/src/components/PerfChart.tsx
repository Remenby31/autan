import { useEffect, useRef } from "react";
import uPlot from "uplot";

const css = (name: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** One benchmark's % of peak over main's commits, oldest left; commits a bisection blamed are
 * marked. `onHover` gets the commit under the cursor, null when it leaves. */
export function PerfChart({
  percent,
  culprits,
  dates,
  onHover,
}: {
  percent: (number | null)[];
  culprits: boolean[];
  dates: string[];
  onHover: (index: number | null) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const hover = useRef(onHover);
  hover.current = onHover;

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const axis = {
      stroke: css("--fg-3"),
      grid: { stroke: css("--line") },
      ticks: { stroke: css("--line") },
    };
    const x = percent.map((_, k) => k);
    const blamed = percent.map((p, k) => (culprits[k] ? p : null));
    const top = Math.max(100, ...percent.filter((p): p is number => p !== null));
    const chart = new uPlot(
      {
        width: el.clientWidth,
        height: el.clientHeight,
        scales: { x: { time: false }, y: { range: [0, Math.ceil(top / 10) * 10] } },
        axes: [
          {
            ...axis,
            // A date where the day changes; ticks between commits stay blank.
            values: (_: uPlot, splits: number[]) => {
              let shown: string | undefined;
              return splits.map((v) => {
                const day = Number.isInteger(v) ? dates[v]?.slice(5, 10) : undefined;
                if (!day || day === shown) return "";
                shown = day;
                return day;
              });
            },
          },
          { ...axis, size: 48, values: (_: uPlot, s: number[]) => s.map((v) => `${v} %`) },
        ],
        legend: { show: false },
        cursor: { points: { size: 8 } },
        padding: [12, 8, 0, 0],
        series: [
          {},
          {
            stroke: css("--accent"),
            width: 1.75,
            spanGaps: true,
            points: { show: true, size: 5, fill: css("--accent") },
          },
          { stroke: "transparent", points: { show: true, size: 9, fill: css("--danger") } },
        ],
        hooks: {
          setCursor: [(u) => hover.current(u.cursor.idx ?? null)],
        },
      },
      [x, percent, blamed],
      el,
    );
    const resize = new ResizeObserver(() =>
      chart.setSize({ width: el.clientWidth, height: el.clientHeight }),
    );
    resize.observe(el);
    chart.over.addEventListener("mouseleave", () => hover.current(null));
    return () => {
      resize.disconnect();
      chart.destroy();
    };
  }, [percent, culprits, dates]);

  return (
    <div
      ref={host}
      className="h-full w-full"
      data-samples={percent.filter((p) => p !== null).length}
    />
  );
}
