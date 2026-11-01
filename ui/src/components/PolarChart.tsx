import type { PolarPoint, PolarSeries } from "../api/polars";

const number = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 4 });
const W = 480;
const H = 240;
const PAD = { left: 52, right: 12, top: 10, bottom: 36 };

// About five round steps covering [low, high] (1, 2 or 5 times a power of ten).
function ticks(low: number, high: number): number[] {
  const span = high - low || Math.abs(high) || 1;
  const raw = span / 5;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 5, 10].find((m) => m * power >= raw) ?? 10) * power;
  const out = [];
  for (let v = Math.ceil(low / step) * step; v <= high + step * 1e-9; v += step)
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

function extent(values: number[]): [number, number] {
  const low = Math.min(...values);
  const high = Math.max(...values);
  const pad = (high - low || Math.abs(high) || 1) * 0.08;
  return [low - pad, high + pad];
}

/** A polar: every run a marked point, a line through each series' points in angle order,
 * coloured per series; a point's title names its run. */
export function PolarChart({
  series,
  x,
  y,
  xLabel,
  colors,
  testId,
}: {
  series: PolarSeries[];
  x: (p: PolarPoint) => number;
  y: (p: PolarPoint) => number;
  xLabel: string;
  colors: string[];
  testId: string;
}) {
  const points = series.flatMap((s) => s.points);
  const [x0, x1] = extent(points.map(x));
  const [y0, y1] = extent(points.map(y));
  const sx = (v: number) => PAD.left + ((v - x0) / (x1 - x0)) * (W - PAD.left - PAD.right);
  const sy = (v: number) => H - PAD.bottom - ((v - y0) / (y1 - y0)) * (H - PAD.top - PAD.bottom);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-full w-full overflow-visible font-mono text-[11px]"
      role="img"
      aria-label={`${xLabel} polar`}
      data-testid={testId}
    >
      {ticks(x0, x1).map((t) => (
        <g key={`x${t}`} className="text-fg-3">
          <line x1={sx(t)} x2={sx(t)} y1={PAD.top} y2={H - PAD.bottom} stroke="var(--line)" />
          <text x={sx(t)} y={H - PAD.bottom + 14} textAnchor="middle" fill="currentColor">
            {number.format(t)}
          </text>
        </g>
      ))}
      {ticks(y0, y1).map((t) => (
        <g key={`y${t}`} className="text-fg-3">
          <line x1={PAD.left} x2={W - PAD.right} y1={sy(t)} y2={sy(t)} stroke="var(--line)" />
          <text x={PAD.left - 6} y={sy(t) + 4} textAnchor="end" fill="currentColor">
            {number.format(t)}
          </text>
        </g>
      ))}
      <text
        x={(PAD.left + W - PAD.right) / 2}
        y={H - 4}
        textAnchor="middle"
        className="fill-fg-2 font-sans"
      >
        {xLabel}
      </text>
      {series.map((s, k) => (
        <g key={s.key} style={{ color: colors[k] }}>
          <polyline
            points={s.points.map((p) => `${sx(x(p))},${sy(y(p))}`).join(" ")}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.75}
          />
          {s.points.map((p) => (
            <circle
              key={p.id}
              data-run={p.id}
              cx={sx(x(p))}
              cy={sy(y(p))}
              r={3.5}
              fill="currentColor"
            >
              <title>{`${p.role} · α ${number.format(p.alpha)}° · CL ${number.format(p.cl)} · CD ${number.format(p.cd)}`}</title>
            </circle>
          ))}
        </g>
      ))}
    </svg>
  );
}
