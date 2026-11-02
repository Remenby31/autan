import type { PolarPoint, PolarSeries } from "../api/polars";
import { coefficient } from "../api/units";

const W = 480;
const H = 240;
const PAD = { left: 56, right: 12, top: 10, bottom: 36 };
/** Line styles telling series apart; points keep their run's colour. */
export const DASHES = ["", "6 3", "1.5 3", "8 3 1.5 3"];

// The shown range: the values with a margin, or a small band around a single value.
function extent(values: number[]): [number, number] {
  const low = Math.min(...values);
  const high = Math.max(...values);
  const scale = Math.max(Math.abs(low), Math.abs(high));
  const span = high - low;
  if (span <= scale * 1e-9 || span === 0) {
    const pad = scale * 0.05 || 1;
    return [low - pad, high + pad];
  }
  return [low - span * 0.08, high + span * 0.08];
}

// About five round steps (1, 2 or 5 times a power of ten) over [low, high], each written with
// the digits its step needs, so that neighbouring ticks never read the same: decimals for
// everyday magnitudes, an exponent for very small or large ones.
function ticks(low: number, high: number): { value: number; text: string }[] {
  const raw = (high - low) / 5;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 5, 10].find((m) => m * power >= raw) ?? 10) * power;
  const order = Math.floor(Math.log10(step));
  const first = Math.ceil(low / step);
  const count = Math.min(20, Math.floor(high / step) - first + 1);
  const largest = Math.max(Math.abs(low), Math.abs(high));
  const scientific = largest < 1e-4 || largest >= 1e7;
  const write = (v: number) =>
    v === 0
      ? "0"
      : scientific
        ? v.toExponential(Math.min(15, Math.max(0, Math.floor(Math.log10(Math.abs(v))) - order)))
        : v.toFixed(Math.max(0, -order));
  return Array.from({ length: Math.max(0, count) }, (_, k) => {
    const value = (first + k) * step;
    return { value, text: write(Math.abs(value) < step / 2 ? 0 : value) };
  });
}

const exact = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 6 });

/** A polar: every run a point in its run's colour, a line through each series' points in angle
 * order, one line style per series; a point's title names its run and values. */
export function PolarChart({
  series,
  x,
  y,
  xLabel,
  yLabel,
  color,
  testId,
}: {
  series: PolarSeries[];
  x: (p: PolarPoint) => number;
  y: (p: PolarPoint) => number;
  xLabel: string;
  yLabel: string;
  color: (p: PolarPoint) => string;
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
      aria-label={`${yLabel} against ${xLabel}`}
      data-testid={testId}
    >
      {ticks(x0, x1).map((t) => (
        <g key={`x${t.value}`} className="text-fg-3">
          <line
            x1={sx(t.value)}
            x2={sx(t.value)}
            y1={PAD.top}
            y2={H - PAD.bottom}
            stroke="var(--line)"
          />
          <text x={sx(t.value)} y={H - PAD.bottom + 14} textAnchor="middle" fill="currentColor">
            {t.text}
          </text>
        </g>
      ))}
      {ticks(y0, y1).map((t) => (
        <g key={`y${t.value}`} className="text-fg-3">
          <line
            x1={PAD.left}
            x2={W - PAD.right}
            y1={sy(t.value)}
            y2={sy(t.value)}
            stroke="var(--line)"
          />
          <text x={PAD.left - 6} y={sy(t.value) + 4} textAnchor="end" fill="currentColor">
            {t.text}
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
        <polyline
          key={s.key}
          points={s.points.map((p) => `${sx(x(p))},${sy(y(p))}`).join(" ")}
          fill="none"
          stroke="var(--fg-3)"
          strokeWidth={1.5}
          strokeDasharray={DASHES[k % DASHES.length]}
        />
      ))}
      {points.map((p) => (
        <circle
          key={p.id}
          data-run={p.id}
          cx={sx(x(p))}
          cy={sy(y(p))}
          r={4}
          fill={color(p)}
          stroke="var(--bg-0)"
          strokeWidth={1}
        >
          <title>{`${p.role} · α ${exact.format(p.alpha)}° · CL ${coefficient(p.cl)} · CD ${coefficient(p.cd)}`}</title>
        </circle>
      ))}
    </svg>
  );
}
