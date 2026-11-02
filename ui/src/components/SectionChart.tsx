const W = 360;
const H = 200;
const PAD = { left: 44, right: 10, top: 22, bottom: 30 };

// A tick value with the digits its step needs, so that neighbouring ticks never read the same.
export function tickLabel(v: number, step: number): string {
  if (v === 0) return "0";
  const digits = Math.max(1, Math.ceil(Math.log10(Math.abs(v) / step)) + 1);
  return String(Number(v.toPrecision(Math.min(15, digits))));
}

/** A field along a section's chord: each cut segment drawn against x/c. Cp is drawn as −Cp,
 * suction up, as aerodynamicists read it. */
export function SectionChart({
  segments,
  name,
}: {
  segments: [number, number, number, number][];
  name: string;
}) {
  const sign = name === "Cp" ? -1 : 1;
  const shownName = sign === -1 ? `−${name}` : name;
  const drawn = segments.map(([x0, v0, x1, v1]) => [x0, sign * v0, x1, sign * v1] as const);
  // Extrema by a loop: a section can hold more points than a call has arguments.
  let least = Number.POSITIVE_INFINITY;
  let greatest = Number.NEGATIVE_INFINITY;
  let peakAt = 0;
  for (const [x0, a, x1, b] of drawn) {
    least = Math.min(least, a, b);
    if (a > greatest) [greatest, peakAt] = [a, x0];
    if (b > greatest) [greatest, peakAt] = [b, x1];
  }
  let low = least;
  let high = greatest;
  if (high - low < 1e-12 * Math.max(1, Math.abs(high))) {
    low -= 0.5;
    high += 0.5;
  }
  const sx = (f: number) => PAD.left + f * (W - PAD.left - PAD.right);
  const sy = (v: number) => {
    const t = (v - low) / (high - low);
    return H - PAD.bottom - t * (H - PAD.top - PAD.bottom);
  };
  const ticks = [low, (low + high) / 2, high];
  const step = (high - low) / 2;
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full font-mono text-[10px]"
      role="img"
      aria-label={`${shownName} along the chord, x/c from 0 to 1`}
      data-testid="section-chart"
      data-segments={segments.length}
      data-least={least}
      data-greatest={greatest}
    >
      <desc>
        {`${shownName} from ${tickLabel(least, step / 100)} to ${tickLabel(greatest, step / 100)}, highest at x/c ${tickLabel(peakAt, 0.001)}; ${segments.length} segments.`}
      </desc>
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <g key={`x${f}`} className="text-fg-3">
          <line x1={sx(f)} x2={sx(f)} y1={PAD.top} y2={H - PAD.bottom} stroke="var(--line)" />
          <text x={sx(f)} y={H - PAD.bottom + 12} textAnchor="middle" fill="currentColor">
            {f}
          </text>
        </g>
      ))}
      {ticks.map((v) => (
        <g key={`y${v}`} className="text-fg-3">
          <line x1={PAD.left} x2={W - PAD.right} y1={sy(v)} y2={sy(v)} stroke="var(--line)" />
          <text x={PAD.left - 4} y={sy(v) + 3} textAnchor="end" fill="currentColor">
            {tickLabel(v, step)}
          </text>
        </g>
      ))}
      <text
        x={(PAD.left + W - PAD.right) / 2}
        y={H - 4}
        textAnchor="middle"
        className="fill-fg-2 font-sans"
      >
        x/c
      </text>
      <text x={4} y={12} className="fill-fg-2 font-sans">
        {shownName}
      </text>
      {drawn.map(([x0, v0, x1, v1], k) => (
        <line
          // biome-ignore lint/suspicious/noArrayIndexKey: segments have no identity of their own
          key={k}
          x1={sx(x0)}
          y1={sy(v0)}
          x2={sx(x1)}
          y2={sy(v1)}
          stroke="var(--accent)"
          strokeWidth={1.5}
          strokeLinecap="round"
        />
      ))}
    </svg>
  );
}
