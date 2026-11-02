/** Where each value sits in a `width` × `height` box: x by its position in `values`, so a
 * missing value (null) leaves its place empty rather than moving the others. */
export function sparkPoints(
  values: (number | null)[],
  width: number,
  height: number,
  [lo, hi]: [number, number],
): [number, number][] {
  const span = hi - lo || 1;
  const x = (k: number) => (k / Math.max(values.length - 1, 1)) * (width - 2) + 1;
  const y = (v: number) => height - 1 - ((v - lo) / span) * (height - 2);
  return values.flatMap((v, k) => (v === null ? [] : [[x(k), y(v)] as [number, number]]));
}

/** A trend without axes, for a row or a tile. `target` draws a dashed line the data aims at;
 * `domain` fixes the vertical range, which otherwise fits the data; null values are gaps. */
export function Sparkline({
  values,
  width = 96,
  height = 24,
  target,
  color = "var(--fg-2)",
  label,
  domain,
}: {
  values: (number | null)[];
  width?: number;
  height?: number;
  target?: number;
  color?: string;
  label: string;
  domain?: [number, number];
}) {
  const present = values.filter((v): v is number => v !== null);
  if (present.length < 2) return <svg width={width} height={height} aria-hidden />;
  const lo = domain?.[0] ?? Math.min(...present, target ?? Number.POSITIVE_INFINITY);
  const hi = domain?.[1] ?? Math.max(...present, target ?? Number.NEGATIVE_INFINITY);
  const y = (v: number) => height - 1 - ((v - lo) / (hi - lo || 1)) * (height - 2);
  const placed = sparkPoints(values, width, height, [lo, hi]);
  const points = placed.map(([px, py]) => `${px.toFixed(1)},${py.toFixed(1)}`).join(" ");
  const end = placed.at(-1) as [number, number];
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={label}
    >
      {target !== undefined && (
        <line
          x1={0}
          x2={width}
          y1={y(target)}
          y2={y(target)}
          stroke="var(--line-strong)"
          strokeDasharray="2 2"
        />
      )}
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinejoin="round"
      />
      <circle cx={end[0]} cy={end[1]} r={2} fill={color} />
    </svg>
  );
}
