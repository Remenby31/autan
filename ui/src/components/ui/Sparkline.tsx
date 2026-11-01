/** A trend without axes, for a row or a tile. `target` draws a dashed line the data aims at. */
export function Sparkline({
  values,
  width = 96,
  height = 24,
  target,
  color = "var(--fg-2)",
  label,
}: {
  values: number[];
  width?: number;
  height?: number;
  target?: number;
  color?: string;
  label: string;
}) {
  if (values.length < 2) return <svg width={width} height={height} aria-hidden />;
  const lo = Math.min(...values, target ?? Number.POSITIVE_INFINITY);
  const hi = Math.max(...values, target ?? Number.NEGATIVE_INFINITY);
  const span = hi - lo || 1;
  const x = (k: number) => (k / (values.length - 1)) * (width - 2) + 1;
  const y = (v: number) => height - 1 - ((v - lo) / span) * (height - 2);
  const points = values.map((v, k) => `${x(k).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
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
      <circle cx={x(values.length - 1)} cy={y(values.at(-1) ?? 0)} r={2} fill={color} />
    </svg>
  );
}
