import { Meter } from "@base-ui/react/meter";

/** Progress towards a target as a ring; the colour says whether the target is met. */
export function Gauge({
  value,
  max,
  size = 32,
  label,
}: {
  value: number;
  max: number;
  size?: number;
  label: string;
}) {
  const ratio = max > 0 ? Math.min(Math.max(value / max, 0), 1) : 0;
  const stroke = Math.max(2.5, size / 10);
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const color = ratio >= 1 ? "var(--success)" : "var(--accent)";
  return (
    <Meter.Root
      value={Math.min(value, max)}
      max={max}
      aria-label={label}
      className="inline-flex flex-none"
    >
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        aria-hidden
        className="-rotate-90 flex-none"
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--bg-3)"
          strokeWidth={stroke}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - ratio)}
          style={{ transition: "stroke-dashoffset var(--dur-slow) var(--ease-out)" }}
        />
      </svg>
    </Meter.Root>
  );
}
