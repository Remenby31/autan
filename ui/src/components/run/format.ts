export function formatBytes(bytes: number): string {
  const units = ["B", "kB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${value.toFixed(unit && value < 10 ? 1 : 0)} ${units[unit]}`;
}

export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)} M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e5 ? 0 : 1)} k`;
  return String(n);
}

/** Milliseconds left for a running steady run: the sooner of the iteration budget at the current
 * pace and the residual target extrapolated from the recent drop rate; null when neither applies. */
export function remainingMs({
  iteration,
  elapsedMs,
  maxIterations,
  orders,
  iterations,
  targetOrders,
}: {
  iteration: number;
  elapsedMs: number;
  maxIterations: number | undefined;
  orders: number[];
  iterations: number[];
  targetOrders: number | undefined;
}): number | null {
  if (iteration <= 0 || elapsedMs <= 0) return null;
  const msPerIteration = elapsedMs / iteration;
  const candidates: number[] = [];
  if (maxIterations !== undefined) candidates.push((maxIterations - iteration) * msPerIteration);
  const n = orders.length;
  if (targetOrders !== undefined && n >= 10) {
    const back = Math.max(0, n - 50);
    const rise = (orders[n - 1] ?? 0) - (orders[back] ?? 0);
    const span = (iterations[n - 1] ?? 0) - (iterations[back] ?? 0);
    if (rise > 0 && span > 0) {
      const left = (targetOrders - (orders[n - 1] ?? 0)) / (rise / span);
      candidates.push(Math.max(0, left) * msPerIteration);
    }
  }
  return candidates.length ? Math.max(0, Math.min(...candidates)) : null;
}
