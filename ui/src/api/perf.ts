import { queryOptions } from "@tanstack/react-query";
import { getJson } from "./runs";

/** main's performance history, as bench/history/history.py writes it (served by /api/perf). */
export interface PerfHistory {
  peak_GBps: number;
  tracked: string[];
  records: PerfRecord[];
}

export interface PerfRecord {
  commit: string;
  measured: string;
  host: string | null;
  gpu: string;
  benchmarks: Record<
    string,
    { median: number; cv: number; unit: string; percent_of_peak?: number }
  >;
  regression?: { status: string; good: string; bad: string };
}

export const perfQuery = queryOptions({
  queryKey: ["perf"],
  queryFn: () => getJson<PerfHistory>("/api/perf"),
  staleTime: Number.POSITIVE_INFINITY,
  retry: false,
});

const TIME_PER_CELL = ".time_per_cell";

/** One bandwidth benchmark over main: its % of peak per record (null where that commit did not
 * measure it), its last measurement, and the time per cell measured with it, if any. */
export interface PerfSeries {
  name: string;
  suite: string;
  kernel: string;
  percent: (number | null)[];
  last: number;
  tracked: boolean;
  nsPerCell: number | null;
}

export function perfSeries(history: PerfHistory): PerfSeries[] {
  const names = new Set<string>();
  for (const r of history.records)
    for (const [name, b] of Object.entries(r.benchmarks))
      if (b.percent_of_peak !== undefined) names.add(name);
  return [...names].sort().map((name) => {
    const percent = history.records.map((r) => r.benchmarks[name]?.percent_of_peak ?? null);
    let last = percent.length - 1;
    while (percent[last] === null) last--;
    const timing = history.records[last]?.benchmarks[name + TIME_PER_CELL];
    const dot = name.indexOf(".");
    return {
      name,
      suite: name.slice(0, dot),
      kernel: name.slice(dot + 1),
      percent,
      last,
      tracked: history.tracked.includes(name),
      nsPerCell: timing?.unit === "ns" ? timing.median : null,
    };
  });
}

/** A vertical range for a series: its own, widened to at least ±5 % around its middle, the CI
 * regression threshold, so run-to-run noise does not read as a trend (as history.py draws it). */
export function trendDomain(values: number[]): [number, number] {
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const mid = (lo + hi) / 2;
  const half = Math.max(((hi - lo) / 2) * 1.2, 0.05 * Math.abs(mid)) || 1;
  return [mid - half, mid + half];
}
