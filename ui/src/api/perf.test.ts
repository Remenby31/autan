import { expect, test } from "vitest";
import { type PerfHistory, perfSeries, trendDomain } from "./perf";

const record = (commit: string, benchmarks: PerfHistory["records"][number]["benchmarks"]) => ({
  commit,
  measured: "2026-11-02T09:00:00+00:00",
  host: "spark-5359",
  gpu: "NVIDIA GB10",
  benchmarks,
});
const bw = (percent: number) => ({
  median: 2 * percent,
  cv: 0.02,
  unit: "GB/s",
  percent_of_peak: percent,
});

test("a bandwidth benchmark is one series over main's commits, gaps where a commit did not measure it", () => {
  const history: PerfHistory = {
    peak_GBps: 200,
    tracked: ["codegen.roe_atomic"],
    records: [
      record("a", { "stream.copy": bw(98) }),
      record("b", {
        "stream.copy": bw(97),
        "codegen.roe_atomic": bw(85),
        "codegen.roe_atomic.time_per_cell": { median: 0.9, cv: 0.02, unit: "ns" },
      }),
      record("c", { "stream.copy": bw(99) }),
    ],
  };
  expect(perfSeries(history)).toEqual([
    {
      name: "codegen.roe_atomic",
      suite: "codegen",
      kernel: "roe_atomic",
      percent: [null, 85, null],
      last: 1,
      tracked: true,
      nsPerCell: 0.9,
    },
    {
      name: "stream.copy",
      suite: "stream",
      kernel: "copy",
      percent: [98, 97, 99],
      last: 2,
      tracked: false,
      nsPerCell: null,
    },
  ]);
});

test("a trend's range is at least the 5 % regression threshold around its middle", () => {
  expect(trendDomain([95, 97])).toEqual([96 - 0.05 * 96, 96 + 0.05 * 96]);
  expect(trendDomain([20, 80])).toEqual([50 - 36, 50 + 36]);
});
