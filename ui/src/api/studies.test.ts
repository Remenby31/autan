import { expect, test } from "vitest";
import type { Run, RunSummary } from "./contracts";
import { groupByStudy, studies } from "./studies";

const run = (id: string, name: string) => ({ run_id: id, case: { name } }) as unknown as Run;
const summary = (study: string) => ({ study }) as unknown as RunSummary;

test("runs group by study, whatever their case names, and a run without a summary stands alone", () => {
  // results.md#studies: a sweep may name each case after its state; one name may hold two
  // studies (another mesh, another budget).
  const runs = [run("a", "naca-m080"), run("b", "naca-m050"), run("c", "naca"), run("d", "naca")];
  const groups = groupByStudy(runs, { a: summary("s1"), b: summary("s1"), c: summary("s2") });

  expect(groups.map((g) => [g.name, g.runs.map((r) => r.run_id)])).toEqual([
    ["naca-m080 · naca-m050", ["a", "b"]],
    ["naca", ["c"]],
    ["naca", ["d"]],
  ]);
});

test("two sweeps under one case name are two studies to compare", () => {
  const runs = ["a", "b", "c", "d"].map((id) => ({
    ...run(id, "naca"),
    started: `2026-11-02T00:00:0${id.charCodeAt(0) - 97}Z`,
  })) as Run[];
  const at = (study: string, aoa: string) =>
    ({
      study,
      settings: { aoa, sideslip: null, mach: 0.8, re: null, turbulence: null, precision: null },
    }) as unknown as RunSummary;
  const offered = studies({
    runs,
    rejected: [],
    solutions: [],
    summaries: {
      a: at("s1", "0 deg"),
      b: at("s1", "2 deg"),
      c: at("s2", "0 deg"),
      d: at("s2", "4 deg"),
    },
  } as unknown as Parameters<typeof studies>[0]);

  expect(offered.map((s) => [s.key, s.name, s.label, s.ids])).toEqual([
    ["s1", "naca", "α 0–2 deg", ["b", "a"]],
    ["s2", "naca", "α 0–4 deg", ["d", "c"]],
  ]);
});
