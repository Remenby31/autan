import { expect, test } from "vitest";
import type { Run, RunSummary } from "./contracts";
import { groupByStudy } from "./studies";

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
