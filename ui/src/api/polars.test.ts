import { expect, test } from "vitest";
import type { Run, RunSettings } from "./contracts";
import { type PolarRun, polars } from "./polars";

const SETTINGS: RunSettings = {
  aoa: "0 deg",
  sideslip: "0 deg",
  mach: 0.8,
  re: null,
  turbulence: "inviscid",
  precision: "fp32+acc64",
};

function compared(
  role: string,
  settings: Partial<RunSettings>,
  run: Partial<Run> = {},
  forces: Record<string, { cl?: number; cd?: number }> = { forces: { cl: 0.1, cd: 0.01 } },
): PolarRun {
  const id = `run-20261101T00000${role.charCodeAt(0) - 65}Z-0000000${role}`;
  return {
    id,
    role,
    run: {
      run_id: id,
      status: "converged",
      mesh: { path: "/m", sha256: "0edcff6577" },
      monitors: Object.fromEntries(
        Object.entries(forces).map(([name, f]) => [name, { ...f, iteration: 9 }]),
      ),
      ...run,
    } as Run,
    settings: { ...SETTINGS, ...settings },
  };
}

test("each fixed condition is its own series, and no run is merged into another", () => {
  // The four NACA 0012 runs of the review: M0.80 at 0° and 1.25° (twice), M0.50 at 1.25°.
  const result = polars([
    compared("A", { aoa: "0 deg" }),
    compared("B", { aoa: "1.25 deg", mach: 0.5 }),
    compared("C", { aoa: "1.25 deg" }),
    compared("D", { aoa: "1.25 deg" }),
  ]);

  expect(result.series.map((s) => [s.label, s.points.map((p) => p.role)])).toEqual([
    ["M 0.5", ["B"]],
    ["M 0.8", ["A", "C", "D"]],
  ]);
  expect(result.fixed).toBe("inviscid · fp32+acc64");
  expect(result.unplotted).toEqual([]);
});

test("a run off the polars says why", () => {
  const result = polars([
    compared("A", {}),
    compared("B", {}, { status: "diverged" }),
    compared("C", {}, {}, { loads: {} }),
    compared("D", {}, {}, { a: { cl: 1, cd: 1 }, b: { cl: 1, cd: 1 } }),
    compared("E", { aoa: "3" }),
    { id: "F", role: "F", run: undefined, settings: undefined },
  ]);

  expect(result.unplotted.map((u) => [u.role, u.reason])).toEqual([
    ["B", "diverged, not converged"],
    ["C", "no forces monitor with CL and CD"],
    ["D", "several forces monitors with CL and CD"],
    ["E", "no angle of attack in deg or rad"],
    ["F", "not available"],
  ]);
  expect(result.series.flatMap((s) => s.points.map((p) => p.role))).toEqual(["A"]);
});

test("another mesh, model or Reynolds number is another series", () => {
  const result = polars([
    compared("A", { re: 5e6 }),
    compared("B", { re: 5e6, turbulence: "sa" }),
    compared("C", { re: 6e6 }, { mesh: { path: "/n", sha256: "abcdef0123" } }),
  ]);

  expect(result.series.map((s) => s.label)).toEqual([
    "mesh 0edcff6 · Re 5.00e+6 · inviscid",
    "mesh 0edcff6 · Re 5.00e+6 · sa",
    "mesh abcdef0 · Re 6.00e+6 · inviscid",
  ]);
});
