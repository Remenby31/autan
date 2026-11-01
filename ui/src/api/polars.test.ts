import { expect, test } from "vitest";
import type { CaseDocument } from "./cases";
import type { Run } from "./contracts";
import { type PolarRun, polars } from "./polars";

// A case as case.toml stores it (case-schema.md), at M0.8, inviscid, one forces monitor.
const CASE = {
  schema: "autan.case/1",
  name: "naca0012",
  mesh: "naca.cgns",
  physics: { kind: "compressible", turbulence: { model: "inviscid" }, precision: "fp32+acc64" },
  freestream: { mach: 0.8, aoa: "0 deg", sideslip: "0 deg", pressure: "101325 Pa" },
  time: { kind: "steady", max_iterations: 500 },
  boundaries: [{ zones: "wall", condition: { kind: "slip" } }],
  monitors: [{ kind: "forces", name: "forces", surfaces: "wall", ref_area: "1 m ** 2", every: 1 }],
};

type Patch = (c: typeof CASE) => void;

function compared(
  role: string,
  patch: Patch = () => {},
  coefficients = { cl: 0.1, cd: 0.01 },
  run: Partial<Run> = {},
): PolarRun {
  const document = structuredClone(CASE);
  patch(document);
  const id = `run-2026110${role.charCodeAt(0) - 64}T000000Z-0000000${role}`;
  const monitor = document.monitors[0]?.name ?? "forces";
  return {
    id,
    role,
    case: document as CaseDocument,
    run: {
      run_id: id,
      status: "converged",
      mesh: { path: "/m/naca.cgns", sha256: "0edcff6577aa" },
      monitors: { [monitor]: { ...coefficients, iteration: 9 } },
      ...run,
    } as Run,
  };
}

const at =
  (aoa: string): Patch =>
  (c) => {
    c.freestream.aoa = aoa;
  };

test("each Mach number is its own series, and no run is merged into another", () => {
  // The four NACA 0012 runs of the review: M0.80 at 0° and 1.25° (twice), M0.50 at 1.25°;
  // given in a shuffled order, with their own coefficients.
  const result = polars([
    compared("A", at("1.25 deg"), { cl: 0.31, cd: 0.021 }),
    compared(
      "B",
      (c) => {
        at("1.25 deg")(c);
        c.freestream.mach = 0.5;
      },
      { cl: 0.17, cd: 0.0004 },
    ),
    compared("C", at("0 deg"), { cl: 0, cd: 0.009 }),
    compared("D", at("1.25 deg"), { cl: 0.3, cd: 0.02 }),
  ]);

  expect(result.series.map((s) => [s.label, s.points])).toEqual([
    ["M 0.5", [{ id: expect.any(String), role: "B", alpha: 1.25, cl: 0.17, cd: 0.0004 }]],
    [
      "M 0.8",
      [
        { id: expect.any(String), role: "C", alpha: 0, cl: 0, cd: 0.009 },
        { id: expect.any(String), role: "A", alpha: 1.25, cl: 0.31, cd: 0.021 },
        { id: expect.any(String), role: "D", alpha: 1.25, cl: 0.3, cd: 0.02 },
      ],
    ],
  ]);
  expect(result.fixed).toBe("inviscid · fp32+acc64");
  expect(result.unplotted).toEqual([]);
});

// Each setting alone, changed in B only: A and B solve different problems.
test.each<[string, Patch]>([
  ["Reynolds number", (c) => Object.assign(c.freestream, { re: 5e6, pressure: undefined })],
  ["sideslip", (c) => Object.assign(c.freestream, { sideslip: "2 deg" })],
  ["turbulence model", (c) => Object.assign(c.physics, { turbulence: { model: "sa" } })],
  ["precision", (c) => Object.assign(c.physics, { precision: "fp64" })],
  [
    "boundary condition",
    (c) => Object.assign(c.boundaries[0] ?? {}, { condition: { kind: "symmetry" } }),
  ],
  ["reference area", (c) => Object.assign(c.monitors[0] ?? {}, { ref_area: "2 m ** 2" })],
  ["monitored surfaces", (c) => Object.assign(c.monitors[0] ?? {}, { surfaces: "wing.*" })],
])("a different %s is another series", (_, patch) => {
  expect(polars([compared("A"), compared("B", patch)]).series).toHaveLength(2);
});

test("another mesh is another series, named by its hash", () => {
  const result = polars([
    compared("A"),
    compared("B", undefined, undefined, { mesh: { path: "/m/naca.cgns", sha256: "9f2c11d" } }),
  ]);
  expect(result.series.map((s) => s.label)).toEqual(["mesh 0edcff6", "mesh 9f2c11d"]);
});

test("names, budgets and other monitors do not split a series", () => {
  const result = polars([
    compared("A"),
    compared("B", (c) => {
      c.name = "naca0012-again";
      c.mesh = "copy-of-naca.cgns";
      c.time.max_iterations = 2000;
      c.monitors = [
        { kind: "forces", name: "lift", surfaces: "wall", ref_area: "1 m ** 2", every: 7 },
        { kind: "forces", name: "loads", surfaces: "flap", ref_area: "1 m ** 2", every: 3 },
      ];
    }),
  ]);
  expect(result.series).toHaveLength(1);
});

test("a run off the polars says why", () => {
  const result = polars([
    compared("A"),
    compared("B", undefined, undefined, { status: "diverged" }),
    compared("C", undefined, undefined, {
      monitors: { loads: { iteration: 9 } } as unknown as Run["monitors"],
    }),
    compared("D", undefined, undefined, {
      monitors: {
        a: { cl: 1, cd: 1, iteration: 9 },
        b: { cl: 1, cd: 1, iteration: 9 },
      } as unknown as Run["monitors"],
    }),
    compared("E", at("3")),
    { id: "F", role: "F", run: undefined, case: undefined },
  ]);

  expect(result.unplotted.map((u) => [u.role, u.reason])).toEqual([
    ["B", "diverged, not converged"],
    ["C", "no forces monitor with CL and CD"],
    ["D", "several forces monitors with CL and CD"],
    ["E", "angle of attack not in deg or rad"],
    ["F", "not available"],
  ]);
  expect(result.series.flatMap((s) => s.points.map((p) => p.role))).toEqual(["A"]);
});
