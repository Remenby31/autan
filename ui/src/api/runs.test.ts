import { expect, test } from "vitest";
import type { HistoryRecord, Run, RunStart } from "./contracts";
import { appendHistory, emptyColumns as empty, soleForces } from "./runs";

const record = (iteration: number, monitors?: HistoryRecord["monitors"]): HistoryRecord => ({
  iteration,
  time: null,
  // Parsed, as the literal 1e-k is: 10 ** -k is not correctly rounded on every platform.
  residuals: { density: Number(`1e-${iteration}`) },
  ...(monitors ? { monitors } : {}),
});

const run = (
  start: Pick<RunStart, "residual_reference" | "woken">,
  floor?: Record<string, number>,
  plateau: string[] = [],
): Pick<Run, "start" | "convergence"> => ({
  start: {
    run_id: "",
    identity: "",
    mode: "continue",
    iterations: 0,
    time: null,
    solution_sha256: "",
    file: "solution",
    ...start,
  },
  convergence: floor ? { floor, plateau } : null,
});

// Drops in orders of magnitude, to round-off of log10 on the platform.
const expectOrders = (actual: number[], expected: number[]) => {
  expect(actual).toHaveLength(expected.length);
  for (const [k, value] of expected.entries()) {
    if (Number.isFinite(value)) expect(actual[k]).toBeCloseTo(value, 12);
    else expect(actual[k]).toBe(value);
  }
};

const forces = (cl: number, cd: number) => ({ fx: 0, fy: 0, fz: 0, mx: 0, my: 0, mz: 0, cl, cd });

test("reads the forces monitor under whatever name the case gave it", () => {
  const history = appendHistory(empty(), [
    record(1),
    record(2, { wing_loads: forces(2.0, 0.3) }),
    record(3),
    record(4, { wing_loads: forces(2.1, 0.31) }),
  ]);
  expect(history.iteration).toEqual([1, 2, 3, 4]);
  expect(history.residuals.get("density")).toEqual([1e-1, 1e-2, 1e-3, 1e-4]);
  expect(soleForces(history)).toEqual({ iteration: [2, 4], cl: [2.0, 2.1], cd: [0.3, 0.31] });

  const two = appendHistory(history, [record(5, { tail: forces(0.1, 0.01) })]);
  expect(soleForces(two)).toBeNull();
});

test("the residual drop is the slowest equation's, undefined once a zero residual grows", () => {
  const drop = (rows: Record<string, number>[]) =>
    appendHistory(
      empty(),
      rows.map((residuals, k) => ({ iteration: k + 1, time: null, residuals })),
    ).orders.at(-1);
  expect(
    drop([
      { density: 1e-300, energy: 1 },
      { density: 1e-310, energy: 1e-8 },
    ]),
  ).toBeCloseTo(8);
  expect(
    drop([
      { density: 1, energy: 0 },
      { density: 1e-8, energy: 0 },
    ]),
  ).toBeCloseTo(8);
  expect(
    drop([
      { density: 1, energy: 0 },
      { density: 1e-8, energy: 1 },
    ]),
  ).toBeNaN();
  expect(
    drop([
      { density: 1, energy: 1 },
      { density: 0, energy: 1e-8 },
    ]),
  ).toBeCloseTo(8);
});

test("a continued run's drop is measured from the reference it carries", () => {
  const reference = { density: 1, energy: 1 };
  const history = appendHistory(
    empty(),
    [{ iteration: 31, time: null, residuals: { density: 1e-3, energy: 1e-4 } }],
    run({ residual_reference: reference, woken: [] }),
  );
  expect(history.orders.at(-1)).toBeCloseTo(3);
});

test("an identically zero equation is exempt at its floor and woken above it, for good", () => {
  // Floor: 1e-12 times the largest first residual (case-schema.md#time).
  const rows = (...my: number[]) =>
    my.map((v, k) => ({
      iteration: k + 1,
      time: null,
      residuals: { density: Number(`1e-${4 * k}`), my: v },
    }));
  const drops = (my: number[], woken: string[] = []) =>
    appendHistory(empty(), rows(...my), run({ residual_reference: null, woken })).orders;
  expectOrders(drops([0, 1e-12, 1e-33]), [0, 4, 8]);
  const exempt = (my: number[]) =>
    appendHistory(empty(), rows(...my), run({ residual_reference: null, woken: [] })).exempt;
  expect([...exempt([0, 1e-33])]).toEqual(["my"]);
  expect([...exempt([0, 1e-11])]).toEqual([]);
  expect(drops([1e-30, 1e-33])[1]).toBeCloseTo(4);
  const woken = drops([0, 1e-11, 1e-33]);
  expect(woken[1]).toBeNaN();
  expect(woken[2]).toBeNaN();
  expect(drops([0, 1e-33], ["my"])[1]).toBeNaN();
  expect(drops([1e-30, 1e-33], ["my"])[1]).toBeCloseTo(3);
});

test("an equation converged on its round-off plateau no longer counts; at its floor alone it does", () => {
  // case-schema.md#time: the floor alone ends nothing; a converged run lists its plateau.
  const rows = [
    { iteration: 1, time: null, residuals: { density: 1, energy: 1 } },
    { iteration: 2, time: null, residuals: { density: 1e-4, energy: 1e-6 } },
    { iteration: 3, time: null, residuals: { density: 1e-4, energy: 1e-8 } },
  ];
  const start = { residual_reference: null, woken: null };
  const floor = { density: 1e-4, energy: 1e-8 };
  expectOrders(appendHistory(empty(), rows, run(start, floor)).orders, [0, 4, 4]);
  expectOrders(appendHistory(empty(), rows, run(start, floor, ["density"])).orders, [0, 6, 8]);
  const both = run(start, floor, ["density", "energy"]);
  const infinite = Number.POSITIVE_INFINITY;
  expectOrders(appendHistory(empty(), rows, both).orders, [infinite, infinite, infinite]);
});

test("a momentum component drops from the largest momentum first residual, whatever the axes", () => {
  // case-schema.md#time (#544): momentum-y starts 5 orders below momentum-x and ends at 1e-9;
  // against momentum-x's 1 it has dropped 9 orders, as momentum-x has, and swapping the two
  // axes gives the same drop. Density keeps its own reference.
  const line = (iteration: number, x: number, y: number) => ({
    iteration,
    time: null,
    residuals: { density: iteration === 1 ? 1 : 1e-10, "momentum-x": x, "momentum-y": y },
  });
  const start = { residual_reference: null, woken: [] };
  const along = appendHistory(empty(), [line(1, 1, 1e-5), line(2, 1e-9, 1e-9)], run(start));
  const across = appendHistory(empty(), [line(1, 1e-5, 1), line(2, 1e-9, 1e-9)], run(start));
  expect(along.orders.at(-1)).toBeCloseTo(9);
  expect(across.orders.at(-1)).toBeCloseTo(9);
});

test("an equation missing from a continued run's reference leaves the drop undefined", () => {
  const history = appendHistory(
    empty(),
    [{ iteration: 31, time: null, residuals: { density: 1e-8, energy: 1e4 } }],
    run({ residual_reference: { density: 1 }, woken: [] }),
  );
  expect(history.orders.at(-1)).toBeNaN();
});
