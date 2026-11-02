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

const run = (start: Pick<RunStart, "residual_reference" | "woken">): Pick<Run, "start"> => ({
  start: {
    run_id: "",
    identity: "",
    mode: "continue",
    iterations: 0,
    time: null,
    solution_sha256: "",
    ...start,
  },
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

test("a residual plateau is a plateau: the drop is the slowest equation's, floor or not", () => {
  // case-schema.md#time: the round-off floor is judged cell by cell, never from these RMS
  // residuals, and reaching it is not convergence.
  const rows = [
    { iteration: 1, time: null, residuals: { density: 1, energy: 1 } },
    { iteration: 2, time: null, residuals: { density: 1e-4, energy: 1e-6 } },
    { iteration: 3, time: null, residuals: { density: 1e-4, energy: 1e-8 } },
  ];
  const plain = run({ residual_reference: null, woken: null });
  expectOrders(appendHistory(empty(), rows, plain).orders, [0, 4, 4]);
});

test("an equation missing from a continued run's reference leaves the drop undefined", () => {
  const history = appendHistory(
    empty(),
    [{ iteration: 31, time: null, residuals: { density: 1e-8, energy: 1e4 } }],
    run({ residual_reference: { density: 1 }, woken: [] }),
  );
  expect(history.orders.at(-1)).toBeNaN();
});
