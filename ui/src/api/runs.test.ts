import { expect, test } from "vitest";
import type { HistoryRecord } from "./contracts";
import { appendHistory, emptyColumns as empty, soleForces } from "./runs";

const record = (iteration: number, monitors?: HistoryRecord["monitors"]): HistoryRecord => ({
  iteration,
  time: null,
  residuals: { density: 10 ** -iteration },
  ...(monitors ? { monitors } : {}),
});

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
