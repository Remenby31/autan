import { expect, test } from "vitest";
import { sparkPoints } from "./Sparkline";

test("a missing value keeps the others at their own positions", () => {
  // Four commits across 0..3 (x = 1 + k): the third value sits at k = 2 whatever precedes it.
  const full = sparkPoints([10, 50, 90, 30], 5, 102, [0, 100]);
  const gap = sparkPoints([10, null, 90, 30], 5, 102, [0, 100]);
  expect(gap).toEqual([full[0], full[2], full[3]]);
  expect(gap.map(([x]) => x)).toEqual([1, 3, 4]);
  expect(gap.map(([, y]) => y)).toEqual([91, 11, 71]);
});
