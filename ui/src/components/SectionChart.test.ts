import { expect, test } from "vitest";
import { tickLabel } from "./SectionChart";

test("neighbouring ticks never read the same", () => {
  // A pressure from 100 000 to 102 000 Pa in three ticks, and a uniform Cp padded by ±0.5.
  for (const [low, high] of [
    [100_000, 102_000],
    [-0.08, 0.92],
    [1e-6, 3e-6],
  ] as const) {
    const step = (high - low) / 2;
    const texts = [low, low + step, high].map((v) => tickLabel(v, step));
    expect(new Set(texts).size).toBe(3);
  }
});
