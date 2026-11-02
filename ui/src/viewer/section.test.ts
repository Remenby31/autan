import { expect, test } from "vitest";
import type { Surface } from "../api/surface";
import { section } from "./section";

// A flat band from x = 2 to x = 4, y from 0 to 1, z = 0, split in triangles; the field is
// linear, 1 + 3x, so its interpolation along any edge is exact.
function band(): { surface: Surface; values: Float32Array } {
  const nodes: number[][] = [];
  for (const x of [2, 2.5, 3, 3.5, 4]) for (const y of [0, 1]) nodes.push([x, y, 0]);
  const indices: number[] = [];
  for (let i = 0; i < 4; i++) {
    const a = 2 * i;
    indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
  }
  const origin: [number, number, number] = [3, 0.5, 0];
  const positions = new Float32Array(nodes.flatMap((n) => n.map((v, k) => v - (origin[k] ?? 0))));
  const values = new Float32Array(nodes.map(([x]) => 1 + 3 * (x ?? 0)));
  return {
    surface: { origin, positions, indices: new Uint32Array(indices), zones: [], fields: [] },
    values,
  };
}

test("a section reads the field along the chord, exactly for a linear field", () => {
  const { surface, values } = band();
  const { segments, chord } = section(surface, values, "y", [0, 0.3, 0]);

  expect(chord).toEqual([2, 4]);
  expect(segments.length).toBeGreaterThan(0);
  for (const [x0, v0, x1, v1] of segments) {
    for (const [f, v] of [
      [x0, v0],
      [x1, v1],
    ] as const) {
      expect(f).toBeGreaterThanOrEqual(0);
      expect(f).toBeLessThanOrEqual(1);
      // x = 2 + 2 f, so the field is 1 + 3 (2 + 2 f) = 7 + 6 f.
      expect(v).toBeCloseTo(7 + 6 * f, 5);
    }
  }
  // The segments cover the chord once: their lengths add up to it.
  expect(segments.reduce((sum, [a, , b]) => sum + Math.abs(b - a), 0)).toBeCloseTo(1, 6);
});

test("a plane that misses the surface has no section", () => {
  const { surface, values } = band();
  expect(section(surface, values, "y", [0, 5, 0]).segments).toEqual([]);
});
