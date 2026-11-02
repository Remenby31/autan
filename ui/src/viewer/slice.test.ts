import { expect, test } from "vitest";
import type { Surface } from "../api/surface";
import { slicePlane } from "./slice";

test("the slider's ends cut inside a mesh its FP32 positions round outward", () => {
  // A box from x = 0.1 to 0.2 written as FP32 offsets from its centre, as the server sends it:
  // 0.05 in FP32 is slightly more than 0.05, so the box reads a little wider than it is.
  const centre = [0.15, 0.5, 0.5];
  const corners = [-0.05, 0.05].flatMap((x) => [x, -0.5, -0.5, x, 0.5, 0.5]);
  const surface = { positions: new Float32Array(corners), origin: centre } as unknown as Surface;

  for (const at of [0, 1]) {
    const { origin, normal } = slicePlane(surface, "x", at);
    expect(normal).toEqual([1, 0, 0]);
    expect(origin[0]).toBeGreaterThan(0.1);
    expect(origin[0]).toBeLessThan(0.2);
    expect(origin.slice(1)).toEqual([0.5, 0.5]);
  }
  expect(slicePlane(surface, "x", 0.5).origin[0]).toBeCloseTo(0.15, 12);
});
