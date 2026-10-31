import { expect, test } from "vitest";
import { COOL_WARM, displayRange, sampleColormap } from "./colormap";

test("a signed field puts zero on the neutral colour of the diverging map", () => {
  const { min, max } = displayRange(-3, 1.15);
  const lut = sampleColormap(COOL_WARM, 257);
  const zero = Math.round(((0 - min) / (max - min)) * 256);
  expect(Array.from(lut.slice(zero * 4, zero * 4 + 3))).toEqual([221, 221, 221]);
  expect(displayRange(0.2, 1.4)).toEqual({ min: 0.2, max: 1.4 });
});
