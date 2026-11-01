import { expect, test } from "vitest";
import { devicesLabel } from "./devices";

test("devices read as ranks of a kind, plural past one (results.md: count is ranks)", () => {
  expect(devicesLabel({ kind: "cpu", count: 1 })).toBe("1 CPU rank");
  expect(devicesLabel({ kind: "cuda", count: 4 })).toBe("4 CUDA ranks");
  expect(devicesLabel({ kind: "hip", count: 1 })).toBe("1 HIP rank");
});
