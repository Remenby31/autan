import { expect, test } from "vitest";
import { devicesLabel } from "./devices";

test("devices read as a count and a kind, plural past one", () => {
  expect(devicesLabel({ kind: "cpu", count: 1 })).toBe("1 CPU");
  expect(devicesLabel({ kind: "cuda", count: 4 })).toBe("4 CUDA GPUs");
  expect(devicesLabel({ kind: "hip", count: 1 })).toBe("1 HIP GPU");
});
