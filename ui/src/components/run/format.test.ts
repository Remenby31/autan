import { expect, test } from "vitest";
import { formatBytes } from "./format";

test("bytes read in SI units, one decimal below 10 of a unit", () => {
  expect(formatBytes(0)).toBe("0 B");
  expect(formatBytes(999)).toBe("999 B");
  expect(formatBytes(1000)).toBe("1.0 kB");
  expect(formatBytes(1234)).toBe("1.2 kB");
  expect(formatBytes(12_345)).toBe("12 kB");
  expect(formatBytes(3_500_000)).toBe("3.5 MB");
  expect(formatBytes(48_200_000_000)).toBe("48 GB");
  expect(formatBytes(2.5e15)).toBe("2500 TB");
});
