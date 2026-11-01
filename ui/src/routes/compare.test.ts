import { expect, test } from "vitest";
import { degrees } from "../api/units";
import { flatten } from "./compare";
import { study } from "./index";

test("angles are read in degrees from the case's units", () => {
  expect(degrees("19.57 deg")).toBe(19.57);
  expect(degrees(`${Math.PI / 2} rad`)).toBeCloseTo(90);
  expect(degrees("19.57")).toBeNull();
  expect(degrees(null)).toBeNull();
});

test("a case document flattens to dotted settings with arrays by index", () => {
  const flat = flatten({
    freestream: { aoa: "7 deg", mach: 0.2 },
    boundaries: [{ zones: "wing.*" }],
  });
  expect(Object.fromEntries(flat)).toEqual({
    "freestream.aoa": "7 deg",
    "freestream.mach": "0.2",
    "boundaries[0].zones": "wing.*",
  });
});

test("a study names the settings that vary across its runs", () => {
  const base = {
    sideslip: "0 deg",
    mach: 0.2,
    re: 5.49e6,
    turbulence: "sa",
    precision: "fp32+acc64",
  };
  expect(study([{ ...base, aoa: "7.05 deg" }, { ...base, aoa: "21.47 deg" }, undefined])).toBe(
    "α 7.05–21.47 deg",
  );
  expect(
    study([
      { ...base, aoa: "1 deg", turbulence: "sst" },
      { ...base, aoa: "1 deg" },
    ]),
  ).toBe("turbulence sst · sa");
  expect(
    study([
      { ...base, aoa: "1 deg" },
      { ...base, aoa: "1 deg" },
    ]),
  ).toBeNull();
});
