import { expect, test } from "vitest";
import { degrees } from "../api/units";
import { convergedForces, flatten } from "./compare";
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

test("a converged run ends with its last forces sample within one sampling interval", () => {
  const samples = { iteration: [10860, 10870, 10880], cl: [0.52, 0.53, 0.5301], cd: [0, 0, 0.004] };
  const ended = (status: "converged" | "completed", iterations: number, last = iterations) =>
    convergedForces({ status, iterations }, { iteration: [1, last] }, samples);
  expect(ended("converged", 10885)).toEqual({ cl: 0.5301, cd: 0.004 });
  expect(ended("converged", 10880)).toEqual({ cl: 0.5301, cd: 0.004 });
  expect(ended("converged", 10890)).toBeNull(); // a sample due at the end is missing
  expect(ended("converged", 10885, 10884)).toBeNull(); // the history has not caught up
  expect(ended("completed", 10885)).toBeNull();
});
