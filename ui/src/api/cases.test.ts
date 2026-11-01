import { expect, test } from "vitest";
import { caseDocument, differences, formFromDocument, sameValue } from "./cases";

test("values are equal whatever the order of their keys", () => {
  expect(sameValue({ a: 1, b: { c: [1, 2] } }, { b: { c: [1, 2] }, a: 1 })).toBe(true);
  expect(sameValue({ a: 1 }, { a: 1, b: 2 })).toBe(false);
  expect(sameValue([1, 2], [2, 1])).toBe(false);
});

test("documents differ path by path, a value hidden elsewhere included", () => {
  // The same value elsewhere does not hide a change: gamma 1.285 on the gas, kept on a zone.
  const before = { physics: { gas: { gamma: 1.285 } }, cell_zones: [{ m: { gamma: 1.285 } }] };
  const after = { physics: { gas: { gamma: 1.4 } }, cell_zones: [{ m: { gamma: 1.285 } }] };
  expect(differences(before, after)).toEqual([
    { path: "physics.gas.gamma", before: 1.285, after: 1.4 },
  ]);
  expect(differences({ a: [1, 2] }, { a: [1] })).toEqual([
    { path: "a[1]", before: 2, after: undefined },
  ]);
  expect(differences({ x: { y: 1 } }, { x: { y: 1 } })).toEqual([]);
});

test("group selectors read into the form and back", () => {
  const document = {
    boundaries: [
      { group: "WING", condition: { kind: "wall" } },
      { zones: "far*", condition: { kind: "farfield" } },
    ],
    monitors: [{ kind: "forces", group: "wing.*", ref_area: "1 m ** 2" }],
  };
  const form = formFromDocument(document, []);
  expect(form.boundaries.map((r) => [r.by, r.zones])).toEqual([
    ["group", "WING"],
    ["zones", "far*"],
  ]);
  const back = caseDocument(form);
  expect(back.boundaries).toEqual(document.boundaries);
  expect(back.monitors).toEqual(document.monitors);
});

test("a monitor name is kept as given, blanks included", () => {
  const document = {
    monitors: [
      { kind: "forces", name: " ", surfaces: "a" },
      { kind: "forces", name: "forces", surfaces: "b" },
    ],
  };
  expect(caseDocument(formFromDocument(document, [])).monitors).toEqual(document.monitors);
});
