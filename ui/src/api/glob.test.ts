import { expect, test } from "vitest";
import { globEscape, globMatches } from "./glob";

// Expected values from Python's fnmatch.fnmatchcase, which the SDK uses (autan/plan.py).
test.each([
  ["wing.*", "wing.main", true],
  ["wing.*", "wing", false],
  ["wing.*", "Wing.main", false],
  ["*", "anything", true],
  ["?min", "xmin", true],
  ["?min", "xxmin", false],
  ["[xy]min", "ymin", true],
  ["[!xy]min", "zmin", true],
  ["[!xy]min", "xmin", false],
  ["[]]", "]", true],
  ["[a", "[a", true],
  ["a.b", "a.b", true],
  ["a.b", "axb", false],
  ["*-outlet", "air-outlet", true],
  ["(x)", "(x)", true],
  ["[!z-a]", "y", true],
  ["[z-a]", "y", false],
  ["[a-c]x", "bx", true],
  ["[a-]", "-", true],
  ["[^a]", "^", true],
])("%s matches %s: %s", (pattern, name, expected) => {
  expect(globMatches(pattern, name)).toBe(expected);
});

test("an escaped zone name matches that name only", () => {
  for (const name of ["wing[1]", "a*b", "what?", "plain"]) {
    expect(globMatches(globEscape(name), name)).toBe(true);
  }
  expect(globMatches(globEscape("wing[1]"), "wing1")).toBe(false);
  expect(globMatches(globEscape("a*b"), "axxb")).toBe(false);
});

// Classes with dashes, as Python 3.12+ rewrites them (fnmatch._translate); the names each
// pattern matches are fnmatchcase's.
test.each([
  ["[a--!]min", ["ymin", "-min", "!min", "bmin", "dmin", "]min", "^min", "amin"]],
  ["[!-a]min", ["ymin", "!min", "bmin", "dmin", "]min", "^min"]],
  ["[a-]min", ["-min", "amin"]],
  ["[--z]min", ["ymin", "-min", "bmin", "dmin", "]min", "^min", "amin"]],
  ["[z-a-c]min", ["-min"]],
  ["[a-c-e]min", ["-min", "bmin", "amin"]],
  ["[!]min", []],
  ["[]-a]min", ["]min", "^min", "amin"]],
])("%s matches exactly %j", (pattern, matched) => {
  const names = ["ymin", "-min", "!min", "bmin", "dmin", "]min", "^min", "amin"];
  expect(names.filter((n) => globMatches(pattern, n))).toEqual(matched);
});

test("ranges are over code points, and runs of stars stay linear", () => {
  expect(globMatches("[😀-🙏]*", "😀wall")).toBe(true);
  expect(globMatches("[!😀-🙏]*", "😀wall")).toBe(false);
  expect(globMatches("************z", "farfield-downstream-outlet")).toBe(false);
  expect(globMatches("*a*a*a*a*a*a*a*a*b", "a".repeat(200))).toBe(false);
});
