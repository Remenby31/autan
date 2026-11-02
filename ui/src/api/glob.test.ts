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
