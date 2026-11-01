import { expect, test } from "vitest";
import { globMatches } from "./glob";

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
])("%s matches %s: %s", (pattern, name, expected) => {
  expect(globMatches(pattern, name)).toBe(expected);
});
