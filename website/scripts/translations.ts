import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { glob } from "node:fs/promises";
import path from "node:path";
import { sourceOf } from "../app/content/paths.ts";

// A translation `page.fr.md` records in its frontmatter `source_hash`: the first 12 hex characters
// of the SHA-256 of its English source's bytes. `check` fails when a translation has no hash, no
// source, or a hash that no longer matches its source; `hash <file>` prints the value to record.
const repoRoot = path.resolve(import.meta.dirname, "../..");
const hashOf = (file: string) =>
  createHash("sha256")
    .update(readFileSync(path.join(repoRoot, file)))
    .digest("hex")
    .slice(0, 12);

const [command, file] = process.argv.slice(2);

if (command === "hash" && file) {
  console.log(hashOf(path.relative(repoRoot, path.resolve(file))));
} else if (command === "check") {
  const problems: string[] = [];
  let count = 0;
  for await (const translation of glob(
    ["README.fr.md", "docs/**/*.fr.{md,mdx}"],
    { cwd: repoRoot },
  )) {
    count++;
    const source = sourceOf(translation);
    const frontmatter =
      readFileSync(path.join(repoRoot, translation), "utf8").match(
        /^---\n([\s\S]*?)\n---/,
      )?.[1] ?? "";
    const recorded = frontmatter.match(
      /^source_hash:\s*["']?([0-9a-f]{12})["']?\s*$/m,
    )?.[1];
    let current: string;
    try {
      current = hashOf(source);
    } catch {
      problems.push(`${translation}: no English source ${source}`);
      continue;
    }
    if (!recorded)
      problems.push(`${translation}: no source_hash (expected ${current})`);
    else if (recorded !== current)
      problems.push(
        `${translation}: out of date with ${source} (source_hash ${recorded}, source now ${current})`,
      );
  }
  for (const problem of problems) console.error(`[translations] ${problem}`);
  console.log(
    `[translations] ${count} translation(s), ${problems.length} problem(s)`,
  );
  process.exit(problems.length > 0 ? 1 : 0);
} else {
  console.error(
    "usage: bun scripts/translations.ts check | hash <english file>",
  );
  process.exit(2);
}
