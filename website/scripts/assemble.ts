import { cpSync, existsSync, mkdirSync, renameSync, rmSync } from "node:fs";
import { glob } from "node:fs/promises";
import path from "node:path";
import {
  assetsDir,
  basePath,
  docsDir,
  htmlAssetPath,
} from "../app/content/paths";

// React Router writes prerendered pages under build/client/<basename>/ and assets under
// build/client/assets/. GitHub Pages serves one directory at the base path: dist/.
const website = path.resolve(import.meta.dirname, "..");
const repoRoot = path.resolve(website, "..");
const client = path.join(website, "build/client");
const dist = path.join(website, "dist");

rmSync(dist, { recursive: true, force: true });
renameSync(path.join(client, basePath), dist);
cpSync(path.join(client, "assets"), path.join(dist, "assets"), {
  recursive: true,
});
// GitHub Pages also serves /x/ (from x/index.html), and React Router then fetches /x/_.data.
for await (const data of glob("**/*.data", { cwd: dist })) {
  const page = data.slice(0, -".data".length);
  if (page !== "_" && existsSync(path.join(dist, page, "index.html")))
    cpSync(path.join(dist, data), path.join(dist, page, "_.data"));
}
// The SPA fallback renders the not-found page for any URL that was not prerendered.
cpSync(path.join(client, "index.html"), path.join(dist, "404.html"));

// Diagrams and images under docs/assets/ are published at their repository path, for the
// Markdown exports and for links to them.
cpSync(path.join(repoRoot, assetsDir), path.join(dist, assetsDir), {
  recursive: true,
});

for await (const file of glob(`${docsDir}/**/*.html`, { cwd: repoRoot })) {
  const asset = htmlAssetPath(file);
  if (asset === null) continue;
  const target = path.join(dist, asset);
  if (existsSync(target))
    throw new Error(
      `[docs] ${file} collides with a generated file at ${asset}`,
    );
  mkdirSync(path.dirname(target), { recursive: true });
  cpSync(path.join(repoRoot, file), target);
}
