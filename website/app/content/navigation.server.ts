import { readFileSync } from "node:fs";
import { glob } from "node:fs/promises";
import path from "node:path";
import type * as PageTree from "fumadocs-core/page-tree";
import { basePath, docsDir, htmlAssetPath } from "./paths";

const repoRoot = path.resolve(import.meta.dirname, "../../..");

/** Sidebar sections, in order. A folder of `docs/` not listed here gets its own section after these. */
const sections: { title?: string; dir?: string; slugs?: string[] }[] = [
  { slugs: ["", "plan", "plan.html"] },
  { title: "Contracts", dir: "contracts" },
  { title: "SDK", dir: "sdk" },
  { title: "Validation", dir: "validation" },
  { title: "Design", dir: "design" },
  { title: "Research", dir: "research" },
];

interface SitePage {
  slug: string;
  title: string;
  url: string;
}

/** Builds the sidebar from the pages and the standalone HTML documents of `docs/`. */
export async function buildPageTree(pages: SitePage[]): Promise<PageTree.Root> {
  const documents = await htmlDocuments();
  const entries = [
    ...pages.map((page) => ({ ...page, external: false })),
    ...documents,
  ];
  const placed = new Set<string>();
  const children: PageTree.Node[] = [];

  const dirs = new Set(
    entries
      .map((entry) => entry.slug.split("/"))
      .filter((s) => s.length > 1)
      .map((s) => s[0]),
  );
  const listed = new Set(sections.map((section) => section.dir));
  const extra = [...dirs].filter((dir) => !listed.has(dir)).sort();

  for (const section of [
    ...sections,
    ...extra.map((dir) => ({ title: capitalize(dir), dir, slugs: undefined })),
  ]) {
    const members = section.slugs
      ? section.slugs.flatMap((slug) =>
          entries.filter((entry) => entry.slug === slug),
        )
      : entries
          .filter((entry) => entry.slug.startsWith(`${section.dir}/`))
          .sort(
            (a, b) =>
              Number(a.external) - Number(b.external) ||
              a.title.localeCompare(b.title),
          );
    if (members.length === 0) continue;
    const nodes = members.map((entry): PageTree.Item => {
      placed.add(entry.slug);
      return {
        type: "page",
        name: sidebarLabel(entry.title),
        url: entry.url,
        external: entry.external,
      };
    });
    if (section.title)
      children.push({
        type: "folder",
        name: section.title,
        defaultOpen: true,
        children: nodes,
      });
    else children.push(...nodes);
  }

  for (const entry of entries) {
    if (!placed.has(entry.slug))
      children.push({
        type: "page",
        name: sidebarLabel(entry.title),
        url: entry.url,
        external: entry.external,
      });
  }

  return { name: "Autan", children };
}

async function htmlDocuments() {
  const documents: (SitePage & { external: true })[] = [];
  for await (const file of glob(`${docsDir}/**/*.html`, { cwd: repoRoot })) {
    const asset = htmlAssetPath(file);
    if (asset === null) continue;
    const html = readFileSync(path.join(repoRoot, file), "utf8");
    const title = html.match(/<title>([^<]+)<\/title>/)?.[1]?.trim() ?? asset;
    documents.push({
      slug: asset.slice(docsDir.length + 1),
      title: `${title} (HTML)`,
      url: `${basePath}${asset}`,
      external: true,
    });
  }
  return documents;
}

/**
 * Short sidebar label from a page title: drops a one-word prefix ("Contract: results" → "Results", the rest kept as written)
 * or a subtitle ("Plan d'implémentation — solveur CFD" → "Plan d'implémentation").
 */
export function sidebarLabel(title: string): string {
  const label = title.split(" — ")[0].trim();
  const colon = label.indexOf(": ");
  if (colon === -1) return label;
  const prefix = label.slice(0, colon);
  return /\s/.test(prefix) ? prefix : capitalize(label.slice(colon + 2));
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
