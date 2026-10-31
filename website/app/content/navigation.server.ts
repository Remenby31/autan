import { existsSync, readFileSync } from "node:fs";
import { glob } from "node:fs/promises";
import path from "node:path";
import type * as PageTree from "fumadocs-core/page-tree";
import { frLabels } from "../lib/translations.fr.ts";
import { basePath, docsDir, htmlAssetPath, type Locale } from "./paths.ts";

const repoRoot = path.resolve(import.meta.dirname, "../../..");

/**
 * Sidebar groups, in order. `slugs` lists pages by slug, `dir` takes every page of a folder
 * under `docs/` in the order of its `meta.json` `pages` (file names without extension), then by
 * title; `children` nests groups. A folder listed nowhere gets its own group at the end.
 */
interface Group {
  /** English title; French in `lib/translations.fr.ts`. */
  title?: string;
  slugs?: string[];
  dir?: string;
  /** Pages of `dir` shown in another group. */
  exclude?: string[];
  children?: Group[];
}

const sidebar: Group[] = [
  { slugs: [""] },
  {
    title: "Getting started",
    dir: "getting-started",
  },
  { title: "Concepts", dir: "concepts" },
  { title: "Guides", dir: "guides" },
  {
    title: "Reference",
    children: [
      {
        title: "Contracts",
        dir: "contracts",
        exclude: ["contracts/mesh"],
      },
      { title: "SDK", dir: "sdk" },
    ],
  },
  {
    title: "Project",
    children: [
      { slugs: ["plan", "plan.html", "style"] },
      {
        title: "Internal contracts",
        slugs: ["contracts/mesh"],
      },
      { title: "Design", dir: "design" },
      { title: "Research", dir: "research" },
      { title: "Validation", dir: "validation" },
    ],
  },
];

export interface SitePage {
  slug: string;
  title: string;
  url: string;
}

interface Entry extends SitePage {
  external: boolean;
}

/** Builds the sidebar of one locale from its pages and the standalone HTML documents of `docs/`. */
export async function buildPageTree(
  pages: SitePage[],
  locale: Locale,
): Promise<PageTree.Root> {
  const entries: Entry[] = [
    ...pages.map((page) => ({ ...page, external: false })),
    ...(await htmlDocuments()),
  ];
  const placed = new Set<string>();

  const node = (entry: Entry): PageTree.Item => {
    placed.add(entry.slug);
    return {
      type: "page",
      name: sidebarLabel(entry.title),
      url: entry.url,
      external: entry.external,
    };
  };

  const build = (group: Group): PageTree.Node[] => {
    let items: PageTree.Node[];
    if (group.children) items = group.children.flatMap(build);
    else if (group.slugs)
      items = group.slugs
        .flatMap((slug) => entries.filter((e) => e.slug === slug))
        .map(node);
    else {
      const dir = group.dir ?? "";
      const order = folderOrder(dir);
      const rank = (e: Entry) => {
        const i = order.indexOf(e.slug.slice(dir.length + 1));
        return i === -1 ? order.length : i;
      };
      items = entries
        .filter(
          (e) =>
            e.slug.startsWith(`${dir}/`) && !group.exclude?.includes(e.slug),
        )
        .sort(
          (a, b) =>
            rank(a) - rank(b) ||
            Number(a.external) - Number(b.external) ||
            a.title.localeCompare(b.title, locale),
        )
        .map(node);
    }
    if (items.length === 0 || !group.title) return items;
    return [
      {
        type: "folder",
        name:
          locale === "fr"
            ? (frLabels[group.title] ?? group.title)
            : group.title,
        defaultOpen: true,
        children: items,
      },
    ];
  };

  const listedDirs = new Set<string>();
  const collect = (group: Group) => {
    if (group.dir) listedDirs.add(group.dir);
    group.children?.forEach(collect);
  };
  sidebar.forEach(collect);
  const extraDirs = [
    ...new Set(
      entries
        .map((e) => e.slug.split("/"))
        .filter((s) => s.length > 1)
        .map((s) => s[0]),
    ),
  ]
    .filter((dir) => !listedDirs.has(dir))
    .sort();
  const groups: Group[] = [
    ...sidebar,
    ...extraDirs.map((dir) => ({
      title: capitalize(dir),
      dir,
    })),
  ];

  const children = groups.flatMap(build);
  for (const entry of entries)
    if (!placed.has(entry.slug)) children.push(node(entry));
  return { name: "Autan", children };
}

/** Page order of a folder, from `docs/<dir>/meta.json`, as slugs relative to the folder. */
function folderOrder(dir: string): string[] {
  const file = path.join(repoRoot, docsDir, dir, "meta.json");
  if (!existsSync(file)) return [];
  const meta = JSON.parse(readFileSync(file, "utf8")) as { pages?: unknown };
  if (
    !Array.isArray(meta.pages) ||
    !meta.pages.every((p) => typeof p === "string")
  )
    throw new Error(
      `[docs] ${path.relative(repoRoot, file)}: "pages" must be a list of file names`,
    );
  return meta.pages.map((p: string) => p.toLowerCase().replace(/\.mdx?$/, ""));
}

async function htmlDocuments(): Promise<Entry[]> {
  const documents: Entry[] = [];
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
 * Short sidebar label from a page title. A subtitle after ` — ` is dropped ("Kernel generator —
 * design research" → "Kernel generator") unless what precedes it is a single word, which is then
 * the dropped part ("Autan — état de l'art" → "État de l'art"). A one-word prefix before `: ` is
 * dropped too ("Contract: results" → "Results").
 */
export function sidebarLabel(title: string): string {
  const [head, ...rest] = title.split(" — ");
  let label =
    rest.length > 0 && !/\s/.test(head.trim())
      ? capitalize(rest.join(" — ").trim())
      : head.trim();
  const colon = label.indexOf(": ");
  if (colon !== -1) {
    const prefix = label.slice(0, colon);
    label = /\s/.test(prefix) ? prefix : capitalize(label.slice(colon + 2));
  }
  return label;
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
