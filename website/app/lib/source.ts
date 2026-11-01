import { llms, loader } from "fumadocs-core/source";
import { pageSchema } from "fumadocs-core/source/schema";
import { applyMdxPreset } from "fumadocs-mdx/config";
import { defineDocs } from "fumadocs-mdx/macro";
import { z } from "zod";
import {
  basePath,
  defaultLocale,
  docsDir,
  fileLocale,
  introductionPath,
  type Locale,
  locales,
  pageSlug,
  sourceOf,
} from "../content/paths.ts";
import {
  remarkRepoDocs,
  rewriteMarkdownLinks,
} from "../content/remark-repo-docs.ts";
import { i18n } from "./i18n.ts";

function isIntroduction(file: string): boolean {
  const normalized = sourceOf(file.replaceAll("\\", "/"));
  return (
    (normalized === introductionPath ||
      normalized.endsWith(`/${introductionPath}`)) &&
    !normalized.includes(`${docsDir}/`)
  );
}

export const docs = defineDocs({
  dir: "..",
  docs: {
    async: true,
    files: [
      "README.md",
      "README.fr.md",
      "docs/**/*.md",
      "docs/**/*.mdx",
      "!docs/assets/**",
    ],
    schema: ({ path, source }) =>
      pageSchema.extend({
        lang: z.enum(locales).optional(),
        title: pageSchema.shape.title.default(
          isIntroduction(path)
            ? "Introduction"
            : (source
                .match(/^# (.+)$/m)?.[1]
                ?.replaceAll("`", "")
                .trim() ?? path),
        ),
      }),
    mdxOptions: applyMdxPreset({
      remarkPlugins: (plugins) => [remarkRepoDocs, ...plugins],
      // Remote images (README badges) keep their own size: measuring them would fetch them
      // at build time, and the build must not depend on the network.
      remarkImageOptions: { external: false },
    }),
    postprocess: { includeProcessedMarkdown: true },
  },
  meta: { files: [] },
});

const repoPaths = new Map<string, string>();
// Fumadocs URL-encodes slugs ("été" → "%C3%A9t%C3%A9"); keys use the decoded form.
const key = (locale: string | undefined, slug: string) =>
  `${locale ?? defaultLocale}:${decodeURIComponent(slug)}`;

// Site paths mirror the slugs (contracts/results.md, index.fr.md), so search results show the section.
const raw = docs.toFumadocsSource();
const files: typeof raw.files = raw.files.map((file) => {
  if (file.type !== "page") return file;
  const slug = pageSlug(file.path);
  if (slug === null) throw new Error(`[docs] no site slug for ${file.path}`);
  const locale = fileLocale(file.path);
  const existing = repoPaths.get(key(locale, slug));
  if (existing)
    throw new Error(`[docs] ${file.path} and ${existing} both map to /${slug}`);
  repoPaths.set(key(locale, slug), file.path);
  const suffix = locale === defaultLocale ? "" : `.${locale}`;
  return { ...file, path: `${slug || "index"}${suffix}.md` };
});

export const source = loader({ source: { files }, baseUrl: "/", i18n });

/** Repository path of a page, e.g. `docs/contracts/results.fr.md`. */
export function repoPath(page: { slugs: string[]; locale?: string }): string {
  const slug = page.slugs.join("/");
  // A locale without a translation of this page serves the English file.
  const path =
    repoPaths.get(key(page.locale, slug)) ??
    repoPaths.get(key(defaultLocale, slug));
  if (!path)
    throw new Error(
      `[docs] unknown page ${page.locale}:${page.slugs.join("/")}`,
    );
  return path;
}

/** Whether a translation exists for the page at `slug`. */
export function isTranslated(slug: string, locale: Locale): boolean {
  return locale === defaultLocale || repoPaths.has(key(locale, slug));
}

const withBase = (text: string) => text.replaceAll("](/", `](${basePath}`);

export const docsLlms = llms(source, {
  renderPage: async (page) => {
    const text = (await page.data.getText("processed")).replace(/^# .*\n+/, "");
    const url = `${basePath}${page.url.slice(1)}`;
    return `# ${page.data.title} ([${url}](${url}))\n\n${rewriteMarkdownLinks(repoPath(page), text)}`;
  },
});

export const llmsIndex = async () =>
  withBase(await docsLlms.index(defaultLocale));
