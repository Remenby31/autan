export const repo = "Remenby31/autan";
export const branch = "main";
export const basePath = "/autan/";
export const githubUrl = `https://github.com/${repo}`;

/** Repository paths, relative to the repository root, that the site renders as pages. */
export const docsDir = "docs";
export const introductionPath = "README.md";
/** Static files under `docs/` (images, diagrams), never pages. */
export const assetsDir = `${docsDir}/assets`;

export const locales = ["en", "fr"] as const;
export type Locale = (typeof locales)[number];
export const defaultLocale: Locale = "en";

/** Locale of a Markdown file: `page.fr.md` is the French translation of `page.md`. */
export function fileLocale(repoPath: string): Locale {
  return /\.fr\.mdx?$/.test(repoPath) ? "fr" : defaultLocale;
}

/** Repository path of the English source of a translation. */
export function sourceOf(repoPath: string): string {
  return repoPath.replace(/\.fr(\.mdx?)$/, "$1");
}

/** Site slug of a Markdown file, from its repository path; `null` if the site does not render it. */
export function pageSlug(repoPath: string): string | null {
  const source = sourceOf(repoPath);
  if (source === introductionPath) return "";
  if (
    !source.startsWith(`${docsDir}/`) ||
    source.startsWith(`${assetsDir}/`) ||
    !/\.mdx?$/.test(source)
  )
    return null;
  return source
    .slice(docsDir.length + 1)
    .replace(/\.mdx?$/, "")
    .toLowerCase()
    .replace(/(^|\/)(readme|index)$/, "");
}

/** Router path of a page in a locale, without the base path: `/x` in English, `/fr/x` in French. */
export function localePath(slug: string, locale: Locale): string {
  const prefix = locale === defaultLocale ? "" : `/${locale}`;
  return `${prefix}/${slug}`.replace(/(.)\/$/, "$1");
}

/**
 * Published path of a standalone HTML document under `docs/`: its repository path, so it never
 * shadows a page (GitHub Pages serves `x.html` for `/x`).
 */
export function htmlAssetPath(repoPath: string): string | null {
  if (!repoPath.startsWith(`${docsDir}/`) || !repoPath.endsWith(".html"))
    return null;
  return repoPath;
}

export function githubBlobUrl(repoPath: string, isDirectory = false): string {
  return `${githubUrl}/${isDirectory ? "tree" : "blob"}/${branch}/${repoPath}`;
}
