export const repo = "Remenby31/autan";
export const branch = "main";
export const basePath = "/autan/";
export const githubUrl = `https://github.com/${repo}`;

/** Repository paths, relative to the repository root, that the site renders as pages. */
export const docsDir = "docs";
export const introductionPath = "README.md";

/** Site slug of a Markdown file, from its repository path; `null` if the site does not render it. */
export function pageSlug(repoPath: string): string | null {
  if (repoPath === introductionPath) return "";
  if (!repoPath.startsWith(`${docsDir}/`) || !/\.mdx?$/.test(repoPath))
    return null;
  return repoPath
    .slice(docsDir.length + 1)
    .replace(/\.mdx?$/, "")
    .toLowerCase()
    .replace(/(^|\/)readme$/, "");
}

/**
 * Published path of a standalone HTML document under `docs/`: its repository path, so it never
 * shadows a page (GitHub Pages serves `plan.html` for `/plan`).
 */
export function htmlAssetPath(repoPath: string): string | null {
  if (!repoPath.startsWith(`${docsDir}/`) || !repoPath.endsWith(".html"))
    return null;
  return repoPath;
}

export function githubBlobUrl(repoPath: string, isDirectory = false): string {
  return `${githubUrl}/${isDirectory ? "tree" : "blob"}/${branch}/${repoPath}`;
}
