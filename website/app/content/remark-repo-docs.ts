import { existsSync, statSync } from "node:fs";
import path from "node:path";
import type { Heading, Link, Nodes, Root } from "mdast";
import type { VFile } from "vfile";
import { basePath, githubBlobUrl, htmlAssetPath, pageSlug } from "./paths";

// Resolved on use: this module is also bundled for the browser (via lib/source.ts), where it never runs.
const repoRoot = () => path.resolve(import.meta.dirname, "../../..");

/**
 * Adapts a repository Markdown file to the site without editing it:
 * drops the first `#` heading (the page title renders it) and rewrites
 * relative links to site pages, published HTML documents or GitHub.
 */
export function remarkRepoDocs() {
  return (tree: Root, file: VFile) => {
    const firstHeading = tree.children.findIndex(
      (node): node is Heading => node.type === "heading" && node.depth === 1,
    );
    if (firstHeading !== -1) tree.children.splice(firstHeading, 1);

    const from = path.relative(repoRoot(), file.path);
    visitLinks(tree, (link) => {
      link.url = resolveLink(from, link.url);
    });
  };
}

function visitLinks(node: Nodes, fn: (link: Link) => void) {
  if (node.type === "link") fn(node);
  if ("children" in node)
    for (const child of node.children) visitLinks(child, fn);
}

/** Site URL for a link written in the repository file `from`. Page URLs omit the base path (the router adds it); HTML documents carry it. */
export function resolveLink(from: string, url: string): string {
  if (
    /^[a-z][a-z0-9+.-]*:/i.test(url) ||
    url.startsWith("#") ||
    url.startsWith("//")
  )
    return url;

  const [target, hash = ""] = url.split("#", 2);
  const anchor = hash ? `#${hash}` : "";
  const repoPath = path.posix.normalize(
    path.posix.join(path.posix.dirname(from), target),
  );
  const absolute = path.join(repoRoot(), repoPath);

  if (!existsSync(absolute)) {
    console.warn(`[docs] ${from}: link to missing file ${repoPath}`);
    return githubBlobUrl(repoPath) + anchor;
  }

  const slug = pageSlug(repoPath);
  if (slug !== null) return `/${slug}${anchor}`;

  const asset = htmlAssetPath(repoPath);
  if (asset !== null) return `${basePath}${asset}${anchor}`;

  return githubBlobUrl(repoPath, statSync(absolute).isDirectory()) + anchor;
}
