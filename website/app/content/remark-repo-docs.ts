import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import type {
  Definition,
  Heading,
  Image,
  Link,
  Nodes,
  Parents,
  Root,
} from "mdast";
import { fromMarkdown } from "mdast-util-from-markdown";
import type { VFile } from "vfile";
import {
  assetsDir,
  basePath,
  fileLocale,
  githubBlobUrl,
  htmlAssetPath,
  localePath,
  pageSlug,
} from "./paths.ts";

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
    inlineDiagrams(tree, from);
  };
}

const diagramsDir = `${assetsDir}/diagrams/`;

/**
 * Replaces images of `docs/assets/diagrams/*.svg` with the SVG itself, so the diagram's
 * `var(--token, fallback)` colours follow the page theme. Its `<style>` blocks, which serve
 * the standalone file (GitHub), are dropped: inlined, they would apply to the whole page.
 */
function inlineDiagrams(node: Nodes, from: string) {
  if (!("children" in node)) return;
  const parent = node as Parents;
  parent.children = parent.children.map((child) => {
    if (child.type !== "image") {
      inlineDiagrams(child, from);
      return child;
    }
    const svg = diagramSource(from, child);
    if (svg === null) return child;
    return {
      type: "mdxJsxTextElement",
      name: "Diagram",
      attributes: [
        { type: "mdxJsxAttribute", name: "svg", value: svg },
        { type: "mdxJsxAttribute", name: "alt", value: child.alt ?? "" },
      ],
      children: [],
    } as unknown as typeof child;
  });
}

function diagramSource(from: string, image: Image): string | null {
  if (/^[a-z][a-z0-9+.-]*:/i.test(image.url) || image.url.startsWith("/"))
    return null;
  const repoPath = path.posix.normalize(
    path.posix.join(path.posix.dirname(from), decodeURI(image.url)),
  );
  if (!repoPath.startsWith(diagramsDir) || !repoPath.endsWith(".svg"))
    return null;
  const absolute = path.join(repoRoot(), repoPath);
  if (!existsSync(absolute)) {
    console.warn(`[docs] ${from}: image of missing file ${repoPath}`);
    return null;
  }
  return readFileSync(absolute, "utf8")
    .replace(/<\?xml[^>]*>/g, "")
    .replace(/<style[\s\S]*?<\/style>/g, "")
    .trim();
}

function visitLinks(node: Nodes, fn: (link: Link | Definition) => void) {
  if (node.type === "link" || node.type === "definition") fn(node);
  if ("children" in node)
    for (const child of node.children) visitLinks(child, fn);
}

/**
 * Site URL for a link written in the repository file `from`. Page URLs omit the base path
 * (the router adds it); HTML documents carry it. `warn: false` silences the missing-file report.
 */
export function resolveLink(from: string, url: string, warn = true): string {
  if (
    /^[a-z][a-z0-9+.-]*:/i.test(url) ||
    url.startsWith("#") ||
    url.startsWith("//")
  )
    return url;

  const match = url.match(/^([^?#]*)(\?[^#]*)?(#.*)?$/);
  const target = decodeURI(match?.[1] ?? url);
  const query = match?.[2] ?? "";
  const anchor = match?.[3] ?? "";
  const repoPath = path.posix.normalize(
    path.posix.join(path.posix.dirname(from), target),
  );
  const absolute = path.join(repoRoot(), repoPath);

  if (repoPath.startsWith("../") || !existsSync(absolute)) {
    if (warn) console.warn(`[docs] ${from}: link to missing file ${repoPath}`);
    return githubBlobUrl(repoPath) + query + anchor;
  }

  const slug = pageSlug(repoPath);
  if (slug !== null)
    return `${localePath(slug, fileLocale(from))}${query}${anchor}`;

  const asset = htmlAssetPath(repoPath);
  if (asset !== null) return `${basePath}${encodeURI(asset)}${query}${anchor}`;

  return (
    githubBlobUrl(repoPath, statSync(absolute).isDirectory()) + query + anchor
  );
}

/**
 * Rewrites the link destinations of raw Markdown from `from` to absolute site URLs, for the
 * Markdown exports. Only the destinations change, located by the parser, so code stays as written.
 */
export function rewriteMarkdownLinks(from: string, markdown: string): string {
  const edits: { start: number; end: number; url: string }[] = [];
  visitLinks(fromMarkdown(markdown), (node) => {
    const start = node.position?.start.offset;
    const end = node.position?.end.offset;
    if (start === undefined || end === undefined) return;
    const resolved = resolveLink(from, node.url, false);
    const url =
      resolved.startsWith("/") && !resolved.startsWith("//")
        ? basePath + resolved.slice(1)
        : resolved;
    if (url === node.url) return;
    const source = markdown.slice(start, end);
    const at =
      node.type === "link"
        ? source.lastIndexOf(node.url)
        : source.indexOf(node.url);
    if (at === -1) {
      console.warn(
        `[docs] ${from}: cannot rewrite link ${node.url} in the Markdown export`,
      );
      return;
    }
    edits.push({ start: start + at, end: start + at + node.url.length, url });
  });
  let out = markdown;
  for (const edit of edits.sort((a, b) => b.start - a.start))
    out = out.slice(0, edit.start) + edit.url + out.slice(edit.end);
  return out;
}
