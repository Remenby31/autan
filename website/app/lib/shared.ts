export const markdownRoute = "/llms.mdx";

/** URL of the Markdown export of the page at `slug` (empty for the introduction). */
export function markdownPath(slug: string): string {
  return `${markdownRoute}/${slug ? `${slug}/` : ""}content.md`;
}
