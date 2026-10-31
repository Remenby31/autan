import { defaultLocale, type Locale } from "../content/paths.ts";

export const markdownRoute = "/llms.mdx";

/** URL of the Markdown export of the page at `slug` (empty for the introduction). */
export function markdownPath(
  slug: string,
  locale: Locale = defaultLocale,
): string {
  const prefix = locale === defaultLocale ? "" : `${locale}/`;
  return `${markdownRoute}/${prefix}${slug ? `${slug}/` : ""}content.md`;
}
