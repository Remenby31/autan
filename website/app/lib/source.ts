import { llms, loader } from "fumadocs-core/source";
import { pageSchema } from "fumadocs-core/source/schema";
import { applyMdxPreset } from "fumadocs-mdx/config";
import { defineDocs } from "fumadocs-mdx/macro";
import {
  basePath,
  docsDir,
  introductionPath,
  pageSlug,
} from "../content/paths";
import { remarkRepoDocs } from "../content/remark-repo-docs";

function isIntroduction(file: string): boolean {
  const normalized = file.replaceAll("\\", "/");
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
    files: ["README.md", "docs/**/*.md", "docs/**/*.mdx"],
    schema: ({ path, source }) =>
      pageSchema.extend({
        title: pageSchema.shape.title.default(
          isIntroduction(path)
            ? "Introduction"
            : (source.match(/^# (.+)$/m)?.[1]?.trim() ?? path),
        ),
      }),
    mdxOptions: applyMdxPreset({
      remarkPlugins: (plugins) => [remarkRepoDocs, ...plugins],
    }),
    postprocess: { includeProcessedMarkdown: true },
  },
  meta: { files: [] },
});

const repoPaths = new Map<string, string>();

// Site paths mirror the slugs (contracts/results.md, index.md), so search results show the section.
const raw = docs.toFumadocsSource();
const files: typeof raw.files = raw.files.map((file) => {
  if (file.type !== "page") return file;
  const slug = pageSlug(file.path);
  if (slug === null) throw new Error(`[docs] no site slug for ${file.path}`);
  const sitePath = `${slug || "index"}.md`;
  repoPaths.set(sitePath, file.path);
  return { ...file, path: sitePath };
});

export const source = loader({ source: { files }, baseUrl: "/" });

/** Repository path of a page, e.g. `docs/contracts/results.md`. */
export function repoPath(page: { path: string }): string {
  const path = repoPaths.get(page.path);
  if (!path) throw new Error(`[docs] unknown page ${page.path}`);
  return path;
}

const withBase = (text: string) => text.replaceAll("](/", `](${basePath}`);

export const docsLlms = llms(source, {
  renderPage: async (page) => {
    const body = (await page.data.getText("processed")).replace(/^# .*\n+/, "");
    return withBase(
      `# ${page.data.title} ([${basePath}${page.url.slice(1)}](/${page.url.slice(1)}))\n\n${body}`,
    );
  },
});

export const llmsIndex = async () => withBase(await docsLlms.index());
