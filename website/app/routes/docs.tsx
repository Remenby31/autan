import { useFumadocsLoader } from "fumadocs-core/source/client";
import { DocsLayout } from "fumadocs-ui/layouts/docs";
import {
  DocsBody,
  DocsDescription,
  DocsPage,
  DocsTitle,
  MarkdownCopyButton,
  ViewOptionsPopover,
} from "fumadocs-ui/layouts/docs/page";
import { use } from "react";
import { useMDXComponents } from "@/components/mdx";
import { buildPageTree } from "@/content/navigation.server";
import { githubBlobUrl } from "@/content/paths";
import { baseOptions } from "@/lib/layout.shared";
import { markdownPath } from "@/lib/shared";
import { docs, repoPath, source } from "@/lib/source";
import type { Route } from "./+types/docs";

export async function loader({ params }: Route.LoaderArgs) {
  const slugs = (params["*"] ?? "").split("/").filter((v) => v.length > 0);
  const page = source.getPage(slugs);
  if (!page) throw new Response("Not found", { status: 404 });
  const path = repoPath(page);
  await docs.getPage(path)?.preload();

  const tree = await buildPageTree(
    source.getPages().map((p) => ({
      slug: p.slugs.join("/"),
      title: p.data.title ?? p.path,
      url: p.url,
    })),
  );
  return {
    path,
    markdownUrl: markdownPath(page.slugs.join("/")),
    pageTree: await source.serializePageTree(tree),
  };
}

function Content({ path, markdownUrl }: { path: string; markdownUrl: string }) {
  const page = docs.getPage(path);
  if (!page) throw new Error(`unknown page: ${path}`);
  const { toc } = use(page.load());
  const Mdx = page.body;

  return (
    <DocsPage toc={toc} breadcrumb={{ includeSeparator: true }}>
      <title>{`${page.title} · Autan`}</title>
      {page.description && (
        <meta name="description" content={page.description} />
      )}
      <DocsTitle>{page.title}</DocsTitle>
      <DocsDescription>{page.description}</DocsDescription>
      <div className="flex flex-row gap-2 items-center border-b -mt-4 pb-6">
        <MarkdownCopyButton markdownUrl={markdownUrl} />
        <ViewOptionsPopover
          markdownUrl={markdownUrl}
          githubUrl={githubBlobUrl(path)}
        />
      </div>
      <DocsBody>
        <Mdx components={useMDXComponents()} />
      </DocsBody>
    </DocsPage>
  );
}

export default function Page({ loaderData }: Route.ComponentProps) {
  const { path, pageTree, markdownUrl } = useFumadocsLoader(loaderData);
  return (
    <DocsLayout {...baseOptions()} tree={pageTree}>
      <Content path={path} markdownUrl={markdownUrl} />
    </DocsLayout>
  );
}
