import { useFumadocsLoader } from "fumadocs-core/source/client";
import { Callout } from "fumadocs-ui/components/callout";
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
import {
  basePath,
  defaultLocale,
  githubBlobUrl,
  type Locale,
  localePath,
} from "@/content/paths";
import { baseOptions } from "@/lib/layout.shared";
import { parsePath } from "@/lib/locale";
import { markdownPath } from "@/lib/shared";
import { docs, isTranslated, repoPath, source } from "@/lib/source";
import { frLabels } from "@/lib/translations.fr";
import type { Route } from "./+types/docs";

type Notice = "untranslated" | "french" | null;

export async function loader({ request }: Route.LoaderArgs) {
  // Data requests carry a `.data` (or `/_.data` for `/`) suffix.
  const pathname = new URL(request.url).pathname.replace(/(\/_)?\.data$/, "");
  const { locale, slug } = parsePath(`/${pathname.slice(basePath.length)}`);
  const slugs = slug ? slug.split("/") : [];
  const translated = isTranslated(slug, locale);
  const page = source.getPage(slugs, translated ? locale : defaultLocale);
  if (!page) throw new Response("Not found", { status: 404 });
  const path = repoPath(page);
  await docs.getPage(path)?.preload();

  const pages = source.getPages(defaultLocale).map((p) => {
    const pageSlug = p.slugs.join("/");
    const shown =
      (isTranslated(pageSlug, locale) && source.getPage(p.slugs, locale)) || p;
    return {
      slug: pageSlug,
      title: shown.data.title ?? pageSlug,
      url: localePath(pageSlug, locale),
    };
  });
  const contentLang: Locale =
    page.data.lang ?? (translated ? locale : defaultLocale);
  const notice: Notice =
    locale !== defaultLocale && contentLang !== locale
      ? "untranslated"
      : locale === defaultLocale && contentLang !== defaultLocale
        ? "french"
        : null;

  return {
    path,
    locale,
    contentLang,
    notice,
    markdownUrl: markdownPath(slug, translated ? locale : defaultLocale),
    pageTree: await source.serializePageTree(
      await buildPageTree(pages, locale),
    ),
  };
}

const notices = {
  untranslated:
    frLabels["This page is not translated yet; it is shown in English."],
  french: "This page is in French.",
};

function Content({
  path,
  markdownUrl,
  contentLang,
  notice,
}: {
  path: string;
  markdownUrl: string;
  contentLang: Locale;
  notice: Notice;
}) {
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
      {notice && <Callout type="info">{notices[notice]}</Callout>}
      <DocsBody lang={contentLang}>
        <Mdx components={useMDXComponents()} />
      </DocsBody>
    </DocsPage>
  );
}

export default function Page({ loaderData }: Route.ComponentProps) {
  const { path, pageTree, markdownUrl, locale, contentLang, notice } =
    useFumadocsLoader(loaderData);
  return (
    <DocsLayout {...baseOptions(locale)} tree={pageTree}>
      <Content
        path={path}
        markdownUrl={markdownUrl}
        contentLang={contentLang}
        notice={notice}
      />
    </DocsLayout>
  );
}
