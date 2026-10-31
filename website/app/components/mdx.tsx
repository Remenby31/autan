import { Step, Steps } from "fumadocs-ui/components/steps";
import { Tab, Tabs } from "fumadocs-ui/components/tabs";
import { TypeTable } from "fumadocs-ui/components/type-table";
import defaultMdxComponents from "fumadocs-ui/mdx";
import type { MDXComponents } from "mdx/types";
import type { ComponentProps } from "react";
import { useLocation } from "react-router";
import { Diagram } from "@/components/diagram";
import { basePath, defaultLocale, localePath } from "@/content/paths";
import { parsePath } from "@/lib/locale";

const DefaultLink = defaultMdxComponents.a;

/**
 * Published files (HTML documents, assets) carry the base path and bypass the router. A page shown
 * in another locale than its file's (an untranslated page on the French site) keeps the reader's
 * locale on its links.
 */
function Link({ href, ...props }: ComponentProps<"a">) {
  const { locale } = parsePath(useLocation().pathname);
  if (href?.startsWith(basePath)) return <a href={href} {...props} />;
  if (
    href?.startsWith("/") &&
    !href.startsWith("//") &&
    locale !== defaultLocale
  ) {
    const path = href.split(/[?#]/)[0];
    const target = parsePath(path);
    if (target.locale !== locale)
      href = localePath(target.slug, locale) + href.slice(path.length);
  }
  return <DefaultLink href={href} {...props} />;
}

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    a: Link,
    Diagram,
    Step,
    Steps,
    Tab,
    Tabs,
    TypeTable,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
