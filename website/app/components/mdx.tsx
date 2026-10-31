import { Step, Steps } from "fumadocs-ui/components/steps";
import { Tab, Tabs } from "fumadocs-ui/components/tabs";
import { TypeTable } from "fumadocs-ui/components/type-table";
import defaultMdxComponents from "fumadocs-ui/mdx";
import type { MDXComponents } from "mdx/types";
import type { ComponentProps } from "react";
import { Diagram } from "@/components/diagram";
import { basePath } from "@/content/paths";

const DefaultLink = defaultMdxComponents.a;

/** Published HTML documents are files, not routes: link to them without the router. */
function Link(props: ComponentProps<"a">) {
  if (props.href?.startsWith(basePath)) return <a {...props} />;
  return <DefaultLink {...props} />;
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
