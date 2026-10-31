import { glob } from "node:fs/promises";
import type { Config } from "@react-router/dev/config";
import {
  basePath,
  defaultLocale,
  fileLocale,
  localePath,
  locales,
  pageSlug,
} from "./app/content/paths.ts";
import { markdownPath } from "./app/lib/shared.ts";

export default {
  ssr: false,
  basename: basePath,
  async prerender({ getStaticPaths }) {
    const paths = getStaticPaths();
    for await (const file of glob(
      ["README.md", "README.fr.md", "docs/**/*.{md,mdx}"],
      { cwd: ".." },
    )) {
      const slug = pageSlug(file);
      if (slug === null) continue;
      const locale = fileLocale(file);
      paths.push(markdownPath(slug, locale));
      // Every page exists in every locale: an untranslated one shows its English version.
      if (locale === defaultLocale)
        for (const l of locales) paths.push(localePath(slug, l));
    }
    return paths;
  },
} satisfies Config;
