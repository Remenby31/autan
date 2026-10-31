import { glob } from "node:fs/promises";
import type { Config } from "@react-router/dev/config";
import { basePath, pageSlug } from "./app/content/paths.ts";
import { markdownPath } from "./app/lib/shared.ts";

export default {
  ssr: false,
  basename: basePath,
  async prerender({ getStaticPaths }) {
    const paths = getStaticPaths();
    for await (const file of glob(["README.md", "docs/**/*.{md,mdx}"], {
      cwd: "..",
    })) {
      const slug = pageSlug(file);
      if (slug === null) continue;
      paths.push(`/${slug}`, markdownPath(slug));
    }
    return paths;
  },
} satisfies Config;
