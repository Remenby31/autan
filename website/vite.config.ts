import { reactRouter } from "@react-router/dev/vite";
import tailwindcss from "@tailwindcss/vite";
import { fumadocsMdx } from "fumadocs-mdx/vite";
import { defineConfig } from "vite";
import { basePath } from "./app/content/paths.ts";

export default defineConfig({
  base: basePath,
  plugins: [fumadocsMdx(), tailwindcss(), reactRouter()],
  resolve: { tsconfigPaths: true },
});
