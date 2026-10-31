import path from "node:path";
import { basePath } from "../app/content/paths";

// Serves dist/ under the base path, with GitHub Pages' lookup rules: <path>, <path>/index.html, then 404.html.
const dist = path.resolve(import.meta.dirname, "../dist");
const port = Number(process.env.PORT ?? 4173);

Bun.serve({
  port,
  async fetch(request) {
    const { pathname } = new URL(request.url);
    if (!pathname.startsWith(basePath)) return Response.redirect(basePath, 302);
    const relative = decodeURIComponent(pathname.slice(basePath.length));
    for (const candidate of [relative, path.join(relative, "index.html")]) {
      const file = Bun.file(path.join(dist, candidate));
      if (candidate && (await file.exists())) return new Response(file);
    }
    const index = Bun.file(path.join(dist, "index.html"));
    if (relative === "" && (await index.exists())) return new Response(index);
    return new Response(Bun.file(path.join(dist, "404.html")), { status: 404 });
  },
});
console.log(`http://localhost:${port}${basePath}`);
