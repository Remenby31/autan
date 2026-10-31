import { parsePath } from "@/lib/locale";
import { docsLlms, source } from "@/lib/source";
import type { Route } from "./+types/mdx";

export async function loader({ params }: Route.LoaderArgs) {
  const { locale, slug } = parsePath(
    params["*"].replace(/\/?content\.md$/, ""),
  );
  const page = source.getPage(slug ? slug.split("/") : [], locale);
  if (!page) throw new Response("Not found", { status: 404 });
  return new Response(await docsLlms.page(page), {
    headers: { "Content-Type": "text/markdown; charset=utf-8" },
  });
}
