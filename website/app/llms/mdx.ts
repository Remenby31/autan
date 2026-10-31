import { docsLlms, source } from "@/lib/source";
import type { Route } from "./+types/mdx";

export async function loader({ params }: Route.LoaderArgs) {
  const slugs = params["*"].split("/").filter((v) => v.length > 0);
  slugs.pop();
  const page = source.getPage(slugs);
  if (!page) throw new Response("Not found", { status: 404 });
  return new Response(await docsLlms.page(page), {
    headers: { "Content-Type": "text/markdown; charset=utf-8" },
  });
}
