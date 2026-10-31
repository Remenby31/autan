import { defaultLocale } from "@/content/paths";
import { docsLlms } from "@/lib/source";

export async function loader() {
  return new Response(await docsLlms.full(defaultLocale), {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
}
