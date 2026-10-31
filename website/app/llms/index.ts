import { llmsIndex } from "@/lib/source";

export async function loader() {
  return new Response(await llmsIndex(), {
    headers: { "Content-Type": "text/plain; charset=utf-8" },
  });
}
