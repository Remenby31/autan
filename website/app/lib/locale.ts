import { defaultLocale, type Locale, locales } from "@/content/paths";

/** Locale and page slug of a router path (without the base path): `/fr/contracts/results` → fr, `contracts/results`. */
export function parsePath(pathname: string): { locale: Locale; slug: string } {
  const segments = pathname.split("/").filter((s) => s.length > 0);
  const first = segments[0] as Locale | undefined;
  if (first && first !== defaultLocale && locales.includes(first))
    return { locale: first, slug: segments.slice(1).join("/") };
  return { locale: defaultLocale, slug: segments.join("/") };
}
