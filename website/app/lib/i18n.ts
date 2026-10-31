import { defineI18n } from "fumadocs-core/i18n";
import { defaultLocale, locales } from "../content/paths.ts";

export const i18n = defineI18n({
  languages: [...locales],
  defaultLanguage: defaultLocale,
  hideLocale: "default-locale",
  parser: "dot",
});
