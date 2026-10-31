import { uiTranslations } from "fumadocs-ui/i18n";
import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";
import { githubUrl, type Locale } from "@/content/paths";
import { i18n } from "@/lib/i18n";
import { fr } from "@/lib/translations.fr";
import logoDark from "../../../docs/assets/logo-dark.svg?url";
import logoLight from "../../../docs/assets/logo-light.svg?url";
import wordmarkDark from "../../../docs/assets/wordmark-dark.svg?url";
import wordmarkLight from "../../../docs/assets/wordmark-light.svg?url";

/** Interface strings. English is fumadocs-ui's default; French is in `translations.fr.ts`. */
export const translations = i18n
  .translations()
  .extend(uiTranslations())
  .add({
    en: { displayName: "English" },
    fr: { displayName: "Français", ...fr },
  });

export function baseOptions(locale: Locale = "en"): BaseLayoutProps {
  return {
    i18n: true,
    nav: {
      url: locale === "en" ? "/" : `/${locale}`,
      title: (
        <>
          <img src={logoLight} alt="" className="size-6 dark:hidden" />
          <img src={logoDark} alt="" className="hidden size-6 dark:block" />
          <img
            src={wordmarkLight}
            alt="Autan"
            className="h-4 w-auto dark:hidden"
          />
          <img
            src={wordmarkDark}
            alt="Autan"
            className="hidden h-4 w-auto dark:block"
          />
        </>
      ),
    },
    githubUrl,
  };
}
