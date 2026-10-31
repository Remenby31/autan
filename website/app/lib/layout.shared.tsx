import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";
import { githubUrl } from "@/content/paths";
import logoDark from "../../../docs/assets/logo-dark.svg?url";
import logoLight from "../../../docs/assets/logo-light.svg?url";
import wordmarkDark from "../../../docs/assets/wordmark-dark.svg?url";
import wordmarkLight from "../../../docs/assets/wordmark-light.svg?url";

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
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
