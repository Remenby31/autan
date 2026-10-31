import type { BaseLayoutProps } from "fumadocs-ui/layouts/shared";
import { githubUrl } from "@/content/paths";

export function baseOptions(): BaseLayoutProps {
  return {
    nav: { title: "Autan" },
    githubUrl,
  };
}
