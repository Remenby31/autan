import { Link } from "@tanstack/react-router";
import { Fragment, type ReactNode } from "react";

export interface Crumb {
  label: ReactNode;
  to?: string;
}

/** Where you are, as a path you can click back up. */
export function TopBar({ crumbs, actions }: { crumbs: Crumb[]; actions?: ReactNode }) {
  return (
    <header className="flex h-14 flex-none items-center gap-3 border-line border-b bg-bg-1 px-6">
      <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-2">
        {crumbs.map((crumb, k) => (
          <Fragment key={crumb.to ?? "current"}>
            {k > 0 && <span className="text-line-strong">/</span>}
            {crumb.to && k < crumbs.length - 1 ? (
              <Link to={crumb.to} className="truncate text-fg-2 transition-colors hover:text-fg">
                {crumb.label}
              </Link>
            ) : (
              <span
                className="truncate font-medium"
                aria-current={k === crumbs.length - 1 ? "page" : undefined}
              >
                {crumb.label}
              </span>
            )}
          </Fragment>
        ))}
      </nav>
      <div className="ml-auto flex items-center gap-2">{actions}</div>
    </header>
  );
}
