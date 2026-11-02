import { Popover } from "@base-ui/react/popover";
import { Link } from "@tanstack/react-router";
import { Ellipsis, Menu } from "lucide-react";
import { Fragment, type ReactNode } from "react";
import { useNavDrawer } from "./nav";

export interface Crumb {
  label: ReactNode;
  to?: string;
}

/** Where you are, as a path you can click back up. */
export function TopBar({ crumbs, actions }: { crumbs: Crumb[]; actions?: ReactNode }) {
  const { open, openNav } = useNavDrawer();
  return (
    <header className="flex h-14 flex-none items-center gap-3 border-line border-b bg-bg-1 px-3 md:px-6">
      <button
        type="button"
        onClick={openNav}
        aria-label="Open navigation"
        aria-expanded={open}
        aria-controls="nav-drawer"
        className="flex size-8 flex-none items-center justify-center rounded-md text-fg-2 hover:bg-bg-2 hover:text-fg md:hidden"
      >
        <Menu className="size-4" />
      </button>
      <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-2">
        {crumbs.map((crumb, k) => (
          <Fragment key={crumb.to ?? "current"}>
            {/* Below 768 px only where you are shows; the drawer leads back up. */}
            {k > 0 && <span className="text-line-strong max-md:hidden">/</span>}
            {crumb.to && k < crumbs.length - 1 ? (
              <Link
                to={crumb.to}
                className="truncate text-fg-2 transition-colors hover:text-fg max-md:hidden"
              >
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
      {actions && (
        <>
          <div className="ml-auto flex items-center gap-2 max-md:hidden">{actions}</div>
          {/* Below 768 px the actions fold behind one button. */}
          <Popover.Root>
            <Popover.Trigger
              aria-label="Actions"
              className="ml-auto flex size-8 flex-none items-center justify-center rounded-md text-fg-2 hover:bg-bg-2 hover:text-fg data-[popup-open]:bg-bg-2 md:hidden"
            >
              <Ellipsis className="size-4" />
            </Popover.Trigger>
            <Popover.Portal>
              <Popover.Positioner side="bottom" align="end" sideOffset={6} className="z-50">
                <Popover.Popup className="flex flex-col items-stretch gap-1 rounded-lg border border-line bg-bg-0 p-2 shadow-tooltip">
                  {actions}
                </Popover.Popup>
              </Popover.Positioner>
            </Popover.Portal>
          </Popover.Root>
        </>
      )}
    </header>
  );
}
