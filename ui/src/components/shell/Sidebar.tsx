import { Toggle } from "@base-ui/react/toggle";
import { ToggleGroup } from "@base-ui/react/toggle-group";
import { Link } from "@tanstack/react-router";
import {
  Box,
  FilePlus2,
  GitCompareArrows,
  ListTree,
  Monitor,
  Moon,
  Search,
  Sun,
} from "lucide-react";
import { type ReactNode, useCallback, useEffect, useRef } from "react";
import { Kbd } from "../ui/Kbd";
import { setTheme, type ThemeChoice, useResolvedTheme, useTheme } from "./theme";

const NAV: {
  to: "/" | "/compare" | "/viewer" | "/cases/new";
  label: string;
  icon: ReactNode;
  exact?: boolean;
}[] = [
  { to: "/", label: "Runs", icon: <ListTree className="size-4" />, exact: true },
  { to: "/compare", label: "Compare", icon: <GitCompareArrows className="size-4" /> },
  { to: "/viewer", label: "Analysis", icon: <Box className="size-4" /> },
  { to: "/cases/new", label: "New case", icon: <FilePlus2 className="size-4" /> },
];

/** The only navigation chrome: Find on top, sections, theme at the bottom. Below 768 px it is
 * a drawer, `open` or not, that closes on a choice, on Escape or on its backdrop. */
export function Sidebar({
  onFind,
  open,
  onClose,
}: {
  onFind: () => void;
  open: boolean;
  onClose: () => void;
}) {
  const theme = useTheme();
  const resolved = useResolvedTheme();
  const drawer = useRef<HTMLElement>(null);
  // Dismissed (Escape, backdrop), the drawer hands focus back to the button that opened it.
  const dismiss = useCallback(() => {
    onClose();
    // Once the page behind is no longer inert.
    requestAnimationFrame(() =>
      document.querySelector<HTMLElement>('[aria-controls="nav-drawer"]')?.focus(),
    );
  }, [onClose]);
  // Opened, it takes focus, and Escape closes it before any page shortcut sees it.
  useEffect(() => {
    if (!open) return;
    drawer.current?.querySelector<HTMLElement>("nav a[href]")?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopPropagation();
      dismiss();
    };
    window.addEventListener("keydown", onKey, { capture: true });
    return () => window.removeEventListener("keydown", onKey, { capture: true });
  }, [open, dismiss]);
  return (
    <>
      {open && (
        <div
          aria-hidden
          className="fixed inset-0 z-30 bg-black/30 md:hidden"
          onClick={dismiss}
          data-testid="nav-backdrop"
        />
      )}
      <aside
        className={`${open ? "flex" : "hidden"} fixed inset-y-0 left-0 z-40 w-60 flex-none flex-col gap-3 border-line border-r bg-bg-1 px-3 py-3 shadow-tooltip md:static md:flex md:shadow-none`}
        ref={drawer}
        id="nav-drawer"
        aria-label="Navigation"
        {...(open ? { role: "dialog", "aria-modal": true } : {})}
      >
        <Link
          to="/"
          className="flex h-8 items-center gap-2 px-2"
          aria-label="Autan, runs"
          onClick={onClose}
        >
          <img src={`/logo-${resolved}.svg`} alt="" className="size-6" />
          <img src={`/wordmark-${resolved}.svg`} alt="Autan" className="h-[18px]" />
        </Link>
        <button
          type="button"
          onClick={() => {
            onClose();
            onFind();
          }}
          className="flex h-9 items-center gap-2 rounded-md border border-line bg-bg-0 px-3 text-fg-3 transition-colors duration-[var(--dur-fast)] hover:border-line-strong hover:text-fg-2"
        >
          <Search className="size-4" />
          <span className="flex-1 text-left">Find…</span>
          <Kbd>⌘K</Kbd>
        </button>
        <nav className="flex flex-col gap-0.5">
          {NAV.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              activeOptions={{ exact: item.exact ?? false }}
              onClick={onClose}
              className="flex h-9 items-center gap-3 rounded-md px-2 text-fg-2 transition-colors duration-[var(--dur-fast)] hover:bg-bg-2 hover:text-fg data-[status=active]:bg-bg-3 data-[status=active]:font-medium data-[status=active]:text-fg"
            >
              {item.icon}
              {item.label}
            </Link>
          ))}
        </nav>
        <div className="mt-auto flex items-center justify-between px-1">
          <span className="font-mono text-fg-3 text-xs">local</span>
          <ToggleGroup
            value={[theme]}
            onValueChange={(value) => value[0] && setTheme(value[0] as ThemeChoice)}
            aria-label="Theme"
            className="flex rounded-full border border-line p-0.5"
          >
            {(
              [
                ["system", <Monitor key="s" className="size-3.5" />],
                ["light", <Sun key="l" className="size-3.5" />],
                ["dark", <Moon key="d" className="size-3.5" />],
              ] as const
            ).map(([value, icon]) => (
              <Toggle
                key={value}
                value={value}
                aria-label={`${value} theme`}
                className="flex size-6 items-center justify-center rounded-full text-fg-3 hover:text-fg data-[pressed]:bg-bg-3 data-[pressed]:text-fg"
              >
                {icon}
              </Toggle>
            ))}
          </ToggleGroup>
        </div>
      </aside>
    </>
  );
}
