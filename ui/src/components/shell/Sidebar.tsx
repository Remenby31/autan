import { Toggle } from "@base-ui/react/toggle";
import { ToggleGroup } from "@base-ui/react/toggle-group";
import { Link } from "@tanstack/react-router";
import { Box, ListTree, Monitor, Moon, Search, Sun } from "lucide-react";
import type { ReactNode } from "react";
import { Kbd } from "../ui/Kbd";
import { setTheme, type ThemeChoice, useResolvedTheme, useTheme } from "./theme";

const NAV: { to: "/" | "/viewer"; label: string; icon: ReactNode; exact?: boolean }[] = [
  { to: "/", label: "Runs", icon: <ListTree className="size-4" />, exact: true },
  { to: "/viewer", label: "Analysis", icon: <Box className="size-4" /> },
];

/** The only navigation chrome: Find on top, sections, theme at the bottom. */
export function Sidebar({ onFind }: { onFind: () => void }) {
  const theme = useTheme();
  const resolved = useResolvedTheme();
  return (
    <aside className="flex w-60 flex-none flex-col gap-3 border-line border-r bg-bg-1 px-3 py-3">
      <Link to="/" className="flex h-8 items-center gap-2 px-2" aria-label="Autan, runs">
        <img src={`/wordmark-${resolved}.svg`} alt="Autan" className="h-5" />
      </Link>
      <button
        type="button"
        onClick={onFind}
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
  );
}
