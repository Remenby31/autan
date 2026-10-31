import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext, Link, Outlet } from "@tanstack/react-router";
import { MotionConfig } from "motion/react";
import { useRegistrySubscription } from "../api/runs";
import { CommandPalette } from "../components/CommandPalette";

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  component: Root,
});

function Root() {
  useRegistrySubscription();
  return (
    <MotionConfig reducedMotion="user" transition={{ duration: 0.12 }}>
      <div className="flex h-full flex-col">
        <header className="flex items-center gap-4 border-line border-b px-4 py-2.5">
          <Link to="/" aria-label="Autan" className="flex">
            <picture>
              <source srcSet="/logo-dark.svg" media="(prefers-color-scheme: dark)" />
              <img src="/logo-light.svg" alt="" className="size-5" />
            </picture>
          </Link>
          <nav className="flex gap-1 text-sm">
            <Link
              to="/"
              className="rounded-md px-2.5 py-1 text-fg-3 hover:text-fg"
              activeProps={{ className: "text-fg bg-glass" }}
              activeOptions={{ exact: true }}
            >
              Runs
            </Link>
            <Link
              to="/viewer"
              className="rounded-md px-2.5 py-1 text-fg-3 hover:text-fg"
              activeProps={{ className: "text-fg bg-glass" }}
            >
              Viewer
            </Link>
          </nav>
          <span className="ml-auto font-mono text-fg-3 text-xs">
            <kbd className="rounded border border-line px-1.5">⌘K</kbd>
          </span>
        </header>
        <main className="relative min-h-0 flex-1">
          <Outlet />
        </main>
        <CommandPalette />
      </div>
    </MotionConfig>
  );
}
