import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext, Link, Outlet } from "@tanstack/react-router";
import { SearchX } from "lucide-react";
import { MotionConfig } from "motion/react";
import { useState } from "react";
import { useRegistrySubscription } from "../api/runs";
import { CommandPalette } from "../components/CommandPalette";
import { Sidebar } from "../components/shell/Sidebar";
import { TopBar } from "../components/shell/TopBar";
import { useTheme } from "../components/shell/theme";
import { Button } from "../components/ui/Button";
import { EmptyState } from "../components/ui/EmptyState";
import { ToastProvider } from "../components/ui/Toast";
import { TooltipProvider } from "../components/ui/Tooltip";

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  component: Root,
  notFoundComponent: NotFound,
});

function NotFound() {
  return (
    <>
      <TopBar crumbs={[{ label: "Not found" }]} />
      <div className="flex flex-1 items-start justify-center px-8 py-16" data-testid="not-found">
        <EmptyState
          icon={<SearchX className="size-5" />}
          title="Nothing here"
          description="This address is not a page of Autan."
        >
          <Link to="/" className="self-center">
            <Button>Go to Runs</Button>
          </Link>
        </EmptyState>
      </div>
    </>
  );
}

function Root() {
  useRegistrySubscription();
  useTheme();
  const [find, setFind] = useState(false);
  return (
    <MotionConfig reducedMotion="user" transition={{ duration: 0.12 }}>
      <TooltipProvider delay={400}>
        <ToastProvider>
          <div className="flex h-full">
            <Sidebar onFind={() => setFind(true)} />
            <main className="relative flex min-w-0 flex-1 flex-col bg-bg-1">
              <Outlet />
            </main>
          </div>
          <CommandPalette open={find} onOpenChange={setFind} />
        </ToastProvider>
      </TooltipProvider>
    </MotionConfig>
  );
}
