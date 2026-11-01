import type { QueryClient } from "@tanstack/react-query";
import { createRootRouteWithContext, Outlet } from "@tanstack/react-router";
import { MotionConfig } from "motion/react";
import { useState } from "react";
import { useRegistrySubscription } from "../api/runs";
import { CommandPalette } from "../components/CommandPalette";
import { Sidebar } from "../components/shell/Sidebar";
import { useTheme } from "../components/shell/theme";
import { ToastProvider } from "../components/ui/Toast";
import { TooltipProvider } from "../components/ui/Tooltip";

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  component: Root,
});

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
