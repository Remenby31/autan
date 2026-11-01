import type { ReactNode } from "react";
import { cn } from "./cn";

export function Kbd({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <kbd
      className={cn(
        "inline-flex h-5 min-w-5 items-center justify-center rounded-xs border border-line bg-bg-1 px-1 font-mono text-[11px] text-fg-2",
        className,
      )}
    >
      {children}
    </kbd>
  );
}
