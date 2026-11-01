import type { ReactNode } from "react";
import { cn } from "./cn";

/** One thing in a list: icon, name, a muted line under it, facts on the right, actions last. */
export function Entity({
  icon,
  title,
  description,
  meta,
  actions,
  className,
}: {
  icon?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex min-h-12 items-center gap-3 py-2", className)}>
      {icon && (
        <span className="flex size-8 flex-none items-center justify-center rounded-sm border border-line text-fg-2">
          {icon}
        </span>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <span className="truncate font-medium">{title}</span>
        {description && <span className="truncate text-fg-3 text-sm">{description}</span>}
      </div>
      {meta && <div className="flex flex-none items-center gap-3 text-fg-2 text-sm">{meta}</div>}
      {actions}
    </div>
  );
}
