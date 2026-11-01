import { CircleAlert, Info, TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "./cn";

type Tone = "info" | "warning" | "error";

const TONE: Record<Tone, { box: string; icon: ReactNode }> = {
  info: { box: "border-line bg-bg-1", icon: <Info className="size-4 text-fg-2" /> },
  warning: {
    box: "border-warning/40 bg-warning-soft",
    icon: <TriangleAlert className="size-4 text-warning" />,
  },
  error: {
    box: "border-danger/50 bg-danger-soft",
    icon: <CircleAlert className="size-4 text-danger" />,
  },
};

/** A message about the thing on screen: what happened, then what to do. */
export function Note({
  tone = "info",
  title,
  children,
  action,
}: {
  tone?: Tone;
  title: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
}) {
  const { box, icon } = TONE[tone];
  return (
    <div
      role={tone === "error" ? "alert" : "note"}
      className={cn("flex gap-3 rounded-md border p-3", box)}
    >
      <span className="mt-0.5 flex-none">{icon}</span>
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="font-medium">{title}</span>
        {children && <div className="text-fg-2 text-sm">{children}</div>}
      </div>
      {action}
    </div>
  );
}
