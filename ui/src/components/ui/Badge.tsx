import type { ReactNode } from "react";
import { cn } from "./cn";

type Tone = "gray" | "blue" | "green" | "amber" | "red" | "purple";

const TONE: Record<Tone, string> = {
  gray: "bg-bg-2 text-fg-2",
  blue: "bg-accent-soft text-accent",
  green: "bg-success-soft text-success",
  amber: "bg-warning-soft text-warning",
  red: "bg-danger-soft text-danger",
  purple: "bg-[color-mix(in_srgb,var(--series-4)_14%,transparent)] text-[var(--series-4)]",
};

/** Short label for a category or count; colour only when it carries state. */
export function Badge({
  tone = "gray",
  size = "sm",
  icon,
  children,
  className,
}: {
  tone?: Tone;
  size?: "sm" | "md";
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex flex-none items-center gap-1 whitespace-nowrap rounded-full font-medium",
        size === "sm" ? "h-5 px-2 text-xs" : "h-6 px-2.5 text-sm",
        TONE[tone],
        className,
      )}
    >
      {icon}
      {children}
    </span>
  );
}
