import { cn } from "./cn";

/** Placeholder with the shape of content still loading; never a spinner in its place. */
export function Skeleton({ className }: { className?: string }) {
  return <span aria-hidden className={cn("block animate-pulse rounded-xs bg-bg-3", className)} />;
}
