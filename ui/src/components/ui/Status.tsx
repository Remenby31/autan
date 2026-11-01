import type { RunStatus } from "../../api/contracts";
import { StateGlyph } from "./StateGlyph";

const LABEL: Record<RunStatus, string> = {
  running: "Running",
  converged: "Converged",
  completed: "Completed",
  stopped: "Stopped",
  diverged: "Diverged",
  failed: "Failed",
};

/** State glyph, its name, and how long the run has taken. */
export function Status({ status, duration }: { status: RunStatus; duration?: string | undefined }) {
  return (
    <span className="inline-flex items-center gap-2 whitespace-nowrap" data-status={status}>
      <StateGlyph status={status} />
      <span className="font-medium">{LABEL[status]}</span>
      {duration && <span className="text-fg-3 tabular-nums">{duration}</span>}
    </span>
  );
}

/** "1h 12m", "4m 05s", "12s". */
export function formatDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${String(sec).padStart(2, "0")}s`;
  return sec ? `${sec}s` : "<1s";
}
