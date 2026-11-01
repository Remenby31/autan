import type { RunStatus } from "../../api/contracts";

// Glyphs of docs/design/ui-inspiration/README.md § Components, state glyph.
const COLOR: Record<RunStatus, string> = {
  running: "var(--state-running)",
  converged: "var(--state-converged)",
  completed: "var(--state-completed)",
  stopped: "var(--state-stopped)",
  diverged: "var(--state-diverged)",
  failed: "var(--state-failed)",
};

/** 8 px run-state glyph; its accessible name is the state. */
export function StateGlyph({ status }: { status: RunStatus }) {
  const color = COLOR[status];
  return (
    <span
      role="img"
      aria-label={status}
      className="relative inline-grid size-2 flex-none place-items-center"
      style={{ color }}
    >
      {status === "running" && (
        <span
          className="absolute inset-0 rounded-full bg-current"
          style={{ animation: "halo var(--dur-pulse) var(--ease-out) infinite" }}
        />
      )}
      {(status === "running" || status === "converged" || status === "completed") && (
        <span className="size-2 rounded-full bg-current" />
      )}
      {status === "completed" && <span className="absolute h-px w-1.5 bg-bg-0" />}
      {status === "stopped" && <span className="size-2 rounded-[1px] bg-current" />}
      {status === "failed" && <span className="font-semibold text-[10px] leading-none">×</span>}
      {status === "diverged" && (
        <span className="grid size-2 place-items-center rounded-full border border-current font-semibold text-[6px] leading-none">
          !
        </span>
      )}
    </span>
  );
}
