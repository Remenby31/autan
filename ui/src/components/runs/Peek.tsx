import { Link } from "@tanstack/react-router";
import { X } from "lucide-react";
import { motion } from "motion/react";
import type { Run, RunSummary } from "../../api/contracts";
import { devicesLabel } from "../../api/devices";
import { formatCount } from "../run/format";
import { Kbd } from "../ui/Kbd";
import { RelativeTime } from "../ui/RelativeTime";
import { Sparkline } from "../ui/Sparkline";
import { Status } from "../ui/Status";

/** Space on a row: the run's state, residual drop and settings, without leaving the list. */
export function Peek({
  run,
  summary,
  onClose,
}: {
  run: Run;
  summary: RunSummary | undefined;
  onClose: () => void;
}) {
  const settings = summary?.settings;
  const facts: [string, React.ReactNode][] = [
    [
      "Iteration",
      `${(summary?.iteration ?? run.iterations).toLocaleString("en")}${summary?.max_iterations ? ` / ${formatCount(summary.max_iterations)}` : ""}`,
    ],
    [
      "Residual drop",
      summary?.orders != null
        ? `${summary.orders.toFixed(2)}${summary.target_orders ? ` / ${summary.target_orders}` : ""}`
        : "—",
    ],
    ["α", settings?.aoa ?? "—"],
    ["Mach", settings?.mach ?? "—"],
    ["Re", settings?.re != null ? settings.re.toExponential(2) : "—"],
    ["Turbulence", settings?.turbulence ?? "—"],
    ["Devices", `${devicesLabel(run.devices)} · ${run.precision}`],
  ];
  return (
    <motion.aside
      aria-label={`Preview of ${run.run_id}`}
      data-testid="peek"
      initial={{ opacity: 0, x: 16 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 16 }}
      className="fixed top-16 right-4 bottom-4 z-30 flex w-80 flex-col gap-5 overflow-auto rounded-lg bg-bg-0 p-5 shadow-dialog"
    >
      <header className="flex items-start gap-2">
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <span className="font-semibold text-md">{run.case.name}</span>
          <span className="truncate font-mono text-fg-3 text-xs">{run.run_id}</span>
        </div>
        <button
          type="button"
          aria-label="Close the preview"
          onClick={onClose}
          className="flex size-7 items-center justify-center rounded-sm text-fg-3 hover:bg-bg-2 hover:text-fg"
        >
          <X className="size-4" />
        </button>
      </header>
      <Status status={run.status} />
      <Sparkline
        values={summary?.trend ?? []}
        {...(summary?.target_orders ? { target: summary.target_orders } : {})}
        width={240}
        height={64}
        color="var(--accent)"
        label="Residual drop along the run"
      />
      <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
        {facts.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-fg-3">{label}</dt>
            <dd className="m-0 text-right font-mono">{value}</dd>
          </div>
        ))}
        <dt className="text-fg-3">Started</dt>
        <dd className="m-0 text-right">
          <RelativeTime iso={run.started} />
        </dd>
      </dl>
      <footer className="mt-auto flex items-center gap-3 text-fg-3 text-sm">
        <Link
          to="/runs/$runId"
          params={{ runId: run.run_id }}
          className="text-accent hover:underline"
        >
          Open run
        </Link>
        <span className="ml-auto flex items-center gap-1.5">
          <Kbd>Space</Kbd> close
        </span>
      </footer>
    </motion.aside>
  );
}
