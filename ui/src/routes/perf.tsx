import { useQuery } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Gauge } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  type PerfHistory,
  type PerfSeries,
  perfQuery,
  perfSeries,
  TIME_PER_CELL,
  trendDomain,
} from "../api/perf";
import { HttpError } from "../api/runs";
import { PerfChart } from "../components/PerfChart";
import { TopBar } from "../components/shell/TopBar";
import { Badge } from "../components/ui/Badge";
import { cn } from "../components/ui/cn";
import { EmptyState } from "../components/ui/EmptyState";
import { RelativeTime } from "../components/ui/RelativeTime";
import { Snippet } from "../components/ui/Snippet";
import { Sparkline } from "../components/ui/Sparkline";

const REPO = "https://github.com/Remenby31/autan";

/** What the history cannot show yet, and where it is to come from. */
const MISSING = [
  {
    what: "Gates",
    why: "% of peak gates are prose (acceptance.md, multigrid-gpu.md)",
    where: "bench/gates.json → history.json",
  },
  {
    what: "Tetrahedra, MG transfers, Euler residual, host",
    why: "main measures codegen on hex n128 and STREAM only",
    where: "gpu-bench on main",
  },
  {
    what: "FUN3D on GB10",
    why: "an estimate, not a measurement; not yet a CI artifact",
    where: "bench e2e report (#336)",
    estimate: true,
  },
];

export const Route = createFileRoute("/perf")({
  validateSearch: (raw: Record<string, unknown>): { b?: string } =>
    typeof raw.b === "string" ? { b: raw.b } : {},
  component: Perf,
});

function Perf() {
  const perf = useQuery(perfQuery);
  return (
    <>
      <TopBar crumbs={[{ label: "Performance" }]} />
      {perf.data ? (
        <PerfView history={perf.data} />
      ) : (
        <div className="flex flex-1 items-start justify-center px-8 py-16">
          {perf.isPending ? null : perf.error instanceof HttpError && perf.error.status === 404 ? (
            <EmptyState
              icon={<Gauge className="size-5" />}
              title="No performance history"
              description="Download main's perf-history artifact, then serve it."
            >
              <Snippet text="gh run download -R Remenby31/autan -n perf-history -D perf" />
              <Snippet text="autan ui --perf perf" />
            </EmptyState>
          ) : (
            <EmptyState
              icon={<Gauge className="size-5" />}
              title="Unreadable performance history"
              description={perf.error instanceof HttpError ? perf.error.detail : String(perf.error)}
            />
          )}
        </div>
      )}
    </>
  );
}

function PerfView({ history }: { history: PerfHistory }) {
  const series = useMemo(() => perfSeries(history), [history]);
  const { b } = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const selected = series.find((s) => s.name === b) ?? series[0];
  const select = (name: string) => void navigate({ search: { b: name }, replace: true });
  const list = useRef<HTMLDivElement>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const culprits = useMemo(() => history.records.map((r) => Boolean(r.regression)), [history]);
  const dates = useMemo(() => history.records.map((r) => r.measured), [history]);
  const last = history.records.at(-1);

  // j/k and the arrows walk the benchmarks, unless a field, a dialog or another handler has them.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (e.defaultPrevented || document.querySelector('[aria-modal="true"]')) return;
      if (target.closest('input, textarea, [contenteditable=true], [role="dialog"]')) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const step =
        e.key === "j" || e.key === "ArrowDown" ? 1 : e.key === "k" || e.key === "ArrowUp" ? -1 : 0;
      if (!step || !selected) return;
      e.preventDefault();
      const at = series.indexOf(selected);
      const next = series[Math.min(Math.max(at + step, 0), series.length - 1)];
      if (!next) return;
      select(next.name);
      list.current?.querySelector<HTMLElement>(`[data-name="${next.name}"]`)?.focus();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!selected || !last)
    return (
      <div className="flex flex-1 items-start justify-center px-8 py-16">
        <EmptyState
          icon={<Gauge className="size-5" />}
          title="No bandwidth measured"
          description="The history holds no commit with a bandwidth benchmark."
        />
      </div>
    );

  // The commit under the cursor, measured or not; the last measurement otherwise.
  const at = hovered ?? selected.last;
  const record = history.records[at] as PerfHistory["records"][number];
  const measure = record.benchmarks[selected.name];
  const timing = record.benchmarks[selected.name + TIME_PER_CELL];
  const suites = [...new Set(series.map((s) => s.suite))];

  return (
    <div className="grid min-h-0 flex-1 grid-rows-[auto_1fr] md:grid-cols-[minmax(320px,400px)_1fr] md:grid-rows-1">
      <div
        ref={list}
        role="listbox"
        aria-label="Benchmarks"
        className="min-h-0 overflow-auto border-line max-md:max-h-[40vh] max-md:border-b md:border-r"
      >
        <div className="sticky top-0 z-10 flex items-baseline justify-between gap-3 border-line border-b bg-bg-1 px-4 py-3 text-sm">
          <span>
            <span className="font-medium tabular-nums" data-testid="perf-peak">
              {history.peak_GBps.toFixed(1)} GB/s
            </span>{" "}
            <span className="text-fg-3">peak, STREAM</span>
          </span>
          <span className="text-fg-3 tabular-nums">{history.records.length} commits</span>
        </div>
        {suites.map((suite) => (
          <div key={suite}>
            <div className="px-4 pt-3 pb-1 font-mono text-fg-3 text-xs uppercase">{suite}</div>
            {series
              .filter((s) => s.suite === suite)
              .map((s) => (
                <Row key={s.name} series={s} active={s === selected} onSelect={select} />
              ))}
          </div>
        ))}
      </div>

      <section className="flex min-h-0 flex-col gap-4 px-6 py-5" aria-label={selected.name}>
        <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-3">
          <div className="flex flex-col gap-1">
            <h1 className="m-0 font-mono font-medium text-md">{selected.kernel}</h1>
            <div className="flex items-baseline gap-3">
              <span className="font-semibold text-2xl tabular-nums" data-testid="perf-percent">
                {measure?.percent_of_peak !== undefined
                  ? `${measure.percent_of_peak.toFixed(1)} %`
                  : "not measured"}
              </span>
              {measure && (
                <span className="text-fg-3 tabular-nums">
                  {measure.median.toFixed(1)} GB/s ± {(measure.cv * 100).toFixed(1)} %
                </span>
              )}
              {timing?.unit === "ns" && (
                <span className="text-fg-3 tabular-nums">{timing.median.toFixed(3)} ns/cell</span>
              )}
            </div>
          </div>
          <dl className="m-0 grid grid-cols-[auto_auto] gap-x-3 gap-y-1 text-sm">
            <dt className="text-fg-3">Commit</dt>
            <dd className="m-0 flex items-center gap-2" data-testid="perf-commit">
              <a
                href={`${REPO}/commit/${record.commit}`}
                target="_blank"
                rel="noreferrer"
                className="font-mono hover:underline"
              >
                {record.commit.slice(0, 8)}
              </a>
              <RelativeTime iso={record.measured} />
              {record.regression && (
                <span data-testid="perf-culprit">
                  <Badge tone="red">regression</Badge>
                </span>
              )}
            </dd>
            <dt className="text-fg-3">Host</dt>
            <dd className="m-0 font-mono">{record.host ?? "—"}</dd>
            <dt className="text-fg-3">Gate</dt>
            <dd className="m-0" data-testid="perf-gate">
              {selected.tracked ? "−5 % vs baseline (CI)" : "—"}
            </dd>
          </dl>
        </div>

        <div className="min-h-[200px] flex-1" data-testid="perf-chart">
          <PerfChart
            percent={selected.percent}
            culprits={culprits}
            dates={dates}
            onHover={setHovered}
          />
        </div>

        <div className="border-line border-t pt-3" data-testid="perf-missing">
          <h2 className="m-0 mb-2 font-medium text-fg-3 text-xs uppercase">Not in the history</h2>
          <ul className="m-0 grid list-none gap-x-6 gap-y-2 p-0 text-sm xl:grid-cols-3">
            {MISSING.map((m) => (
              <li key={m.what} className="flex flex-col">
                <span className="flex items-center gap-2 font-medium">
                  {m.what}
                  {m.estimate && <Badge tone="amber">ESTIMATE</Badge>}
                </span>
                <span className="text-fg-3">
                  {m.why} · {m.where}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </section>
    </div>
  );
}

function Row({
  series,
  active,
  onSelect,
}: {
  series: PerfSeries;
  active: boolean;
  onSelect: (name: string) => void;
}) {
  const values = series.percent.filter((p): p is number => p !== null);
  const latest = series.percent[series.last] as number;
  return (
    <button
      type="button"
      role="option"
      aria-selected={active}
      data-name={series.name}
      tabIndex={active ? 0 : -1}
      onClick={() => onSelect(series.name)}
      className={cn(
        "flex h-9 w-full items-center gap-3 px-4 text-left hover:bg-bg-2",
        active && "bg-bg-3",
      )}
    >
      <span className="min-w-0 flex-1 truncate font-mono text-sm">
        {series.kernel}
        {series.tracked && (
          <span className="ml-2 text-fg-3" title="Gated in CI against its baseline">
            ●
          </span>
        )}
      </span>
      <Sparkline
        values={series.percent}
        width={80}
        height={20}
        label={`${series.kernel} % of peak`}
        domain={trendDomain(values)}
      />
      <span className="w-14 text-right tabular-nums">{latest.toFixed(1)} %</span>
    </button>
  );
}
