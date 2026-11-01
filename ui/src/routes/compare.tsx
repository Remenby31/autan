import { useQueries, useQuery } from "@tanstack/react-query";
import { createFileRoute, Link, useNavigate } from "@tanstack/react-router";
import { GitCompareArrows, Plus, X } from "lucide-react";
import { useMemo, useState } from "react";
import type { Run } from "../api/contracts";
import { type LiveRun, runQuery, runsQuery, soleForces, useRunsHistory } from "../api/runs";
import { OverlayChart, type OverlaySeries } from "../components/OverlayChart";
import { TopBar } from "../components/shell/TopBar";
import { Button } from "../components/ui/Button";
import { cn } from "../components/ui/cn";
import { EmptyState } from "../components/ui/EmptyState";
import { Menu, MenuItem } from "../components/ui/Menu";
import { StateGlyph } from "../components/ui/StateGlyph";

const MAX_RUNS = 8;

export const Route = createFileRoute("/compare")({
  validateSearch: (raw: Record<string, unknown>): { runs: string[] } => ({
    runs: Array.isArray(raw.runs)
      ? raw.runs.filter((r): r is string => typeof r === "string").slice(0, MAX_RUNS)
      : [],
  }),
  component: Compare,
});

const seriesColor = (k: number) => `var(--series-${(k % 8) + 1})`;
const resolvedColor = (k: number) =>
  getComputedStyle(document.documentElement)
    .getPropertyValue(`--series-${(k % 8) + 1}`)
    .trim();

/** An angle written as the case writes it ("19.57 deg", "0.34 rad"), in degrees. */
export function degrees(value: string | null | undefined): number | null {
  const match = /^\s*(-?[\d.eE+-]+)\s*(deg|degree|degrees|rad|radian|radians)\s*$/.exec(
    value ?? "",
  );
  if (!match) return null;
  const number = Number(match[1]);
  if (!Number.isFinite(number)) return null;
  return match[2]?.startsWith("rad") ? (number * 180) / Math.PI : number;
}

/** Every leaf of a case document under its dotted path; arrays of tables by index. */
export function flatten(value: unknown, prefix = ""): Map<string, string> {
  const out = new Map<string, string>();
  if (value !== null && typeof value === "object") {
    for (const [key, child] of Object.entries(value)) {
      const path = prefix ? (Array.isArray(value) ? `${prefix}[${key}]` : `${prefix}.${key}`) : key;
      for (const [k, v] of flatten(child, path)) out.set(k, v);
    }
  } else {
    out.set(
      prefix,
      value === undefined ? "—" : typeof value === "string" ? value : JSON.stringify(value),
    );
  }
  return out;
}

function Compare() {
  const { runs: ids } = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const registry = useQuery(runsQuery);
  const live = useRunsHistory(ids);
  const details = useQueries({ queries: ids.map((id) => runQuery(id)) });
  const [all, setAll] = useState(false);
  const runs = ids.map((id) => registry.data?.runs.find((r) => r.run_id === id));
  const setRuns = (next: string[]) => void navigate({ search: { runs: next }, replace: true });
  const version = ids.map((id) => live.get(id)?.history.version ?? 0).join(",");

  // Histories grow in place: `version` is what changes when they do.
  // biome-ignore lint/correctness/useExhaustiveDependencies: see above
  const charts = useMemo(() => {
    const drop: OverlaySeries[] = [];
    const cl: OverlaySeries[] = [];
    const cd: OverlaySeries[] = [];
    ids.forEach((id, k) => {
      const history = live.get(id)?.history;
      if (!history) return;
      const label = id.slice(-8);
      const color = resolvedColor(k);
      drop.push({ label, color, x: history.iteration, y: history.orders });
      const forces = soleForces(history);
      if (forces) {
        cl.push({ label, color, x: forces.iteration, y: forces.cl });
        cd.push({ label, color, x: forces.iteration, y: forces.cd });
      }
    });
    return { drop, cl, cd };
  }, [ids, live, version]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: histories grow in place, see charts
  const polar = useMemo(
    () => polars(ids, runs, live, registry.data?.summaries ?? {}),
    [ids, runs, live, registry.data, version],
  );

  const diff = useMemo(() => {
    const flat = details.map((d) => (d.data ? flatten(d.data.case) : null));
    const keys = [...new Set(flat.flatMap((f) => (f ? [...f.keys()] : [])))].sort();
    return keys
      .map((key) => ({ key, values: flat.map((f) => (f ? (f.get(key) ?? "—") : "…")) }))
      .map((row) => ({ ...row, differs: new Set(row.values).size > 1 }));
  }, [details]);

  const others = (registry.data?.runs ?? []).filter((r) => !ids.includes(r.run_id));

  return (
    <>
      <TopBar crumbs={[{ label: "Runs", to: "/" }, { label: "Compare" }]} />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto flex max-w-[1200px] flex-col gap-6 px-6 py-8">
          <h1 className="m-0 font-semibold text-xl">Compare</h1>
          <div className="flex flex-wrap items-center gap-2" data-testid="compared">
            {ids.map((id, k) => (
              <RunChip
                key={id}
                id={id}
                run={runs[k]}
                role={String.fromCharCode(65 + k)}
                color={seriesColor(k)}
                onRemove={() => setRuns(ids.filter((other) => other !== id))}
              />
            ))}
            {ids.length < MAX_RUNS && others.length > 0 && (
              <Menu
                label="Add a run"
                trigger={<Button size="sm" variant="tertiary" icon={<Plus className="size-4" />} />}
              >
                {others.slice(0, 20).map((run) => (
                  <MenuItem
                    key={run.run_id}
                    icon={<StateGlyph status={run.status} />}
                    onClick={() => setRuns([...ids, run.run_id])}
                  >
                    {run.case.name}{" "}
                    <span className="font-mono text-fg-3">{run.run_id.slice(-8)}</span>
                  </MenuItem>
                ))}
              </Menu>
            )}
          </div>

          {ids.length < 2 ? (
            <EmptyState
              icon={<GitCompareArrows className="size-5" />}
              title="Pick two runs or more"
              description="Select runs on the Runs page with x, then press c."
            >
              <Link to="/" className="self-center">
                <Button>Go to Runs</Button>
              </Link>
            </EmptyState>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
                <ChartCard title="Residual drop" unit="orders" testId="chart-drop">
                  <OverlayChart
                    series={charts.drop}
                    xLabel="iteration"
                    sync="compare"
                    version={version}
                  />
                </ChartCard>
                <ChartCard title="CL" unit="iteration">
                  <OverlayChart
                    series={charts.cl}
                    xLabel="iteration"
                    sync="compare"
                    version={version}
                  />
                </ChartCard>
                <ChartCard title="CD" unit="iteration">
                  <OverlayChart
                    series={charts.cd}
                    xLabel="iteration"
                    sync="compare"
                    version={version}
                  />
                </ChartCard>
              </div>

              {polar && (
                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2" data-testid="polar">
                  <ChartCard title="Lift polar" unit="CL against α (deg), converged runs">
                    <OverlayChart series={[polar.lift]} xLabel="α (deg)" points version={version} />
                  </ChartCard>
                  <ChartCard title="Drag polar" unit="CD against CL">
                    <OverlayChart series={[polar.drag]} xLabel="CL" points version={version} />
                  </ChartCard>
                </div>
              )}

              <section className="overflow-hidden rounded-lg border border-line bg-bg-0">
                <header className="flex items-center gap-3 border-line border-b px-4 py-3">
                  <span className="font-medium">Settings</span>
                  <span className="text-fg-3 text-sm">
                    {diff.filter((d) => d.differs).length} differ of {diff.length}
                  </span>
                  <Button
                    size="sm"
                    variant="tertiary"
                    className="ml-auto"
                    aria-pressed={all}
                    onClick={() => setAll(!all)}
                  >
                    {all ? "Only differences" : "Show all"}
                  </Button>
                </header>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse text-sm" data-testid="settings-diff">
                    <thead>
                      <tr className="text-left text-fg-3">
                        <th className="h-9 px-4 font-normal">Setting</th>
                        {ids.map((id, k) => (
                          <th key={id} className="px-4 font-normal">
                            <span className="inline-flex items-center gap-2">
                              <i
                                className="inline-block size-2 rounded-full"
                                style={{ background: seriesColor(k) }}
                              />
                              {String.fromCharCode(65 + k)}
                            </span>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {diff
                        .filter((row) => all || row.differs)
                        .map((row) => (
                          <tr key={row.key} className="border-line border-t">
                            <td className="h-9 px-4 font-mono text-fg-2">{row.key}</td>
                            {row.values.map((value, k) => (
                              <td
                                key={ids[k]}
                                className={cn(
                                  "px-4 font-mono",
                                  row.differs &&
                                    k > 0 &&
                                    value !== row.values[0] &&
                                    "bg-accent-soft",
                                )}
                              >
                                {value}
                              </td>
                            ))}
                          </tr>
                        ))}
                    </tbody>
                  </table>
                  {!all && !diff.some((d) => d.differs) && (
                    <p className="m-0 px-4 py-6 text-center text-fg-3">The cases are identical.</p>
                  )}
                </div>
              </section>
            </>
          )}
        </div>
      </div>
    </>
  );
}

function polars(
  ids: string[],
  runs: (Run | undefined)[],
  live: Map<string, LiveRun>,
  summaries: Record<string, { settings?: { aoa: string | null } }>,
): { lift: OverlaySeries; drag: OverlaySeries } | null {
  // A polar point is a converged run's final coefficients; the newest run wins at a repeated α.
  const points = ids
    .filter((_, k) => runs[k]?.status === "converged")
    .map((id) => {
      const forces = live.get(id)?.history
        ? soleForces(live.get(id)?.history as LiveRun["history"])
        : null;
      const alpha = degrees(summaries[id]?.settings?.aoa);
      const cl = forces?.cl.at(-1);
      const cd = forces?.cd.at(-1);
      return alpha === null || cl === undefined || cd === undefined ? null : { alpha, cl, cd };
    })
    .filter((p): p is { alpha: number; cl: number; cd: number } => p !== null);
  if (points.length < 2 || new Set(points.map((p) => p.alpha)).size < 2 || runs.length < 2)
    return null;
  const latest = new Map(points.map((p) => [p.alpha, p]));
  const unique = [...latest.values()];
  if (unique.length < 2) return null;
  const byAlpha = [...unique].sort((a, b) => a.alpha - b.alpha);
  const byCl = [...unique].sort((a, b) => a.cl - b.cl);
  const color = getComputedStyle(document.documentElement).getPropertyValue("--fg").trim();
  return {
    lift: { label: "CL", color, x: byAlpha.map((p) => p.alpha), y: byAlpha.map((p) => p.cl) },
    drag: { label: "CD", color, x: byCl.map((p) => p.cl), y: byCl.map((p) => p.cd) },
  };
}

function ChartCard({
  title,
  unit,
  testId,
  children,
}: {
  title: string;
  unit: string;
  testId?: string;
  children: React.ReactNode;
}) {
  return (
    <section
      className="flex h-72 flex-col gap-2 rounded-lg border border-line bg-bg-0 p-4"
      data-testid={testId}
    >
      <header className="flex items-baseline gap-2">
        <span className="font-medium">{title}</span>
        <span className="text-fg-3 text-sm">{unit}</span>
      </header>
      <div className="min-h-0 flex-1">{children}</div>
    </section>
  );
}

function RunChip({
  id,
  run,
  role,
  color,
  onRemove,
}: {
  id: string;
  run: Run | undefined;
  role: string;
  color: string;
  onRemove: () => void;
}) {
  return (
    <span className="inline-flex h-8 items-center gap-2 rounded-full border border-line bg-bg-0 pr-1 pl-3 text-sm">
      <i className="inline-block size-2.5 rounded-full" style={{ background: color }} />
      <span className="font-mono text-fg-3">{role}</span>
      <Link
        to="/runs/$runId"
        params={{ runId: id }}
        className="flex items-center gap-2 hover:underline"
      >
        {run && <StateGlyph status={run.status} />}
        <span className="font-medium">{run?.case.name ?? "…"}</span>
        <span className="font-mono text-fg-3">{id.slice(-8)}</span>
      </Link>
      <button
        type="button"
        aria-label={`Remove ${id} from the comparison`}
        onClick={onRemove}
        className="flex size-6 items-center justify-center rounded-full text-fg-3 hover:bg-bg-2 hover:text-fg"
      >
        <X className="size-3.5" />
      </button>
    </span>
  );
}
