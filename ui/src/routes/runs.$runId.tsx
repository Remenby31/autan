import { useQuery, useQueryClient } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Box, Copy } from "lucide-react";
import { useEffect, useSyncExternalStore } from "react";
import type { Run } from "../api/contracts";
import { type HistoryColumns, planQuery, runQuery, soleForces, useLiveRun } from "../api/runs";
import { ResidualChart } from "../components/ResidualChart";
import { formatBytes, formatCount, remainingMs } from "../components/run/format";
import { LogView } from "../components/run/LogView";
import { MiddleTruncate } from "../components/run/MiddleTruncate";
import { MonitorTable } from "../components/run/MonitorTable";
import { OutputList } from "../components/run/OutputList";
import { PlanTable } from "../components/run/PlanTable";
import { TopBar } from "../components/shell/TopBar";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Collapse } from "../components/ui/Collapse";
import { Gauge } from "../components/ui/Gauge";
import { LoadingDots } from "../components/ui/LoadingDots";
import { MetaGrid } from "../components/ui/MetaGrid";
import { Note } from "../components/ui/Note";
import { RelativeTime } from "../components/ui/RelativeTime";
import { Skeleton } from "../components/ui/Skeleton";
import { Snippet } from "../components/ui/Snippet";
import { formatDuration, Status } from "../components/ui/Status";
import { TabList, TabPanel, TabsRoot } from "../components/ui/Tabs";
import { useToast } from "../components/ui/Toast";
import { Tooltip } from "../components/ui/Tooltip";

const TABS = ["run", "log", "case", "outputs"] as const;
type Tab = (typeof TABS)[number];

export const Route = createFileRoute("/runs/$runId")({
  validateSearch: (raw: Record<string, unknown>): { tab?: Tab | undefined } =>
    TABS.includes(raw.tab as Tab) && raw.tab !== "run" ? { tab: raw.tab as Tab } : {},
  component: RunView,
});

const second = {
  subscribe: (listener: () => void) => {
    const timer = setInterval(listener, 1000);
    return () => clearInterval(timer);
  },
  now: () => Math.floor(Date.now() / 1000) * 1000,
};

function RunView() {
  const { runId } = Route.useParams();
  const { tab = "run" } = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const { run, history, connected, invalid } = useLiveRun(runId);
  const details = useQuery({
    ...runQuery(runId),
    // Outputs grow while the run writes them; the status change brings solution.cgns.
    refetchInterval: run?.status === "running" ? 5000 : false,
  });
  const client = useQueryClient();
  // biome-ignore lint/correctness/useExhaustiveDependencies: refetch once per status change
  useEffect(() => {
    void client.invalidateQueries({ queryKey: runQuery(runId).queryKey });
  }, [run?.status]);
  const plan = useQuery(planQuery(runId));
  const toast = useToast();
  const now = useSyncExternalStore(second.subscribe, second.now);
  const time = details.data?.case.time;
  const targetOrders = time?.kind === "steady" ? time.residual_orders : undefined;
  const maxIterations = time?.kind === "steady" ? time.max_iterations : time?.steps;
  const iteration = history.iteration.at(-1) ?? 0;
  const orders = history.orders.at(-1);
  const forces = soleForces(history);

  const crumbs = [
    { label: "Runs", to: "/" },
    {
      label: run?.case.name ?? "…",
      to: `/?case=${encodeURIComponent(JSON.stringify([run?.case.name ?? ""]))}`,
    },
    { label: <span className="font-mono">{runId.slice(-8)}</span> },
  ];

  return (
    <>
      <TopBar
        crumbs={crumbs}
        actions={
          <>
            <Button
              size="sm"
              icon={<Copy className="size-4" />}
              onClick={() =>
                void navigator.clipboard.writeText(runId).then(() => toast("Run id copied"))
              }
            >
              Copy Run Id
            </Button>
            <Tooltip content="3D fields arrive with solution.cgns surfaces">
              <span>
                <Button size="sm" icon={<Box className="size-4" />} disabled>
                  Open in Analysis
                </Button>
              </span>
            </Tooltip>
          </>
        }
      />
      <TabsRoot
        value={tab}
        onValueChange={(value) =>
          void navigate({
            search: { tab: value === "run" ? undefined : (value as Tab) },
            replace: true,
          })
        }
        className="flex min-h-0 flex-1 flex-col"
      >
        <div className="border-line border-b bg-bg-1 px-6 py-2">
          <TabList
            tabs={[
              { value: "run", label: "Run" },
              { value: "log", label: "Log" },
              { value: "case", label: "Case" },
              {
                value: "outputs",
                label: (
                  <span className="flex items-center gap-2">
                    Outputs
                    {details.data && <Badge>{details.data.outputs.length}</Badge>}
                  </span>
                ),
              },
            ]}
          />
        </div>
        <div className="min-h-0 flex-1 overflow-auto">
          <div className="mx-auto flex max-w-[1200px] flex-col gap-6 px-6 py-8">
            <header className="flex items-center gap-4">
              <h1 className="m-0 font-semibold text-xl">
                {run?.case.name ?? <Skeleton className="h-8 w-40" />}
              </h1>
              {run && (
                <span data-testid="status">
                  <Status status={run.status} />
                </span>
              )}
              {run?.status === "running" && (
                <span className="flex items-center gap-2 text-fg-3 text-sm">
                  {connected ? (
                    "Live"
                  ) : (
                    <>
                      Reconnecting <LoadingDots />
                    </>
                  )}
                </span>
              )}
            </header>

            {invalid && (
              <Note tone="error" title="history.jsonl breaks the results contract">
                {invalid}
              </Note>
            )}
            {run?.diagnostics.map((d) => (
              <Note
                key={`${d.code}-${String(d.data.iteration)}`}
                tone={d.severity === "warning" ? "warning" : "error"}
                title={
                  <span>
                    <span className="font-mono">{d.code}</span> {d.message}
                  </span>
                }
              >
                {d.fix && <p className="m-0">{d.fix}</p>}
                {d.doc && (
                  <a
                    href={d.doc}
                    className="text-accent hover:underline"
                    target="_blank"
                    rel="noreferrer"
                  >
                    {d.code} reference
                  </a>
                )}
              </Note>
            ))}

            <TabPanel value="run" className="flex flex-col gap-6 outline-none">
              <section className="grid grid-cols-1 gap-6 rounded-lg border border-line bg-bg-0 p-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
                <div className="h-72 min-w-0 rounded-md border border-line bg-bg-1 p-3">
                  <ResidualChart history={history} />
                </div>
                {run ? (
                  <Facts
                    run={run}
                    now={now}
                    iteration={iteration}
                    maxIterations={maxIterations}
                    orders={orders}
                    targetOrders={targetOrders}
                    history={history}
                    cells={details.data?.mesh_info.counts.cells}
                    cl={forces?.cl.at(-1)}
                    cd={forces?.cd.at(-1)}
                  />
                ) : (
                  <Skeleton className="h-48 w-full" />
                )}
              </section>

              <section className="rounded-lg border border-line bg-bg-0">
                <Collapse
                  title="Plan"
                  meta={
                    plan.data ? (
                      plan.data.ok ? (
                        <Badge tone="green">{plan.data.boundaries.length} zones covered</Badge>
                      ) : (
                        <Badge tone="red">{plan.data.diagnostics.length} problems</Badge>
                      )
                    ) : (
                      <LoadingDots />
                    )
                  }
                >
                  {plan.data && <PlanTable plan={plan.data} />}
                </Collapse>
                <Collapse
                  title="Monitors"
                  defaultOpen
                  meta={
                    <span className="font-mono text-fg-2 text-sm tabular-nums">
                      {forces
                        ? `CL ${forces.cl.at(-1)?.toFixed(3)} · CD ${forces.cd.at(-1)?.toFixed(4)}`
                        : "—"}
                    </span>
                  }
                >
                  <MonitorTable history={history} />
                </Collapse>
                <Collapse
                  title="Outputs"
                  meta={
                    details.data && (
                      <span className="text-fg-2 text-sm">
                        {details.data.outputs.length} files ·{" "}
                        {formatBytes(details.data.outputs.reduce((s, o) => s + o.bytes, 0))}
                      </span>
                    )
                  }
                >
                  {details.data && <OutputList outputs={details.data.outputs} />}
                </Collapse>
                <Collapse
                  title="Log"
                  meta={
                    <span className="text-fg-2 text-sm tabular-nums">
                      {history.iteration.length.toLocaleString("en")} lines
                    </span>
                  }
                >
                  <LogView
                    history={history}
                    diagnostics={run?.diagnostics ?? []}
                    running={run?.status === "running"}
                    height={240}
                  />
                </Collapse>
              </section>
            </TabPanel>

            <TabPanel value="log" className="outline-none">
              <LogView
                history={history}
                diagnostics={run?.diagnostics ?? []}
                running={run?.status === "running"}
              />
            </TabPanel>

            <TabPanel value="case" className="flex flex-col gap-4 outline-none">
              <Snippet text={`autan run case.toml --mesh-info mesh-info.json`} />
              <pre className="m-0 overflow-auto rounded-lg border border-line bg-bg-0 p-4 font-mono text-sm leading-6">
                {details.data?.case_toml}
              </pre>
            </TabPanel>

            <TabPanel
              value="outputs"
              className="rounded-lg border border-line bg-bg-0 px-4 outline-none"
            >
              {details.data && <OutputList outputs={details.data.outputs} />}
            </TabPanel>
          </div>
        </div>
      </TabsRoot>
    </>
  );
}

function Facts({
  run,
  now,
  iteration,
  maxIterations,
  orders,
  targetOrders,
  history,
  cells,
  cl,
  cd,
}: {
  run: Run;
  now: number;
  iteration: number;
  maxIterations: number | undefined;
  orders: number | undefined;
  targetOrders: number | undefined;
  history: HistoryColumns;
  cells: number | undefined;
  cl: number | undefined;
  cd: number | undefined;
}) {
  const started = new Date(run.started).getTime();
  const end = run.finished ? new Date(run.finished).getTime() : now;
  const left =
    run.status === "running"
      ? remainingMs({
          iteration,
          elapsedMs: now - started,
          maxIterations,
          orders: history.orders,
          iterations: history.iteration,
          targetOrders,
        })
      : null;
  return (
    <MetaGrid
      items={[
        { label: "Case", value: <span className="font-medium">{run.case.name}</span> },
        { label: "State", value: <Status status={run.status} /> },
        {
          label: "Iteration",
          value: (
            <>
              {maxIterations !== undefined && (
                <Gauge
                  value={iteration}
                  max={maxIterations}
                  size={20}
                  label="Iteration budget used"
                />
              )}
              <span className="font-mono tabular-nums" data-testid="iteration">
                {iteration}
              </span>
              {maxIterations !== undefined && (
                <span className="font-mono text-fg-3 tabular-nums">
                  / {maxIterations.toLocaleString("en")}
                </span>
              )}
            </>
          ),
        },
        {
          label: "Residual drop",
          value:
            orders !== undefined && Number.isFinite(orders) ? (
              <>
                {targetOrders !== undefined && (
                  <Gauge
                    value={orders}
                    max={targetOrders}
                    size={20}
                    label="Residual drop towards the target"
                  />
                )}
                <span className="font-mono tabular-nums">{orders.toFixed(2)}</span>
                {targetOrders !== undefined && (
                  <span className="font-mono text-fg-3">/ {targetOrders} orders</span>
                )}
              </>
            ) : (
              "—"
            ),
        },
        {
          label: "Devices",
          value: (
            <span className="font-mono">
              {run.devices.count}× {run.devices.kind} · {run.precision}
            </span>
          ),
        },
        { label: "Started", value: <RelativeTime iso={run.started} /> },
        {
          label: run.status === "running" ? "Elapsed · left" : "Duration",
          value: (
            <span className="tabular-nums">
              {formatDuration(end - started)}
              {left !== null && <span className="text-fg-3"> · ≤ {formatDuration(left)}</span>}
            </span>
          ),
        },
        {
          label: "Mesh",
          value: (
            <>
              <MiddleTruncate value={run.mesh.sha256} />
              {cells !== undefined && <span className="text-fg-3">{formatCount(cells)} cells</span>}
            </>
          ),
        },
        {
          label: "Forces",
          value:
            cl !== undefined && cd !== undefined ? (
              <span className="flex flex-col font-mono tabular-nums">
                <span>CL {cl.toFixed(4)}</span>
                <span>CD {cd.toFixed(5)}</span>
              </span>
            ) : (
              "—"
            ),
        },
        {
          label: "Solver",
          value: (
            <span className="flex items-center gap-2">
              <MiddleTruncate value={run.solver.commit} keep={8} />
              {run.solver.backend === "fake" && <Badge tone="amber">fake</Badge>}
            </span>
          ),
        },
      ]}
    />
  );
}
