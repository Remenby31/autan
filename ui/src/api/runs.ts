import { type QueryClient, queryOptions, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import type { Diagnostic, Forces, HistoryRecord, LogLine, Run, RunRegistry } from "./contracts";

/** A response the server refused: its status, and the reason it gave (FastAPI's `detail`). */
export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    url: string,
  ) {
    super(`${url}: HTTP ${status}${detail ? `: ${detail}` : ""}`);
  }
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : "";
    throw new HttpError(response.status, detail, url);
  }
  return (await response.json()) as T;
}

export const runsQuery = queryOptions({
  queryKey: ["runs"],
  queryFn: () => getJson<RunRegistry>("/api/runs"),
  staleTime: Number.POSITIVE_INFINITY,
});

export interface RunDetails {
  run: Run;
  case: CaseDocument;
  /** case.toml as stored in the run directory. */
  case_toml: string;
  mesh_info: MeshInfo;
  outputs: RunOutput[];
}

export interface CaseDocument {
  name: string;
  time: {
    kind: "steady" | "unsteady";
    max_iterations?: number;
    residual_orders?: number;
    steps?: number;
  };
  [key: string]: unknown;
}

export interface MeshInfo {
  source: { path: string; format: string; sha256: string };
  counts: { cells: number; faces: number; nodes: number };
  boundary_zones: { name: string; faces: number }[];
}

export interface RunOutput {
  name: string;
  bytes: number;
  /** Seconds since the epoch. */
  modified: number;
}

export interface Plan {
  ok: boolean;
  boundaries: {
    zone: string;
    faces: number;
    mesh_type: string | null;
    condition: { kind: string } | null;
    provenance: "code" | "mesh" | "default" | null;
  }[];
  cell_zones: {
    zone: string;
    cells: number;
    medium: { kind: string } | null;
    provenance: string | null;
  }[];
  memory: { bytes_total: number; bytes_per_gpu: number; gpus: number; equations: number };
  diagnostics: Diagnostic[];
}

export const planQuery = (runId: string) =>
  queryOptions({
    queryKey: ["plan", runId],
    queryFn: () => getJson<Plan>(`/api/runs/${runId}/plan`),
    staleTime: Number.POSITIVE_INFINITY,
  });

export const runQuery = (runId: string) =>
  queryOptions({
    queryKey: ["run", runId],
    queryFn: () => getJson<RunDetails>(`/api/runs/${runId}`),
    // A run the server does not have stays missing: no retry before saying so.
    retry: (failures, error) =>
      !(error instanceof HttpError && error.status === 404) && failures < 3,
  });

const EMPTY: RunRegistry = { runs: [], rejected: [], summaries: {}, solutions: [] };

function upsert(registry: RunRegistry | undefined, run: Run): RunRegistry {
  const runs = (registry?.runs ?? []).filter((r) => r.run_id !== run.run_id);
  runs.push(run);
  runs.sort((a, b) => b.started.localeCompare(a.started));
  return { ...EMPTY, ...registry, runs };
}

/** Keeps the run registry query live from the server's event stream. */
export function subscribeRegistry(client: QueryClient): () => void {
  const source = new EventSource("/api/runs/events");
  const update = (change: (old: RunRegistry | undefined) => RunRegistry | undefined) => {
    // A /api/runs response still in flight is older than this event and must not land after it.
    void client.cancelQueries({ queryKey: runsQuery.queryKey });
    client.setQueryData(runsQuery.queryKey, change);
  };
  source.addEventListener("runs", (e) => {
    const registry = JSON.parse(e.data) as RunRegistry;
    update(() => registry);
  });
  source.addEventListener("run", (e) => {
    const run = JSON.parse(e.data) as Run;
    update((old) => upsert(old, run));
    client.setQueryData(runQuery(run.run_id).queryKey, (old) => (old ? { ...old, run } : old));
  });
  source.addEventListener("removed", (e) => {
    const runId = JSON.parse(e.data) as string;
    update((old) => (old ? { ...old, runs: old.runs.filter((r) => r.run_id !== runId) } : old));
  });
  source.addEventListener("rejected", (e) => {
    const rejected = JSON.parse(e.data) as RunRegistry["rejected"];
    update((old) => ({ ...EMPTY, ...old, rejected }));
  });
  source.addEventListener("summaries", (e) => {
    const summaries = JSON.parse(e.data) as RunRegistry["summaries"];
    update((old) => ({ ...EMPTY, ...old, summaries }));
  });
  source.addEventListener("solutions", (e) => {
    const solutions = JSON.parse(e.data) as RunRegistry["solutions"];
    update((old) => ({ ...EMPTY, ...old, solutions }));
  });
  return () => source.close();
}

export function useRegistrySubscription(): void {
  const client = useQueryClient();
  useEffect(() => subscribeRegistry(client), [client]);
}

export interface ForceColumns {
  iteration: number[];
  cl: number[];
  cd: number[];
}

/** Columnar history: `iteration`, one column per residual, and coefficients per forces monitor. */
export interface HistoryColumns {
  iteration: number[];
  residuals: Map<string, number[]>;
  forces: Map<string, ForceColumns>;
  /** Last recorded value of every monitor, by name. */
  monitors: Map<string, Forces>;
  /** Orders of magnitude the slowest residual has dropped since the first record. */
  orders: number[];
  /** Identically zero equations that rose above their floor: no longer exempt. */
  woken: Set<string>;
  /** Identically zero equations still exempt from the convergence rule. */
  exempt: Set<string>;
  version: number;
}

export function emptyColumns(): HistoryColumns {
  return {
    iteration: [],
    residuals: new Map(),
    forces: new Map(),
    monitors: new Map(),
    orders: [],
    woken: new Set(),
    exempt: new Set(),
    version: 0,
  };
}

function column<T>(map: Map<string, T>, name: string, create: () => T): T {
  let value = map.get(name);
  if (value === undefined) {
    value = create();
    map.set(name, value);
  }
  return value;
}

// An equation below FLOOR times the largest first residual is identically zero
// (case-schema.md#time); the SDK's autan.restart.FLOOR.
const FLOOR = 1e-12;

/** A drop as a display shows it: one with every equation converged reads as the target. */
export function shownOrders(orders: number | undefined, target: number | undefined): number {
  if (orders === Number.POSITIVE_INFINITY) return target ?? Number.NaN;
  return orders ?? Number.NaN;
}

type Rule = Pick<Run, "start" | "convergence"> | null | undefined;

/** The inputs of the convergence rule a run's drop depends on, as a comparable key. */
export function ruleKey(run: Rule): string {
  return JSON.stringify([run?.start?.residual_reference, run?.start?.woken, run?.convergence]);
}

/** `orders` from row `from` on, by the steady convergence rule of case-schema.md#time, as the
 * server's residual_orders: the drop of the slowest equation not yet converged, +Infinity once
 * every equation is, NaN when one cannot be compared. */
function computeOrders(columns: HistoryColumns, run: Rule, from: number): void {
  const reference = run?.start?.residual_reference ?? null;
  const floors = run?.convergence?.floor ?? {};
  if (from === 0) {
    columns.orders.length = 0;
    columns.woken = new Set(run?.start?.woken ?? []);
  }
  const series = [...columns.residuals];
  const firsts: Record<string, number> =
    reference ?? Object.fromEntries(series.map(([name, values]) => [name, values[0] ?? 0]));
  const floor = FLOOR * Math.max(0, ...Object.values(firsts).filter((v) => v > 0 && v < Infinity));
  for (let k = from; k < columns.iteration.length; k++) {
    let slowest = Number.POSITIVE_INFINITY;
    let comparable = series.length > 0;
    for (const [name, values] of series) {
      const value = values[k];
      const first = firsts[name];
      if (value === undefined || first === undefined) {
        comparable = false;
        continue;
      }
      if (!columns.woken.has(name) && first >= 0 && first < floor) {
        if (value >= 0 && value <= floor) continue;
        columns.woken.add(name);
      }
      const limit = floors[name];
      if (limit !== undefined && value >= 0 && value <= limit) continue;
      if (value === 0 && first >= 0) continue;
      if (first > 0 && value > 0 && first < Infinity && value < Infinity)
        slowest = Math.min(slowest, Math.log10(first) - Math.log10(value));
      else comparable = false;
    }
    columns.orders.push(comparable ? slowest : Number.NaN);
  }
  columns.exempt = new Set(
    series
      .map(([name]) => name)
      .filter((name) => {
        const first = firsts[name];
        return first !== undefined && first >= 0 && first < floor && !columns.woken.has(name);
      }),
  );
}

/** Appends in place, so a long history is never copied; returns a new handle for React.
 * `run` gives the convergence rule's inputs (its `start` and `convergence`). */
export function appendHistory(
  columns: HistoryColumns,
  records: HistoryRecord[],
  run?: Rule,
): HistoryColumns {
  const from = columns.iteration.length;
  for (const record of records) {
    columns.iteration.push(record.iteration);
    for (const [name, value] of Object.entries(record.residuals)) {
      column(columns.residuals, name, () => []).push(value);
    }
    for (const [name, forces] of Object.entries(record.monitors ?? {})) {
      columns.monitors.set(name, forces);
      if (forces.cl === undefined || forces.cd === undefined) continue;
      const series = column(columns.forces, name, () => ({ iteration: [], cl: [], cd: [] }));
      series.iteration.push(record.iteration);
      series.cl.push(forces.cl);
      series.cd.push(forces.cd);
    }
  }
  computeOrders(columns, run, from);
  return { ...columns, version: columns.version + 1 };
}

/** Recomputes the whole drop when the rule's inputs of `run` changed (`floors` arrive once the
 * runtime has built its physics, results.md); the same handle otherwise. */
export function reapplyRule(columns: HistoryColumns, before: Rule, run: Rule): HistoryColumns {
  if (ruleKey(before) === ruleKey(run)) return columns;
  computeOrders(columns, run, 0);
  return { ...columns, version: columns.version + 1 };
}

export function soleForces(history: HistoryColumns): ForceColumns | null {
  return history.forces.size === 1 ? ([...history.forces.values()][0] ?? null) : null;
}

/** log.jsonl lines, appended in place; `version` changes on every append. */
export interface LogLines {
  lines: LogLine[];
  version: number;
}

export interface LiveRun {
  run: Run | null;
  history: HistoryColumns;
  log: LogLines;
  connected: boolean;
  invalid: string | null;
}

/** run.json and history.jsonl of one run, streamed: appended lines arrive as they are written. */
export function useLiveRun(runId: string): LiveRun {
  const [state, setState] = useState<LiveRun>(() => ({
    run: null,
    history: emptyColumns(),
    log: { lines: [], version: 0 },
    connected: false,
    invalid: null,
  }));
  useEffect(() => {
    let history = emptyColumns();
    let run: Run | null = null;
    const log: LogLines = { lines: [], version: 0 };
    setState({ run: null, history, log, connected: false, invalid: null });
    const source = new EventSource(`/api/runs/${runId}/history/events`);
    source.onopen = () => setState((s) => ({ ...s, connected: true }));
    source.onerror = () => setState((s) => ({ ...s, connected: false }));
    source.addEventListener("run", (e) => {
      const next = JSON.parse(e.data) as Run;
      history = reapplyRule(history, run, next);
      run = next;
      const columns = history;
      setState((s) => ({ ...s, run: next, history: columns }));
    });
    source.addEventListener("history", (e) => {
      const records = JSON.parse(e.data) as HistoryRecord[];
      if (!records.length) return;
      history = appendHistory(history, records, run);
      const next = history;
      setState((s) => ({ ...s, history: next }));
    });
    source.addEventListener("log", (e) => {
      const lines = JSON.parse(e.data) as LogLine[];
      if (!lines.length) return;
      // One by one: spreading a large first batch into push() exceeds the argument limit.
      for (const line of lines) log.lines.push(line);
      const next = { lines: log.lines, version: log.version + lines.length };
      log.version = next.version;
      setState((s) => ({ ...s, log: next }));
    });
    source.addEventListener("invalid", (e) => {
      const reason = JSON.parse(e.data) as string;
      source.close();
      setState((s) => ({ ...s, connected: false, invalid: reason }));
    });
    return () => source.close();
  }, [runId]);
  return state;
}

/** Live history of several runs at once over one connection, keyed by run id. */
export function useRunsHistory(runIds: readonly string[]): Map<string, LiveRun> {
  const [state, setState] = useState<Map<string, LiveRun>>(() => new Map());
  const key = runIds.join(",");
  // biome-ignore lint/correctness/useExhaustiveDependencies: `key` stands for the id list
  useEffect(() => {
    setState(new Map());
    if (!runIds.length) return;
    const histories = new Map(runIds.map((id) => [id, emptyColumns()]));
    const set = (runId: string, change: (old: LiveRun) => LiveRun) =>
      setState((s) => {
        const next = new Map(s);
        const old = s.get(runId) ?? {
          run: null,
          history: histories.get(runId) ?? emptyColumns(),
          log: { lines: [], version: 0 },
          connected: true,
          invalid: null,
        };
        next.set(runId, change(old));
        return next;
      });
    const starts = new Map<string, Run>();
    const query = runIds.map((id) => `run=${encodeURIComponent(id)}`).join("&");
    const source = new EventSource(`/api/runs/history/events?${query}`);
    source.addEventListener("run", (e) => {
      const { run_id, data } = JSON.parse(e.data) as { run_id: string; data: Run };
      const history = histories.get(run_id);
      const columns = history && reapplyRule(history, starts.get(run_id), data);
      if (columns) histories.set(run_id, columns);
      starts.set(run_id, data);
      set(run_id, (old) => ({ ...old, run: data, ...(columns ? { history: columns } : {}) }));
    });
    source.addEventListener("history", (e) => {
      const { run_id, data } = JSON.parse(e.data) as { run_id: string; data: HistoryRecord[] };
      const history = histories.get(run_id);
      if (!history || !data.length) return;
      const next = appendHistory(history, data, starts.get(run_id));
      histories.set(run_id, next);
      set(run_id, (old) => ({ ...old, history: next }));
    });
    source.addEventListener("invalid", (e) => {
      const { run_id, data } = JSON.parse(e.data) as { run_id: string | null; data?: string };
      if (run_id) set(run_id, (old) => ({ ...old, invalid: data ?? "invalid" }));
    });
    return () => source.close();
  }, [key]);
  return state;
}
