import { type QueryClient, queryOptions, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import type { HistoryRecord, Run, RunRegistry } from "./contracts";

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return (await response.json()) as T;
}

export const runsQuery = queryOptions({
  queryKey: ["runs"],
  queryFn: () => getJson<RunRegistry>("/api/runs"),
  staleTime: Number.POSITIVE_INFINITY,
});

export interface RunDetails {
  run: Run;
  case: Record<string, unknown>;
  mesh_info: Record<string, unknown>;
}

export const runQuery = (runId: string) =>
  queryOptions({
    queryKey: ["run", runId],
    queryFn: () => getJson<RunDetails>(`/api/runs/${runId}`),
  });

function upsert(registry: RunRegistry | undefined, run: Run): RunRegistry {
  const runs = (registry?.runs ?? []).filter((r) => r.run_id !== run.run_id);
  runs.push(run);
  runs.sort((a, b) => b.started.localeCompare(a.started));
  return { runs, rejected: registry?.rejected ?? [], progress: registry?.progress ?? {} };
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
    update((old) => ({ runs: old?.runs ?? [], rejected, progress: old?.progress ?? {} }));
  });
  source.addEventListener("progress", (e) => {
    const progress = JSON.parse(e.data) as RunRegistry["progress"];
    update((old) => ({ runs: old?.runs ?? [], rejected: old?.rejected ?? [], progress }));
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
  version: number;
}

function emptyColumns(): HistoryColumns {
  return { iteration: [], residuals: new Map(), forces: new Map(), version: 0 };
}

function column<T>(map: Map<string, T>, name: string, create: () => T): T {
  let value = map.get(name);
  if (value === undefined) {
    value = create();
    map.set(name, value);
  }
  return value;
}

/** Appends in place, so a long history is never copied; returns a new handle for React. */
export function appendHistory(columns: HistoryColumns, records: HistoryRecord[]): HistoryColumns {
  for (const record of records) {
    columns.iteration.push(record.iteration);
    for (const [name, value] of Object.entries(record.residuals)) {
      column(columns.residuals, name, () => []).push(value);
    }
    for (const [name, forces] of Object.entries(record.monitors ?? {})) {
      if (forces.cl === undefined || forces.cd === undefined) continue;
      const series = column(columns.forces, name, () => ({ iteration: [], cl: [], cd: [] }));
      series.iteration.push(record.iteration);
      series.cl.push(forces.cl);
      series.cd.push(forces.cd);
    }
  }
  return { ...columns, version: columns.version + 1 };
}

/** The coefficients of the case's forces monitor, when exactly one reports them. */
export function soleForces(history: HistoryColumns): ForceColumns | null {
  return history.forces.size === 1 ? ([...history.forces.values()][0] ?? null) : null;
}

export interface LiveRun {
  run: Run | null;
  history: HistoryColumns;
  connected: boolean;
  invalid: string | null;
}

/** run.json and history.jsonl of one run, streamed: appended lines arrive as they are written. */
export function useLiveRun(runId: string): LiveRun {
  const [state, setState] = useState<LiveRun>(() => ({
    run: null,
    history: emptyColumns(),
    connected: false,
    invalid: null,
  }));
  useEffect(() => {
    let history = emptyColumns();
    setState({ run: null, history, connected: false, invalid: null });
    const source = new EventSource(`/api/runs/${runId}/history/events`);
    source.onopen = () => setState((s) => ({ ...s, connected: true }));
    source.onerror = () => setState((s) => ({ ...s, connected: false }));
    source.addEventListener("run", (e) => {
      const run = JSON.parse(e.data) as Run;
      setState((s) => ({ ...s, run }));
    });
    source.addEventListener("history", (e) => {
      const records = JSON.parse(e.data) as HistoryRecord[];
      if (!records.length) return;
      history = appendHistory(history, records);
      const next = history;
      setState((s) => ({ ...s, history: next }));
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
