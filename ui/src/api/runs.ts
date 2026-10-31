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
  return { runs, rejected: registry?.rejected ?? [] };
}

/** Keeps the run registry query live from the server's event stream. */
export function subscribeRegistry(client: QueryClient): () => void {
  const source = new EventSource("/api/runs/events");
  source.addEventListener("runs", (e) => {
    client.setQueryData(runsQuery.queryKey, JSON.parse(e.data) as RunRegistry);
  });
  source.addEventListener("run", (e) => {
    const run = JSON.parse(e.data) as Run;
    client.setQueryData(runsQuery.queryKey, (old) => upsert(old, run));
    client.setQueryData(runQuery(run.run_id).queryKey, (old) => (old ? { ...old, run } : old));
  });
  source.addEventListener("removed", (e) => {
    const runId = JSON.parse(e.data) as string;
    client.setQueryData(runsQuery.queryKey, (old) =>
      old ? { ...old, runs: old.runs.filter((r) => r.run_id !== runId) } : old,
    );
  });
  return () => source.close();
}

export function useRegistrySubscription(): void {
  const client = useQueryClient();
  useEffect(() => subscribeRegistry(client), [client]);
}

/** Columnar history: `iteration` plus one column per residual and per force coefficient. */
export interface HistoryColumns {
  iteration: number[];
  residuals: Map<string, number[]>;
  forces: { iteration: number[]; cl: number[]; cd: number[] };
  version: number;
}

function emptyColumns(): HistoryColumns {
  return {
    iteration: [],
    residuals: new Map(),
    forces: { iteration: [], cl: [], cd: [] },
    version: 0,
  };
}

/** Appends in place, so a long history is never copied; returns a new handle for React. */
export function appendHistory(columns: HistoryColumns, records: HistoryRecord[]): HistoryColumns {
  for (const record of records) {
    columns.iteration.push(record.iteration);
    for (const [name, value] of Object.entries(record.residuals)) {
      let column = columns.residuals.get(name);
      if (!column) {
        column = [];
        columns.residuals.set(name, column);
      }
      column.push(value);
    }
    const forces = record.monitors?.forces;
    if (forces?.cl !== undefined && forces.cd !== undefined) {
      columns.forces.iteration.push(record.iteration);
      columns.forces.cl.push(forces.cl);
      columns.forces.cd.push(forces.cd);
    }
  }
  return { ...columns, version: columns.version + 1 };
}

export interface LiveRun {
  run: Run | null;
  history: HistoryColumns;
  connected: boolean;
}

/** run.json and history.jsonl of one run, streamed: appended lines arrive as they are written. */
export function useLiveRun(runId: string): LiveRun {
  const [state, setState] = useState<LiveRun>(() => ({
    run: null,
    history: emptyColumns(),
    connected: false,
  }));
  useEffect(() => {
    let history = emptyColumns();
    setState({ run: null, history, connected: false });
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
    return () => source.close();
  }, [runId]);
  return state;
}
