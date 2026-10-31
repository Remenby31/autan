import { Toggle } from "@base-ui/react/toggle";
import { ToggleGroup } from "@base-ui/react/toggle-group";
import { useQuery } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import { RUN_STATUSES, type Run, type RunStatus } from "../api/contracts";
import { runsQuery } from "../api/runs";
import { StateGlyph } from "../components/StateGlyph";

interface Search {
  status?: RunStatus[] | undefined;
  q?: string | undefined;
}

export const Route = createFileRoute("/")({
  validateSearch: (raw: Record<string, unknown>): Search => {
    const search: Search = {};
    if (Array.isArray(raw.status)) {
      search.status = raw.status.filter((s): s is RunStatus =>
        RUN_STATUSES.includes(s as RunStatus),
      );
    }
    if (typeof raw.q === "string" && raw.q) search.q = raw.q;
    return search;
  },
  component: Registry,
});

function matches(run: Run, search: Search): boolean {
  if (search.status?.length && !search.status.includes(run.status)) return false;
  if (!search.q) return true;
  const q = search.q.toLowerCase();
  return [run.case.name, run.run_id, run.precision, run.solver.backend].some((v) =>
    v.toLowerCase().includes(q),
  );
}

function Registry() {
  const search = Route.useSearch();
  const navigate = useNavigate({ from: Route.fullPath });
  const { data, error } = useQuery(runsQuery);
  const [cursor, setCursor] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const runs = useMemo(() => (data?.runs ?? []).filter((r) => matches(r, search)), [data, search]);
  const counts = useMemo(() => {
    const c = new Map<RunStatus, number>();
    for (const run of data?.runs ?? []) c.set(run.status, (c.get(run.status) ?? 0) + 1);
    return c;
  }, [data]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) {
        if (e.key === "Escape") input.current?.blur();
        return;
      }
      if (e.key === "/") {
        e.preventDefault();
        input.current?.focus();
      } else if (e.key === "j" || e.key === "ArrowDown") {
        setCursor((c) => Math.min(c + 1, runs.length - 1));
      } else if (e.key === "k" && !e.metaKey && !e.ctrlKey) {
        setCursor((c) => Math.max(c - 1, 0));
      } else if (e.key === "ArrowUp") {
        setCursor((c) => Math.max(c - 1, 0));
      } else if (e.key === "Enter" && runs[cursor]) {
        void navigate({ to: "/runs/$runId", params: { runId: runs[cursor].run_id } });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [runs, cursor, navigate]);

  return (
    <div className="mx-auto flex h-full max-w-6xl flex-col gap-4 overflow-auto px-4 py-6">
      <div className="flex flex-wrap items-center gap-3">
        <input
          ref={input}
          value={search.q ?? ""}
          onChange={(e) =>
            void navigate({
              search: (s: Search) => ({ ...s, q: e.target.value || undefined }),
              replace: true,
            })
          }
          placeholder="Filter  /"
          className="w-64 rounded-md border border-line bg-bg-1 px-3 py-1.5 outline-none focus:border-accent"
        />
        <ToggleGroup
          multiple
          value={search.status ?? []}
          onValueChange={(value) =>
            void navigate({
              search: (s: Search) => ({
                ...s,
                status: value.length ? (value as RunStatus[]) : undefined,
              }),
              replace: true,
            })
          }
          className="flex gap-1"
        >
          {RUN_STATUSES.filter((s) => counts.has(s)).map((status) => (
            <Toggle
              key={status}
              value={status}
              className="flex items-center gap-2 rounded-md border border-line px-2.5 py-1 text-fg-3 text-sm data-[pressed]:border-accent data-[pressed]:text-fg"
            >
              <StateGlyph status={status} />
              {status}
              <span className="font-mono text-xs">{counts.get(status)}</span>
            </Toggle>
          ))}
        </ToggleGroup>
      </div>
      {error && <p className="text-danger">{String(error)}</p>}
      {data?.rejected.length ? (
        <p className="text-warning text-sm" data-testid="rejected">
          {data.rejected.length} unreadable run directories:{" "}
          {data.rejected.map((r) => `${r.dir} (${r.reason})`).join(", ")}
        </p>
      ) : null}
      <table className="w-full border-collapse text-sm" data-testid="runs">
        <thead className="text-left font-mono text-fg-3 text-xs uppercase tracking-wider">
          <tr>
            <th className="py-2 font-normal">Case</th>
            <th className="font-normal">Status</th>
            <th className="text-right font-normal">Iter.</th>
            <th className="text-right font-normal">Precision</th>
            <th className="text-right font-normal">Devices</th>
            <th className="text-right font-normal">Started</th>
          </tr>
        </thead>
        <tbody>
          <AnimatePresence initial={false}>
            {runs.map((run, i) => (
              <motion.tr
                key={run.run_id}
                layout
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                aria-selected={i === cursor}
                onClick={() => void navigate({ to: "/runs/$runId", params: { runId: run.run_id } })}
                onMouseEnter={() => setCursor(i)}
                className="cursor-pointer border-line border-t aria-selected:bg-accent-soft"
              >
                <td className="py-2">
                  <span className="font-semibold">{run.case.name}</span>
                  <span className="ml-2 font-mono text-fg-3 text-xs">{run.run_id.slice(-8)}</span>
                </td>
                <td>
                  <span className="flex items-center gap-2" data-status={run.status}>
                    <StateGlyph status={run.status} />
                    {run.status}
                    {run.diagnostics[0] && (
                      <span className="font-mono text-danger text-xs">
                        {run.diagnostics[0].code}
                      </span>
                    )}
                  </span>
                </td>
                <td className="text-right font-mono tabular-nums">{run.iterations}</td>
                <td className="text-right font-mono">{run.precision}</td>
                <td className="text-right font-mono">
                  {run.devices.count}× {run.devices.kind}
                  {run.solver.backend === "fake" && <span className="ml-1 text-warning">fake</span>}
                </td>
                <td className="text-right font-mono text-fg-3 tabular-nums">
                  {run.started.replace("T", " ").slice(0, 16)}
                </td>
              </motion.tr>
            ))}
          </AnimatePresence>
        </tbody>
      </table>
      {data && !data.runs.length && (
        <p className="text-fg-3">No runs yet. `autan run` writes them here.</p>
      )}
    </div>
  );
}
