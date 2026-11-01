import { useVirtualizer } from "@tanstack/react-virtual";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Diagnostic } from "../../api/contracts";
import type { HistoryColumns } from "../../api/runs";
import { cn } from "../ui/cn";
import { LiveToggle } from "../ui/LiveToggle";

const ROW = 32;
const BUCKETS = 120;

interface Line {
  index: number;
  iteration: number;
  orders: number;
  slowest: string;
  cl?: number | undefined;
  cd?: number | undefined;
  diagnostics: Diagnostic[];
}

function slowestEquation(history: HistoryColumns, k: number): string {
  let name = "";
  let worst = Number.POSITIVE_INFINITY;
  for (const [equation, values] of history.residuals) {
    const first = values[0] ?? 0;
    const value = values[k] ?? 0;
    if (first > 0 && value > 0) {
      const drop = Math.log10(first) - Math.log10(value);
      if (drop < worst) [worst, name] = [drop, equation];
    }
  }
  return name;
}

/** history.jsonl as a log: one line per recorded iteration, diagnostics at their iteration, a
 * histogram of the residual drop above it. Live follows the end; scrolling up pauses it. */
export function LogView({
  history,
  diagnostics,
  running,
  height = 480,
}: {
  history: HistoryColumns;
  diagnostics: Diagnostic[];
  running: boolean;
  height?: number;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  // Set while the view scrolls itself to follow new lines, so that only a reader's own scroll
  // (wheel, keys, scrollbar) pauses Live.
  const following = useRef(false);
  const [live, setLive] = useState(true);
  const count = history.iteration.length;
  const forces = history.forces.size === 1 ? [...history.forces.values()][0] : undefined;
  // A diagnostic sits on the last recorded line at or before its iteration: a divergence is
  // raised at an iteration that never reaches history.jsonl.
  // biome-ignore lint/correctness/useExhaustiveDependencies: columns grow in place, see forceAt
  const byIteration = useMemo(() => {
    const map = new Map<number, Diagnostic[]>();
    const iterations = history.iteration;
    for (const d of diagnostics) {
      const at = d.data.iteration;
      if (typeof at !== "number" || !iterations.length) continue;
      let lo = 0;
      let hi = iterations.length - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if ((iterations[mid] ?? 0) <= at) lo = mid;
        else hi = mid - 1;
      }
      const line = iterations[lo] ?? at;
      map.set(line, [...(map.get(line) ?? []), d]);
    }
    return map;
  }, [diagnostics, history.version]);
  // History columns grow in place: `history.version` is what changes on every append.
  // biome-ignore lint/correctness/useExhaustiveDependencies: see above
  const forceAt = useMemo(() => {
    const map = new Map<number, number>();
    for (const [k, it] of (forces?.iteration ?? []).entries()) map.set(it, k);
    return map;
  }, [forces, history.version]);

  const line = (k: number): Line => {
    const iteration = history.iteration[k] ?? 0;
    const f = forceAt.get(iteration);
    return {
      index: k,
      iteration,
      orders: history.orders[k] ?? Number.NaN,
      slowest: slowestEquation(history, k),
      cl: f === undefined ? undefined : forces?.cl[f],
      cd: f === undefined ? undefined : forces?.cd[f],
      diagnostics: byIteration.get(iteration) ?? [],
    };
  };

  const virtualizer = useVirtualizer({
    count,
    getScrollElement: () => scroller.current,
    estimateSize: () => ROW,
    overscan: 12,
  });

  // biome-ignore lint/correctness/useExhaustiveDependencies: follow on every append while live
  useEffect(() => {
    if (!live || !count) return;
    following.current = true;
    virtualizer.scrollToIndex(count - 1, { align: "end" });
  }, [history.version, live, count]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea, [contenteditable]")) return;
      if (e.key === "End") setLive(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: columns grow in place, see forceAt
  const buckets = useMemo(() => {
    const n = Math.min(BUCKETS, count);
    return Array.from({ length: n }, (_, b) => {
      const first = Math.floor((b * count) / n);
      const last = Math.max(first, Math.floor(((b + 1) * count) / n) - 1);
      let flagged = false;
      for (let k = first; k <= last; k++) {
        if (byIteration.has(history.iteration[k] ?? -1)) flagged = true;
      }
      return { first, last, orders: history.orders[last] ?? 0, flagged };
    });
  }, [history.version, count, byIteration]);
  const peak = Math.max(1, ...buckets.map((b) => (Number.isFinite(b.orders) ? b.orders : 0)));

  return (
    <div className="flex flex-col overflow-hidden rounded-lg border border-line bg-bg-0">
      <div className="flex items-center gap-3 border-line border-b px-4 py-3">
        <span className="font-medium">Log</span>
        <span className="text-fg-3 text-sm tabular-nums">
          {count.toLocaleString("en")} lines
          {diagnostics.length > 0 && ` · ${diagnostics.length} diagnostics`}
        </span>
        <div className="ml-auto flex items-center gap-2">
          {running && <LiveToggle live={live} onChange={setLive} />}
        </div>
      </div>
      <div
        className="flex h-16 items-end gap-px border-line border-b px-4 pt-3 pb-1"
        aria-label="Residual drop along the run"
        role="img"
      >
        {buckets.map((b) => (
          <button
            type="button"
            key={b.first}
            tabIndex={-1}
            aria-hidden
            title={`iterations ${history.iteration[b.first]}–${history.iteration[b.last]}`}
            onClick={() => {
              setLive(false);
              virtualizer.scrollToIndex(b.first, { align: "start" });
            }}
            className={cn(
              "min-w-px flex-1 rounded-t-[1px] transition-opacity hover:opacity-70",
              b.flagged ? "bg-danger" : "bg-fg-3/40",
            )}
            style={{ height: `${Math.max(4, (Math.max(0, b.orders) / peak) * 100)}%` }}
          />
        ))}
      </div>
      <div className="grid grid-cols-[96px_96px_140px_96px_96px_1fr] gap-4 border-line border-b bg-bg-1 px-4 py-2 text-fg-3 text-sm">
        <span className="text-right">Iteration</span>
        <span className="text-right">Drop</span>
        <span>Slowest</span>
        <span className="text-right">CL</span>
        <span className="text-right">CD</span>
        <span>Message</span>
      </div>
      <div
        ref={scroller}
        style={{ height }}
        className="overflow-auto font-mono text-sm"
        data-testid="log"
        onScroll={(e) => {
          const el = e.currentTarget;
          const atEnd = el.scrollHeight - el.scrollTop - el.clientHeight < ROW;
          if (following.current) following.current = atEnd ? false : following.current;
          else if (!atEnd) setLive(false);
        }}
      >
        <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
          {virtualizer.getVirtualItems().map((item) => {
            const l = line(item.index);
            return (
              <div
                key={item.key}
                className={cn(
                  "absolute inset-x-0 grid grid-cols-[96px_96px_140px_96px_96px_1fr] items-center gap-4 px-4 hover:bg-bg-1",
                  l.diagnostics.length > 0 && "bg-danger-soft",
                )}
                style={{ height: ROW, transform: `translateY(${item.start}px)` }}
              >
                <span className="text-right tabular-nums">{l.iteration}</span>
                <span className="text-right text-fg-2 tabular-nums">
                  {Number.isFinite(l.orders) ? l.orders.toFixed(2) : "—"}
                </span>
                <span className="truncate text-fg-3">{l.slowest || "—"}</span>
                <span className="text-right tabular-nums">{l.cl?.toFixed(4) ?? ""}</span>
                <span className="text-right tabular-nums">{l.cd?.toFixed(5) ?? ""}</span>
                <span className="truncate font-sans">
                  {l.diagnostics.map((d) => (
                    <span key={d.code} className="text-danger">
                      <span className="font-mono">{d.code}</span> {d.message}
                    </span>
                  ))}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
