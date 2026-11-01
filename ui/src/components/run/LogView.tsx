import { useVirtualizer } from "@tanstack/react-virtual";
import { useEffect, useMemo, useRef, useState } from "react";
import type { LogLine } from "../../api/contracts";
import type { HistoryColumns, LogLines } from "../../api/runs";
import { cn } from "../ui/cn";
import { LiveToggle } from "../ui/LiveToggle";
import { SearchInput } from "../ui/SearchInput";

const ROW = 32;
const BUCKETS = 120;
const LEVELS = ["info", "warning", "error"] as const;
type Level = (typeof LEVELS)[number];
const RANK: Record<Level, number> = { info: 0, warning: 1, error: 2 };
const DOT: Record<Level, string> = { info: "bg-fg-3", warning: "bg-warning", error: "bg-danger" };
const BAR: Record<Level, string> = {
  info: "bg-fg-3/40",
  warning: "bg-warning",
  error: "bg-danger",
};

const clock = new Intl.DateTimeFormat("en-GB", {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  fractionalSecondDigits: 3,
});

/** The solver log (log.jsonl): time, iteration, level, source, message and code, the code
 * linking to its documentation. Above it, the residual drop along the run (history.jsonl),
 * each bar coloured by the worst level logged in its iterations. Live follows the end;
 * a reader's scroll pauses it, End resumes. */
export function LogView({
  history,
  log,
  running,
  height = 480,
}: {
  history: HistoryColumns;
  log: LogLines;
  running: boolean;
  height?: number;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  // Set while the view scrolls itself to follow new lines, so that only a reader's own scroll
  // (wheel, keys, scrollbar) pauses Live.
  const following = useRef(false);
  const [live, setLive] = useState(true);
  const [levels, setLevels] = useState<ReadonlySet<Level>>(() => new Set(LEVELS));
  const [query, setQuery] = useState("");

  // Lines grow in place: `log.version` is what changes on every append.
  // biome-ignore lint/correctness/useExhaustiveDependencies: see above
  const counts = useMemo(() => {
    const c: Record<Level, number> = { info: 0, warning: 0, error: 0 };
    for (const line of log.lines) c[line.level] += 1;
    return c;
  }, [log.version]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: lines grow in place, see counts
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return log.lines.filter(
      (line) =>
        levels.has(line.level) &&
        (!q ||
          line.message.toLowerCase().includes(q) ||
          line.source.includes(q) ||
          (line.code ?? "").toLowerCase().includes(q) ||
          String(line.iteration ?? "") === q),
    );
  }, [log.version, levels, query]);
  const count = shown.length;

  const virtualizer = useVirtualizer({
    count,
    getScrollElement: () => scroller.current,
    estimateSize: () => ROW,
    overscan: 12,
  });

  // biome-ignore lint/correctness/useExhaustiveDependencies: follow on every append while live
  useEffect(() => {
    if (!live || !count) return;
    const el = scroller.current;
    // Armed only when the view will actually move, so that the next scroll event clears it.
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight >= ROW) following.current = true;
    virtualizer.scrollToIndex(count - 1, { align: "end" });
  }, [log.version, live, count]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea, [contenteditable]")) return;
      if (e.key === "End") setLive(true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // biome-ignore lint/correctness/useExhaustiveDependencies: columns and lines grow in place
  const buckets = useMemo(() => {
    const iterations = history.iteration;
    const n = Math.min(BUCKETS, iterations.length);
    const worst = new Map<number, Level>();
    for (const line of log.lines) {
      if (line.iteration === null || line.level === "info") continue;
      const seen = worst.get(line.iteration);
      if (!seen || RANK[line.level] > RANK[seen]) worst.set(line.iteration, line.level);
    }
    return Array.from({ length: n }, (_, b) => {
      const first = Math.floor((b * iterations.length) / n);
      const last = Math.max(first, Math.floor(((b + 1) * iterations.length) / n) - 1);
      const from = iterations[first] ?? 0;
      // The last bucket also takes lines logged after the last recorded iteration (a
      // divergence is raised at an iteration that never reaches history.jsonl).
      const to = b === n - 1 ? Number.POSITIVE_INFINITY : (iterations[last] ?? 0);
      let level: Level = "info";
      for (const [iteration, l] of worst) {
        if (iteration >= from && iteration <= to && RANK[l] > RANK[level]) level = l;
      }
      return { from, to, orders: history.orders[last] ?? 0, level };
    });
  }, [history.version, log.version]);
  const peak = Math.max(1, ...buckets.map((b) => (Number.isFinite(b.orders) ? b.orders : 0)));

  const toggle = (level: Level) =>
    setLevels((s) => {
      const next = new Set(s);
      if (next.has(level)) next.delete(level);
      else next.add(level);
      return next;
    });

  return (
    <div className="flex flex-col overflow-hidden rounded-lg border border-line bg-bg-0">
      <div className="flex flex-wrap items-center gap-2 border-line border-b px-4 py-3">
        <div className="w-72">
          <SearchInput
            value={query}
            onChange={setQuery}
            placeholder="Search the log…"
            label="Search the log"
          />
        </div>
        <fieldset className="m-0 flex items-center gap-1 border-0 p-0">
          <legend className="sr-only">Levels</legend>
          {LEVELS.map((level) => (
            <button
              type="button"
              key={level}
              aria-pressed={levels.has(level)}
              onClick={() => toggle(level)}
              className="flex h-8 items-center gap-2 rounded-full border border-line px-3 text-fg-3 text-sm transition-colors hover:text-fg aria-pressed:border-line-strong aria-pressed:text-fg"
            >
              <i className={cn("inline-block size-2 rounded-full", DOT[level])} />
              {level}
              <span className="font-mono text-xs tabular-nums">{counts[level]}</span>
            </button>
          ))}
        </fieldset>
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
            key={b.from}
            tabIndex={-1}
            aria-hidden
            title={`iterations ${b.from}–${Number.isFinite(b.to) ? b.to : "end"}`}
            onClick={() => {
              const index = shown.findIndex(
                (line) => line.iteration !== null && line.iteration >= b.from,
              );
              if (index < 0) return;
              setLive(false);
              virtualizer.scrollToIndex(index, { align: "start" });
            }}
            className={cn(
              "min-w-px flex-1 rounded-t-[1px] transition-opacity hover:opacity-70",
              BAR[b.level],
            )}
            style={{ height: `${Math.max(4, (Math.max(0, b.orders) / peak) * 100)}%` }}
          />
        ))}
      </div>
      <div className="grid grid-cols-[112px_80px_88px_96px_1fr_72px] gap-4 border-line border-b bg-bg-1 px-4 py-2 text-fg-3 text-sm">
        <span>Time</span>
        <span className="text-right">Iteration</span>
        <span>Level</span>
        <span>Source</span>
        <span>Message</span>
        <span>Code</span>
      </div>
      <div
        ref={scroller}
        style={{ height }}
        className="overflow-auto text-sm"
        data-testid="log"
        onScroll={(e) => {
          const el = e.currentTarget;
          const atEnd = el.scrollHeight - el.scrollTop - el.clientHeight < ROW;
          if (following.current) {
            if (atEnd) following.current = false;
          } else if (!atEnd) setLive(false);
        }}
      >
        {count === 0 ? (
          <p className="m-0 px-4 py-8 text-center text-fg-3">
            {log.lines.length ? "No line matches these filters" : "No log line yet"}
          </p>
        ) : (
          <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
            {virtualizer.getVirtualItems().map((item) => {
              const line = shown[item.index] as LogLine;
              return (
                <div
                  key={item.key}
                  className={cn(
                    "absolute inset-x-0 grid grid-cols-[112px_80px_88px_96px_1fr_72px] items-center gap-4 px-4 hover:bg-bg-1",
                    line.level === "error" && "bg-danger-soft",
                    line.level === "warning" && "bg-warning-soft",
                  )}
                  style={{ height: ROW, transform: `translateY(${item.start}px)` }}
                >
                  <time
                    dateTime={line.time}
                    title={line.time}
                    className="font-mono text-fg-3 tabular-nums"
                  >
                    {clock.format(new Date(line.time))}
                  </time>
                  <span className="text-right font-mono tabular-nums">{line.iteration ?? "—"}</span>
                  <span className="flex items-center gap-2">
                    <i className={cn("inline-block size-2 rounded-full", DOT[line.level])} />
                    {line.level}
                  </span>
                  <span className="truncate font-mono text-fg-2">{line.source}</span>
                  <span className="truncate" title={line.message}>
                    {line.message}
                  </span>
                  <span className="font-mono">
                    {line.code &&
                      (line.doc ? (
                        <a
                          href={line.doc}
                          target="_blank"
                          rel="noreferrer"
                          className={cn(
                            "hover:underline",
                            line.level === "error"
                              ? "text-danger"
                              : line.level === "warning"
                                ? "text-warning"
                                : "text-accent",
                          )}
                        >
                          {line.code}
                        </a>
                      ) : (
                        line.code
                      ))}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
