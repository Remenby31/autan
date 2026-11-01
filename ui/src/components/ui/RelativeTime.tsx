import { PreviewCard } from "@base-ui/react/preview-card";
import { useSyncExternalStore } from "react";

const listeners = new Set<() => void>();
let now = Date.now();
setInterval(() => {
  now = Date.now();
  for (const listener of listeners) listener();
}, 15_000);

function useNow(): number {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => now,
  );
}

const UNITS: [Intl.RelativeTimeFormatUnit, number][] = [
  ["year", 31_536_000],
  ["month", 2_592_000],
  ["day", 86_400],
  ["hour", 3_600],
  ["minute", 60],
];
const relative = new Intl.RelativeTimeFormat("en", { numeric: "auto", style: "narrow" });

export function formatRelative(date: Date, reference: number): string {
  const seconds = Math.round((date.getTime() - reference) / 1000);
  for (const [unit, size] of UNITS) {
    if (Math.abs(seconds) >= size) return relative.format(Math.trunc(seconds / size), unit);
  }
  return "just now";
}

const ABSOLUTE = {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
} as const;

/** "3m ago", kept current; hovering or focusing shows the instant in UTC and local time.
 * Inside a link (`card={false}`), the absolute UTC time is the native title instead, since a
 * control cannot nest in a link. */
export function RelativeTime({ iso, card = true }: { iso: string; card?: boolean }) {
  const reference = useNow();
  const date = new Date(iso);
  const relativeText = formatRelative(date, reference);
  const utc = date.toLocaleString("en-GB", { ...ABSOLUTE, timeZone: "UTC" });
  const local = date.toLocaleString("en-GB", { ...ABSOLUTE, timeZoneName: "short" });
  if (!card) {
    return (
      <time dateTime={iso} title={`${utc} UTC`} className="whitespace-nowrap tabular-nums">
        {relativeText}
      </time>
    );
  }
  return (
    <PreviewCard.Root>
      <PreviewCard.Trigger
        render={<button type="button" aria-label={`${relativeText}, ${utc} UTC`} />}
        delay={400}
        className="cursor-default whitespace-nowrap rounded-xs tabular-nums"
      >
        <time dateTime={iso}>{relativeText}</time>
      </PreviewCard.Trigger>
      <PreviewCard.Portal>
        <PreviewCard.Positioner sideOffset={6} className="z-50">
          <PreviewCard.Popup className="grid grid-cols-[auto_auto] gap-x-4 gap-y-1 rounded-sm bg-bg-0 p-3 text-sm shadow-tooltip transition-opacity duration-[var(--dur-fast)] data-[ending-style]:opacity-0 data-[starting-style]:opacity-0">
            <span className="text-fg-3">UTC</span>
            <span className="font-mono">{utc}</span>
            <span className="text-fg-3">Local</span>
            <span className="font-mono">{local}</span>
          </PreviewCard.Popup>
        </PreviewCard.Positioner>
      </PreviewCard.Portal>
    </PreviewCard.Root>
  );
}
