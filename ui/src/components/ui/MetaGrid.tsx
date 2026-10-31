import type { ReactNode } from "react";

/** Facts about one thing in a grid: a muted label above each value. */
export function MetaGrid({ items }: { items: { label: string; value: ReactNode }[] }) {
  return (
    <dl className="m-0 grid grid-cols-[repeat(auto-fill,minmax(170px,1fr))] gap-x-6 gap-y-5">
      {items.map((item) => (
        <div key={item.label} className="flex min-w-0 flex-col gap-1.5">
          <dt className="text-fg-3 text-sm">{item.label}</dt>
          <dd className="m-0 flex min-h-6 min-w-0 items-center gap-2 truncate">{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}
