import type { ReactNode } from "react";

/** What would be here, and the next step to fill it; never only "no data". */
export function EmptyState({
  icon,
  title,
  description,
  children,
}: {
  icon: ReactNode;
  title: string;
  description?: string;
  children?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-md border border-line bg-bg-0 px-16 py-12 text-center">
      <span className="mb-2 flex size-12 items-center justify-center rounded-md border border-line bg-bg-0 text-fg-2">
        {icon}
      </span>
      <h2 className="m-0 font-medium text-md">{title}</h2>
      {description && <p className="m-0 max-w-md text-fg-2">{description}</p>}
      {children && <div className="mt-3 flex w-full max-w-md flex-col gap-2">{children}</div>}
    </div>
  );
}
