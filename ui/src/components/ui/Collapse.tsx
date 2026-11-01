import { Collapsible } from "@base-ui/react/collapsible";
import { ChevronRight } from "lucide-react";
import type { ReactNode } from "react";

/** A titled section that folds; its state or count sits on the right so it reads while closed. */
export function Collapse({
  title,
  meta,
  defaultOpen = false,
  children,
}: {
  title: ReactNode;
  meta?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  return (
    <Collapsible.Root defaultOpen={defaultOpen} className="border-line border-t first:border-t-0">
      <Collapsible.Trigger className="group flex h-12 w-full items-center gap-2 px-4 text-left font-medium text-fg-2 transition-colors duration-[var(--dur-fast)] hover:text-fg data-[panel-open]:text-fg">
        <ChevronRight
          aria-hidden
          className="size-4 flex-none transition-transform duration-[var(--dur-base)] group-data-[panel-open]:rotate-90"
        />
        <span className="flex-1">{title}</span>
        {meta}
      </Collapsible.Trigger>
      <Collapsible.Panel className="h-[var(--collapsible-panel-height)] overflow-hidden transition-[height] duration-[var(--dur-base)] ease-[var(--ease-out)] data-[ending-style]:h-0 data-[starting-style]:h-0">
        <div className="px-4 pb-4">{children}</div>
      </Collapsible.Panel>
    </Collapsible.Root>
  );
}
