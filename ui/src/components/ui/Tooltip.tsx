import { Tooltip as Base } from "@base-ui/react/tooltip";
import type { ReactElement, ReactNode } from "react";

export const TooltipProvider = Base.Provider;

const POPUP =
  "rounded-md bg-fg px-2 py-1.5 text-bg-0 text-sm shadow-tooltip origin-[var(--transform-origin)] transition-[opacity,transform] duration-[var(--dur-fast)] data-[starting-style]:scale-95 data-[starting-style]:opacity-0 data-[ending-style]:opacity-0";

/** Names an icon-only control or shows a value's exact form; never the only place a fact lives. */
export function Tooltip({
  content,
  children,
  side = "top",
}: {
  content: ReactNode;
  children: ReactElement;
  side?: "top" | "bottom" | "left" | "right";
}) {
  return (
    <Base.Root>
      <Base.Trigger render={children} />
      <Base.Portal>
        <Base.Positioner side={side} sideOffset={6} className="z-50">
          <Base.Popup className={POPUP}>{content}</Base.Popup>
        </Base.Positioner>
      </Base.Portal>
    </Base.Root>
  );
}
