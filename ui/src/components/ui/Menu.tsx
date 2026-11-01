import { Menu as Base } from "@base-ui/react/menu";
import { Check, Ellipsis } from "lucide-react";
import type { ReactElement, ReactNode } from "react";
import { Kbd } from "./Kbd";

const POPUP =
  "min-w-52 rounded-lg bg-bg-0 p-1.5 shadow-popover outline-none origin-[var(--transform-origin)] transition-[opacity,transform] duration-[var(--dur-fast)] data-[starting-style]:scale-95 data-[starting-style]:opacity-0 data-[ending-style]:opacity-0";
const ITEM =
  "flex h-9 cursor-default select-none items-center gap-2 rounded-sm px-2 text-base outline-none data-[highlighted]:bg-bg-2 data-[disabled]:text-fg-3";

/** Actions on one thing, opened from its trigger; each item may show its key. */
export function Menu({
  trigger,
  label,
  children,
}: {
  trigger?: ReactElement;
  label: string;
  children: ReactNode;
}) {
  return (
    <Base.Root>
      <Base.Trigger
        aria-label={label}
        render={
          trigger ?? (
            <button
              type="button"
              className="flex size-8 items-center justify-center rounded-sm text-fg-2 hover:bg-bg-2 hover:text-fg data-[popup-open]:bg-bg-2"
            />
          )
        }
      >
        {trigger ? undefined : <Ellipsis className="size-4" />}
      </Base.Trigger>
      <Base.Portal>
        <Base.Positioner sideOffset={4} align="end" className="z-50">
          <Base.Popup className={POPUP}>{children}</Base.Popup>
        </Base.Positioner>
      </Base.Portal>
    </Base.Root>
  );
}

export function MenuItem({
  onClick,
  shortcut,
  icon,
  children,
}: {
  onClick: () => void;
  shortcut?: string;
  icon?: ReactNode;
  children: ReactNode;
}) {
  return (
    <Base.Item className={ITEM} onClick={onClick}>
      <span className="flex size-4 items-center text-fg-2">{icon}</span>
      <span className="flex-1">{children}</span>
      {shortcut && <Kbd>{shortcut}</Kbd>}
    </Base.Item>
  );
}

export function MenuCheckboxItem({
  checked,
  onCheckedChange,
  children,
}: {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  children: ReactNode;
}) {
  return (
    <Base.CheckboxItem
      checked={checked}
      onCheckedChange={onCheckedChange}
      className={ITEM}
      closeOnClick={false}
    >
      <span className="flex size-4 items-center">
        <Base.CheckboxItemIndicator>
          <Check className="size-3.5" />
        </Base.CheckboxItemIndicator>
      </span>
      {children}
    </Base.CheckboxItem>
  );
}

export const MenuSeparator = () => <Base.Separator className="-mx-1 my-1 h-px bg-line" />;
