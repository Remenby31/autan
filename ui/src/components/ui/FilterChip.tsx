import { Menu as Base } from "@base-ui/react/menu";
import { X } from "lucide-react";
import type { ReactNode } from "react";
import { cn } from "./cn";
import { MenuCheckboxItem } from "./Menu";

/** A filter on one field: dashed while empty, solid with its values once set; its menu toggles values. */
export function FilterChip<T extends string>({
  label,
  options,
  selected,
  onChange,
  render,
}: {
  label: string;
  options: readonly T[];
  selected: readonly T[];
  onChange: (values: T[]) => void;
  render: (value: T) => ReactNode;
}) {
  const active = selected.length > 0;
  return (
    <span
      className={cn(
        "inline-flex h-8 items-center rounded-full border text-sm",
        active ? "border-line-strong bg-bg-0" : "border-line border-dashed text-fg-2",
      )}
    >
      <Base.Root>
        <Base.Trigger className="flex h-full items-center gap-1.5 rounded-full pr-2 pl-3 hover:text-fg data-[popup-open]:text-fg">
          <span className={active ? "text-fg-2" : undefined}>{label}</span>
          {active && (
            <span className="flex items-center gap-1.5 font-medium text-fg">
              {selected.map((v) => (
                <span key={v} className="inline-flex items-center">
                  {render(v)}
                </span>
              ))}
            </span>
          )}
        </Base.Trigger>
        <Base.Portal>
          <Base.Positioner sideOffset={4} align="start" className="z-50">
            <Base.Popup className="min-w-52 rounded-lg bg-bg-0 p-1.5 shadow-popover outline-none transition-opacity duration-[var(--dur-fast)] data-[ending-style]:opacity-0 data-[starting-style]:opacity-0">
              {options.map((option) => (
                <MenuCheckboxItem
                  key={option}
                  checked={selected.includes(option)}
                  onCheckedChange={(checked) =>
                    onChange(checked ? [...selected, option] : selected.filter((v) => v !== option))
                  }
                >
                  {render(option)}
                </MenuCheckboxItem>
              ))}
            </Base.Popup>
          </Base.Positioner>
        </Base.Portal>
      </Base.Root>
      {active && (
        <button
          type="button"
          aria-label={`Clear ${label}`}
          onClick={() => onChange([])}
          className="mr-1 flex size-6 items-center justify-center rounded-full text-fg-3 hover:bg-bg-2 hover:text-fg"
        >
          <X className="size-3.5" />
        </button>
      )}
    </span>
  );
}
