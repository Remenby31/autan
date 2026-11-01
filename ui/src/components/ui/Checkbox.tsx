import { Checkbox as Base } from "@base-ui/react/checkbox";
import { Check } from "lucide-react";

/** A 16 px box; its accessible name says what it selects. */
export function Checkbox({
  checked,
  onChange,
  label,
  className,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  className?: string;
}) {
  return (
    <Base.Root
      checked={checked}
      onCheckedChange={onChange}
      aria-label={label}
      className={`flex size-4 items-center justify-center rounded-xs border border-line-strong bg-bg-0 transition-colors duration-[var(--dur-fast)] data-[checked]:border-fg data-[checked]:bg-fg ${className ?? ""}`}
    >
      <Base.Indicator className="text-bg-0">
        <Check className="size-3" strokeWidth={3} />
      </Base.Indicator>
    </Base.Root>
  );
}
