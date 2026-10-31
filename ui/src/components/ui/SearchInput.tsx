import { Search } from "lucide-react";
import { forwardRef } from "react";
import { Kbd } from "./Kbd";

/** Filter field: search icon left, its key right, Escape clears focus. */
export const SearchInput = forwardRef<
  HTMLInputElement,
  { value: string; onChange: (value: string) => void; placeholder: string; shortcut?: string }
>(function SearchInput({ value, onChange, placeholder, shortcut }, ref) {
  return (
    <label className="flex h-9 min-w-64 flex-1 items-center gap-2 rounded-sm bg-bg-0 px-3 shadow-[0_0_0_1px_var(--line)] transition-shadow duration-[var(--dur-base)] hover:shadow-[0_0_0_1px_var(--line-strong)] focus-within:shadow-[var(--field-ring)]">
      <Search aria-hidden className="size-4 flex-none text-fg-3" />
      <input
        ref={ref}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Escape" && e.currentTarget.blur()}
        placeholder={placeholder}
        className="min-w-0 flex-1 bg-transparent outline-none placeholder:text-fg-3 focus-visible:shadow-none"
      />
      {shortcut && <Kbd>{shortcut}</Kbd>}
    </label>
  );
});
