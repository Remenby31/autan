import { createContext, type ReactNode, useContext, useId } from "react";
import { cn } from "./cn";

const BOX =
  "h-9 rounded-sm bg-bg-0 shadow-[0_0_0_1px_var(--line)] transition-shadow duration-[var(--dur-base)] hover:shadow-[0_0_0_1px_var(--line-strong)] focus-within:shadow-[var(--field-ring)]";
const INPUT =
  "h-full min-w-0 flex-1 bg-transparent px-3 outline-none placeholder:text-fg-3 focus-visible:shadow-none";

// The control a Field labels: its id, and the id of its hint as its description.
const Control = createContext<{ id?: string; hint?: string }>({});
const useControl = () => useContext(Control);

/** A label above one control; the hint says the default the SDK applies when it is empty. */
export function Field({
  label,
  hint,
  children,
  className,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  const id = useId();
  const hintId = `${id}-hint`;
  return (
    <div className={cn("flex min-w-0 flex-col gap-1.5", className)}>
      <span className="flex items-baseline justify-between gap-2 text-fg-2 text-sm">
        <label htmlFor={id}>{label}</label>
        {hint && (
          <span id={hintId} className="truncate text-fg-3 text-xs">
            {hint}
          </span>
        )}
      </span>
      <Control.Provider value={hint ? { id, hint: hintId } : { id }}>{children}</Control.Provider>
    </div>
  );
}

/** A text field; `mono` for identifiers and patterns, `options` for suggestions. */
export function TextInput({
  value,
  onChange,
  placeholder,
  mono,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  mono?: boolean;
  options?: { id: string; values: string[] };
}) {
  const control = useControl();
  return (
    <span className={cn("flex", BOX)}>
      <input
        id={control.id}
        aria-describedby={control.hint}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        list={options?.id}
        spellCheck={false}
        className={cn(INPUT, mono && "font-mono")}
      />
      {options && (
        <datalist id={options.id}>
          {options.values.map((v) => (
            <option key={v} value={v} />
          ))}
        </datalist>
      )}
    </span>
  );
}

/** A number typed as text, so that an unfinished entry ("0.") stays as typed. */
export function NumberInput({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  const control = useControl();
  return (
    <span className={cn("flex", BOX)}>
      <input
        id={control.id}
        aria-describedby={control.hint}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        inputMode="decimal"
        className={cn(INPUT, "font-mono tabular-nums")}
      />
    </span>
  );
}

/** One of a few choices; the empty choice is the SDK's default, named by `placeholder`. */
export function SelectInput<T extends string>({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: T | "";
  onChange: (value: T | "") => void;
  options: readonly { value: T; label: string }[];
  placeholder?: string;
}) {
  const control = useControl();
  return (
    <span className={cn("flex", BOX)}>
      <select
        id={control.id}
        aria-describedby={control.hint}
        value={value}
        onChange={(e) => onChange(e.target.value as T | "")}
        className={cn(INPUT, "cursor-pointer appearance-none pr-8")}
      >
        {placeholder !== undefined && <option value="">{placeholder}</option>}
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </span>
  );
}

/** A quantity as the SDK reads it, `"<number> <unit>"`; empty when no number is typed. */
export function QuantityInput({
  value,
  onChange,
  units,
  placeholder,
}: {
  value: string;
  onChange: (value: string) => void;
  units: readonly string[];
  placeholder?: string;
}) {
  const control = useControl();
  const space = value.indexOf(" ");
  const magnitude = space < 0 ? value : value.slice(0, space);
  const unit = space < 0 ? (units[0] ?? "") : value.slice(space + 1);
  const emit = (m: string, u: string) => onChange(m.trim() === "" ? "" : `${m.trim()} ${u}`);
  return (
    <span className={cn("flex", BOX)}>
      <input
        id={control.id}
        aria-describedby={control.hint}
        value={magnitude}
        onChange={(e) => emit(e.target.value, unit)}
        placeholder={placeholder}
        inputMode="decimal"
        className={cn(INPUT, "font-mono tabular-nums")}
      />
      <select
        value={unit}
        onChange={(e) => emit(magnitude, e.target.value)}
        aria-label="Unit"
        className="cursor-pointer border-line border-l bg-transparent px-2 font-mono text-fg-2 text-sm outline-none"
      >
        {units.map((u) => (
          <option key={u} value={u}>
            {u}
          </option>
        ))}
      </select>
    </span>
  );
}
