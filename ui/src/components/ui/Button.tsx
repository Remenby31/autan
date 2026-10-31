import type { ComponentProps, ReactNode } from "react";
import { cn } from "./cn";
import { LoadingDots } from "./LoadingDots";

type Variant = "primary" | "secondary" | "tertiary" | "danger";
type Size = "sm" | "md" | "lg";

const VARIANT: Record<Variant, string> = {
  primary: "bg-fg text-bg-0 hover:bg-fg/80",
  secondary: "bg-bg-0 text-fg shadow-[0_0_0_1px_var(--line)] hover:bg-bg-2",
  tertiary: "text-fg hover:bg-bg-2",
  danger: "bg-danger text-white hover:bg-danger/85",
};
const SIZE: Record<Size, string> = {
  sm: "h-8 px-3 text-base gap-1.5 rounded-sm",
  md: "h-9 px-4 text-base gap-2 rounded-sm",
  lg: "h-10 px-5 text-md gap-2 rounded-md",
};

export interface ButtonProps extends ComponentProps<"button"> {
  variant?: Variant;
  size?: Size;
  icon?: ReactNode;
  loading?: boolean;
  /** Icon-only button: square, needs an aria-label. */
  square?: boolean;
}

/** One action. Primary once per view; icon-only buttons are square and carry aria-label. */
export function Button({
  variant = "secondary",
  size = "md",
  icon,
  loading,
  square,
  className,
  children,
  disabled,
  ...props
}: ButtonProps) {
  return (
    <button
      type="button"
      {...props}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cn(
        "inline-flex flex-none select-none items-center justify-center whitespace-nowrap font-medium transition-[background-color,color,box-shadow] duration-[var(--dur-fast)] disabled:cursor-not-allowed disabled:opacity-50",
        VARIANT[variant],
        SIZE[size],
        square && "aspect-square px-0",
        className,
      )}
    >
      {loading ? <LoadingDots /> : icon}
      {children}
    </button>
  );
}
