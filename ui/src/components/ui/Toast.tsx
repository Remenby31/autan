import { Toast as Base } from "@base-ui/react/toast";
import type { ReactNode } from "react";

/** Confirms an action that left no visible trace (a copy); state changes stay inline. */
export const useToast = () => {
  const manager = Base.useToastManager();
  return (title: string, description?: string) =>
    manager.add({ title, ...(description ? { description } : {}), timeout: 2500 });
};

function Toasts() {
  const { toasts } = Base.useToastManager();
  return toasts.map((toast) => (
    <Base.Root
      key={toast.id}
      toast={toast}
      className="absolute right-0 bottom-0 w-full rounded-lg bg-bg-0 p-4 shadow-popover transition-[transform,opacity] duration-[var(--dur-base)] ease-[var(--ease-out)] [transform:translateY(calc(var(--toast-index)*-8px))_scale(calc(1-var(--toast-index)*0.04))] data-[ending-style]:opacity-0 data-[starting-style]:[transform:translateY(100%)] data-[starting-style]:opacity-0"
    >
      <Base.Title className="font-medium" />
      <Base.Description className="text-fg-2 text-sm" />
    </Base.Root>
  ));
}

export function ToastProvider({ children }: { children: ReactNode }) {
  return (
    <Base.Provider>
      {children}
      <Base.Portal>
        <Base.Viewport className="fixed right-4 bottom-4 z-50 w-80">
          <Toasts />
        </Base.Viewport>
      </Base.Portal>
    </Base.Provider>
  );
}
