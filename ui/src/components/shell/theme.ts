import { useEffect, useSyncExternalStore } from "react";

export type ThemeChoice = "system" | "light" | "dark";
const KEY = "autan.theme";
const listeners = new Set<() => void>();

function read(): ThemeChoice {
  try {
    const value = localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

let current = read();

export function setTheme(choice: ThemeChoice): void {
  current = choice;
  try {
    if (choice === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, choice);
  } catch {
    // Private mode: the choice lasts for this page only.
  }
  for (const listener of listeners) listener();
}

/** The viewer's theme choice, applied as `data-theme` on <html> (tokens.css reads it). */
export function useTheme(): ThemeChoice {
  const choice = useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => current,
  );
  useEffect(() => {
    if (choice === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = choice;
  }, [choice]);
  return choice;
}

const dark =
  typeof window === "undefined" ? null : window.matchMedia("(prefers-color-scheme: dark)");

/** "light" or "dark" as displayed: the choice, or the system's when the choice is "system". */
export function useResolvedTheme(): "light" | "dark" {
  const choice = useTheme();
  const systemDark = useSyncExternalStore(
    (listener) => {
      dark?.addEventListener("change", listener);
      return () => dark?.removeEventListener("change", listener);
    },
    () => dark?.matches ?? false,
  );
  if (choice === "system") return systemDark ? "dark" : "light";
  return choice;
}
