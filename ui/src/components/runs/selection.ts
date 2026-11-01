import { createContext, useContext } from "react";

export interface Selection {
  selected: ReadonlySet<string>;
  toggle: (runId: string) => void;
  peek: string | null;
}

export const SelectionContext = createContext<Selection>({
  selected: new Set(),
  toggle: () => {},
  peek: null,
});

export const useSelection = () => useContext(SelectionContext);
