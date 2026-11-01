import { Tabs as Base } from "@base-ui/react/tabs";
import type { ReactNode } from "react";

export const TabsRoot = Base.Root;
export const TabPanel = Base.Panel;

/** Views of the same thing; the active tab is a filled pill that slides between tabs. */
export function TabList({ tabs }: { tabs: { value: string; label: ReactNode }[] }) {
  return (
    <Base.List className="relative flex items-center gap-1">
      {tabs.map((tab) => (
        <Base.Tab
          key={tab.value}
          value={tab.value}
          className="relative z-10 flex h-8 items-center rounded-sm px-3 text-fg-2 transition-colors duration-[var(--dur-fast)] hover:text-fg data-[active]:text-fg"
        >
          {tab.label}
        </Base.Tab>
      ))}
      <Base.Indicator className="absolute top-1/2 left-[var(--active-tab-left)] h-8 w-[var(--active-tab-width)] -translate-y-1/2 rounded-sm bg-bg-2 transition-[left,width] duration-[var(--dur-base)] ease-[var(--ease-out)]" />
    </Base.List>
  );
}
