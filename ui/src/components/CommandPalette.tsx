import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { Command } from "cmdk";
import {
  Box,
  CornerDownLeft,
  FilePlus2,
  GitCompareArrows,
  ListTree,
  Monitor,
  Moon,
  Search,
  Sun,
} from "lucide-react";
import { type ReactNode, useEffect, useMemo, useState } from "react";
import type { Run } from "../api/contracts";
import { devicesLabel } from "../api/devices";
import { runsQuery } from "../api/runs";
import { studies } from "../api/studies";
import { setTheme } from "./shell/theme";
import { Kbd } from "./ui/Kbd";
import { formatRelative } from "./ui/RelativeTime";
import { Sparkline } from "./ui/Sparkline";
import { StateGlyph } from "./ui/StateGlyph";
import { Status } from "./ui/Status";

/** ⌘K: every run, screen and command, grouped; a preview of the highlighted run on the right. */
export function CommandPalette({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const navigate = useNavigate();
  const { data } = useQuery(runsQuery);
  const offered = useMemo(() => (open ? studies(data) : []), [open, data]);
  const [value, setValue] = useState("");

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        onOpenChange(!open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onOpenChange]);

  const close = (action: () => unknown) => {
    onOpenChange(false);
    void action();
  };
  const preview = data?.runs.find((run) => value === `run ${run.run_id}`);

  return (
    <Command.Dialog
      open={open}
      onOpenChange={onOpenChange}
      value={value}
      onValueChange={setValue}
      label="Command menu"
      overlayClassName="fixed inset-0 z-40 bg-scrim"
      contentClassName="fixed top-[14vh] left-1/2 z-50 flex w-[min(760px,calc(100%-32px))] -translate-x-1/2 flex-col overflow-hidden rounded-lg border border-line bg-bg-0 shadow-dialog"
    >
      <div className="flex h-12 items-center gap-3 border-line border-b px-4">
        <Search className="size-4 flex-none text-fg-3" />
        <Command.Input
          placeholder="Find a run, screen or command…"
          className="h-full flex-1 bg-transparent text-md outline-none placeholder:text-fg-3 focus-visible:shadow-none"
        />
        <Kbd>Esc</Kbd>
      </div>
      <div className="flex min-h-0">
        <Command.List className="max-h-[min(420px,60vh)] min-w-0 flex-1 overflow-auto p-2 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:pt-2 [&_[cmdk-group-heading]]:pb-1 [&_[cmdk-group-heading]]:text-fg-3 [&_[cmdk-group-heading]]:text-xs">
          <Command.Empty className="px-3 py-6 text-center text-fg-3">No match</Command.Empty>
          <Command.Group heading="Runs">
            {data?.runs.map((run) => (
              <Item
                key={run.run_id}
                value={`run ${run.run_id}`}
                keywords={[run.case.name, run.status, run.precision]}
                icon={<StateGlyph status={run.status} />}
                title={run.case.name}
                subtitle={`${run.status} · ${formatRelative(new Date(run.started), Date.now())} · ${run.run_id.slice(-8)}`}
                onSelect={() =>
                  close(() => navigate({ to: "/runs/$runId", params: { runId: run.run_id } }))
                }
              />
            ))}
          </Command.Group>
          <Command.Group heading="Go to">
            <Item
              value="go runs"
              icon={<ListTree className="size-4" />}
              title="Runs"
              onSelect={() => close(() => navigate({ to: "/" }))}
            />
            <Item
              value="go compare"
              icon={<GitCompareArrows className="size-4" />}
              title="Compare"
              onSelect={() => close(() => navigate({ to: "/compare", search: { runs: [] } }))}
            />
            <Item
              value="go analysis"
              icon={<Box className="size-4" />}
              title="Analysis"
              onSelect={() => close(() => navigate({ to: "/viewer" }))}
            />
            <Item
              value="new case"
              icon={<FilePlus2 className="size-4" />}
              title="New case"
              onSelect={() => close(() => navigate({ to: "/cases/new" }))}
            />
          </Command.Group>
          {offered.length > 0 && (
            <Command.Group heading="Compare a study">
              {offered.map((s) => (
                <Item
                  key={s.name}
                  // Unique by the study's newest run: two studies never share an entry.
                  value={`compare ${s.name} ${s.label} ${s.ids[0]}`}
                  icon={<GitCompareArrows className="size-4" />}
                  title={`${s.name} · ${s.label}`}
                  onSelect={() =>
                    close(() => navigate({ to: "/compare", search: { runs: s.ids } }))
                  }
                />
              ))}
            </Command.Group>
          )}
          <Command.Group heading="Theme">
            <Item
              value="theme system"
              icon={<Monitor className="size-4" />}
              title="Use system theme"
              onSelect={() => close(() => setTheme("system"))}
            />
            <Item
              value="theme light"
              icon={<Sun className="size-4" />}
              title="Light theme"
              onSelect={() => close(() => setTheme("light"))}
            />
            <Item
              value="theme dark"
              icon={<Moon className="size-4" />}
              title="Dark theme"
              onSelect={() => close(() => setTheme("dark"))}
            />
          </Command.Group>
        </Command.List>
        {preview && <Preview run={preview} orders={data?.summaries[preview.run_id]?.trend ?? []} />}
      </div>
      <footer className="flex h-10 items-center gap-4 border-line border-t bg-bg-1 px-4 text-fg-3 text-sm">
        <span className="flex items-center gap-1.5">
          <Kbd>↑</Kbd>
          <Kbd>↓</Kbd> move
        </span>
        <span className="flex items-center gap-1.5">
          <Kbd>
            <CornerDownLeft className="size-3" />
          </Kbd>
          open
        </span>
      </footer>
    </Command.Dialog>
  );
}

function Item({
  value,
  keywords,
  icon,
  title,
  subtitle,
  onSelect,
}: {
  value: string;
  keywords?: string[];
  icon: ReactNode;
  title: string;
  subtitle?: string;
  onSelect: () => void;
}) {
  return (
    <Command.Item
      value={value}
      {...(keywords ? { keywords } : {})}
      onSelect={onSelect}
      className="flex min-h-10 cursor-default items-center gap-3 rounded-md px-2 py-1.5 data-[selected=true]:bg-bg-2"
    >
      <span className="flex size-6 flex-none items-center justify-center text-fg-2">{icon}</span>
      <span className="flex min-w-0 flex-col">
        <span className="truncate">{title}</span>
        {subtitle && <span className="truncate text-fg-3 text-sm">{subtitle}</span>}
      </span>
    </Command.Item>
  );
}

function Preview({ run, orders }: { run: Run; orders: number[] }) {
  return (
    <aside className="hidden w-64 flex-none flex-col gap-4 border-line border-l p-4 md:flex">
      <div className="flex flex-col gap-1">
        <span className="font-medium">{run.case.name}</span>
        <span className="font-mono text-fg-3 text-xs">{run.run_id}</span>
      </div>
      <Status status={run.status} />
      <Sparkline
        values={orders}
        width={200}
        height={48}
        label="Residual drop"
        color="var(--accent)"
      />
      <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm">
        <dt className="text-fg-3">Precision</dt>
        <dd className="m-0 font-mono">{run.precision}</dd>
        <dt className="text-fg-3">Devices</dt>
        <dd className="m-0 font-mono">{devicesLabel(run.devices)}</dd>
        <dt className="text-fg-3">Backend</dt>
        <dd className="m-0 font-mono">{run.solver.backend}</dd>
      </dl>
    </aside>
  );
}
