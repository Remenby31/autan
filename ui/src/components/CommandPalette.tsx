import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { Command } from "cmdk";
import { useEffect, useState } from "react";
import { runsQuery } from "../api/runs";

/** ⌘K: jump to any run or screen. */
export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const { data } = useQuery(runsQuery);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const go = (to: () => Promise<void>) => {
    setOpen(false);
    void to();
  };

  return (
    <Command.Dialog
      open={open}
      onOpenChange={setOpen}
      label="Command palette"
      overlayClassName="fixed inset-0 bg-scrim"
      contentClassName="fixed left-1/2 top-[14vh] w-[min(620px,calc(100%-32px))] -translate-x-1/2 overflow-hidden rounded-xl border border-line bg-bg-1 shadow-dialog"
    >
      <Command.Input
        placeholder="Run, screen…"
        className="w-full border-0 border-b border-line bg-transparent px-4 py-4 text-base outline-none"
      />
      <Command.List className="max-h-[50vh] overflow-auto p-1.5">
        <Command.Empty className="px-3 py-2 text-fg-3">No match</Command.Empty>
        <Command.Group heading="Go to" className="text-fg-3 text-xs">
          <Item onSelect={() => go(() => navigate({ to: "/" }))}>Runs</Item>
          <Item onSelect={() => go(() => navigate({ to: "/viewer" }))}>Viewer</Item>
        </Command.Group>
        <Command.Group heading="Runs" className="text-fg-3 text-xs">
          {data?.runs.map((run) => (
            <Item
              key={run.run_id}
              value={`${run.case.name} ${run.run_id} ${run.status}`}
              onSelect={() =>
                go(() => navigate({ to: "/runs/$runId", params: { runId: run.run_id } }))
              }
            >
              <span>{run.case.name}</span>
              <span className="ml-auto font-mono text-fg-3 text-xs">
                {run.status} · {run.run_id.slice(4, 19)}
              </span>
            </Item>
          ))}
        </Command.Group>
      </Command.List>
    </Command.Dialog>
  );
}

function Item({
  children,
  onSelect,
  value,
}: {
  children: React.ReactNode;
  onSelect: () => void;
  value?: string;
}) {
  return (
    <Command.Item
      {...(value ? { value } : {})}
      onSelect={onSelect}
      className="flex cursor-pointer items-center gap-2.5 rounded-md px-3 py-2.5 text-fg text-sm data-[selected=true]:bg-accent-soft"
    >
      {children}
    </Command.Item>
  );
}
