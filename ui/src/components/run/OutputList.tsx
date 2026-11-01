import { FileText } from "lucide-react";
import type { RunOutput } from "../../api/runs";
import { Entity } from "../ui/Entity";
import { RelativeTime } from "../ui/RelativeTime";
import { formatBytes } from "./format";

const ROLE: Record<string, string> = {
  "run.json": "Run metadata",
  "case.toml": "Case as run, every default filled in",
  "mesh-info.json": "Mesh summary",
  "history.jsonl": "Residuals and monitors per iteration",
  "solution.cgns": "Solution (CGNS/HDF5)",
};

/** The files of the run directory. */
export function OutputList({ outputs }: { outputs: RunOutput[] }) {
  return (
    <ul className="m-0 flex list-none flex-col divide-y divide-line p-0">
      {outputs.map((o) => (
        <li key={o.name}>
          <Entity
            icon={<FileText className="size-4" />}
            title={<span className="font-mono">{o.name}</span>}
            description={ROLE[o.name] ?? "—"}
            meta={
              <>
                <span className="font-mono tabular-nums">{formatBytes(o.bytes)}</span>
                <RelativeTime iso={new Date(o.modified * 1000).toISOString()} />
              </>
            }
          />
        </li>
      ))}
    </ul>
  );
}
