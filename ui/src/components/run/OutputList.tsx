import { Download, FileText } from "lucide-react";
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
  "log.jsonl": "Log: warnings, errors and diagnostics",
  "manufactured.json": "Manufactured solution of the verification case",
};

/** The files of the run directory, each one a download. */
export function OutputList({ runId, outputs }: { runId: string; outputs: RunOutput[] }) {
  return (
    <ul className="m-0 flex list-none flex-col divide-y divide-line p-0">
      {outputs.map((o) => (
        <li key={o.name}>
          <Entity
            icon={<FileText className="size-4" />}
            title={
              <a
                href={`/api/runs/${runId}/files/${encodeURIComponent(o.name)}`}
                download={o.name}
                className="inline-flex items-center gap-1.5 font-mono hover:underline"
              >
                {o.name}
                <Download aria-hidden className="size-3.5 text-fg-3" />
              </a>
            }
            description={ROLE[o.name] ?? "Written by the run"}
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
