import type { Plan } from "../../api/runs";
import { Badge } from "../ui/Badge";
import { formatBytes, formatCount } from "./format";

const PROVENANCE = { code: "blue", mesh: "gray", default: "purple" } as const;

/** `autan plan` of the stored case: each boundary zone, its condition and where it comes from. */
export function PlanTable({ plan }: { plan: Plan }) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-x-8 gap-y-2 text-sm">
        {/* One rank's host and device memory (one pool on a CPU or a GB10), then all ranks'. */}
        <span data-testid="plan-memory">
          <span className="text-fg-3">Memory </span>
          <span className="font-mono">
            {formatBytes(plan.memory.bytes_per_gpu)} per rank
            <span className="text-fg-3">
              {" "}
              ({formatBytes(plan.memory.bytes_host)} host
              {plan.memory.device_kind === "cpu"
                ? ", cpu"
                : ` · ${formatBytes(plan.memory.bytes_device)} ${plan.memory.device_kind}`}
              )
            </span>{" "}
            × {plan.memory.gpus} {plan.memory.gpus === 1 ? "rank" : "ranks"} ={" "}
            {formatBytes(plan.memory.bytes_total)}
          </span>
        </span>
        <span>
          <span className="text-fg-3">Equations </span>
          <span className="font-mono">{plan.memory.equations}</span>
        </span>
      </div>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="text-left text-fg-3">
            <th className="h-9 font-normal">Boundary zone</th>
            <th className="font-normal">Condition</th>
            <th className="font-normal">From</th>
            <th className="text-right font-normal">Faces</th>
          </tr>
        </thead>
        <tbody>
          {plan.boundaries.map((b) => (
            <tr key={b.zone} className="border-line border-t">
              <td className="h-9 font-mono">{b.zone}</td>
              <td>{b.condition?.kind ?? <span className="text-danger">none</span>}</td>
              <td>
                {b.provenance && <Badge tone={PROVENANCE[b.provenance]}>{b.provenance}</Badge>}
              </td>
              <td className="text-right font-mono tabular-nums">{formatCount(b.faces)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
