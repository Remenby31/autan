import { useState } from "react";
import type { Forces } from "../../api/contracts";
import type { HistoryColumns } from "../../api/runs";
import { coefficient } from "../../api/units";
import { cn } from "../ui/cn";
import { Sparkline } from "../ui/Sparkline";

const LIFT_DRAG = ["cl", "cd"] as const;
// Which moment is the pitching one depends on the mesh's axes, and a 2D section's side force
// and out-of-plane moments are artefacts of its extrusion: these show on request.
const OTHERS = ["cs", "cmx", "cmy", "cmz"] as const;

/** Last value of every forces monitor, with the CL trend; greyed, with the reason, while they
 * are not the coefficients of a converged run. */
export function MonitorTable({
  history,
  caveat,
}: {
  history: HistoryColumns;
  caveat: string | null;
}) {
  const [all, setAll] = useState(false);
  if (!history.monitors.size) {
    return <p className="m-0 py-2 text-fg-3 text-sm">The case records no forces monitor.</p>;
  }
  const keys = all ? [...LIFT_DRAG, ...OTHERS] : LIFT_DRAG;
  return (
    <div className="flex flex-col gap-2">
      <table className="w-full border-collapse text-sm" data-testid="monitor-table">
        <thead>
          <tr className="text-left text-fg-3">
            <th className="h-9 font-normal">Monitor</th>
            {keys.map((k) => (
              <th key={k} className="text-right font-normal uppercase">
                {k}
              </th>
            ))}
            <th className="pl-4 font-normal">CL trend</th>
          </tr>
        </thead>
        <tbody className={cn(caveat && "text-fg-3")}>
          {[...history.monitors].map(([name, forces]: [string, Forces]) => (
            <tr key={name} className="border-line border-t">
              <td className="h-10 font-mono">{name}</td>
              {keys.map((k) => (
                <td key={k} className="text-right font-mono tabular-nums">
                  {forces[k] === undefined ? "—" : coefficient(forces[k])}
                </td>
              ))}
              <td className="pl-4">
                <Sparkline
                  values={history.forces.get(name)?.cl.slice(-200) ?? []}
                  label={`${name} CL trend`}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex items-center gap-4 text-sm">
        <button
          type="button"
          aria-pressed={all}
          onClick={() => setAll(!all)}
          className="text-fg-2 hover:text-fg aria-pressed:text-fg"
        >
          Side force and moments
        </button>
        {caveat && (
          <span className="text-fg-3" data-testid="monitor-caveat">
            {caveat}
          </span>
        )}
      </div>
    </div>
  );
}
