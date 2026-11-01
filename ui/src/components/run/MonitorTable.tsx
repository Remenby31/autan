import type { Forces } from "../../api/contracts";
import type { HistoryColumns } from "../../api/runs";
import { Sparkline } from "../ui/Sparkline";

const KEYS = ["cl", "cd", "cs", "cmx", "cmy", "cmz"] as const;

/** Last value of every forces monitor, with the CL trend. */
export function MonitorTable({ history }: { history: HistoryColumns }) {
  if (!history.monitors.size) {
    return <p className="m-0 py-2 text-fg-3 text-sm">The case records no forces monitor.</p>;
  }
  return (
    <table className="w-full border-collapse text-sm">
      <thead>
        <tr className="text-left text-fg-3">
          <th className="h-9 font-normal">Monitor</th>
          {KEYS.map((k) => (
            <th key={k} className="text-right font-normal uppercase">
              {k}
            </th>
          ))}
          <th className="pl-4 font-normal">CL trend</th>
        </tr>
      </thead>
      <tbody>
        {[...history.monitors].map(([name, forces]: [string, Forces]) => (
          <tr key={name} className="border-line border-t">
            <td className="h-10 font-mono">{name}</td>
            {KEYS.map((k) => (
              <td key={k} className="text-right font-mono tabular-nums">
                {forces[k]?.toFixed(4) ?? "—"}
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
  );
}
