import { createFileRoute } from "@tanstack/react-router";
import { soleForces, useLiveRun } from "../api/runs";
import { ResidualChart } from "../components/ResidualChart";
import { StateGlyph } from "../components/StateGlyph";

export const Route = createFileRoute("/runs/$runId")({ component: RunView });

function RunView() {
  const { runId } = Route.useParams();
  const { run, history, connected, invalid } = useLiveRun(runId);
  const forces = soleForces(history);

  return (
    <div className="flex h-full flex-col gap-4 px-4 py-4">
      <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
        <h1 className="m-0 font-semibold text-lg">{run?.case.name ?? runId}</h1>
        {run && (
          <span className="flex items-center gap-2" data-testid="status">
            <StateGlyph status={run.status} />
            {run.status}
          </span>
        )}
        <Metric label="iteration" value={history.iteration.at(-1)} testId="iteration" />
        <Metric label="CL" value={forces?.cl.at(-1)} digits={4} />
        <Metric label="CD" value={forces?.cd.at(-1)} digits={5} />
        <span className="ml-auto font-mono text-fg-3 text-xs">
          {run?.status === "running" && (connected ? "live · " : "offline · ")}
          {run?.precision} · {run?.solver.backend}
        </span>
      </div>
      {invalid && <p className="m-0 font-mono text-danger text-sm">{invalid}</p>}
      {run?.diagnostics.map((d) => (
        <p key={`${d.code}-${String(d.data.iteration)}`} className="m-0 font-mono text-sm">
          <span className="text-danger">{d.code}</span> {d.message}
          {d.fix && <span className="text-fg-3"> · {d.fix}</span>}
        </p>
      ))}
      <div className="min-h-0 flex-1">
        <ResidualChart history={history} />
      </div>
    </div>
  );
}

function Metric({
  label,
  value,
  digits,
  testId,
}: {
  label: string;
  value: number | undefined;
  digits?: number;
  testId?: string;
}) {
  return (
    <span className="flex items-baseline gap-2">
      <span className="font-mono text-fg-3 text-xs uppercase">{label}</span>
      <span className="font-mono text-base tabular-nums" data-testid={testId}>
        {value === undefined ? "–" : digits ? value.toFixed(digits) : value}
      </span>
    </span>
  );
}
