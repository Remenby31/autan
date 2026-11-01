// Shapes of docs/contracts/results.md (autan.results/1) and case-schema.md diagnostics.

export type RunStatus = "running" | "converged" | "completed" | "stopped" | "diverged" | "failed";

export const RUN_STATUSES: readonly RunStatus[] = [
  "running",
  "converged",
  "completed",
  "stopped",
  "diverged",
  "failed",
];

export interface Diagnostic {
  code: string;
  severity: "error" | "warning" | "divergence";
  message: string;
  location: { file?: string; line?: number; path?: string } | null;
  fix: string | null;
  doc: string | null;
  data: Record<string, unknown>;
}

export interface Run {
  schema: "autan.results/1";
  run_id: string;
  identity: string;
  status: RunStatus;
  case: { name: string; sha256: string };
  mesh: { path: string; sha256: string };
  solver: { version: string; commit: string; backend: "cuda" | "hip" | "cpu" | "fake" };
  precision: "fp32+acc64" | "fp64";
  deterministic: boolean;
  devices: { kind: "cuda" | "hip" | "cpu"; count: number };
  started: string;
  finished: string | null;
  iterations: number;
  time: number | null;
  averaging: Record<string, number> | null;
  diagnostics: Diagnostic[];
}

export interface Forces {
  fx: number;
  fy: number;
  fz: number;
  mx: number;
  my: number;
  mz: number;
  cl?: number;
  cd?: number;
  cs?: number;
  cmx?: number;
  cmy?: number;
  cmz?: number;
}

export interface HistoryRecord {
  iteration: number;
  time: number | null;
  residuals: Record<string, number>;
  monitors?: Record<string, Forces>;
}

export interface RunRegistry {
  runs: Run[];
  rejected: { dir: string; reason: string }[];
  /** Per run id: last recorded iteration and residual drop, from history.jsonl. */
  summaries: Record<string, RunSummary>;
}

export interface RunSummary {
  iteration: number;
  /** Orders of magnitude the slowest residual has dropped since iteration 1. */
  orders: number | null;
  /** `orders` along the run, at most 48 points. */
  trend: number[];
  /** The case's `time.residual_orders` (steady) and iteration or step budget. */
  target_orders: number | null;
  max_iterations: number | null;
}
