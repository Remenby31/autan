import type { Run, RunSettings } from "./contracts";
import { degrees } from "./units";

/** One compared run, with its letter on the Compare screen and the settings it was run at. */
export interface PolarRun {
  id: string;
  role: string;
  run: Run | undefined;
  settings: RunSettings | undefined;
}

export interface PolarPoint {
  id: string;
  role: string;
  alpha: number;
  cl: number;
  cd: number;
}

/** The runs of one fixed condition (mesh, Mach, Reynolds number, sideslip, model, precision),
 * in angle-of-attack order; `label` names what sets it apart from the other series. */
export interface PolarSeries {
  key: string;
  label: string;
  points: PolarPoint[];
}

export interface Polars {
  series: PolarSeries[];
  /** What every series shares, e.g. "M 0.8 · inviscid". */
  fixed: string;
  /** The compared runs not on the polars, each with the reason. */
  unplotted: { id: string; role: string; reason: string }[];
}

// What a polar holds fixed, and how a value of it reads in a label.
const CONDITIONS: { name: string; of: (run: Run, s: RunSettings) => string | null }[] = [
  { name: "mesh", of: (run) => `mesh ${run.mesh.sha256.slice(0, 7)}` },
  { name: "mach", of: (_, s) => (s.mach === null ? null : `M ${s.mach}`) },
  { name: "re", of: (_, s) => (s.re === null ? null : `Re ${s.re.toExponential(2)}`) },
  {
    name: "sideslip",
    of: (_, s) => {
      const beta = degrees(s.sideslip);
      return beta === null || beta === 0 ? null : `β ${beta}°`;
    },
  },
  { name: "turbulence", of: (_, s) => s.turbulence },
  { name: "precision", of: (_, s) => s.precision },
];

/** The coefficients a converged run ends with: its one forces monitor giving CL and CD, on the
 * final state run.json records (results.md#runjson); otherwise why there are none. */
function ended(run: Run): { cl: number; cd: number } | string {
  if (run.status !== "converged") return `${run.status}, not converged`;
  const coefficients = Object.values(run.monitors ?? {}).filter(
    (m) => m.cl !== undefined && m.cd !== undefined,
  );
  const [only] = coefficients;
  if (coefficients.length === 0) return "no forces monitor with CL and CD";
  if (coefficients.length > 1) return "several forces monitors with CL and CD";
  return { cl: only?.cl as number, cd: only?.cd as number };
}

/** Lift and drag polars of the compared runs: one series per fixed condition, every converged
 * run with coefficients a point of its series (none merged, whatever its angle), and every
 * other run listed with the reason it is not there. */
export function polars(compared: PolarRun[]): Polars {
  const unplotted: Polars["unplotted"] = [];
  const groups = new Map<string, { values: (string | null)[]; points: PolarPoint[] }>();
  for (const { id, role, run, settings } of compared) {
    if (!run || !settings) {
      unplotted.push({ id, role, reason: "not available" });
      continue;
    }
    const coefficients = ended(run);
    if (typeof coefficients === "string") {
      unplotted.push({ id, role, reason: coefficients });
      continue;
    }
    const alpha = degrees(settings.aoa);
    if (alpha === null) {
      unplotted.push({ id, role, reason: "no angle of attack in deg or rad" });
      continue;
    }
    const values = CONDITIONS.map((c) => c.of(run, settings));
    const key = JSON.stringify(values);
    const group = groups.get(key) ?? { values, points: [] };
    group.points.push({ id, role, alpha, ...coefficients });
    groups.set(key, group);
  }
  const all = [...groups.entries()];
  // A condition names a series only where the series differ; the others are what is fixed.
  const varies = CONDITIONS.map((_, k) => new Set(all.map(([, g]) => g.values[k])).size > 1);
  // The mesh hash says nothing when it is shared: only a differing mesh is named.
  const words = (values: (string | null)[], differing: boolean) =>
    values
      .filter((v, k) => v !== null && varies[k] === differing && (differing || k > 0))
      .join(" · ");
  const series = all.map(([key, g]) => ({
    key,
    label: words(g.values, true),
    points: g.points.sort((a, b) => a.alpha - b.alpha || a.role.localeCompare(b.role)),
  }));
  series.sort((a, b) => a.label.localeCompare(b.label, "en", { numeric: true }));
  return { series, fixed: all[0] ? words(all[0][1].values, false) : "", unplotted };
}
