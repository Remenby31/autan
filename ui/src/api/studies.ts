import type { Run, RunRegistry, RunSettings } from "./contracts";
import { degrees } from "./units";

/** "α 7.05–21.47 deg" for the settings that vary across a group; null when none does. Angles
 * are compared in degrees, whatever unit each case wrote. */
export function study(settings: (RunSettings | undefined)[]): string | null {
  const parts: string[] = [];
  const labels = {
    aoa: "α",
    sideslip: "β",
    mach: "Mach",
    re: "Re",
    turbulence: "turbulence",
    precision: "precision",
  };
  for (const key of ["aoa", "sideslip", "mach", "re", "turbulence", "precision"] as const) {
    const raw = settings.map((s) => s?.[key]).filter((v) => v != null);
    const angle = key === "aoa" || key === "sideslip";
    const numbers = raw.map((v) => {
      const n = angle ? degrees(String(v)) : typeof v === "number" ? v : null;
      // Equal settings written in other units must not look like a sweep after conversion.
      return n === null ? null : Math.round(n * 1e9) / 1e9;
    });
    if (numbers.length && numbers.every((n) => n !== null)) {
      const distinct = [...new Set(numbers as number[])];
      if (distinct.length < 2) continue;
      const range = `${Math.min(...distinct)}–${Math.max(...distinct)}`;
      parts.push(`${labels[key]} ${range}${angle ? " deg" : ""}`);
    } else {
      const distinct = [...new Set(raw.map(String))];
      if (distinct.length > 1) parts.push(`${labels[key]} ${distinct.join(" · ")}`);
    }
  }
  return parts.length ? parts.join(", ") : null;
}

/** The studies of a registry: runs grouped by case name whose summaries vary in a swept
 * setting, newest run first, with their label (`α 7.05–21.47 deg`); at most `limit` runs each. */
export function studies(
  registry: RunRegistry | undefined,
  limit = 8,
): { name: string; label: string; ids: string[] }[] {
  const byCase = new Map<string, Run[]>();
  for (const run of registry?.runs ?? []) {
    const group = byCase.get(run.case.name);
    if (group) group.push(run);
    else byCase.set(run.case.name, [run]);
  }
  return [...byCase].flatMap(([name, runs]) => {
    const label = study(runs.map((r) => registry?.summaries[r.run_id]?.settings));
    const ids = [...runs]
      .sort((a, b) => b.started.localeCompare(a.started))
      .slice(0, limit)
      .map((r) => r.run_id);
    return label && runs.length > 1 ? [{ name, label, ids }] : [];
  });
}
