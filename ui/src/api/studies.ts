import type { Run, RunRegistry, RunSettings, RunSummary } from "./contracts";
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

/** The runs of a registry in their studies (results.md#studies), in registry order; a run
 * without a summary yet stands alone. `name` lists the case names its runs carry. */
export function groupByStudy(
  runs: Run[],
  summaries: Record<string, RunSummary> | undefined,
): { key: string; name: string; runs: Run[] }[] {
  const groups = new Map<string, Run[]>();
  for (const run of runs) {
    const key = summaries?.[run.run_id]?.study ?? run.run_id;
    groups.set(key, [...(groups.get(key) ?? []), run]);
  }
  return [...groups].map(([key, members]) => ({
    key,
    name: [...new Set(members.map((r) => r.case.name))].join(" · "),
    runs: members,
  }));
}

/** The studies of a registry that sweep a setting, newest run first, with their label
 * (`α 7.05–21.47 deg`); at most `limit` runs each. */
export function studies(
  registry: RunRegistry | undefined,
  limit = 8,
): { name: string; label: string; ids: string[] }[] {
  return groupByStudy(registry?.runs ?? [], registry?.summaries).flatMap(({ name, runs }) => {
    const label = study(runs.map((r) => registry?.summaries[r.run_id]?.settings));
    const ids = [...runs]
      .sort((a, b) => b.started.localeCompare(a.started))
      .slice(0, limit)
      .map((r) => r.run_id);
    return label && runs.length > 1 ? [{ name, label, ids }] : [];
  });
}
