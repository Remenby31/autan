import { queryOptions } from "@tanstack/react-query";
import type { Diagnostic } from "./contracts";
import type { MeshInfo, Plan } from "./runs";

/** A MeshInfo the server lists for planning (`GET /api/meshes`). */
export interface MeshSummary {
  name: string;
  source: MeshInfo["source"];
  counts: MeshInfo["counts"];
  boundary_zones: { name: string; faces: number; type?: string | null }[];
  cell_zones: { name: string; cells: number }[];
}

export const meshesQuery = queryOptions({
  queryKey: ["meshes"],
  queryFn: async (): Promise<MeshSummary[]> => {
    const response = await fetch("/api/meshes");
    if (!response.ok) throw new Error(`/api/meshes: HTTP ${response.status}`);
    return response.json();
  },
});

/** The SDK's reading of a case document (`POST /api/cases/render`). */
export type Rendered =
  | { ok: true; python: string; toml: string; plan: Plan | null; diagnostics: [] }
  | { ok: false; diagnostics: Diagnostic[] };

export const renderQuery = (document: CaseDocument, mesh: string | null) =>
  queryOptions({
    queryKey: ["render", document, mesh],
    queryFn: async ({ signal }): Promise<Rendered> => {
      const response = await fetch("/api/cases/render", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ case: document, mesh }),
        signal,
      });
      if (!response.ok) throw new Error(`/api/cases/render: HTTP ${response.status}`);
      return response.json();
    },
    placeholderData: (previous) => previous,
  });

export type CaseDocument = Record<string, unknown>;

/** Condition kinds the editor offers, with the quantities each needs (case-schema.md). */
export const CONDITIONS = {
  wall: [{ key: "temperature", dimension: "temperature", optional: true }],
  slip: [],
  symmetry: [],
  farfield: [],
  inlet: [
    { key: "total_pressure", dimension: "pressure", optional: false },
    { key: "total_temperature", dimension: "temperature", optional: false },
  ],
  outlet: [{ key: "static_pressure", dimension: "pressure", optional: false }],
} as const satisfies Record<
  string,
  readonly { key: string; dimension: Dimension; optional: boolean }[]
>;
export type ConditionKind = keyof typeof CONDITIONS;

/** Units a quantity of each dimension is written in, as the SDK reads them. */
export const UNITS = {
  angle: ["deg", "rad"],
  length: ["m", "mm", "cm", "in", "ft"],
  area: ["m ** 2", "mm ** 2", "in ** 2", "ft ** 2"],
  pressure: ["Pa", "kPa", "bar", "atm"],
  temperature: ["K"],
} as const;
export type Dimension = keyof typeof UNITS;

export interface BoundaryForm {
  zones: string;
  kind: ConditionKind;
  values: Record<string, string>;
}

export interface MonitorForm {
  surfaces: string;
  ref_area: string;
  every: string;
}

/** What the editor holds: every field as typed, empty when unset. */
export interface CaseForm {
  name: string;
  mesh: string;
  meshFile: string;
  physics: { kind: "compressible"; turbulence: string; precision: string };
  freestream: Record<
    "mach" | "aoa" | "sideslip" | "re" | "ref_length" | "pressure" | "temperature",
    string
  >;
  time: { max_iterations: string; residual_orders: string };
  boundaries: BoundaryForm[];
  monitors: MonitorForm[];
}

export const EMPTY_CASE: CaseForm = {
  name: "",
  mesh: "",
  meshFile: "",
  physics: { kind: "compressible", turbulence: "", precision: "" },
  freestream: {
    mach: "",
    aoa: "",
    sideslip: "",
    re: "",
    ref_length: "",
    pressure: "",
    temperature: "",
  },
  time: { max_iterations: "", residual_orders: "" },
  boundaries: [],
  monitors: [],
};

// A number the SDK reads, or the text as typed when it is not one, which the SDK refuses with
// its code; empty is unset.
const number = (text: string) => {
  if (text.trim() === "") return undefined;
  const value = Number(text);
  return Number.isFinite(value) ? value : text.trim();
};
const text = (value: string) => (value.trim() === "" ? undefined : value.trim());
// A quantity `"<number> <unit>"`, unset while its number is empty (its unit is kept).
const quantity = (value: string) => {
  const space = value.indexOf(" ");
  const magnitude = (space < 0 ? value : value.slice(0, space)).trim();
  return magnitude === "" ? undefined : `${magnitude} ${value.slice(space + 1).trim()}`;
};

function sparse(entries: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(entries).filter(([, v]) => v !== undefined));
}

/** The case document the form describes: only what was set, the SDK fills its defaults. */
export function caseDocument(form: CaseForm): CaseDocument {
  const fs = form.freestream;
  const time = sparse({
    max_iterations: number(form.time.max_iterations),
    residual_orders: number(form.time.residual_orders),
  });
  return sparse({
    schema: "autan.case/1",
    name: form.name,
    mesh: form.meshFile,
    physics: sparse({
      kind: form.physics.kind,
      turbulence: text(form.physics.turbulence) && { model: form.physics.turbulence },
      precision: text(form.physics.precision),
    }),
    freestream: sparse({
      mach: number(fs.mach),
      aoa: quantity(fs.aoa),
      sideslip: quantity(fs.sideslip),
      re: number(fs.re),
      ref_length: quantity(fs.ref_length),
      pressure: quantity(fs.pressure),
      temperature: quantity(fs.temperature),
    }),
    time: Object.keys(time).length ? { kind: "steady", ...time } : undefined,
    boundaries: form.boundaries.length
      ? form.boundaries.map((rule) => ({
          zones: rule.zones,
          condition: sparse({
            kind: rule.kind,
            ...Object.fromEntries(
              CONDITIONS[rule.kind].map(({ key }) => [key, quantity(rule.values[key] ?? "")]),
            ),
          }),
        }))
      : undefined,
    monitors: form.monitors.length
      ? form.monitors.map((m, k) =>
          sparse({
            kind: "forces",
            // Names are unique in a case; the first keeps the SDK's default, "forces".
            name: k === 0 ? undefined : `forces-${k + 1}`,
            surfaces: m.surfaces,
            ref_area: quantity(m.ref_area),
            every: number(m.every),
          }),
        )
      : undefined,
  });
}
