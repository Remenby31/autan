import { queryOptions } from "@tanstack/react-query";
import type { Diagnostic } from "./contracts";
import { globMatches } from "./glob";
import type { MeshInfo, Plan } from "./runs";

/** A MeshInfo the server lists for planning (`GET /api/meshes`). */
export interface MeshSummary {
  name: string;
  source: MeshInfo["source"];
  counts: MeshInfo["counts"];
  boundary_zones: { name: string; faces: number; type?: string | null }[];
  cell_zones: { name: string; cells: number }[];
  /** The mesh's zone groups (mesh-info.md#zone-groups): each its boundary zones, in order. */
  groups: { name: string; zones: string[] }[];
}

/** What a selection of boundary zones matches: their names, or the names of the mesh's zone
 * groups (case-schema.md#boundaries). */
export type Selector = "zones" | "group";

/** A material of the SDK's libraries (`GET /api/materials`): `autan.<library>.<name>`. */
export interface Material {
  library: "gases" | "materials";
  name: string;
  document: { kind: "ideal-gas" | "solid"; name: string } & Record<string, unknown>;
}

export const materialsQuery = queryOptions({
  queryKey: ["materials"],
  queryFn: async (): Promise<Material[]> => {
    const response = await fetch("/api/materials");
    if (!response.ok) throw new Error(`/api/materials: HTTP ${response.status}`);
    return response.json();
  },
  staleTime: Number.POSITIVE_INFINITY,
});

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
  | {
      ok: true;
      python: string;
      toml: string;
      /** The SDK's document of the case, every default filled in. */
      document: CaseDocument;
      plan: Plan | null;
      diagnostics: [];
    }
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
  by: Selector;
  /** A glob over zone names, or over group names when `by` is "group". */
  zones: string;
  kind: ConditionKind;
  values: Record<string, string>;
}

export interface CellZoneForm {
  zones: string;
  kind: "fluid" | "solid";
  /** A library material's name, e.g. CO2 or COPPER. */
  material: string;
}

export interface MonitorForm {
  /** Empty: "forces" for the first monitor, "forces-<k>" for the k-th. */
  name: string;
  by: Selector;
  surfaces: string;
  ref_area: string;
  every: string;
}

/** What the editor holds: every field as typed, empty when unset. */
export interface CaseForm {
  name: string;
  mesh: string;
  meshFile: string;
  physics: { kind: "compressible"; turbulence: string; precision: string; gas: string };
  freestream: Record<
    "mach" | "aoa" | "sideslip" | "re" | "ref_length" | "pressure" | "temperature",
    string
  >;
  time: { max_iterations: string; residual_orders: string };
  cellZones: CellZoneForm[];
  boundaries: BoundaryForm[];
  monitors: MonitorForm[];
}

export const EMPTY_CASE: CaseForm = {
  name: "",
  mesh: "",
  meshFile: "",
  physics: { kind: "compressible", turbulence: "", precision: "", gas: "" },
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
  cellZones: [],
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

/** The case document the form describes: only what was set, the SDK fills its defaults. A
 * material is the library document `materials` lists under its name. */
export function caseDocument(form: CaseForm, materials: Material[] = []): CaseDocument {
  // A name the library does not list (not loaded yet) goes as typed: the SDK refuses it with its
  // code rather than taking a default for it.
  const material = (name: string) => materials.find((m) => m.name === name)?.document ?? name;
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
      gas: text(form.physics.gas) && material(form.physics.gas),
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
    cell_zones: form.cellZones.length
      ? form.cellZones.map((zone) => ({
          zones: zone.zones,
          medium: sparse({ kind: zone.kind, material: material(zone.material) }),
        }))
      : undefined,
    boundaries: form.boundaries.length
      ? form.boundaries.map((rule) => ({
          [rule.by]: rule.zones,
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
            // A name given is kept verbatim: " loads " and "loads" are two monitors.
            name: m.name.trim() !== "" ? m.name : k === 0 ? undefined : `forces-${k + 1}`,
            [m.by === "group" ? "group" : "surfaces"]: m.surfaces,
            ref_area: quantity(m.ref_area),
            every: number(m.every),
          }),
        )
      : undefined,
  });
}

const str = (value: unknown) =>
  value === undefined || value === null ? "" : typeof value === "string" ? value : String(value);
const record = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
const list = (value: unknown): Record<string, unknown>[] =>
  Array.isArray(value) ? value.map(record) : [];

/** The form a case document reads as, as far as the form can say it; what it cannot (an
 * unsteady time, a condition or material outside the editor's choices) is found by comparing
 * the SDK's reading of both documents, not here. A material is named when it equals a
 * library one. */
export function formFromDocument(document: CaseDocument, materials: Material[]): CaseForm {
  const physics = record(document.physics);
  const freestream = record(document.freestream);
  const time = record(document.time);
  const libraryName = (value: unknown) =>
    materials.find((m) => sameValue(m.document, value))?.name ?? "";
  const kinds = Object.keys(CONDITIONS);
  return {
    name: str(document.name),
    mesh: "",
    meshFile: str(document.mesh),
    physics: {
      kind: "compressible",
      turbulence: str(record(physics.turbulence).model),
      precision: str(physics.precision),
      gas: physics.gas === undefined ? "" : libraryName(physics.gas),
    },
    freestream: {
      mach: str(freestream.mach),
      aoa: str(freestream.aoa),
      sideslip: str(freestream.sideslip),
      re: str(freestream.re),
      ref_length: str(freestream.ref_length),
      pressure: str(freestream.pressure),
      temperature: str(freestream.temperature),
    },
    time: {
      max_iterations: str(time.max_iterations),
      residual_orders: str(time.residual_orders),
    },
    cellZones: list(document.cell_zones).map((zone) => {
      const medium = record(zone.medium);
      return {
        zones: str(zone.zones),
        kind: medium.kind === "solid" ? "solid" : "fluid",
        material: libraryName(medium.material),
      };
    }),
    boundaries: list(document.boundaries).map((rule) => {
      const condition = record(rule.condition);
      const kind = (kinds.includes(str(condition.kind)) ? condition.kind : "wall") as ConditionKind;
      return {
        by: rule.group === undefined ? "zones" : "group",
        zones: str(rule.group ?? rule.zones),
        kind,
        values: Object.fromEntries(CONDITIONS[kind].map(({ key }) => [key, str(condition[key])])),
      };
    }),
    monitors: list(document.monitors)
      .filter((m) => m.kind === "forces")
      .map((m) => ({
        name: str(m.name),
        by: m.group === undefined ? "zones" : "group",
        surfaces: str(m.group ?? m.surfaces),
        ref_area: str(m.ref_area),
        every: str(m.every),
      })),
  };
}

/** The boundary zones a selection takes, as the SDK resolves it (`autan.plan.selected`): those
 * whose name matches the glob, or every zone of the groups whose name matches it. */
export function selectedZones(by: Selector, pattern: string, mesh: MeshSummary): Set<string> {
  if (by === "zones")
    return new Set(mesh.boundary_zones.map((z) => z.name).filter((n) => globMatches(pattern, n)));
  return new Set(mesh.groups.filter((g) => globMatches(pattern, g.name)).flatMap((g) => g.zones));
}

/** Whether two JSON values are equal, whatever the order of their keys. */
export function sameValue(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) || Array.isArray(b)) {
    return (
      Array.isArray(a) &&
      Array.isArray(b) &&
      a.length === b.length &&
      a.every((v, k) => sameValue(v, b[k]))
    );
  }
  if (a !== null && b !== null && typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a);
    const kb = Object.keys(b);
    return (
      ka.length === kb.length &&
      ka.every((k) =>
        sameValue((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]),
      )
    );
  }
  return a === b;
}

/** Where two SDK documents of a case differ, path by path (`physics.gas.gamma`,
 * `boundaries[2].condition.kind`), with the value in each. */
export function differences(
  a: unknown,
  b: unknown,
  path = "",
): { path: string; before: unknown; after: unknown }[] {
  const object = (v: unknown) => v !== null && typeof v === "object" && !Array.isArray(v);
  if (Array.isArray(a) && Array.isArray(b)) {
    const out = [];
    for (let k = 0; k < Math.max(a.length, b.length); k++)
      out.push(...differences(a[k], b[k], `${path}[${k}]`));
    return out;
  }
  if (object(a) && object(b)) {
    const ra = a as Record<string, unknown>;
    const rb = b as Record<string, unknown>;
    return [...new Set([...Object.keys(ra), ...Object.keys(rb)])]
      .sort()
      .flatMap((k) => differences(ra[k], rb[k], path ? `${path}.${k}` : k));
  }
  return sameValue(a, b) ? [] : [{ path, before: a, after: b }];
}
