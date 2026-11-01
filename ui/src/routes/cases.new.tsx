import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { Plus, Trash2 } from "lucide-react";
import { type ReactNode, useDeferredValue, useEffect, useRef, useState } from "react";
import {
  type BoundaryForm,
  type CaseDocument,
  type CaseForm,
  type CellZoneForm,
  CONDITIONS,
  type ConditionKind,
  caseDocument,
  differences,
  EMPTY_CASE,
  formFromDocument,
  materialsQuery,
  meshesQuery,
  renderQuery,
  selectedZones,
  UNITS,
} from "../api/cases";
import type { Diagnostic } from "../api/contracts";
import { globEscape } from "../api/glob";
import { type Plan, runQuery } from "../api/runs";
import { MeshView } from "../components/case/MeshView";
import { TopBar } from "../components/shell/TopBar";
import { Button } from "../components/ui/Button";
import { cn } from "../components/ui/cn";
import {
  Field,
  NumberInput,
  QuantityInput,
  SelectInput,
  SelectorInput,
  TextInput,
} from "../components/ui/Field";
import { Note } from "../components/ui/Note";
import { TabList, TabPanel, TabsRoot } from "../components/ui/Tabs";

export const Route = createFileRoute("/cases/new")({
  // `webgl`: draw the mesh with WebGL2, as on CPU-only machines without WebGPU. `run`: start
  // from a copy of that run's case.
  validateSearch: (raw: Record<string, unknown>): { webgl?: boolean; run?: string } => ({
    ...(raw.webgl === true || raw.webgl === "1" ? { webgl: true } : {}),
    ...(typeof raw.run === "string" && raw.run ? { run: raw.run } : {}),
  }),
  component: CaseEditorRoute,
});

// A new editor for each source: a copy's values never outlive the run they came from.
function CaseEditorRoute() {
  const { run } = Route.useSearch();
  return <CaseEditor key={run ?? "new"} />;
}

const TURBULENCE = [
  { value: "inviscid", label: "Inviscid" },
  { value: "laminar", label: "Laminar" },
  { value: "sa", label: "Spalart–Allmaras" },
  { value: "sst", label: "k-ω SST" },
] as const;
const PRECISION = [
  { value: "fp32+acc64", label: "fp32 + fp64 accumulation" },
  { value: "fp64", label: "fp64" },
] as const;
const KINDS = Object.keys(CONDITIONS).map((kind) => ({
  value: kind as ConditionKind,
  label: kind,
}));
const LABEL: Record<string, string> = {
  temperature: "Wall temperature",
  total_pressure: "Total pressure",
  total_temperature: "Total temperature",
  static_pressure: "Static pressure",
};

/** A case built field by field against a mesh, read back by the SDK as it is typed. */
function CaseEditor() {
  const { webgl = false, run: from } = Route.useSearch();
  const meshes = useQuery(meshesQuery);
  const materials = useQuery(materialsQuery);
  const library = materials.data ?? [];
  const gases = library.filter((m) => m.document.kind === "ideal-gas");
  const solids = library.filter((m) => m.document.kind === "solid");
  const [form, setForm] = useState<CaseForm>(EMPTY_CASE);
  // A copy of a run's case: read once the library materials are known, its mesh chosen among
  // the listed MeshInfos by the run's mesh hash and groups hash: the same selectors on other
  // groups would select other zones.
  const source = useQuery({ ...runQuery(from ?? ""), enabled: from !== undefined });
  const [imported, setImported] = useState<CaseDocument | null>(null);
  useEffect(() => {
    if (!source.data || !materials.isSuccess || !meshes.isSuccess || imported) return;
    const copy = formFromDocument(source.data.case, materials.data);
    const { sha256, groups_sha256 } = source.data.run.mesh;
    const listed = meshes.data.find(
      (m) =>
        m.source.sha256 === sha256 && (m.groups_source?.sha256 ?? null) === (groups_sha256 ?? null),
    );
    setForm({ ...copy, mesh: listed?.name ?? "" });
    setImported(source.data.case);
  }, [source.data, materials, meshes, imported]);
  // What the copy cannot carry: the SDK's case.toml of the run's case against that of the form
  // it was read into; a line only in the first is a setting editing here would drop.
  const original = useQuery({ ...renderQuery(imported ?? {}, null), enabled: imported !== null });
  const reread = useQuery({
    ...renderQuery(
      imported ? caseDocument(formFromDocument(imported, library), library) : {},
      null,
    ),
    enabled: imported !== null,
  });
  // Settled only when both readings are in: a pending or failed comparison says so.
  const comparison: Comparison =
    imported === null
      ? { state: "none" }
      : original.isError || reread.isError
        ? { state: "failed", message: String(original.error ?? reread.error) }
        : !original.data || !reread.data
          ? { state: "checking" }
          : !original.data.ok
            ? { state: "failed", message: "the SDK refuses the run's own case" }
            : !reread.data.ok
              ? { state: "unrepresentable", diagnostics: reread.data.diagnostics }
              : {
                  state: "compared",
                  dropped: differences(original.data.document, reread.data.document),
                };
  const importing = from !== undefined && imported === null;
  const deferred = useDeferredValue(form);
  const [settled, setSettled] = useState(form);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(deferred), 250);
    return () => clearTimeout(timer);
  }, [deferred]);
  const rendered = useQuery(renderQuery(caseDocument(settled, library), settled.mesh || null));
  const mesh = meshes.data?.find((m) => m.name === form.mesh);
  const zones = mesh?.boundary_zones.map((z) => z.name) ?? [];
  const groups = mesh?.groups.map((g) => g.name) ?? [];
  const [picked, setPicked] = useState<string | null>(null);

  // A zone picked in 3D: the rule whose pattern matches it gets the focus, or a new rule for
  // it, its condition suggested by the type the mesh file gives the zone. A periodic zone takes
  // no condition (case-schema.md#boundaries): nothing is added.
  const [focusRule, setFocusRule] = useState<number | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // The groups holding a zone just picked into a new rule, which the rule may select instead.
  const [offer, setOffer] = useState<{ rule: number; zone: string; groups: string[] } | null>(null);
  const pick = (zone: string) => {
    setPicked(zone);
    setNotice(null);
    setOffer(null);
    const type = mesh?.boundary_zones.find((z) => z.name === zone)?.type ?? "";
    if (type === "periodic") {
      setNotice(
        `${zone} is periodic: it takes no condition, the solver pairs it with its partner.`,
      );
      return;
    }
    const index = mesh
      ? form.boundaries.findIndex((r) => selectedZones(r.by, r.zones, mesh).has(zone))
      : -1;
    if (index >= 0) {
      setFocusRule(index);
      return;
    }
    const typed: ConditionKind = type in CONDITIONS ? (type as ConditionKind) : "wall";
    // A no-slip wall needs viscous flow (E0414): an inviscid case walls it with slip.
    const kind = typed === "wall" && form.physics.turbulence === "inviscid" ? "slip" : typed;
    set("boundaries", [
      ...form.boundaries,
      { by: "zones", zones: globEscape(zone), kind, values: {} },
    ]);
    setFocusRule(form.boundaries.length);
    const holding = mesh?.groups.filter((g) => g.zones.includes(zone)).map((g) => g.name) ?? [];
    if (holding.length) setOffer({ rule: form.boundaries.length, zone, groups: holding });
  };
  // Focused once React has committed the rule's fieldset.
  useEffect(() => {
    if (focusRule === null) return;
    document
      .querySelector<HTMLInputElement>(
        `fieldset[aria-label="Boundary rule ${focusRule + 1}"] input`,
      )
      ?.focus();
    setFocusRule(null);
  }, [focusRule]);

  const set = <K extends keyof CaseForm>(key: K, value: CaseForm[K]) =>
    setForm((f) => ({ ...f, [key]: value }));
  const setIn = <K extends "physics" | "freestream" | "time">(
    key: K,
    field: keyof CaseForm[K],
    value: string,
  ) => setForm((f) => ({ ...f, [key]: { ...f[key], [field]: value } }));
  const cellZones = mesh?.cell_zones.map((z) => z.name) ?? [];
  const setCellZone = (k: number, zone: CellZoneForm) =>
    set(
      "cellZones",
      form.cellZones.map((z, i) => (i === k ? zone : z)),
    );
  const setRule = (k: number, rule: BoundaryForm) =>
    set(
      "boundaries",
      form.boundaries.map((r, i) => (i === k ? rule : r)),
    );

  return (
    <>
      <TopBar crumbs={[{ label: "Cases" }, { label: "New case" }]} />
      <div className="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_minmax(0,1fr)] overflow-hidden">
        <form
          className="flex flex-col gap-8 overflow-auto px-8 py-6"
          onSubmit={(e) => e.preventDefault()}
          aria-label="Case"
        >
          {source.isError && (
            <Note
              tone="error"
              title={`The run ${from} could not be read`}
              action={
                <Button size="sm" onClick={() => source.refetch()}>
                  Retry
                </Button>
              }
            >
              {source.error.message}
            </Note>
          )}
          {source.data && <ImportNote run={source.data.run.run_id} comparison={comparison} />}
          {materials.isError && (
            <Note
              tone="error"
              title="The material libraries could not be listed"
              action={
                <Button size="sm" onClick={() => materials.refetch()}>
                  Retry
                </Button>
              }
            >
              Gases and solids come from the SDK: {materials.error.message}
            </Note>
          )}
          {meshes.isError && (
            <Note tone="error" title="The meshes could not be listed">
              {meshes.error.message}
            </Note>
          )}
          <fieldset disabled={importing} className="m-0 flex flex-col gap-8 border-0 p-0">
            <Section title="Case">
              <Field label="Name">
                <TextInput value={form.name} onChange={(v) => set("name", v)} mono />
              </Field>
              <Field label="Mesh" hint={mesh && `${mesh.counts.cells.toLocaleString("en")} cells`}>
                <SelectInput
                  value={form.mesh}
                  onChange={(name) => {
                    const chosen = meshes.data?.find((m) => m.name === name);
                    setPicked(null);
                    setNotice(null);
                    setOffer(null);
                    setForm((f) => ({
                      ...f,
                      mesh: name,
                      meshFile: chosen ? (chosen.source.path.split("/").pop() ?? "") : "",
                    }));
                  }}
                  options={(meshes.data ?? []).map((m) => ({ value: m.name, label: m.name }))}
                  placeholder="Choose a MeshInfo…"
                />
              </Field>
            </Section>

            <Section title="Physics">
              <Field label="Turbulence">
                <SelectInput
                  value={form.physics.turbulence as (typeof TURBULENCE)[number]["value"] | ""}
                  onChange={(v) => setIn("physics", "turbulence", v)}
                  options={TURBULENCE}
                  placeholder="Choose…"
                />
              </Field>
              <Field label="Precision">
                <SelectInput
                  value={form.physics.precision as (typeof PRECISION)[number]["value"] | ""}
                  onChange={(v) => setIn("physics", "precision", v)}
                  options={PRECISION}
                  placeholder="Default (fp32 + fp64 accumulation)"
                />
              </Field>
              <Field label="Gas">
                <SelectInput
                  value={form.physics.gas}
                  onChange={(v) => setIn("physics", "gas", v)}
                  options={gases.map((m) => ({ value: m.name, label: m.document.name }))}
                  placeholder="Default (air)"
                />
              </Field>
            </Section>

            <Section
              title="Cell zones"
              action={
                <Button
                  size="sm"
                  icon={<Plus className="size-4" />}
                  onClick={() =>
                    set("cellZones", [
                      ...form.cellZones,
                      { zones: "", kind: "fluid", material: "" },
                    ])
                  }
                >
                  Add zone
                </Button>
              }
            >
              {form.cellZones.map((zone, k) => (
                <fieldset
                  // biome-ignore lint/suspicious/noArrayIndexKey: rules have no identity but their place
                  key={k}
                  aria-label={`Cell zone ${k + 1}`}
                  className="col-span-2 m-0 grid grid-cols-[minmax(0,1fr)_8rem_minmax(0,1fr)_auto] items-end gap-3 border-0 p-0"
                >
                  <Field label="Zones">
                    <TextInput
                      value={zone.zones}
                      onChange={(v) => setCellZone(k, { ...zone, zones: v })}
                      placeholder="plate"
                      mono
                      options={{ id: `cells-${k}`, values: cellZones }}
                    />
                  </Field>
                  <Field label="Medium">
                    <SelectInput
                      value={zone.kind}
                      onChange={(v) => v && setCellZone(k, { ...zone, kind: v, material: "" })}
                      options={[
                        { value: "fluid", label: "fluid" },
                        { value: "solid", label: "solid" },
                      ]}
                    />
                  </Field>
                  <Field label="Material">
                    <SelectInput
                      value={zone.material}
                      onChange={(v) => setCellZone(k, { ...zone, material: v })}
                      options={(zone.kind === "fluid" ? gases : solids).map((m) => ({
                        value: m.name,
                        label: m.document.name,
                      }))}
                      placeholder="Choose…"
                    />
                  </Field>
                  <Button
                    square
                    variant="tertiary"
                    aria-label={`Remove cell zone ${k + 1}`}
                    icon={<Trash2 className="size-4" />}
                    onClick={() =>
                      set(
                        "cellZones",
                        form.cellZones.filter((_, i) => i !== k),
                      )
                    }
                  />
                </fieldset>
              ))}
            </Section>

            <Section title="Freestream">
              <Field label="Mach">
                <NumberInput
                  value={form.freestream.mach}
                  onChange={(v) => setIn("freestream", "mach", v)}
                />
              </Field>
              <Field label="Angle of attack">
                <QuantityInput
                  value={form.freestream.aoa}
                  onChange={(v) => setIn("freestream", "aoa", v)}
                  units={UNITS.angle}
                />
              </Field>
              <Field label="Reynolds number" hint="per reference length">
                <NumberInput
                  value={form.freestream.re}
                  onChange={(v) => setIn("freestream", "re", v)}
                />
              </Field>
              <Field label="Reference length">
                <QuantityInput
                  value={form.freestream.ref_length}
                  onChange={(v) => setIn("freestream", "ref_length", v)}
                  units={UNITS.length}
                />
              </Field>
              <Field label="Pressure" hint="optional">
                <QuantityInput
                  value={form.freestream.pressure}
                  onChange={(v) => setIn("freestream", "pressure", v)}
                  units={UNITS.pressure}
                />
              </Field>
              <Field label="Temperature" hint="default 288.15 K">
                <QuantityInput
                  value={form.freestream.temperature}
                  onChange={(v) => setIn("freestream", "temperature", v)}
                  units={UNITS.temperature}
                />
              </Field>
            </Section>

            <Section title="Time">
              <Field label="Maximum iterations" hint="default 10 000">
                <NumberInput
                  value={form.time.max_iterations}
                  onChange={(v) => setIn("time", "max_iterations", v)}
                />
              </Field>
              <Field label="Residual drop (orders)" hint="default 8">
                <NumberInput
                  value={form.time.residual_orders}
                  onChange={(v) => setIn("time", "residual_orders", v)}
                />
              </Field>
            </Section>

            <Section
              title="Boundaries"
              action={
                <Button
                  size="sm"
                  icon={<Plus className="size-4" />}
                  onClick={() =>
                    set("boundaries", [
                      ...form.boundaries,
                      { by: "zones", zones: "", kind: "wall", values: {} },
                    ])
                  }
                >
                  Add rule
                </Button>
              }
            >
              {form.boundaries.map((rule, k) => (
                <fieldset
                  // biome-ignore lint/suspicious/noArrayIndexKey: rules have no identity but their place
                  key={k}
                  aria-label={`Boundary rule ${k + 1}`}
                  className="col-span-2 m-0 grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] items-end gap-3 border-0 p-0"
                >
                  <Field label="Zones">
                    <SelectorInput
                      by={rule.by}
                      pattern={rule.zones}
                      onChange={(by, v) => setRule(k, { ...rule, by, zones: v })}
                      placeholder={rule.by === "group" ? "WING" : "wing.*"}
                      names={{ id: `zones-${k}`, zones, groups }}
                    />
                  </Field>
                  <Field label="Condition">
                    <SelectInput
                      value={rule.kind}
                      onChange={(v) => v && setRule(k, { ...rule, kind: v, values: {} })}
                      options={KINDS}
                    />
                  </Field>
                  <Button
                    square
                    variant="tertiary"
                    aria-label={`Remove boundary rule ${k + 1}`}
                    icon={<Trash2 className="size-4" />}
                    onClick={() => {
                      set(
                        "boundaries",
                        form.boundaries.filter((_, i) => i !== k),
                      );
                      setOffer((o) =>
                        o === null || o.rule === k
                          ? null
                          : { ...o, rule: o.rule > k ? o.rule - 1 : o.rule },
                      );
                    }}
                  />
                  {CONDITIONS[rule.kind].map(({ key, dimension, optional }) => (
                    <Field
                      key={key}
                      label={LABEL[key] ?? key}
                      hint={optional ? "optional" : undefined}
                    >
                      <QuantityInput
                        value={rule.values[key] ?? ""}
                        onChange={(v) =>
                          setRule(k, { ...rule, values: { ...rule.values, [key]: v } })
                        }
                        units={UNITS[dimension]}
                      />
                    </Field>
                  ))}
                </fieldset>
              ))}
            </Section>

            <Section
              title="Monitors"
              action={
                <Button
                  size="sm"
                  icon={<Plus className="size-4" />}
                  onClick={() =>
                    set("monitors", [
                      ...form.monitors,
                      { name: "", by: "zones", surfaces: "", ref_area: "", every: "" },
                    ])
                  }
                >
                  Add forces
                </Button>
              }
            >
              {form.monitors.map((monitor, k) => (
                <fieldset
                  // biome-ignore lint/suspicious/noArrayIndexKey: monitors have no identity but their place
                  key={k}
                  aria-label={`Forces monitor ${k + 1}`}
                  className="col-span-2 m-0 grid grid-cols-[8rem_minmax(0,1fr)_minmax(0,1fr)_6rem_auto] items-end gap-3 border-0 p-0"
                >
                  <Field label="Name">
                    <TextInput
                      value={monitor.name}
                      onChange={(v) =>
                        set(
                          "monitors",
                          form.monitors.map((m, i) => (i === k ? { ...m, name: v } : m)),
                        )
                      }
                      placeholder={k === 0 ? "forces" : `forces-${k + 1}`}
                      mono
                    />
                  </Field>
                  <Field label="Surfaces">
                    <SelectorInput
                      by={monitor.by}
                      pattern={monitor.surfaces}
                      onChange={(by, v) =>
                        set(
                          "monitors",
                          form.monitors.map((m, i) => (i === k ? { ...m, by, surfaces: v } : m)),
                        )
                      }
                      placeholder={monitor.by === "group" ? "WING" : "wing.*"}
                      names={{ id: `surfaces-${k}`, zones, groups }}
                    />
                  </Field>
                  <Field label="Reference area">
                    <QuantityInput
                      value={monitor.ref_area}
                      onChange={(v) =>
                        set(
                          "monitors",
                          form.monitors.map((m, i) => (i === k ? { ...m, ref_area: v } : m)),
                        )
                      }
                      units={UNITS.area}
                    />
                  </Field>
                  <Field label="Every" hint="iterations">
                    <NumberInput
                      value={monitor.every}
                      onChange={(v) =>
                        set(
                          "monitors",
                          form.monitors.map((m, i) => (i === k ? { ...m, every: v } : m)),
                        )
                      }
                    />
                  </Field>
                  <Button
                    square
                    variant="tertiary"
                    aria-label={`Remove forces monitor ${k + 1}`}
                    icon={<Trash2 className="size-4" />}
                    onClick={() =>
                      set(
                        "monitors",
                        form.monitors.filter((_, i) => i !== k),
                      )
                    }
                  />
                </fieldset>
              ))}
            </Section>
          </fieldset>
        </form>

        <aside className="flex min-h-0 min-w-0 flex-col gap-4 overflow-y-auto border-line border-l bg-bg-1 px-6 py-6">
          {form.mesh && <MeshView mesh={form.mesh} selected={picked} onPick={pick} webgl={webgl} />}
          {notice && (
            <Note tone="info" title="Periodic zone">
              {notice}
            </Note>
          )}
          {offer && form.boundaries[offer.rule]?.zones === globEscape(offer.zone) && (
            <div
              data-testid="group-offer"
              className="flex flex-wrap items-center gap-2 text-fg-2 text-sm"
            >
              <span>
                Rule {offer.rule + 1} could select a group holding{" "}
                <span className="font-mono">{offer.zone}</span>:
              </span>
              {offer.groups.map((g) => (
                <Button
                  key={g}
                  size="sm"
                  onClick={() => {
                    setForm((f) => ({
                      ...f,
                      boundaries: f.boundaries.map((r, i) =>
                        i === offer.rule ? { ...r, by: "group", zones: globEscape(g) } : r,
                      ),
                    }));
                    setOffer(null);
                  }}
                >
                  <span className="font-mono">{g}</span>
                </Button>
              ))}
            </div>
          )}
          {rendered.isError && (
            <Note tone="error" title="The SDK could not be reached">
              The code below is from an earlier version of the case: {rendered.error.message}
            </Note>
          )}
          {rendered.data?.ok === false ? (
            <Diagnostics items={rendered.data.diagnostics} testId="diagnostics" />
          ) : (
            <TabsRoot
              defaultValue="python"
              className={cn(
                "flex min-h-0 min-w-0 flex-col gap-3 transition-opacity",
                (rendered.isError || rendered.isPlaceholderData) && "opacity-60",
              )}
            >
              <div className="flex items-center gap-3">
                <TabList
                  tabs={[
                    { value: "python", label: <span className="font-mono">case.py</span> },
                    { value: "toml", label: <span className="font-mono">case.toml</span> },
                  ]}
                />
                <span className="text-fg-3 text-sm">written by the SDK</span>
              </div>
              <TabPanel value="python" className="min-w-0 outline-none">
                <Code testId="case-py" text={rendered.data?.ok ? rendered.data.python : ""} />
              </TabPanel>
              <TabPanel value="toml" className="min-w-0 outline-none">
                <Code testId="case-toml" text={rendered.data?.ok ? rendered.data.toml : ""} />
              </TabPanel>
            </TabsRoot>
          )}
          {rendered.data?.ok && rendered.data.plan && (
            <PlanSummary plan={rendered.data.plan} current={!rendered.isPlaceholderData} />
          )}
        </aside>
      </div>
    </>
  );
}

// Lines never wrap: the code's own line breaks are part of what it shows. A line longer than the
// panel scrolls, and a fade on the right edge says so until its end is in view (overlay
// scrollbars stay hidden until a reader scrolls).
type Comparison =
  | { state: "none" }
  | { state: "checking" }
  | { state: "failed"; message: string }
  | { state: "unrepresentable"; diagnostics: Diagnostic[] }
  | { state: "compared"; dropped: ReturnType<typeof differences> };

const show = (value: unknown) => (value === undefined ? "absent" : JSON.stringify(value));

function ImportNote({ run, comparison }: { run: string; comparison: Comparison }) {
  const title = `A copy of ${run}`;
  if (comparison.state === "none" || comparison.state === "checking")
    return (
      <Note tone="info" title={title}>
        Comparing the copy with the run's case…
      </Note>
    );
  if (comparison.state === "failed")
    return (
      <Note tone="error" title={title}>
        The copy could not be compared with the run's case: {comparison.message}
      </Note>
    );
  if (comparison.state === "unrepresentable")
    return (
      <Note tone="warning" title={title}>
        The form cannot say this case yet; read into it, the case is refused:
        <Diagnostics items={comparison.diagnostics} testId="unrepresentable" />
      </Note>
    );
  if (comparison.dropped.length === 0)
    return (
      <Note tone="info" title={title}>
        Every setting of the run's case is in the form.
      </Note>
    );
  return (
    <Note tone="warning" title={title}>
      The form cannot show these settings of the run's case yet; this copy changes them:
      <ul className="m-0 mt-2 p-0 font-mono text-sm" data-testid="dropped">
        {comparison.dropped.map((d) => (
          <li key={d.path} className="list-none">
            {d.path}: {show(d.before)} → {show(d.after)}
          </li>
        ))}
      </ul>
    </Note>
  );
}

function Code({ text, testId }: { text: string; testId: string }) {
  const ref = useRef<HTMLPreElement>(null);
  const [more, setMore] = useState(false);
  const measure = (el: HTMLElement) => setMore(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
  // biome-ignore lint/correctness/useExhaustiveDependencies: measured again when the text changes
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    measure(el);
    const resize = new ResizeObserver(() => measure(el));
    resize.observe(el);
    return () => resize.disconnect();
  }, [text]);
  return (
    <div className="relative min-w-0">
      <pre
        ref={ref}
        onScroll={(e) => measure(e.currentTarget)}
        className="m-0 max-w-full overflow-x-auto whitespace-pre rounded-lg border border-line bg-bg-0 p-4 font-mono text-sm leading-6"
        data-testid={testId}
      >
        {text}
      </pre>
      {more && (
        <span
          aria-hidden
          data-testid={`${testId}-more`}
          className="pointer-events-none absolute inset-y-px right-px w-12 rounded-r-lg bg-gradient-to-l from-bg-0 to-transparent"
        />
      )}
    </div>
  );
}

function Section({
  title,
  action,
  children,
}: {
  title: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h2 className="m-0 font-medium text-base">{title}</h2>
        {action}
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-4">{children}</div>
    </section>
  );
}

// `current`: the plan is of the case as it now reads; otherwise it is being checked again.
function PlanSummary({ plan, current }: { plan: Plan; current: boolean }) {
  const uncovered = plan.boundaries.filter((b) => b.condition === null);
  return (
    <section className="flex flex-col gap-2" data-testid="plan">
      <h2 className="m-0 flex items-baseline gap-2 font-medium text-base">
        Plan
        {current ? (
          <span className={plan.ok ? "text-success" : "text-danger"}>
            {plan.ok ? "ready" : "refused"}
          </span>
        ) : (
          <span className="text-fg-3">checking…</span>
        )}
        <span className="font-normal text-fg-3 text-sm">
          {plan.boundaries.length - uncovered.length} of {plan.boundaries.length} zones covered
        </span>
      </h2>
      {uncovered.length > 0 && (
        <p className="m-0 font-mono text-fg-2 text-sm">
          {uncovered.map((b) => b.zone).join(" · ")}
        </p>
      )}
      <p className="m-0 text-fg-3 text-sm">
        Planned against the MeshInfo; the mesh file itself is read when the case runs.
      </p>
      <Diagnostics
        // The editor plans without the mesh file and its groups file by design: W0205 and W0211
        // say so of every case.
        items={plan.diagnostics.filter(
          (d) => !((d.code === "W0205" || d.code === "W0211") && d.location?.path === "mesh"),
        )}
        testId="plan-diagnostics"
      />
    </section>
  );
}

function Diagnostics({ items, testId }: { items: Diagnostic[]; testId: string }) {
  return (
    <div className="flex flex-col gap-2" data-testid={testId}>
      {items.map((d, k) => (
        <Note
          // biome-ignore lint/suspicious/noArrayIndexKey: diagnostics repeat; their place is their identity
          key={k}
          tone={d.severity === "warning" ? "warning" : "error"}
          title={d.location?.path ? `${d.code} · ${d.location.path}` : d.code}
        >
          {d.message}
        </Note>
      ))}
    </div>
  );
}
