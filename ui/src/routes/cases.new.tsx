import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { Plus, Trash2 } from "lucide-react";
import { type ReactNode, useDeferredValue, useEffect, useState } from "react";
import {
  type BoundaryForm,
  type CaseForm,
  CONDITIONS,
  type ConditionKind,
  caseDocument,
  EMPTY_CASE,
  meshesQuery,
  renderQuery,
  UNITS,
} from "../api/cases";
import type { Plan } from "../api/runs";
import { TopBar } from "../components/shell/TopBar";
import { Button } from "../components/ui/Button";
import { Field, NumberInput, QuantityInput, SelectInput, TextInput } from "../components/ui/Field";
import { Note } from "../components/ui/Note";

export const Route = createFileRoute("/cases/new")({ component: CaseEditor });

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
  const meshes = useQuery(meshesQuery);
  const [form, setForm] = useState<CaseForm>(EMPTY_CASE);
  const deferred = useDeferredValue(form);
  const [settled, setSettled] = useState(form);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(deferred), 250);
    return () => clearTimeout(timer);
  }, [deferred]);
  const rendered = useQuery(renderQuery(caseDocument(settled), settled.mesh || null));
  const mesh = meshes.data?.find((m) => m.name === form.mesh);
  const zones = mesh?.boundary_zones.map((z) => z.name) ?? [];

  const set = <K extends keyof CaseForm>(key: K, value: CaseForm[K]) =>
    setForm((f) => ({ ...f, [key]: value }));
  const setIn = <K extends "physics" | "freestream" | "time">(
    key: K,
    field: keyof CaseForm[K],
    value: string,
  ) => setForm((f) => ({ ...f, [key]: { ...f[key], [field]: value } }));
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
          <Section title="Case">
            <Field label="Name">
              <TextInput value={form.name} onChange={(v) => set("name", v)} mono />
            </Field>
            <Field label="Mesh" hint={mesh && `${mesh.counts.cells.toLocaleString("en")} cells`}>
              <SelectInput
                value={form.mesh}
                onChange={(name) => {
                  const chosen = meshes.data?.find((m) => m.name === name);
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
                  set("boundaries", [...form.boundaries, { zones: "", kind: "wall", values: {} }])
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
                  <TextInput
                    value={rule.zones}
                    onChange={(v) => setRule(k, { ...rule, zones: v })}
                    placeholder="wing.*"
                    mono
                    options={{ id: `zones-${k}`, values: zones }}
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
                  onClick={() =>
                    set(
                      "boundaries",
                      form.boundaries.filter((_, i) => i !== k),
                    )
                  }
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
                  set("monitors", [...form.monitors, { surfaces: "", ref_area: "", every: "" }])
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
                className="col-span-2 m-0 grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_8rem] items-end gap-3 border-0 p-0"
              >
                <Field label="Surfaces">
                  <TextInput
                    value={monitor.surfaces}
                    onChange={(v) =>
                      set(
                        "monitors",
                        form.monitors.map((m, i) => (i === k ? { ...m, surfaces: v } : m)),
                      )
                    }
                    placeholder="wing.*"
                    mono
                    options={{ id: `surfaces-${k}`, values: zones }}
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
              </fieldset>
            ))}
          </Section>
        </form>

        <aside className="flex min-h-0 flex-col gap-4 overflow-auto border-line border-l bg-bg-1 px-6 py-6">
          <div className="flex items-center gap-2">
            <h2 className="m-0 font-medium text-base">case.toml</h2>
            <span className="text-fg-3 text-sm">as the SDK reads it</span>
          </div>
          {rendered.data?.ok === false ? (
            <div className="flex flex-col gap-2" data-testid="diagnostics">
              {rendered.data.diagnostics.map((d) => (
                <Note
                  key={`${d.code}-${d.location?.path ?? ""}`}
                  tone={d.severity === "warning" ? "warning" : "error"}
                  title={d.code}
                >
                  {d.message}
                </Note>
              ))}
            </div>
          ) : (
            <pre
              className="m-0 overflow-auto rounded-lg border border-line bg-bg-0 p-4 font-mono text-sm leading-6"
              data-testid="case-toml"
            >
              {rendered.data?.ok ? rendered.data.toml : ""}
            </pre>
          )}
          {rendered.data?.ok && rendered.data.plan && <PlanSummary plan={rendered.data.plan} />}
        </aside>
      </div>
    </>
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

function PlanSummary({ plan }: { plan: Plan }) {
  const uncovered = plan.boundaries.filter((b) => b.condition === null);
  return (
    <section className="flex flex-col gap-2" data-testid="plan">
      <h2 className="m-0 font-medium text-base">
        Plan{" "}
        <span className={plan.ok ? "text-success" : "text-danger"}>
          {plan.boundaries.length - uncovered.length} of {plan.boundaries.length} zones covered
        </span>
      </h2>
      {uncovered.length > 0 && (
        <p className="m-0 font-mono text-fg-2 text-sm">
          {uncovered.map((b) => b.zone).join(" · ")}
        </p>
      )}
    </section>
  );
}
