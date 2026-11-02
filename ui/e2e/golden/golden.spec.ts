// The golden path: every page on what autan-run and autan-mesh-info themselves wrote. Expected
// values are read from those files, never written here: whatever the solver writes, the UI reads.
import { execFileSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import { GOLDEN } from "../../playwright.golden.config";
import type { Run } from "../../src/api/contracts";

const RUNS = join(GOLDEN, "runs");
const written: Run[] = readdirSync(RUNS)
  .map((dir) => JSON.parse(readFileSync(join(RUNS, dir, "run.json"), "utf8")) as Run)
  .sort((a, b) => a.started.localeCompare(b.started));
const history = (run: Run) =>
  readFileSync(join(RUNS, run.run_id, "history.jsonl"), "utf8")
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line) as { iteration: number; residuals: Record<string, number> });
// The angle of attack a run's stored case writes, in degrees ("6 deg").
const aoa = (run: Run) =>
  readFileSync(join(RUNS, run.run_id, "case.toml"), "utf8").match(/^aoa = "([\d.]+) deg"$/m)?.[1];
// The cells' values of a field of a run's solution.cgns, read with h5py: the producer's file,
// not the UI's reading of it.
const cellValues = (run: Run, field: string): number[] =>
  JSON.parse(
    execFileSync("uv", [
      ...["run", "--locked", "--project", "server", "python", "-c"],
      `import sys, json, h5py\nwith h5py.File(sys.argv[1]) as f: print(json.dumps(f["Base"]["Zone"]["FlowSolution"][sys.argv[2]][" data"][()].tolist()))`,
      join(RUNS, run.run_id, "solution.cgns"),
      field,
    ]).toString(),
  );
const four = (v: number | undefined) => (v ?? Number.NaN).toFixed(4).replace(/^-(0\.0+)$/, "$1");

test("the runs list and a run page show what the solver wrote", async ({ page }) => {
  const registry = await (await page.request.get("/api/runs")).json();
  expect(registry.rejected).toEqual([]);
  expect((registry.runs as Run[]).map((r) => r.run_id).sort()).toEqual(
    written.map((r) => r.run_id).sort(),
  );

  // Two angles of one case: one study.
  await page.goto("/");
  for (const run of written)
    await expect(page.locator(`a[data-run="${run.run_id}"]`)).toBeVisible();
  await expect(page.getByTestId("study")).toHaveCount(1);
  await expect(page.getByTestId("study")).toContainText("α 3–6 deg");

  const run = written[0] as Run;
  const lines = history(run);
  await page.goto(`/runs/${run.run_id}`);
  await expect(page.getByTestId("conditions")).toContainText("M 0.3");
  await expect(page.getByTestId("conditions")).toContainText(`α ${Number(aoa(run))}°`);
  await expect(page.getByTestId("status")).toContainText(
    run.status.charAt(0).toUpperCase() + run.status.slice(1),
  );
  for (const equation of Object.keys(lines.at(-1)?.residuals ?? {})) {
    await expect(page.getByText(equation, { exact: true }).first()).toBeVisible();
  }
  const forces = run.monitors?.forces;
  await expect(page.getByTestId("forces")).toContainText(`CL ${four(forces?.cl)}`);
  await expect(page.getByTestId("forces")).toContainText(`CD ${four(forces?.cd)}`);
});

test("Analysis reads the wall, a slice, a probe and the wall section", async ({ page }) => {
  const run = written[0] as Run;
  await page.goto(`/viewer?run=${run.run_id}`);
  // The walls are the case's wall and slip zones: ymin, slip.
  await expect(page.getByTestId("zones")).toHaveText(/^ymin/);
  // Inviscid: no skin friction to show.
  await expect(page.getByRole("tab", { name: "Cp" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Cf" })).toHaveCount(0);

  await page.goto(`/viewer?run=${run.run_id}&field=Cp&slice=z&at=0.5&sliceField=Pressure`);
  await expect(page.getByTestId("slice-legend")).toContainText("slice · Pressure");
  await expect(page.getByTestId("viewport")).toHaveAttribute("data-slice-field", "Pressure");
  // The plane z = 0.5 cuts the slip wall y = 0 along its chord.
  const chart = page.getByTestId("section-chart");
  await expect(chart).toHaveAttribute("aria-label", /^−Cp along the chord/);
  expect(Number(await chart.getAttribute("data-segments"))).toBeGreaterThan(0);

  const box = await page.getByTestId("viewport").boundingBox();
  if (!box) throw new Error("no viewport");
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.45);
  // On the slice, a probe reads one cell's pressure as the solver wrote it, to FP32 rounding.
  const probe = page.getByTestId("probe");
  await expect(probe).toContainText("slice ·");
  const read = await probe.getAttribute("data-value");
  expect(read).not.toBeNull();
  const value = Number(read);
  const pressures = cellValues(run, "Pressure");
  expect(pressures.some((p) => Math.abs(p - value) <= 4 * 2 ** -23 * Math.abs(p))).toBe(true);
});

test("Compare and the case editor read the same outputs", async ({ page }) => {
  const ids = encodeURIComponent(JSON.stringify(written.map((r) => r.run_id)));
  await page.goto(`/compare?runs=${ids}`);
  // Each run's drop is drawn with one point per line of its history.jsonl.
  const drop = page.getByTestId("chart-drop").locator("[data-samples]");
  for (const run of written) {
    await expect(drop).toHaveAttribute(
      "data-samples",
      new RegExp(`(^| )${run.run_id.slice(-8)}:${history(run).length}( |$)`),
    );
  }
  // Runs stopped at their budget are no polar points, and say why.
  for (const run of written.filter((r) => r.status !== "converged")) {
    await expect(page.getByTestId("polar-missing")).toContainText(`${run.status}, not converged`);
  }
  await expect(page.getByTestId("settings-diff")).toContainText("freestream.aoa");

  // The editor draws the mesh from the surface autan-mesh-info wrote, zones from its MeshInfo.
  const info = JSON.parse(readFileSync(join(GOLDEN, "meshes", "hybrid.meshinfo.json"), "utf8"));
  await page.goto("/cases/new");
  await page.getByLabel("Name").fill("golden");
  await page.getByLabel("Mesh", { exact: true }).selectOption("hybrid");
  await page.getByLabel("Turbulence").selectOption("inviscid");
  await page.getByLabel("Mach").fill("0.3");
  await page.getByLabel("Pressure", { exact: true }).fill("101325");
  await expect(page.getByTestId("mesh-view")).toHaveAttribute("data-ready", "true");
  await expect(page.getByLabel("Mesh", { exact: true })).toHaveValue("hybrid");
  const typed = (info.boundary_zones as { name: string; type: string | null }[]).filter(
    (z) => z.type,
  );
  for (const zone of typed)
    await expect(page.getByTestId("condition-legend")).toContainText(zone.name);
});
