import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { RUNS } from "../playwright.config";

// py/examples/crm_hl.py without its module docstring and the blank line after it, as the
// SDK's own gate compares it with Case.to_python (py/tests/test_render.py).
function withoutDocstring(source: string): string {
  const lines = source.split(/(?<=\n)/);
  const end = lines.findIndex(
    (line, k) => (k === 0 ? line.trim().length > 6 : true) && line.trimEnd().endsWith('"""'),
  );
  return lines.slice(end + 2).join("");
}
const CRM_HL = withoutDocstring(readFileSync("../py/examples/crm_hl.py", "utf8"));
const HEAT_EXCHANGER = withoutDocstring(readFileSync("../py/examples/heat_exchanger.py", "utf8"));

async function rule(page: Page, k: number, zones: string, kind: string) {
  await page.getByRole("button", { name: "Add rule" }).click();
  const fieldset = page.getByRole("group", { name: `Boundary rule ${k}` });
  await fieldset.getByLabel("Zones", { exact: true }).fill(zones);
  await fieldset.getByLabel("Condition").selectOption(kind);
}

async function forces(page: Page, k: number, surfaces: string, area: string, every: string) {
  await page.getByRole("button", { name: "Add forces" }).click();
  const monitor = page.getByRole("group", { name: `Forces monitor ${k}` });
  await monitor.getByLabel("Surfaces", { exact: true }).fill(surfaces);
  await monitor.getByLabel("Reference area", { exact: true }).fill(area);
  await monitor.getByLabel("Every").fill(every);
}

// The CRM-HL case of py/examples/crm_hl.py, field by field.
async function crmHl(page: Page) {
  await page.getByLabel("Name").fill("crm-hl");
  await page.getByLabel("Mesh", { exact: true }).selectOption("crm-hl");
  await page.getByLabel("Turbulence").selectOption("sa");
  await page.getByLabel("Mach").fill("0.2");
  await page.getByLabel("Angle of attack", { exact: true }).fill("19.57");
  await page.getByLabel("Reynolds number", { exact: true }).fill("5490000.0");
  await page.getByLabel("Reference length", { exact: true }).fill("7.0");
  await rule(page, 1, "wing.*", "wall");
  await rule(page, 2, "symmetry", "symmetry");
  await forces(page, 1, "wing.*", "191.84", "10");
}

test("the CRM-HL case built in the editor writes py/examples/crm_hl.py", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("a[data-row]").first()).toBeVisible();
  await page.keyboard.press("ControlOrMeta+k");
  await page.keyboard.type("new case");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/cases\/new$/);

  await crmHl(page);

  // Byte for byte: toHaveText would collapse whitespace.
  await expect.poll(() => page.getByTestId("case-py").textContent()).toBe(CRM_HL);
  await expect(page.getByTestId("plan")).toContainText("ready");
  await expect(page.getByTestId("plan-diagnostics")).toBeEmpty();
});

test("a value the SDK refuses shows its code at its field", async ({ page }) => {
  await page.goto("/cases/new");
  await crmHl(page);
  await expect(page.getByTestId("plan")).toContainText("ready");

  await page.getByLabel("Mach").fill("-1");
  await expect(page.getByTestId("diagnostics")).toContainText("E0304 · freestream.mach");

  await page.getByLabel("Mach").fill("0.2");
  await page.getByLabel("Reynolds number", { exact: true }).fill("abc");
  await expect(page.getByTestId("diagnostics")).toContainText("E0301 · freestream.re");
});

test("a unit chosen first is kept, and monitors get names of their own", async ({ page }) => {
  await page.goto("/cases/new");
  await crmHl(page);
  await page.getByLabel("Reference length unit").selectOption("mm");
  await page.getByLabel("Reference length", { exact: true }).fill("");
  await page.getByLabel("Reference length", { exact: true }).fill("7000.0");
  await forces(page, 2, "wing.*", "191.84", "10");

  await expect(page.getByTestId("case-py")).toContainText("ref_length=7000.0 * mm");
  await expect(page.getByTestId("case-py")).toContainText('name="forces-2"');
  await expect(page.getByTestId("plan")).toContainText("ready");
});

test("errors leave with the rule that raised them", async ({ page }) => {
  await page.goto("/cases/new");
  await crmHl(page);
  await rule(page, 3, "wing.*", "slip");
  const errors = page.getByTestId("plan-diagnostics").getByText(/^E0411/);
  await expect(errors).toHaveCount(3); // wing.slat, wing.main and wing.flap, matched twice

  const zones = page.getByRole("group", { name: "Boundary rule 3" }).getByLabel("Zones");
  await zones.fill("missing");
  await expect(page.getByTestId("plan")).not.toContainText("checking");
  await zones.fill("symmetry");
  await expect(page.getByTestId("plan")).toContainText("refused");
  await expect(errors).toHaveCount(1); // symmetry alone

  await page.getByRole("button", { name: "Remove boundary rule 3" }).click();
  await expect(page.getByTestId("plan")).toContainText("ready");
  await expect(errors).toHaveCount(0);
});

test("a line longer than the code panel says so and scrolls to its end", async ({ page }) => {
  await page.goto("/cases/new");
  await crmHl(page);
  const code = page.getByTestId("case-py");
  await expect.poll(() => code.textContent()).toBe(CRM_HL);
  await expect(page.getByTestId("case-py-more")).toBeVisible();

  const end = await code.evaluate((el) => {
    el.scrollLeft = el.scrollWidth;
    el.dispatchEvent(new Event("scroll"));
    const aside = el.closest("aside")?.getBoundingClientRect();
    return {
      inside: aside !== undefined && el.getBoundingClientRect().right <= aside.right,
      atEnd: Math.abs(el.scrollLeft + el.clientWidth - el.scrollWidth) <= 1,
    };
  });
  expect(end).toEqual({ inside: true, atEnd: true });
  await expect(page.getByTestId("case-py-more")).toHaveCount(0);
});

const WEBGL = process.env.AUTAN_UI_E2E_BACKEND === "webgl2" ? "?webgl=1" : "";

// The default view looks at the box centre from direction (-0.7, -1, 0.75) (SurfaceViewer
// frameAll): the ray through the middle of the view enters the unit cube through ymin.
async function clickCentre(page: Page) {
  const view = page.getByTestId("mesh-view");
  await expect(view).toHaveAttribute("data-ready", "true");
  const box = await view.boundingBox();
  if (!box) throw new Error("the mesh view has no box");
  await view.click({ position: { x: box.width / 2, y: box.height / 2 } });
}

test("a zone clicked in 3D gets a boundary rule, its condition from the mesh", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  await clickCentre(page);

  // polyMesh.meshinfo.json types ymin as a wall.
  const rule = page.getByRole("group", { name: "Boundary rule 1" });
  await expect(rule.getByLabel("Zones")).toHaveValue("ymin");
  await expect(rule.getByLabel("Condition")).toHaveValue("wall");
  await expect(rule.getByLabel("Zones")).toBeFocused();

  await clickCentre(page);
  await expect(page.getByRole("group", { name: /^Boundary rule/ })).toHaveCount(1);
});

test("a zone a pattern already covers focuses that rule", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  await rule(page, 1, "*", "slip");
  await page.getByLabel("Name").focus();

  await clickCentre(page);

  await expect(page.getByRole("group", { name: /^Boundary rule/ })).toHaveCount(1);
  await expect(
    page.getByRole("group", { name: "Boundary rule 1" }).getByLabel("Zones"),
  ).toBeFocused();
});

test("picking keeps working on another mesh", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  await expect(page.getByTestId("mesh-view")).toHaveAttribute("data-ready", "true");
  await page.getByLabel("Mesh", { exact: true }).selectOption("hybrid-cgns");

  await clickCentre(page);

  // hybrid.cgns gives its zones no type: the suggestion is a wall.
  const rule = page.getByRole("group", { name: "Boundary rule 1" });
  await expect(rule.getByLabel("Zones")).toHaveValue("ymin");
  await expect(rule.getByLabel("Condition")).toHaveValue("wall");
});

test("a mesh without a surface draws nothing to pick", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  await expect(page.getByTestId("mesh-view")).toHaveAttribute("data-ready", "true");
  await page.getByLabel("Mesh", { exact: true }).selectOption("crm-hl");
  await expect(page.getByText("No boundary surface for this mesh")).toBeVisible();

  const view = page.getByTestId("mesh-view");
  const box = await view.boundingBox();
  if (!box) throw new Error("the mesh view has no box");
  await view.click({ position: { x: box.width / 2, y: box.height / 2 } });

  await expect(page.getByRole("group", { name: /^Boundary rule/ })).toHaveCount(0);
});

test("an inviscid case walls a picked wall with slip", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Turbulence").selectOption("inviscid");
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  await clickCentre(page);

  const rule = page.getByRole("group", { name: "Boundary rule 1" });
  await expect(rule.getByLabel("Zones")).toHaveValue("ymin");
  await expect(rule.getByLabel("Condition")).toHaveValue("slip");
});

async function cellZone(page: Page, k: number, zones: string, kind: string, material: string) {
  await page.getByRole("button", { name: "Add zone" }).click();
  const fieldset = page.getByRole("group", { name: `Cell zone ${k}` });
  await fieldset.getByLabel("Zones", { exact: true }).fill(zones);
  await fieldset.getByLabel("Medium").selectOption(kind);
  await fieldset.getByLabel("Material").selectOption(material);
}

async function quantity(page: Page, group: string, label: string, value: string, unit: string) {
  const fieldset = page.getByRole("group", { name: group });
  await fieldset.getByLabel(`${label} unit`).selectOption(unit);
  await fieldset.getByLabel(label, { exact: true }).fill(value);
}

test("the heat exchanger built in the editor writes py/examples/heat_exchanger.py", async ({
  page,
}) => {
  await page.goto("/cases/new");
  await page.getByLabel("Name").fill("heat-exchanger");
  await page.getByLabel("Mesh", { exact: true }).selectOption("heat-exchanger");
  await page.getByLabel("Turbulence").selectOption("sst");
  await page.getByLabel("Mach").fill("0.1");
  await page.getByLabel("Temperature", { exact: true }).fill("300.0");
  await page.getByLabel("Pressure", { exact: true }).fill("101325.0");
  await cellZone(page, 1, "co2", "fluid", "CO2");
  await cellZone(page, 2, "plate", "solid", "COPPER");
  for (const [k, zone, temperature] of [
    [1, "air-inlet", "400.0"],
    [2, "co2-inlet", "280.0"],
  ] as const) {
    await rule(page, k, zone, "inlet");
    await quantity(page, `Boundary rule ${k}`, "Total pressure", "103.0", "kPa");
    await quantity(page, `Boundary rule ${k}`, "Total temperature", temperature, "K");
  }
  await rule(page, 3, "*-outlet", "outlet");
  await quantity(page, "Boundary rule 3", "Static pressure", "101325.0", "Pa");

  await expect.poll(() => page.getByTestId("case-py").textContent()).toBe(HEAT_EXCHANGER);
  await expect(page.getByTestId("plan")).toContainText("ready");
});

test("gas and medium choices change the case and can be undone", async ({ page }) => {
  await page.goto("/cases/new");
  await crmHl(page);
  await page.getByLabel("Gas").selectOption("CO2");
  await expect(page.getByTestId("case-py")).toContainText("gas=autan.gases.CO2");
  await page.getByLabel("Gas").selectOption("");
  await expect.poll(() => page.getByTestId("case-py").textContent()).toBe(CRM_HL);

  await cellZone(page, 1, "fluid", "fluid", "CO2");
  const zone = page.getByRole("group", { name: "Cell zone 1" });
  await zone.getByLabel("Medium").selectOption("solid");
  await expect(zone.getByLabel("Material")).toHaveValue("");
  await zone.getByLabel("Material").selectOption("COPPER");
  await expect(page.getByTestId("case-py")).toContainText(
    'case.cell_zone("fluid").solid(autan.materials.COPPER)',
  );
});

test("material libraries that fail to load say so and can be retried", async ({ page }) => {
  let fail = true;
  await page.route("**/api/materials", (route) =>
    fail ? route.fulfill({ status: 503, body: "" }) : route.continue(),
  );
  await page.goto("/cases/new");
  await expect(page.getByText("The material libraries could not be listed")).toBeVisible({
    timeout: 15_000,
  });

  fail = false;
  await page.getByRole("button", { name: "Retry" }).click();
  await expect(page.getByText("The material libraries could not be listed")).toHaveCount(0);
  await expect(page.getByLabel("Gas").locator("option", { hasText: "co2" })).toHaveCount(1);
});

test("a copy of every sample run's case writes that case's SDK code", async ({ page }) => {
  const listed: { run_id: string; status: string }[] = (
    await (await page.request.get("/api/runs")).json()
  ).runs;
  expect(listed.length).toBeGreaterThan(0);
  for (const { run_id } of listed) {
    // The oracle: the SDK's code of the case the run stored, rendered by the server.
    const stored = (await (await page.request.get(`/api/runs/${run_id}`)).json()).case;
    const sdk = await (
      await page.request.post("/api/cases/render", { data: { case: stored } })
    ).json();
    expect(sdk.ok, run_id).toBe(true);

    await page.goto(`/runs/${run_id}`);
    await page.getByRole("button", { name: "Edit a copy" }).click();
    await expect(page).toHaveURL(new RegExp(`/cases/new\\?run=${run_id}`));
    await expect.poll(() => page.getByTestId("case-py").textContent()).toBe(sdk.python);
    await expect(page.getByText("Every setting of the run's case is in the form.")).toBeVisible();
  }
});

test("a copy of a case the form cannot say lists what it changes", async ({ page }) => {
  // A real run of a case with an unsteady time, which the editor does not offer yet, found by
  // its own case name and removed afterwards.
  const dir = mkdtempSync(join(tmpdir(), "autan-unsteady-"));
  const name = `unsteady-${Date.now()}`;
  writeFileSync(
    join(dir, "case.py"),
    readFileSync("../py/examples/crm_hl.py", "utf8").replace('"crm-hl"', `"${name}"`) +
      "case.time = autan.Unsteady(steps=3, dt=1e-3 * autan.units.s)\n",
  );
  execFileSync("uv", [
    ...["run", "--locked", "--project", "server", "autan", "run", join(dir, "case.py")],
    ...["--mesh-info", "../py/tests/fixtures/crm-hl.meshinfo.json", "--runs", RUNS],
  ]);
  const listed: { run_id: string; case: { name: string } }[] = (
    await (await page.request.get("/api/runs")).json()
  ).runs;
  const run = listed.find((r) => r.case.name === name);
  if (!run) throw new Error("the unsteady run is not listed");
  try {
    await page.goto(`/cases/new?run=${run.run_id}`);

    await expect(page.getByTestId("dropped")).toContainText('time.kind: "unsteady" → "steady"');
  } finally {
    rmSync(join(RUNS, run.run_id), { recursive: true, force: true });
    rmSync(dir, { recursive: true, force: true });
  }
});

test("the form waits for the copy, and a run it cannot read says so", async ({ page }) => {
  let release = () => {};
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/materials", async (route) => {
    await held;
    await route.continue();
  });
  const listed: { run_id: string }[] = (await (await page.request.get("/api/runs")).json()).runs;
  await page.goto(`/cases/new?run=${listed[0]?.run_id}`);
  // The case's own name, not a monitor's.
  const caseName = page
    .getByRole("group")
    .filter({ hasNotText: "Forces monitor" })
    .getByLabel("Name")
    .first();
  await expect(caseName).toBeDisabled();
  release();
  await expect(caseName).toBeEnabled();
  await expect(caseName).not.toHaveValue("");

  await page.goto("/cases/new?run=run-20000101T000000Z-00000000");
  await expect(
    page.getByText("The run run-20000101T000000Z-00000000 could not be read"),
  ).toBeVisible({
    timeout: 15_000,
  });
});

// server/tests/data/surface/grouped: the unit cube's polyMesh with the groups of
// ui/scripts/surface-fixtures.sh: plates = ymin, ymax; box.x = xmin, xmax; walls = ymin.
test("rules and monitors select the mesh's zone groups", async ({ page }) => {
  await page.goto("/cases/new");
  await page.getByLabel("Name").fill("box");
  await page.getByLabel("Mesh", { exact: true }).selectOption("grouped");
  await page.getByLabel("Turbulence").selectOption("inviscid");
  await page.getByLabel("Mach").fill("0.3");
  await page.getByLabel("Pressure", { exact: true }).fill("101325");
  await page.getByRole("button", { name: "Add rule" }).click();
  const plates = page.getByRole("group", { name: "Boundary rule 1" });
  await plates.getByLabel("Zones by").selectOption("group");
  await plates.getByLabel("Zones", { exact: true }).fill("plates");
  await plates.getByLabel("Condition").selectOption("symmetry");
  await rule(page, 2, "[xz]*", "farfield");
  await page.getByRole("button", { name: "Add forces" }).click();
  const monitor = page.getByRole("group", { name: "Forces monitor 1" });
  await monitor.getByLabel("Surfaces by").selectOption("group");
  await monitor.getByLabel("Surfaces", { exact: true }).fill("wall?");
  await monitor.getByLabel("Reference area", { exact: true }).fill("1");

  // case-schema.md#boundaries: a group glob is the `group` keyword.
  const code = page.getByTestId("case-py");
  await expect(code).toContainText('case.boundary(group="plates").symmetry()');
  await expect(code).toContainText('autan.forces(group="wall?"');
  await expect(page.getByTestId("plan")).toContainText("6 of 6 zones covered");

  await plates.getByLabel("Zones", { exact: true }).fill("plate");
  await expect(page.getByTestId("plan-diagnostics")).toContainText("E0426");
});

test("a picked zone's rule can select a group holding it instead", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Mesh", { exact: true }).selectOption("grouped");
  await clickCentre(page);

  const rule = page.getByRole("group", { name: "Boundary rule 1" });
  await expect(rule.getByLabel("Zones", { exact: true })).toHaveValue("ymin");
  const offer = page.getByTestId("group-offer");
  await expect(offer.getByRole("button")).toHaveText(["plates", "walls"]);
  await offer.getByRole("button", { name: "plates" }).click();

  await expect(rule.getByLabel("Zones by")).toHaveValue("group");
  await expect(rule.getByLabel("Zones", { exact: true })).toHaveValue("plates");
  await expect(offer).toHaveCount(0);
  await page.getByLabel("Name").focus();
  await clickCentre(page);
  await expect(page.getByRole("group", { name: /^Boundary rule/ })).toHaveCount(1);
  await expect(rule.getByLabel("Zones", { exact: true })).toBeFocused();
});

test("a group offer follows its rule and leaves with its mesh", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Mesh", { exact: true }).selectOption("grouped");
  await rule(page, 1, "xmin", "farfield");
  await clickCentre(page);
  const offer = page.getByTestId("group-offer");
  await expect(offer).toContainText("Rule 2");

  await page.getByRole("button", { name: "Remove boundary rule 1" }).click();
  await expect(offer).toContainText("Rule 1");
  await offer.getByRole("button", { name: "walls" }).click();
  const first = page.getByRole("group", { name: "Boundary rule 1" });
  await expect(first.getByLabel("Zones", { exact: true })).toHaveValue("walls");

  await page.getByRole("button", { name: "Add rule" }).click();
  await page.getByRole("button", { name: "Remove boundary rule 2" }).click();
  await clickCentre(page);
  await page.getByRole("button", { name: "Remove boundary rule 1" }).click();
  await clickCentre(page);
  await expect(offer).toBeVisible();
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  await expect(offer).toHaveCount(0);
});

test("dragging the view orbits it and picks nothing", async ({ page }) => {
  await page.goto(`/cases/new${WEBGL}`);
  await page.getByLabel("Mesh", { exact: true }).selectOption("polyMesh");
  const view = page.getByTestId("mesh-view");
  await expect(view).toHaveAttribute("data-ready", "true");
  const box = await view.boundingBox();
  if (!box) throw new Error("the mesh view has no box");
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;

  await page.mouse.move(x, y);
  await page.mouse.down();
  await page.mouse.move(x + 35, y + 20, { steps: 5 });
  await page.mouse.up();
  // Out and back: released where it was pressed, still a drag.
  await page.mouse.down();
  await page.mouse.move(x + 35, y + 20, { steps: 5 });
  await page.mouse.move(x, y, { steps: 5 });
  await page.mouse.up();

  await expect(page.getByRole("group", { name: /^Boundary rule/ })).toHaveCount(0);
  await clickCentre(page);
  await expect(page.getByRole("group", { name: /^Boundary rule/ })).toHaveCount(1);
});

test("a copy keeps its run's zone groups: same mesh hash, other groups is another mesh", async ({
  page,
}) => {
  // grouped and polyMesh are one mesh file; only grouped has groups. A real run on polyMesh.
  const dir = mkdtempSync(join(tmpdir(), "autan-groups-"));
  const name = `ungrouped-${Date.now()}`;
  writeFileSync(
    join(dir, "case.py"),
    [
      "import autan",
      "from autan.units import Pa",
      "",
      `case = autan.Case("${name}", mesh="polyMesh")`,
      "case.physics = autan.Compressible(turbulence=autan.Inviscid())",
      "case.freestream(mach=0.3, pressure=101325.0 * Pa)",
      'case.boundary("*").farfield()',
      "",
    ].join("\n"),
  );
  execFileSync("uv", [
    ...["run", "--locked", "--project", "server", "autan", "run", join(dir, "case.py")],
    ...["--mesh-info", "server/tests/data/surface/polyMesh.meshinfo.json", "--runs", RUNS],
  ]);
  const listed: { run_id: string; case: { name: string } }[] = (
    await (await page.request.get("/api/runs")).json()
  ).runs;
  const run = listed.find((r) => r.case.name === name);
  if (!run) throw new Error("the polyMesh run is not listed");
  try {
    await page.goto(`/cases/new?run=${run.run_id}`);

    await expect(page.getByLabel("Name")).toHaveValue(name);
    await expect(page.getByLabel("Mesh", { exact: true })).toHaveValue("polyMesh");
  } finally {
    rmSync(join(RUNS, run.run_id), { recursive: true, force: true });
    rmSync(dir, { recursive: true, force: true });
  }
});
