import { readFileSync } from "node:fs";
import { expect, type Page, test } from "@playwright/test";

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

async function rule(page: Page, k: number, zones: string, kind: string) {
  await page.getByRole("button", { name: "Add rule" }).click();
  const fieldset = page.getByRole("group", { name: `Boundary rule ${k}` });
  await fieldset.getByLabel("Zones").fill(zones);
  await fieldset.getByLabel("Condition").selectOption(kind);
}

async function forces(page: Page, k: number, surfaces: string, area: string, every: string) {
  await page.getByRole("button", { name: "Add forces" }).click();
  const monitor = page.getByRole("group", { name: `Forces monitor ${k}` });
  await monitor.getByLabel("Surfaces").fill(surfaces);
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
