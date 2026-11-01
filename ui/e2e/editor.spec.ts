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

test("the CRM-HL case built in the editor writes py/examples/crm_hl.py", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("a[data-row]").first()).toBeVisible();
  await page.keyboard.press("ControlOrMeta+k");
  await page.keyboard.type("new case");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/cases\/new$/);

  await page.getByLabel("Name").fill("crm-hl");
  await page.getByLabel("Mesh").selectOption("crm-hl");
  await page.getByLabel("Turbulence").selectOption("sa");
  await page.getByLabel("Mach").fill("0.2");
  await page.getByLabel("Angle of attack", { exact: true }).fill("19.57");
  await page.getByLabel("Reynolds number", { exact: true }).fill("5490000.0");
  await page.getByLabel("Reference length", { exact: true }).fill("7.0");
  await rule(page, 1, "wing.*", "wall");
  await rule(page, 2, "symmetry", "symmetry");
  await page.getByRole("button", { name: "Add forces" }).click();
  const forces = page.getByRole("group", { name: "Forces monitor 1" });
  await forces.getByLabel("Surfaces").fill("wing.*");
  await forces.getByLabel("Reference area", { exact: true }).fill("191.84");
  await forces.getByLabel("Every").fill("10");

  // Byte for byte: toHaveText would collapse whitespace.
  await expect.poll(() => page.getByTestId("case-py").textContent()).toBe(CRM_HL);
  await expect(page.getByTestId("plan")).toContainText("zones covered");
});

test("a case the SDK refuses shows its coded diagnostics", async ({ page }) => {
  await page.goto("/cases/new");
  await page.getByLabel("Name").fill("broken");
  await page.getByLabel("Mach").fill("-1");

  await expect(page.getByTestId("diagnostics")).toContainText(/E\d{4}/);
});
