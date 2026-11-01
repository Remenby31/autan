import { execFileSync } from "node:child_process";
import { expect, type Page, test } from "@playwright/test";

// The SDK's own reading of py/examples/crm_hl.py: what the editor must produce.
const CRM_HL_TOML = execFileSync(
  "uv",
  [
    "run",
    "--locked",
    "--project",
    "server",
    "python",
    "-c",
    "import autan; print(autan.load('../py/examples/crm_hl.py').to_toml(), end='')",
  ],
  { encoding: "utf8" },
);

async function rule(page: Page, k: number, zones: string, kind: string) {
  await page.getByRole("button", { name: "Add rule" }).click();
  const fieldset = page.getByRole("group", { name: `Boundary rule ${k}` });
  await fieldset.getByLabel("Zones").fill(zones);
  await fieldset.getByLabel("Condition").selectOption(kind);
}

test("the CRM-HL case built in the editor is the SDK's crm_hl.py", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("a[data-row]").first()).toBeVisible();
  await page.keyboard.press("ControlOrMeta+k");
  await page.keyboard.type("new case");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/cases\/new$/);

  await page.getByLabel("Name").fill("crm-hl");
  await page.getByLabel("Mesh").selectOption("crm-hl");
  await page.getByLabel("Turbulence").selectOption("sa");
  await page.getByLabel("Precision").selectOption("fp32+acc64");
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

  await expect(page.getByTestId("case-toml")).toHaveText(CRM_HL_TOML);
  await expect(page.getByTestId("plan")).toContainText("zones covered");
});

test("a case the SDK refuses shows its coded diagnostics", async ({ page }) => {
  await page.goto("/cases/new");
  await page.getByLabel("Name").fill("broken");
  await page.getByLabel("Mach").fill("-1");

  await expect(page.getByTestId("diagnostics")).toContainText(/E\d{4}/);
});
