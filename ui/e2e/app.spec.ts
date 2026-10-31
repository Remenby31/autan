import { appendFileSync, cpSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { RUNS } from "../playwright.config";
import type { Run } from "../src/api/contracts";

async function runs(page: Page): Promise<Run[]> {
  return (await (await page.request.get("/api/runs")).json()).runs;
}

test("home lists every run directory and filters by state", async ({ page }) => {
  await page.goto("/");
  const rows = page.getByTestId("runs").locator("tbody tr");
  await expect(rows).toHaveCount(4);
  await page.getByRole("button", { name: /diverged/ }).click();
  await expect(rows).toHaveCount(1);
  await expect(page).toHaveURL(/status/);
  await expect(rows.first()).toContainText("D0107");
});

test("a run written by another process appears and updates live", async ({ page }) => {
  await page.goto("/");
  const rows = page.getByTestId("runs").locator("tbody tr");
  await expect(rows).toHaveCount(4);

  const source = (await runs(page))[0] as Run;
  const runId = "run-20261031T235959Z-0123abcd";
  const directory = join(RUNS, runId);
  const staging = join(RUNS, ".staging");
  cpSync(join(RUNS, source.run_id), staging, { recursive: true });
  const run: Run = {
    ...source,
    run_id: runId,
    status: "running",
    started: "2026-10-31T23:59:59Z",
    finished: null,
  };
  writeFileSync(join(staging, "run.json"), JSON.stringify(run));
  renameSync(staging, directory);
  await expect(rows).toHaveCount(5);
  await expect(rows.first().getByRole("img", { name: "running" })).toBeVisible();

  await rows.first().click();
  const last = readFileSync(join(directory, "history.jsonl"), "utf8").trim().split("\n").length;
  await expect(page.getByTestId("iteration")).toHaveText(String(last));
  appendFileSync(
    join(directory, "history.jsonl"),
    `${JSON.stringify({ iteration: last + 1, time: null, residuals: { density: 1e-9 } })}\n`,
  );
  await expect(page.getByTestId("iteration")).toHaveText(String(last + 1));
  const finished = { ...run, status: "converged", finished: "2026-11-01T00:10:00Z" };
  writeFileSync(join(directory, "run.json"), JSON.stringify(finished));
  await expect(page.getByTestId("status")).toContainText("converged");
});

test("⌘K opens a run by name and state", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("runs").locator("tbody tr").first()).toBeVisible();
  await page.keyboard.press("ControlOrMeta+k");
  await page.keyboard.type("diverged");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/runs\/run-/);
  await expect(page.getByTestId("status")).toContainText("diverged");
});

test("viewer renders on WebGPU and picks the zone under the cursor", async ({ page }) => {
  await page.goto("/viewer?triangles=500000");
  await expect(page.getByTestId("viewer-status")).toHaveAttribute("data-backend", "webgpu");
  const box = await page.getByTestId("viewport").boundingBox();
  if (!box) throw new Error("no viewport");
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await expect(page.getByRole("button", { pressed: true })).toHaveText(/^wing\./);
  await page.mouse.click(box.x + 20, box.y + box.height - 20);
  await expect(page.getByRole("button", { pressed: true })).toHaveCount(0);
});
