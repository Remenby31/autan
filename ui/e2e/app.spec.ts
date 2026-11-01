import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  appendFileSync,
  cpSync,
  mkdtempSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { expect, type Page, test } from "@playwright/test";
import { RUNS } from "../playwright.config";
import type { Run } from "../src/api/contracts";

async function runs(page: Page): Promise<Run[]> {
  return (await (await page.request.get("/api/runs")).json()).runs;
}

test("home lists every run directory and filters by state", async ({ page }) => {
  await page.goto("/");
  const rows = page.locator("a[data-row]");
  await expect(rows).toHaveCount(8);
  await page.getByRole("button", { name: "State" }).click();
  await page.getByRole("menuitemcheckbox", { name: /diverged/ }).click();
  await page.keyboard.press("Escape");
  await expect(rows).toHaveCount(1);
  await expect(page).toHaveURL(/status/);
  await expect(rows.first()).toContainText("D0107");
});

test("a run written by another process appears and updates live", async ({ page }) => {
  await page.goto("/");
  const rows = page.locator("a[data-row]");
  await expect(rows).toHaveCount(8);

  const source = (await runs(page))[0] as Run;
  const started = new Date(Date.now() + 60_000).toISOString().replace(/\.\d+Z$/, "Z");
  const runId = `run-${started.replace(/[-:]/g, "")}-0123abcd`;
  const directory = join(RUNS, runId);
  const staging = join(RUNS, ".staging");
  cpSync(join(RUNS, source.run_id), staging, { recursive: true });
  const run: Run = {
    ...source,
    run_id: runId,
    status: "running",
    started,
    finished: null,
    monitors: null,
  };
  writeFileSync(join(staging, "run.json"), JSON.stringify(run));
  renameSync(staging, directory);
  await expect(rows).toHaveCount(9);
  await expect(rows.first().getByRole("img", { name: "running" })).toBeVisible();

  const history = join(directory, "history.jsonl");
  const lines = readFileSync(history, "utf8").trim().split("\n");
  const last = lines.length;
  const append = (iteration: number) =>
    appendFileSync(
      history,
      `${JSON.stringify({ ...JSON.parse(lines[last - 1] as string), iteration })}\n`,
    );
  await expect(rows.first().getByTestId("iterations")).toHaveText(String(last));
  append(last + 1);
  await expect(rows.first().getByTestId("iterations")).toHaveText(String(last + 1));

  await rows.first().click();
  await expect(page.getByTestId("iteration")).toHaveText(String(last + 1));
  const budget = page.getByRole("meter", { name: "Iteration budget used" });
  await expect(budget).toBeVisible();
  append(last + 2);
  await expect(page.getByTestId("iteration")).toHaveText(String(last + 2));
  // The final record of every monitor (results.md#runjson), equal to the last line where it
  // carries one: the appended line repeats the last.
  const records = {
    ...source.monitors,
    ...JSON.parse(lines[last - 1] as string).monitors,
  };
  const monitors = Object.fromEntries(
    Object.entries(records).map(([name, record]) => [
      name,
      { ...(record as object), iteration: last + 2 },
    ]),
  );
  const finished = {
    ...run,
    status: "converged",
    finished: "2026-11-01T00:10:00Z",
    iterations: last + 2,
    monitors,
  };
  writeFileSync(join(directory, "run.json"), JSON.stringify(finished));
  await expect(page.getByTestId("status")).toContainText(/converged/i);
  await expect
    .poll(async () => (await runs(page)).find((r) => r.run_id === runId)?.status)
    .toBe("converged");
  await expect(budget).toHaveCount(0);
  await expect(page.getByTestId("iteration").locator("..")).toHaveText(String(last + 2));
});

test("⌘K opens a run by name and state", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("a[data-row]").first()).toBeVisible();
  await page.keyboard.press("ControlOrMeta+k");
  await page.keyboard.type("diverged");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/runs\/run-/);
  await expect(page.getByTestId("status")).toContainText(/diverged/i);
});

test("a diverged run shows its diagnostic in the log and the plan of its case", async ({
  page,
}) => {
  const diverged = (await runs(page)).find((r) => r.status === "diverged") as Run;
  await page.goto(`/runs/${diverged.run_id}`);
  await expect(page.getByRole("alert")).toContainText("D0107");
  await page.getByRole("button", { name: /Plan/ }).click();
  await expect(page.getByRole("cell", { name: "wing.main" })).toBeVisible();
  await page.getByRole("tab", { name: "Log" }).click();
  await expect(page).toHaveURL(/tab=log/);
  const log = page.getByTestId("log");
  await expect(log.getByRole("link", { name: "D0107" })).toHaveAttribute("href", /#d0107$/);
  const lines = log.locator("time");
  const all = await lines.count();
  await page.getByRole("button", { name: /^info/ }).click();
  await expect(lines).not.toHaveCount(all);
  await expect(log.getByRole("link", { name: "D0107" })).toBeVisible();
});

test("viewer renders on WebGPU and picks the zone under the cursor", async ({ page }) => {
  await page.goto("/viewer?triangles=500000");
  // CPU-only CI sets AUTAN_UI_E2E_BACKEND=webgl2; GPU runners keep the WebGPU default.
  const backend = process.env.AUTAN_UI_E2E_BACKEND ?? "webgpu";
  await expect(page.getByTestId("viewer-status")).toHaveAttribute("data-backend", backend);
  // The backend is known before the surface arrives: pick once its zones are listed.
  await expect(page.getByTestId("zones")).toBeVisible();
  const box = await page.getByTestId("viewport").boundingBox();
  if (!box) throw new Error("no viewport");
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  const zones = page.getByTestId("zones");
  await expect(zones.getByRole("button", { pressed: true })).toHaveText(/^wing\./);
  await page.mouse.click(box.x + 20, box.y + box.height - 20);
  await expect(zones.getByRole("button", { pressed: true })).toHaveCount(0);

  const corner = { x: box.x + 4, y: box.y + 4, width: 8, height: 8 };
  await page.getByRole("button", { name: "light theme" }).click();
  const light = await page.screenshot({ clip: corner });
  await page.getByRole("button", { name: "dark theme" }).click();
  await expect
    .poll(async () => (await page.screenshot({ clip: corner })).equals(light))
    .toBe(false);
});

test("runs selected with x compare with c, with a polar for an angle sweep", async ({ page }) => {
  await page.goto("/");
  const group = page.locator("li", { has: page.getByTestId("study") });
  await expect(group.getByTestId("study")).toContainText("α 7.05–21.47 deg");
  const rows = group.locator("a[data-row]");
  await rows.first().focus();
  await page.keyboard.press(" ");
  await expect(page.getByTestId("peek")).toBeVisible();
  await page.keyboard.press(" ");
  await expect(page.getByTestId("peek")).toHaveCount(0);
  await page.keyboard.press("x");
  await page.keyboard.press("j");
  await page.keyboard.press("x");
  await expect(page.getByTestId("selected-count")).toHaveText("2 selected");
  await page.keyboard.press(" ");
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("peek")).toHaveCount(0);
  await expect(page.getByTestId("selected-count")).toHaveText("2 selected");
  await page.keyboard.press("c");
  await expect(page).toHaveURL(/\/compare\?runs=/);
  await expect(page.getByTestId("compared").getByRole("link")).toHaveCount(2);
  await expect(page.getByTestId("chart-drop").locator("canvas")).toBeVisible();

  await page.goto("/");
  await group.getByRole("button", { name: "Compare" }).click();
  // One condition, one series; every converged run of the study a point, none merged by angle.
  await expect(page).toHaveURL(/\/compare\?runs=/);
  const compared: string[] = JSON.parse(new URL(page.url()).searchParams.get("runs") ?? "[]");
  const all = await runs(page);
  const converged = compared.filter(
    (id) => all.find((r) => r.run_id === id)?.status === "converged",
  );
  expect(converged.length).toBeGreaterThan(4);
  await expect(page.getByTestId("lift-polar").locator("circle")).toHaveCount(converged.length);
  await expect(page.getByTestId("polar-series")).toHaveCount(0);
  await expect(page.getByTestId("settings-diff")).toContainText("freestream.aoa");
});

test("polars keep each Mach number apart and every run, and say which runs are missing", async ({
  page,
}) => {
  // As in the review: M0.80 at 0° and twice at 1.25°, M0.50 at 1.25°; and a diverged run.
  const dir = mkdtempSync(join(tmpdir(), "autan-polars-"));
  const tag = Date.now();
  const cases = [
    ["0.8", "0.0"],
    ["0.5", "1.25"],
    ["0.8", "1.25"],
    ["0.8", "1.25"],
  ] as const;
  const example = readFileSync("../py/examples/crm_hl.py", "utf8");
  const write = (k: number, mach: string, aoa: string, extra: string[] = []) => {
    const file = join(dir, `case${k}.py`);
    writeFileSync(
      file,
      example
        .replace('"crm-hl"', `"polar-${tag}-${k}"`)
        .replace("mach=0.2", `mach=${mach}`)
        .replace("aoa=19.57", `aoa=${aoa}`),
    );
    execFileSync("uv", [
      ...["run", "--locked", "--project", "server", "autan", "run", file, ...extra],
      ...["--mesh-info", "../py/tests/fixtures/crm-hl.meshinfo.json", "--runs", RUNS],
    ]);
  };
  for (const [k, [mach, aoa]] of cases.entries()) write(k, mach, aoa);
  try {
    write(4, "0.8", "2.5", ["--diverge-at", "40"]);
  } catch {
    // A diverged run exits 1, as autan run documents.
  }
  const ids = (await runs(page))
    .filter((r) => r.case.name.startsWith(`polar-${tag}-`))
    .sort((a, b) => a.case.name.localeCompare(b.case.name))
    .map((r) => r.run_id);
  try {
    expect(ids).toHaveLength(5);
    await page.goto(`/compare?runs=${encodeURIComponent(JSON.stringify(ids))}`);

    const series = page.getByTestId("polar-series");
    await expect(series).toHaveText([/^M 0\.5\s*B$/, /^M 0\.8\s*A C D$/]);
    await expect(page.getByTestId("lift-polar").locator("circle")).toHaveCount(4);
    await expect(page.getByTestId("drag-polar").locator("circle")).toHaveCount(4);
    await expect(page.getByTestId("polar-missing")).toHaveText(
      "Not on the polars: E diverged, not converged",
    );
    // Each point is its run's final coefficients as run.json records them.
    const recorded = await runs(page);
    const rows = page.getByTestId("polar-table").locator("tbody tr");
    await expect(rows).toHaveCount(4);
    for (const [k, id] of ids.slice(0, 4).entries()) {
      const forces = recorded.find((r) => r.run_id === id)?.monitors?.forces;
      const row = rows.filter({ hasText: new RegExp(`^${String.fromCharCode(65 + k)}`) });
      await expect(row.locator("td").nth(2)).toHaveText(String(forces?.cl));
      await expect(row.locator("td").nth(3)).toHaveText(String(forces?.cd));
    }
  } finally {
    for (const id of ids) rmSync(join(RUNS, id), { recursive: true, force: true });
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a converged run opens its wall surface in Analysis and switches fields", async ({ page }) => {
  const converged = (await runs(page)).find((r) => r.status === "converged") as Run;
  await page.goto(`/runs/${converged.run_id}`);
  await page.getByRole("button", { name: "Open in Analysis" }).click();
  await expect(page).toHaveURL(new RegExp(`/viewer\\?run=${converged.run_id}`));
  await expect(page.getByTestId("zones")).toContainText("wing.main");
  await expect(page.getByTestId("zones")).not.toContainText("farfield");
  await expect(page.getByTestId("legend")).toHaveText("Cp");
  await page.getByRole("tab", { name: "Cf" }).click();
  await expect(page.getByTestId("legend")).toHaveText("Cf");
  await expect(page).toHaveURL(/field=Cf/);
});

test("residuals at zero and far below 1 draw on the log chart", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const source = (await runs(page)).find((r) => r.status === "converged") as Run;
  const started = new Date(Date.now() - 3_600_000).toISOString().replace(/\.\d+Z$/, "Z");
  const suffix = Math.floor(Math.random() * 2 ** 32)
    .toString(16)
    .padStart(8, "0");
  const runId = `run-${started.replace(/[-:]/g, "")}-${suffix}`;
  const directory = join(RUNS, runId);
  const staging = join(RUNS, `.staging-${suffix}`);
  cpSync(join(RUNS, source.run_id), staging, { recursive: true });
  writeFileSync(join(staging, "run.json"), JSON.stringify({ ...source, run_id: runId, started }));
  // A one-cell-thick slab: the spanwise momentum residual starts at exactly zero, then sits at
  // round-off; the others are scaled far below 1, as a converged FP64 run can reach.
  const history = join(staging, "history.jsonl");
  const lines = readFileSync(history, "utf8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l));
  for (const [k, line] of lines.entries()) {
    for (const name of Object.keys(line.residuals)) line.residuals[name] *= 1e-20;
    line.residuals["momentum-y"] = k === 0 ? 0 : 1e-33;
  }
  writeFileSync(history, `${lines.map((l) => JSON.stringify(l)).join("\n")}\n`);
  renameSync(staging, directory);

  await page.goto(`/runs/${runId}`);
  await expect(page.getByTestId("residual-chart").locator("canvas")).toBeVisible();
  await expect(page.getByTestId("iteration")).toHaveText(String(lines.at(-1).iteration));
  await expect(page.locator("[data-exempt]")).toHaveText(/momentum-y\s*exempt/);
  expect(errors).toEqual([]);
});

test("floors written while a run page is open recompute its residual drop", async ({ page }) => {
  // One monitor, recorded every 10 iterations: line 30 carries its final record.
  const source = (await runs(page)).find(
    (r) => r.status === "converged" && Object.keys(r.monitors ?? {}).join() === "forces",
  ) as Run;
  const started = new Date(Date.now() - 7_200_000).toISOString().replace(/\.\d+Z$/, "Z");
  const suffix = Math.floor(Math.random() * 2 ** 32)
    .toString(16)
    .padStart(8, "0");
  const runId = `run-${started.replace(/[-:]/g, "")}-${suffix}`;
  const staging = join(RUNS, `.staging-${suffix}`);
  cpSync(join(RUNS, source.run_id), staging, { recursive: true });
  // Stopped short of its target: its first 30 lines.
  const lines = readFileSync(join(staging, "history.jsonl"), "utf8")
    .trim()
    .split("\n")
    .slice(0, 30);
  writeFileSync(join(staging, "history.jsonl"), `${lines.join("\n")}\n`);
  const at30 = JSON.parse(lines.at(-1) as string).monitors ?? {};
  const monitors = Object.fromEntries(
    Object.entries(at30).map(([name, record]) => [name, { ...(record as object), iteration: 30 }]),
  );
  const run = { ...source, run_id: runId, started, status: "completed", iterations: 30, monitors };
  writeFileSync(join(staging, "run.json"), JSON.stringify({ ...run, convergence: null }));
  renameSync(staging, join(RUNS, runId));
  const last = JSON.parse(lines.at(-1) as string).residuals;
  const target = Number(
    /residual_orders = ([0-9.]+)/.exec(readFileSync(join(RUNS, runId, "case.toml"), "utf8"))?.[1],
  );

  await page.goto(`/runs/${runId}`);
  const orders = page.getByTestId("orders");
  await expect(orders).not.toHaveText(target.toFixed(2));
  writeFileSync(
    join(RUNS, runId, "run.json"),
    JSON.stringify({ ...run, convergence: { floor: last } }),
  );

  await expect(orders).toHaveText(target.toFixed(2));
});

test("Compare lists where each run ran when that differs", async ({ page }) => {
  const source = (await runs(page)).find((r) => r.status === "converged") as Run;
  const started = new Date(Date.now() - 10_800_000).toISOString().replace(/\.\d+Z$/, "Z");
  const suffix = Math.floor(Math.random() * 2 ** 32)
    .toString(16)
    .padStart(8, "0");
  const runId = `run-${started.replace(/[-:]/g, "")}-${suffix}`;
  const solver = { ...source.solver, backend: "cuda" as const, commit: "f".repeat(40) };
  const devices = { kind: "cuda" as const, count: 4 };
  // results.md#runjson: identity over what the run computes, start fields absent.
  const identity = createHash("sha256")
    .update(
      [
        source.case.sha256,
        source.mesh.sha256,
        solver.commit,
        source.precision,
        solver.backend,
        devices.kind,
        devices.count,
      ].join("\n"),
    )
    .digest("hex");
  const staging = join(RUNS, `.staging-${suffix}`);
  cpSync(join(RUNS, source.run_id), staging, { recursive: true });
  const run = {
    ...source,
    run_id: runId,
    started,
    identity,
    solver,
    devices,
    deterministic: !source.deterministic,
  };
  writeFileSync(join(staging, "run.json"), JSON.stringify(run));
  renameSync(staging, join(RUNS, runId));
  await expect.poll(async () => (await runs(page)).some((r) => r.run_id === runId)).toBe(true);

  await page.goto(`/compare?runs=${encodeURIComponent(JSON.stringify([source.run_id, runId]))}`);

  const diff = page.getByTestId("settings-diff");
  for (const [key, a, b] of [
    ["run.devices", "1 CPU", "4 CUDA GPUs"],
    ["run.backend", source.solver.backend, "cuda"],
    ["run.solver", source.solver.commit, solver.commit],
    ["run.deterministic", String(source.deterministic), String(!source.deterministic)],
  ]) {
    await expect(diff.getByRole("row", { name: new RegExp(`^${key} `) })).toContainText(
      a as string,
    );
    await expect(diff.getByRole("row", { name: new RegExp(`^${key} `) })).toContainText(
      b as string,
    );
  }
});
