import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";

// What the bench recorded, read here independently of history.py and of the server.
const RECORDED = "../bench/tests/data/history";
const newest = readdirSync(RECORDED).sort().at(-1) as string;
const results = readdirSync(join(RECORDED, newest)).map((f) =>
  JSON.parse(readFileSync(join(RECORDED, newest, f), "utf8")),
);
const measured = new Map<string, { median: number; unit: string }>(
  results.flatMap((r) =>
    r.benchmarks.map((b: { name: string; unit: string; stats: { median: number } }) => [
      b.name,
      { median: b.stats.median, unit: b.unit },
    ]),
  ),
);
const commit: string = results[0].provenance.solver.commit;
const oldest: string = JSON.parse(
  readFileSync(join(RECORDED, readdirSync(RECORDED).sort()[0] as string, "stream.json"), "utf8"),
).provenance.solver.commit;
// scripts/sample-perf.sh puts between them a commit that measured STREAM only, and has a bisection
// blame the newest.
const BETWEEN = "b".repeat(40);
// bench.md: the GB10 peak is the highest STREAM median of bench/baselines/gb10.json.
const peak = Math.max(
  ...JSON.parse(readFileSync("../bench/baselines/gb10.json", "utf8")).benchmarks.map(
    (b: { stats: { median: number } }) => b.stats.median,
  ),
);
const tracked = new Set(
  JSON.parse(
    readFileSync("../bench/baselines/gb10-codegen-hex-n128-fp32.json", "utf8"),
  ).benchmarks.map((b: { name: string }) => b.name),
);

test("each bandwidth benchmark of main reads as a % of the GB10 peak, with its last commit and gate", async ({
  page,
}) => {
  await page.goto("/perf");
  await expect(page.getByTestId("perf-peak")).toHaveText(`${peak.toFixed(1)} GB/s`);
  const bandwidths = [...measured].filter(([, b]) => b.unit === "GB/s").map(([name]) => name);
  await expect(page.getByRole("option")).toHaveCount(bandwidths.length);

  const name = "codegen.roe_store_gather";
  await page.locator(`[data-name="${name}"]`).click();
  const roe = measured.get(name) as { median: number };
  await expect(page.getByTestId("perf-percent")).toHaveText(
    `${((100 * roe.median) / peak).toFixed(1)} %`,
  );
  await expect(page.getByTestId("perf-commit").getByRole("link")).toHaveAttribute(
    "href",
    new RegExp(`/commit/${commit}$`),
  );
  await expect(page.getByTestId("perf-gate")).toHaveText(tracked.has(name) ? /baseline/ : "—");
  await expect(page.getByTestId("perf-culprit")).toBeVisible();
  await expect(page.getByTestId("perf-chart").locator("[data-samples]")).toHaveAttribute(
    "data-samples",
    "2",
  );

  // Hovering the chart reads the commit under the cursor; leaving it, the last one again.
  const plot = page.getByTestId("perf-chart").locator(".u-over");
  const box = await plot.boundingBox();
  if (!box) throw new Error("no plot area");
  await page.mouse.move(box.x + 2, box.y + box.height / 2);
  await expect(page.getByTestId("perf-commit").getByRole("link")).toHaveText(oldest.slice(0, 8));
  await expect(page.getByTestId("perf-culprit")).toHaveCount(0);
  // A commit that did not measure this benchmark says so, under its own name.
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await expect(page.getByTestId("perf-commit").getByRole("link")).toHaveText(BETWEEN.slice(0, 8));
  await expect(page.getByTestId("perf-percent")).toHaveText("not measured");
  await page.mouse.move(box.x + box.width / 2, box.y + box.height + 80);
  await expect(page.getByTestId("perf-commit").getByRole("link")).toHaveText(commit.slice(0, 8));

  // The keyboard walks the list, and the URL keeps the selection.
  await page.keyboard.press("j");
  const next = [...bandwidths].sort()[[...bandwidths].sort().indexOf(name) + 1] as string;
  await expect(page.locator(`[data-name="${next}"]`)).toHaveAttribute("aria-selected", "true");
  await expect(page).toHaveURL(new RegExp(`b=${next}`));

  const gated = [...tracked].find((t) => measured.get(t as string)?.unit === "GB/s") as string;
  await page.locator(`[data-name="${gated}"]`).click();
  await expect(page.getByTestId("perf-gate")).toHaveText(/baseline/);

  await expect(page.getByTestId("perf-missing")).toContainText("ESTIMATE");
});

test("the benchmark shortcuts leave the selection alone behind the navigation drawer", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/perf?b=stream.copy");
  await page.getByRole("button", { name: "Open navigation" }).click();
  await expect(page.getByRole("dialog", { name: "Navigation" })).toBeVisible();
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("j");
  await expect(page).toHaveURL(/b=stream\.copy$/);
});
