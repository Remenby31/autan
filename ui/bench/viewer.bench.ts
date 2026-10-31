import { mkdirSync, writeFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import type { BenchReport } from "../src/routes/viewer";

const SIZES = [1_000_000, 5_000_000, 10_000_000, 20_000_000];
const BACKENDS = ["webgpu", "webgl2"] as const;

function quantile(sorted: number[], q: number): number {
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] ?? Number.NaN;
}

for (const backend of BACKENDS) {
  for (const triangles of SIZES) {
    test(`${backend} ${triangles / 1e6} M triangles`, async ({ page }) => {
      const webgl = backend === "webgl2" ? "&webgl=1" : "";
      await page.goto(`/viewer?bench=1&triangles=${triangles}${webgl}`);
      const handle = await page.waitForFunction(() => window.__bench, null, { timeout: 240_000 });
      const report = (await handle.jsonValue()) as BenchReport;
      expect(report.backend).toBe(backend);
      const frames = report.frameMs.slice(20).sort((a, b) => a - b);
      const result = {
        backend,
        triangles: report.triangles,
        nodes: report.nodes,
        fetchMs: Math.round(report.fetchMs),
        firstFrameMs: Math.round(report.firstFrameMs),
        frameP50Ms: +quantile(frames, 0.5).toFixed(2),
        frameP95Ms: +quantile(frames, 0.95).toFixed(2),
        fpsP50: +(1000 / quantile(frames, 0.5)).toFixed(1),
        gpu: await page.evaluate(async () => {
          const info = (await navigator.gpu?.requestAdapter())?.info;
          return info ? `${info.vendor} ${info.architecture}` : "none";
        }),
        userAgent: await page.evaluate(() => navigator.userAgent),
        date: new Date().toISOString(),
      };
      console.log(JSON.stringify(result));
      mkdirSync("bench/results", { recursive: true });
      writeFileSync(
        `bench/results/viewer-${backend}-${triangles}.json`,
        `${JSON.stringify(result, null, 2)}\n`,
      );
    });
  }
}
