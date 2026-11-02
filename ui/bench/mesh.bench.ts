import { mkdirSync, writeFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import type { BenchReport } from "../src/routes/viewer";

// A listed mesh's boundary surface (AUTAN_UI_BENCH_MESH, in AUTAN_UI_BENCH_MESHES): the size of
// its message and the server's time to build it, cold then cached; the time to the first frame
// and the frame times while orbiting, on WebGPU or WebGL2 (AUTAN_UI_BENCH_BACKEND).
const mesh = process.env.AUTAN_UI_BENCH_MESH;
const backend = process.env.AUTAN_UI_BENCH_BACKEND === "webgl2" ? "webgl2" : "webgpu";

function quantile(sorted: number[], q: number): number {
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] ?? Number.NaN;
}

test(`${mesh ?? "no mesh"} surface on ${backend}`, async ({ page }) => {
  test.skip(!mesh, "AUTAN_UI_BENCH_MESH names no mesh");
  const url = `/api/meshes/${encodeURIComponent(mesh ?? "")}/surface`;
  const timed = async () => {
    const start = performance.now();
    const response = await page.request.get(url);
    const body = await response.body();
    expect(response.status()).toBe(200);
    return { ms: performance.now() - start, bytes: body.length };
  };
  // The first request is cold only on a server just started (as the config's own is).
  const first = await timed();
  const repeat = await timed();

  const webgl = backend === "webgl2" ? "&webgl=1" : "";
  await page.goto(`/viewer?bench=1&mesh=${encodeURIComponent(mesh ?? "")}${webgl}`);
  const handle = await page.waitForFunction(() => window.__bench, null, { timeout: 240_000 });
  const report = (await handle.jsonValue()) as BenchReport;
  expect(report.backend).toBe(backend);
  const frames = report.frameMs.slice(20).sort((a, b) => a - b);
  const result = {
    mesh,
    backend,
    messageBytes: first.bytes,
    serverFirstMs: Math.round(first.ms),
    serverRepeatMs: Math.round(repeat.ms),
    triangles: report.triangles,
    nodes: report.nodes,
    sinceNavigationMs: Math.round(report.sinceNavigationMs),
    fetchMs: Math.round(report.fetchMs),
    firstFrameMs: Math.round(report.firstFrameMs),
    frameP50Ms: +quantile(frames, 0.5).toFixed(2),
    frameP95Ms: +quantile(frames, 0.95).toFixed(2),
    fpsP50: +(1000 / quantile(frames, 0.5)).toFixed(1),
    gpu: await page.evaluate(async () => {
      const info = (await navigator.gpu?.requestAdapter())?.info;
      return info ? `${info.vendor} ${info.architecture} ${info.device}` : "none";
    }),
    userAgent: await page.evaluate(() => navigator.userAgent),
    date: new Date().toISOString(),
  };
  console.log(JSON.stringify(result));
  mkdirSync("bench/results", { recursive: true });
  writeFileSync(
    `bench/results/mesh-${mesh}-${backend}.json`,
    `${JSON.stringify(result, null, 2)}\n`,
  );
});
