import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";

const port = 8767;
// On a machine without Google Chrome (Linux arm64), AUTAN_UI_BENCH_CHANNEL=chromium and the flags
// WebGPU needs there go in AUTAN_UI_BENCH_ARGS; AUTAN_UI_BENCH_MESHES lists meshes to load.
const channel = process.env.AUTAN_UI_BENCH_CHANNEL ?? "chrome";
const args = (process.env.AUTAN_UI_BENCH_ARGS ?? "").split(" ").filter(Boolean);
const meshes = process.env.AUTAN_UI_BENCH_MESHES;
// A server already running elsewhere (another container): no server is started here.
const external = process.env.AUTAN_UI_BENCH_URL;

export default defineConfig({
  testDir: ".",
  testMatch: "*.bench.ts",
  timeout: 300_000,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: external ?? `http://127.0.0.1:${port}`,
    ...(channel === "chromium" ? {} : { channel }),
    headless: true,
    viewport: { width: 1600, height: 1000 },
    launchOptions: { args: ["--disable-gpu-vsync", "--disable-frame-rate-limit", ...args] },
  },
  ...(external
    ? {}
    : {
        webServer: {
          command: `bun run build && uv run --locked --project server python -m autan_ui --runs bench/.runs${meshes ? ` --meshes ${meshes}` : ""} --port ${port} --no-open`,
          cwd: fileURLToPath(new URL("..", import.meta.url)),
          url: `http://127.0.0.1:${port}/api/runs`,
          reuseExistingServer: false,
          timeout: 120_000,
        },
      }),
});
