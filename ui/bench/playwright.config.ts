import { fileURLToPath } from "node:url";
import { defineConfig } from "@playwright/test";

const port = 8767;

export default defineConfig({
  testDir: ".",
  testMatch: "*.bench.ts",
  timeout: 300_000,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    channel: "chrome",
    headless: true,
    viewport: { width: 1600, height: 1000 },
    launchOptions: { args: ["--disable-gpu-vsync", "--disable-frame-rate-limit"] },
  },
  webServer: {
    command: `bun run build && uv run --locked --project server python -m autan_ui --runs bench/.runs --port ${port} --no-open`,
    cwd: fileURLToPath(new URL("..", import.meta.url)),
    url: `http://127.0.0.1:${port}/api/runs`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
