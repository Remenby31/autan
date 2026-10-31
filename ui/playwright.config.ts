import { defineConfig } from "@playwright/test";

const port = 8766;
export const RUNS = "e2e/.runs";

export default defineConfig({
  testDir: "e2e",
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: { baseURL: `http://127.0.0.1:${port}`, channel: "chrome", headless: true },
  webServer: {
    command: `rm -rf ${RUNS} && bun run build && scripts/sample-runs.sh ${RUNS} && uv run --locked --project server python -m autan_ui --runs ${RUNS} --port ${port} --no-open`,
    url: `http://127.0.0.1:${port}/api/runs`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
