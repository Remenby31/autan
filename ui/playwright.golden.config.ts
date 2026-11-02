import { defineConfig } from "@playwright/test";

// The golden path: the UI driven on what the solver itself writes (scripts/golden-runs.sh, with
// the built autan-mesh-info and autan-run). AUTAN_UI_GOLDEN_DIR serves outputs written earlier.
const port = 8767;
export const GOLDEN = process.env.AUTAN_UI_GOLDEN_DIR ?? "e2e/.golden";
const write = process.env.AUTAN_UI_GOLDEN_DIR
  ? ""
  : `rm -rf ${GOLDEN} && scripts/golden-runs.sh ${GOLDEN} && `;

export default defineConfig({
  testDir: "e2e/golden",
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: { baseURL: `http://127.0.0.1:${port}`, channel: "chrome", headless: true },
  webServer: {
    command: `${write}bun run build && uv run --locked --project server python -m autan_ui --runs ${GOLDEN}/runs --meshes ${GOLDEN}/meshes --port ${port} --no-open`,
    url: `http://127.0.0.1:${port}/api/runs`,
    reuseExistingServer: false,
    timeout: 180_000,
  },
});
