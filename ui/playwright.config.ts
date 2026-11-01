import { defineConfig } from "@playwright/test";

const port = 8766;
export const RUNS = "e2e/.runs";
export const MESHES = "e2e/.meshes";

export default defineConfig({
  testDir: "e2e",
  fullyParallel: false,
  workers: 1,
  reporter: "list",
  use: { baseURL: `http://127.0.0.1:${port}`, channel: "chrome", headless: true },
  webServer: {
    command: `rm -rf ${RUNS} ${MESHES} && mkdir ${MESHES} && cp ../py/tests/fixtures/crm-hl.meshinfo.json server/tests/data/surface/polyMesh.* server/tests/data/surface/hybrid-cgns.* ${MESHES}/ && bun run build && scripts/sample-runs.sh ${RUNS} && uv run --locked --project server python -m autan_ui --runs ${RUNS} --meshes ${MESHES} --port ${port} --no-open`,
    url: `http://127.0.0.1:${port}/api/runs`,
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
