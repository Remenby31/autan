/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import { tanstackRouter } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [tanstackRouter({ target: "react", autoCodeSplitting: true }), react(), tailwindcss()],
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
  publicDir: "../docs/assets",
  // The viewer chunk carries three.js (~0.9 MB) and loads only on 3D screens.
  build: { outDir: "server/src/autan_ui/static", emptyOutDir: true, chunkSizeWarningLimit: 1024 },
  test: { include: ["src/**/*.test.ts"] },
});
