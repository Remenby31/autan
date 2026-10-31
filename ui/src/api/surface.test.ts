import { execFileSync } from "node:child_process";
import { expect, test } from "vitest";
import { decodeSurface } from "./surface";

function pythonSurface(triangles: number): ArrayBuffer {
  const bytes = execFileSync(
    "uv",
    [
      "run",
      "--project",
      "server",
      "python",
      "-c",
      `import sys; from autan_ui.surface import synthetic_wing; sys.stdout.buffer.write(synthetic_wing(${triangles}))`,
    ],
    { maxBuffer: 1 << 30 },
  );
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

test("decodes the server's surface message without copying", () => {
  const bytes = pythonSurface(20_000);
  const surface = decodeSurface(bytes);

  expect(surface.positions.buffer).toBe(bytes);
  expect(surface.indices.length % 3).toBe(0);
  const triangles = surface.indices.length / 3;
  expect(surface.zones.reduce((sum, z) => sum + z.triangles, 0)).toBe(triangles);
  surface.zones.reduce((first, zone) => {
    expect(zone.first_triangle).toBe(first);
    return first + zone.triangles;
  }, 0);
  const nodes = surface.positions.length / 3;
  expect(Math.max(...surface.indices)).toBeLessThan(nodes);
  for (const field of surface.fields) {
    expect(field.values.length).toBe(nodes);
    expect(Math.min(...field.values)).toBe(field.min);
    expect(Math.max(...field.values)).toBe(field.max);
  }
});

test("rejects another schema", () => {
  const header = new TextEncoder().encode('{"schema":"other/1"}');
  const bytes = new Uint8Array(4 + header.length);
  new DataView(bytes.buffer).setUint32(0, header.length, true);
  bytes.set(header, 4);
  expect(() => decodeSurface(bytes.buffer)).toThrow();
});
