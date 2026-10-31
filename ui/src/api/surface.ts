export interface SurfaceZone {
  name: string;
  first_triangle: number;
  triangles: number;
}

export interface SurfaceField {
  name: string;
  location: "node";
  min: number;
  max: number;
  values: Float32Array;
}

export interface Surface {
  origin: [number, number, number];
  positions: Float32Array;
  indices: Uint32Array;
  zones: SurfaceZone[];
  fields: SurfaceField[];
}

interface Header {
  schema: string;
  origin: [number, number, number];
  nodes: number;
  triangles: number;
  zones: SurfaceZone[];
  fields: Omit<SurfaceField, "values">[];
  buffers: { name: string; dtype: "f32" | "u32"; offset: number; length: number }[];
}

export const SURFACE_SCHEMA = "autan-ui.surface/1";

/** Decode a autan-ui.surface/1 message: views into `bytes`, no copy. */
export function decodeSurface(bytes: ArrayBuffer): Surface {
  const headerLength = new DataView(bytes).getUint32(0, true);
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(bytes, 4, headerLength)),
  ) as Header;
  if (header.schema !== SURFACE_SCHEMA) {
    throw new Error(`unsupported surface schema ${header.schema}`);
  }
  const base = 4 + headerLength;
  const view = (name: string) => {
    const buffer = header.buffers.find((b) => b.name === name);
    if (!buffer) throw new Error(`surface message has no ${name} buffer`);
    const Typed = buffer.dtype === "f32" ? Float32Array : Uint32Array;
    return new Typed(bytes, base + buffer.offset, buffer.length);
  };
  return {
    origin: header.origin,
    positions: view("positions") as Float32Array,
    indices: view("indices") as Uint32Array,
    zones: header.zones,
    fields: header.fields.map((f) => ({ ...f, values: view(`field:${f.name}`) as Float32Array })),
  };
}

export async function fetchSurface(url: string, signal?: AbortSignal): Promise<Surface> {
  const response = await fetch(url, signal ? { signal } : {});
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return decodeSurface(await response.arrayBuffer());
}
