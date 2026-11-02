import type { Surface } from "../api/surface";
import type { Axis } from "./slice";

/** The cut of a surface by the plane normal to `axis` through `origin` (absolute coordinates):
 * one segment per triangle the plane crosses, each end at its chord fraction x/c and with the
 * field linearly interpolated along the triangle's edge. The chord runs along x (along y for a
 * plane normal to x), from the section's least to its greatest coordinate there. Empty when the
 * plane misses the surface. */
export function section(
  surface: Surface,
  values: Float32Array,
  axis: Axis,
  origin: readonly number[],
): { segments: [number, number, number, number][]; chord: [number, number] } {
  const k = "xyz".indexOf(axis);
  const along = k === 0 ? 1 : 0;
  const p = surface.positions;
  const level = (origin[k] as number) - (surface.origin[k] as number);
  const offset = surface.origin[along] as number;
  const ends: [number, number][][] = [];
  const index = surface.indices;
  for (let t = 0; t < index.length; t += 3) {
    const crossings: [number, number][] = [];
    for (let e = 0; e < 3; e++) {
      const a = index[t + e] as number;
      const b = index[t + ((e + 1) % 3)] as number;
      const da = (p[3 * a + k] as number) - level;
      const db = (p[3 * b + k] as number) - level;
      // An edge crossed in its interior or at its first node (each node then counts once).
      if ((da <= 0 && db > 0) || (da > 0 && db <= 0)) {
        const s = da / (da - db);
        const x =
          (p[3 * a + along] as number) +
          s * ((p[3 * b + along] as number) - (p[3 * a + along] as number));
        const v = (values[a] as number) + s * ((values[b] as number) - (values[a] as number));
        crossings.push([x + offset, v]);
      }
    }
    if (crossings.length === 2) ends.push(crossings);
  }
  if (!ends.length) return { segments: [], chord: [0, 0] };
  let low = Number.POSITIVE_INFINITY;
  let high = Number.NEGATIVE_INFINITY;
  for (const segment of ends)
    for (const [x] of segment) {
      low = Math.min(low, x);
      high = Math.max(high, x);
    }
  const c = high - low || 1;
  return {
    segments: ends.map((segment) => {
      const [x0, v0] = segment[0] as [number, number];
      const [x1, v1] = segment[1] as [number, number];
      return [(x0 - low) / c, v0, (x1 - low) / c, v1];
    }),
    chord: [low, high],
  };
}
