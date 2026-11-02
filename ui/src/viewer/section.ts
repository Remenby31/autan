import type { Surface } from "../api/surface";
import { type Axis, bounds } from "./slice";

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
  // A plane through nodes is moved off them, by 2e-6 of the extent, to the side where it cuts
  // something: every crossing is then inside an edge, and no face lies in the plane.
  const k = "xyz".indexOf(axis);
  const { min, max } = bounds(surface);
  const shift = 2e-6 * ((max[k] as number) - (min[k] as number) || 1);
  const level = (origin[k] as number) - (surface.origin[k] as number);
  const p = surface.positions;
  let onNode = false;
  for (let i = k; i < p.length; i += 3)
    if (Math.abs((p[i] as number) - level) < shift / 2) {
      onNode = true;
      break;
    }
  if (!onNode) return cut(surface, values, k, level);
  const above = cut(surface, values, k, level + shift);
  return above.segments.length ? above : cut(surface, values, k, level - shift);
}

function cut(
  surface: Surface,
  values: Float32Array,
  k: number,
  level: number,
): { segments: [number, number, number, number][]; chord: [number, number] } {
  const along = k === 0 ? 1 : 0;
  const p = surface.positions;
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
      if ((da < 0 && db > 0) || (da > 0 && db < 0)) {
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
