import type { Surface } from "../api/surface";

export type Axis = "x" | "y" | "z";

// The surface's positions are FP32 offsets from its origin, each rounded by up to 2^-24 of its
// magnitude: a plane at a rounded end of the extent could fall just outside the mesh. Ends are
// kept inside by four times that, and by 1e-6 of the extent.
const ROUNDING = 4 * 2 ** -24;
const INSET = 1e-6;

/** The plane normal to `axis` at the fraction `at` (0 to 1) of the surface's extent along it,
 * through the middle of the other two; its ends are kept just inside the extent. */
export function slicePlane(
  surface: Surface,
  axis: Axis,
  at: number,
): { origin: number[]; normal: number[] } {
  const min = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  const p = surface.positions;
  const offset = [0, 0, 0];
  for (let i = 0; i < p.length; i += 3)
    for (let k = 0; k < 3; k++) {
      offset[k] = Math.max(offset[k] as number, Math.abs(p[i + k] as number));
      const v = (p[i + k] as number) + (surface.origin[k] as number);
      if (v < (min[k] as number)) min[k] = v;
      if (v > (max[k] as number)) max[k] = v;
    }
  const k = "xyz".indexOf(axis);
  const origin = min.map((low, j) => (low + (max[j] as number)) / 2);
  const low = min[k] as number;
  const span = (max[k] as number) - low;
  const pad = Math.min(span / 2, Math.max(INSET * span, ROUNDING * (offset[k] as number)));
  origin[k] = low + pad + Math.min(1, Math.max(0, at)) * (span - 2 * pad);
  const normal = [0, 0, 0];
  normal[k] = 1;
  return { origin, normal };
}
