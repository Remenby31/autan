export interface Colormap {
  name: string;
  /** sRGB control points, evenly spaced over [0, 1]. */
  stops: readonly (readonly [number, number, number])[];
}

// Moreland (2009), "Diverging Color Maps for Scientific Visualization", cool-warm, 5 samples.
export const COOL_WARM: Colormap = {
  name: "cool-warm",
  stops: [
    [59, 76, 192],
    [141, 176, 254],
    [221, 221, 221],
    [244, 154, 123],
    [180, 4, 38],
  ],
};

/** `size` RGBA8 texels sampled by linear interpolation between the stops. */
export function sampleColormap(map: Colormap, size = 256): Uint8Array {
  const out = new Uint8Array(size * 4);
  const last = map.stops.length - 1;
  for (let i = 0; i < size; i++) {
    const t = (i / (size - 1)) * last;
    const k = Math.min(Math.floor(t), last - 1);
    const f = t - k;
    const a = map.stops[k] as readonly number[];
    const b = map.stops[k + 1] as readonly number[];
    for (let c = 0; c < 3; c++)
      out[i * 4 + c] = Math.round((a[c] ?? 0) * (1 - f) + (b[c] ?? 0) * f);
    out[i * 4 + 3] = 255;
  }
  return out;
}

export function cssGradient(map: Colormap): string {
  return `linear-gradient(90deg, ${map.stops.map(([r, g, b]) => `rgb(${r} ${g} ${b})`).join(", ")})`;
}
