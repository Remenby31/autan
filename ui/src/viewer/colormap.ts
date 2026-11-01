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

// Viridis (van der Walt and Smith, matplotlib 2015), 10 samples.
export const VIRIDIS: Colormap = {
  name: "viridis",
  stops: [
    [68, 1, 84],
    [72, 40, 120],
    [62, 73, 137],
    [49, 104, 142],
    [38, 130, 142],
    [31, 158, 137],
    [53, 183, 121],
    [110, 206, 88],
    [181, 222, 43],
    [253, 231, 37],
  ],
};

/** Cool-warm for a field that changes sign (its zero on the neutral colour), viridis for a
 * magnitude: the design direction's rule. */
export function colormapFor(min: number, max: number): Colormap {
  return min < 0 && max > 0 ? COOL_WARM : VIRIDIS;
}

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

/** Display range of a field: symmetric about zero when the field changes sign, so the neutral
 * colour of a diverging map marks zero. */
export function displayRange(min: number, max: number): { min: number; max: number } {
  // A uniform field sits in the middle of the map instead of at one end.
  if (max - min <= 1e-9 * Math.max(1, Math.abs(max))) return { min: min - 1, max: max + 1 };
  if (min < 0 && max > 0) {
    const bound = Math.max(-min, max);
    return { min: -bound, max: bound };
  }
  return { min, max };
}
