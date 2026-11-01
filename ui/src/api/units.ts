/** An angle written as the case writes it ("19.57 deg", "0.34 rad"), in degrees; null if the
 * value has no angle unit. */
export function degrees(value: string | null | undefined): number | null {
  const match = /^\s*(-?[\d.eE+-]+)\s*(deg|degree|degrees|rad|radian|radians)\s*$/.exec(
    value ?? "",
  );
  if (!match) return null;
  const number = Number(match[1]);
  if (!Number.isFinite(number)) return null;
  return match[2]?.startsWith("rad") ? (number * 180) / Math.PI : number;
}
