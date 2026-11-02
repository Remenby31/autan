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

/** A force coefficient to four decimals, one drag count; never "-0.0000". */
export function coefficient(value: number): string {
  const text = value.toFixed(4);
  return /^-0\.0+$/.test(text) ? text.slice(1) : text;
}

/** Why a run's last coefficients are not those of a converged steady solution, or null when
 * they are (results.md#runjson status). */
export function coefficientsCaveat(status: string, unsteady: boolean): string | null {
  if (status === "converged") return null;
  if (status === "running") return "still running";
  if (status === "completed" && unsteady) return "last step of an unsteady run";
  return "not converged";
}
