import { Tooltip } from "../ui/Tooltip";

/** A hash or path cut in the middle so both ends stay readable; the whole value on hover. */
export function MiddleTruncate({ value, keep = 8 }: { value: string; keep?: number }) {
  const short =
    value.length > 2 * keep + 1 ? `${value.slice(0, keep)}…${value.slice(-keep / 2)}` : value;
  return (
    <Tooltip content={<span className="font-mono">{value}</span>}>
      <span className="cursor-default font-mono">{short}</span>
    </Tooltip>
  );
}
