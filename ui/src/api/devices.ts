import type { Run } from "./contracts";

const NAME: Record<Run["devices"]["kind"], string> = { cpu: "CPU", cuda: "CUDA", hip: "HIP" };

/** Where a run ran: `devices.count` is its number of ranks, each on a device of that kind
 * (results.md#runjson), so four ranks may share one GPU. */
export function devicesLabel({ kind, count }: Run["devices"]): string {
  return `${count} ${NAME[kind]} ${count === 1 ? "rank" : "ranks"}`;
}
