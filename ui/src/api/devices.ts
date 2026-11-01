import type { Run } from "./contracts";

const NAME: Record<Run["devices"]["kind"], [string, string]> = {
  cpu: ["CPU", "CPUs"],
  cuda: ["CUDA GPU", "CUDA GPUs"],
  hip: ["HIP GPU", "HIP GPUs"],
};

export function devicesLabel({ kind, count }: Run["devices"]): string {
  const [one, many] = NAME[kind];
  return `${count} ${count === 1 ? one : many}`;
}
