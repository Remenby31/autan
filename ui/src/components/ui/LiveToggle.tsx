import { Toggle } from "@base-ui/react/toggle";
import { Pause, Play } from "lucide-react";

/** Follow new data as it arrives; scrolling back turns it off, End turns it on again. */
export function LiveToggle({
  live,
  onChange,
}: {
  live: boolean;
  onChange: (live: boolean) => void;
}) {
  return (
    <Toggle
      pressed={live}
      onPressedChange={onChange}
      className="inline-flex h-8 items-center gap-2 rounded-sm border border-line px-3 font-medium text-fg-2 transition-colors duration-[var(--dur-fast)] hover:bg-bg-2 data-[pressed]:border-accent data-[pressed]:text-accent"
    >
      {live ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
      Live
    </Toggle>
  );
}
