import { motion } from "motion/react";
import { Button } from "../ui/Button";
import { Kbd } from "../ui/Kbd";

/** Shown while runs are selected: how many, compare them, clear the selection. */
export function SelectionBar({
  count,
  onCompare,
  onClear,
}: {
  count: number;
  onCompare: () => void;
  onClear: () => void;
}) {
  return (
    <motion.div
      role="toolbar"
      aria-label="Selected runs"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 12 }}
      className="fixed bottom-6 left-1/2 z-30 flex -translate-x-1/2 items-center gap-3 rounded-full bg-bg-0 py-1.5 pr-1.5 pl-4 shadow-dialog"
    >
      <span className="font-medium tabular-nums" data-testid="selected-count">
        {count} selected
      </span>
      <Button
        size="sm"
        variant="primary"
        onClick={onCompare}
        disabled={count < 2}
        className="rounded-full"
      >
        Compare <Kbd className="border-transparent bg-bg-0/20 text-bg-0">c</Kbd>
      </Button>
      <Button size="sm" variant="tertiary" onClick={onClear} className="rounded-full">
        Clear <Kbd>Esc</Kbd>
      </Button>
    </motion.div>
  );
}
