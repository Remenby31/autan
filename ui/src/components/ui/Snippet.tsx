import { Check, Copy } from "lucide-react";
import { useState } from "react";
import { useToast } from "./Toast";
import { Tooltip } from "./Tooltip";

/** A command to copy: monospace, prompt sign, one click to the clipboard. */
export function Snippet({ text, prompt = true }: { text: string; prompt?: boolean }) {
  const [copied, setCopied] = useState(false);
  const toast = useToast();
  const copy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    toast("Copied to clipboard");
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <div className="flex h-10 items-center gap-2 rounded-sm border border-line bg-bg-0 pr-1 pl-3 font-mono text-sm">
      {prompt && <span className="select-none text-fg-3">$</span>}
      <span className="min-w-0 flex-1 truncate">{text}</span>
      <Tooltip content={copied ? "Copied" : "Copy"}>
        <button
          type="button"
          aria-label="Copy"
          onClick={copy}
          className="flex size-8 items-center justify-center rounded-sm text-fg-2 hover:bg-bg-2 hover:text-fg"
        >
          {copied ? <Check className="size-4 text-success" /> : <Copy className="size-4" />}
        </button>
      </Tooltip>
    </div>
  );
}
