# `autan logs`

`autan logs` prints the events of a run directory's `log.jsonl` ([results.md](../contracts/results.md#log)), and with `-f` follows them while the run is running.

```text
autan logs RUN [-f] [--level info|warning|error] [--json]
```

In Python: `autan.logs.read(path, follow=..., level=...)` yields each line's text and its parsed form, a `LogLine`, in the order written.

- **Output.** Without `--json`, one line per event: time, level, source, iteration (`-` before the first), code when there is one, then the message. With `--json`, each line exactly as the run wrote it, so `autan logs RUN --json` reproduces `log.jsonl` byte for byte.
- **`--level`.** Keeps the events at that level and above (`info` < `warning` < `error`). The default, `info`, keeps every event.
- **`-f`.** Waits for the lines a running run appends and prints each one within a second of its being written. It returns once `run.json` holds a final status, after printing every line written before that status. Both writers log the run's last event before they rewrite `run.json`, so no line is lost at the end. Without `-f`, a running run's last line, which may still be incomplete, is skipped, as results.md requires of readers. Interrupting `-f` (Ctrl-C) exits 130 and leaves the run untouched.
- **Exit status.** 0 once the events are printed. 1 with `E0701` when `RUN` is not a run directory ([results.md](../contracts/results.md)), when a line is not in the shape of results.md#log, or when a finished run's last line is incomplete. With `--json`, the diagnostic is printed as every `autan` command prints one.

The acceptance gate is [acceptance.md](../validation/acceptance.md), M4.
