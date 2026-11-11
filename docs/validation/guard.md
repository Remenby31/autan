# Nightly convergence guard

Every night, the guard runs the TMR flat plate on `main` and checks that it converges as it did when the guard was declared (#611, from #597's bisection). It reads per-equation residual drops and final residuals, not the run status: a run of this case ends `completed` until #544 lands, whether its convergence regressed or not.

## Case

- **What it runs:** the TMR flat plate 137×97, SA (`nu_tilde_ratio = 3`), fp64, steady multigrid, on `vv/guard/flatplate-case.toml`.
  - It is #597's bisection case with `residual_orders = 14`, so that no build stops early and every night is measured at step 20 000.
  - The mesh is `flatplate-137x97.msh`, sha256 `ceb75abc…a7a59e`, checked before each night.
- **Build:** the `origin/main` sha resolved at 23:30 UTC is pinned for the night. Release, CPU, one rank, one thread.
- **Two runs per night,** at the same time from one binary, whose histories must be bitwise identical in their numbers.

## Gates

Each gate is checked on run 1. Gates (a), (b) and (d)'s m3f come from `main` 179e316c, #597's B4 history (sha256 `b65468a7…`). Gate (c) and the rest of (d) come from #611's null search: candidates and the rule deriving their bounds were declared before any null history was read ([declaration](https://github.com/Remenby31/autan/issues/611#issuecomment-6785712335)); the bounds come from `main`'s 7 training runs, and the candidates were selected on 8 held-out runs ([selection](https://github.com/Remenby31/autan/issues/611#issuecomment-6801480590)), with `vv/guard/null_eval.py`, the evaluator frozen before any null read. They replace the last-step drops and m3s, which round-off-level inputs alone move across their bounds. Removing m3s from the gate is an exception: https://github.com/Remenby31/autan/pull/701#issuecomment-6801958242. m3s stays computed and reported. They hold from the merge of this revision on; earlier nights keep their verdicts. Gates change only through a PR that V&V and the orchestrator review.

| Gate | Condition |
|---|---|
| (a) rule | The case-schema rule at 10 orders, recomputed from the history (`autan_vv.guard`), is met at some step ≤ 20 000 once #544 is in the night's commit, with the shared momentum reference. Before #544, it is recorded and not gated: momentum-y does not meet it against its own step-1 value. The night has #544 when its pinned history holds a commit whose subject ends `(#N)`, N being `PR` in `vv/guard/has544.conf`, which #544's PR sets. |
| (b) final residuals | ≤ the reference run's × 10^0.5: density 4.579e-15, momentum-x 1.2719e-12, momentum-y 5.355e-13, energy 1.8778e-9, `nu-tilde` 3.625e-18; momentum-z (zero in this slab) ≤ its identically-zero floor at every step, not only the last. |
| (c) drops, orders | Each drop measured to the median residual of steps 18 001–20 000, not to the last step: energy ≥ 11.61 below its peak and ≥ 10.07 below its step-1 value; momentum-x ≥ 12.83 below step 1; momentum-y ≥ 13.04 below the shared momentum reference; `nu-tilde` ≥ 13.56 below step 1. The last step's drops are reported. |
| (d) speed | Energy first 10 orders below its peak by step 2 174 (m3f); above that level on at most 5.1 % of steps 10 001–20 000 (the tail fraction); and its peak at most 10^1.84 times its step-1 value (the peak cap). The step from which it stays below the level (m3s) is reported. |
| (e) repeat | The two runs give the same history, and both exit 0. With the same `autan-run` as the last green night (before any, the reference run, `vv/guard/reference.json`), so does the night against that night. With another binary, the first differing step and the changes of m3f, m3s, the tail fraction, the peak cap and every drop are reported. |

## Outcomes

| Outcome | Meaning |
|---|---|
| green | Every gate passes and the two runs are identical. |
| red | A gate fails, the runs differ, a run exits non-zero, or a run cannot be measured: a build failure, a crash, a non-finite or truncated history or monitor, a missing binary fingerprint, or a foreign commit. Inputs the enqueue refuses (the mesh's sha256, a failed fetch) give a red verdict without a job. |
| void | A named infrastructure failure only: the night's deadline (06:30 UTC the next day), which stops the build or the runs; a job still pending in the queue at 07:00, which the watchdog withdraws; a drain that stops the runs, which writes `DRAINED` (who, when, why) in the night's directory before its TERM, the marker's content going into `STATUS`, and a drained night left without a verdict is void too; or a cgroup OOM kill that the job's own counter records. The case runs a fixed 20 000 steps, so a missed deadline measures the queue and the host's load, not the numerics; the verdict gives the steps reached, the steps per minute, the queue start and the publication time. |
| MISSING | No verdict by 07:00 UTC. |

**Reading the results:** every morning, V&V reads `spark-5512:~/cfd/vv-guard/STATUS`, one line per night, with the orchestrator as backup. A red or MISSING night, or a second void in a row (`STATUS` flags it), gets an issue with both runs' paths. The host has no `gh`, so this step is manual.

**What is kept:** each night's directory holds `verdict.json` (schema `autan.vv-guard-night/1`), both runs, the logs and `previous.json`, the last green night it was compared with. A green night's sources and build are removed after 14 days. Red and void nights stay whole.

## A lead the guard keeps in view

On the reference run, energy first goes 10 orders below its peak at step 1977. It then goes back above that level in excursions until step 12 086, for about 10 000 steps, before staying below it to step 20 000. This is a lead, possibly a late limit cycle, and is not gated beyond (d). If #544 or a later change alters it, m3s and the tail fraction show it.

## Installation on spark-5512

Everything lives under `~/cfd/vv-guard`; `~/cfd/guard` belongs to the shared memory guard. The units run from a dedicated clone, `~/cfd/vv-guard/repo`, updated at each merge of this directory. The inputs sit in `~/cfd/vv-guard/inputs`: the mesh (its sha256 is fixed in `enqueue.sh`) and `reference-history.jsonl`, the reference run's history, for the first differing step before the first green night.

```sh
git clone https://github.com/Remenby31/autan ~/cfd/vv-guard/repo
mkdir -p ~/cfd/vv-guard/inputs ~/.config/systemd/user
cp <the mesh> <the reference run's history.jsonl> ~/cfd/vv-guard/inputs/   # names: flatplate-137x97.msh, reference-history.jsonl
cp ~/cfd/vv-guard/repo/vv/guard/autan-guard-*.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now autan-guard-enqueue.timer autan-guard-watchdog.timer
```

**What runs where:**
- **The enqueue timer** (23:30 UTC, `Persistent=true`) only prepares the night's directory and submits its job file, `05-guard-<date>.job`, to the legacy queue through `~/cfd/bin/cfd-submit`, the only way into the queue; its output is in the night's `submit.log`, and a refused submission leaves the night unpublished. PRIO 05 puts the guard ahead of the campaigns (08 and above); only the declared critical path (00–04) goes first. A night is the date of the latest 23:30 UTC slot, so a catch-up after midnight prepares the right night. A lock covers it, and a night prepared but not yet published is published by the next run.
- **The job,** THREADS 2, CAP 6 GiB, does the build and the two runs.
- **The watchdog** (07:00 UTC) appends one line per night to `STATUS`, for the night whose deadline the latest 07:00 slot closed, and trims green nights older than 14 days to their verdicts and histories.
