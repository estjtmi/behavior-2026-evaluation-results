# BEHAVIOR 2026 evaluation results

This repository preserves **all locally available evaluation artifacts** from the existing project: original evaluator JSON, videos, logs, benchmark runs, validation attempts, and checkpoint compatibility audits. Evaluation was stopped at the user's request. **No evaluation runs were started to create this repository.**

## Results available

| Policy / run | Task | Completed scored instances | Successes |
|---|---|---:|---:|
| `behavior-2026-five-winner-ckpt3-10k` | 31 — `clean_boxing_gloves` | 10 (public 0–9) | 1 |
| same checkpoint | 27 — `sorting_household_items` | 10 (public 0–9) | 0 |
| same checkpoint | 32 — `wash_a_baseball_cap` | 10 (public 0–9) | 0 |
| `behavior-2026-pi05-15tasks`, checkpoint 40000 | Requested 15-task scored evaluation | **0 of 150** | Not evaluated |

Separate from scored results:

- Two task-32 speed-benchmark rollouts, preserved with their original step limits and metrics.
- One completed task-5 validation at public index 10 / instance 311. The command requested a 300-step cap; the original evaluator JSON reports 301 steps. This is a compatibility validation, **not an official scored instance** and not evidence of task completion.
- The incomplete first task-5 validation attempt and its failure diagnostics are preserved. The newly added diagnostic initially rejected intentional negative-infinity masks in stage logits; only that diagnostic was corrected. See [diagnostic evidence](project-evaluations/pi05-15tasks/VALIDATION_DIAGNOSTIC_FIX.md).

The 15-task checkpoint's exact historical training configuration is recorded as **UNRECOVERABLE**. The user authorized the verified local `pi_behavior_b1k_fast` inference implementation with bundled checkpoint assets. The checkpoint revision, task mapping, asset checks, deployment settings, and source hashes are preserved under [the 15-task audit](project-evaluations/pi05-15tasks/).

## Browse the artifacts

- [Every completed rollout: JSON, video and log](RESULTS.md)
- [Original evaluation output trees](results/)
- [Evaluation scripts, manifests and audit evidence](project-evaluations/)
- [Preparation scripts and logs](evaluation-preparation/)
- [Simulator logs](simulator-logs/)
- [Complete artifact inventory with source paths and SHA-256 hashes](ARTIFACTS.json)
- [SHA-256 checksums](SHA256SUMS)

There are 33 completed rollout JSON files and 33 MP4 files in the raw result trees: 30 scored, 2 benchmark, and 1 validation. All original artifacts were copied byte-for-byte. No evaluator JSON, score, log or video was edited. Summary files in this repository are separate from evaluator outputs.

Previous project documentation is preserved verbatim and may describe an earlier queued/running state or link to the previous repository. This README records the final publication state: **evaluation stopped; the requested 15-task scored evaluation has not run**.

The complete artifact set is stored directly in Git, including videos; no external release download or Git LFS service is required. For a full copy:

```bash
git clone https://github.com/estjtmi/behavior-2026-evaluation-results.git
cd behavior-2026-evaluation-results
sha256sum -c SHA256SUMS
```

Upstream project: [estjtmi/behavior-1k-solution](https://github.com/estjtmi/behavior-1k-solution).
