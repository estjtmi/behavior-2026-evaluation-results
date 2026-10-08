# Checkpoint evaluation: tasks 31 and 27

Status: setup complete; task 31 first public instance restarted after a dependency repair. No completed episode scores yet.

Policy: `estjtmi/behavior-2026-five-winner-ckpt3-10k`, revision `674056b95708df9d7a9e47faef84c9187f5014d0`.

Evaluation order: `clean_boxing_gloves` (checkpoint task 31), then `sorting_household_items` (checkpoint task 27). Each task uses public-test indices 0–9, one rollout per instance, one environment at a time, seed 0, and the evaluator's default task timeout (1.5× mean demonstration length). This follows the reported-results guidance in the official 2026 evaluation documentation. Videos and per-episode JSON metrics will be retained.

The simulator uses BEHAVIOR-1K `v3.9.3-post1` (`bd049de31`). The policy retains the training repository's pinned OpenPI and legacy OmniGibson Python interfaces in its separate environment. `serve_checkpoint.py` selects the checkpoint's bundled 2026 normalization and tokenizer assets, and maps the official 2026 R1Pro proprioception feature order into the legacy policy's input layout without adding privileged observations. The checkpoint's original task IDs are supplied explicitly.

The official RGB+depth full-resolution observation wrapper is used; the policy consumes RGB resized to 224×224 and its original proprioception inputs. The original action interpolation, rolling inpainting, stage voting, and correction rules remain enabled.

Results must distinguish simulation/setup failures from completed episodes; failures are not counted as policy successes or silently scored as zero.

## Startup failure and repair

The first attempt failed before policy rollouts because Isaac Sim installation replaced `websockets` with version 12.0, while the 2026 evaluator imports `websockets.asyncio.server` and requires version 15 or newer. The resulting import exception was followed by a segmentation fault during simulator shutdown. Installed `websockets==15.0.1`, verified the evaluator policy import, and added this import to the supervisor preflight check. The retry started at 2026-10-05 02:54:44 UTC.
