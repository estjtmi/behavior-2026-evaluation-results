# 15-task evaluation preflight — 2026-10-08

Evaluation has not started. No server or evaluator was running at the initial tmux/process/port check. Existing JSON, logs, and videos were preserved. No training or code fixes were performed.

Repository: estjtmi/behavior-2026-pi05-15tasks, pinned revision 479fd5240a5150d69903506d01c84cd77d25bbd2. It contains checkpoints at steps 10000, 20000, 30000, and 40000, with no README, training configuration, task map, or designated inference checkpoint. All four `_CHECKPOINT_METADATA` files have empty `custom_metadata` and metrics.

## What the available code and data establish

`src/b1k/training/config.py` constructs `TaskIndexToTaskId()` without a mapping. `src/b1k/transforms.py` preserves `task_index` in that case. The dataset loader filters task names without renumbering metadata indices (`BEHAVIOR-1K/OmniGibson/omnigibson/learning/datas/lerobot_dataset.py`, lines 480–500). Task indices are not positions in the Python task-name list.

Downloaded `meta/tasks.jsonl` from both IliaLarchenko/behavior_224_rgb and behavior-1k/2026-challenge-demos agree on the requested task names and indices. Under this available code the embedding rows are 5,7,11,13,14,15,16,17,18,21,24,25,29,30,45 respectively, not 0–14. This does not establish that the specific checkpoint training run used unmodified code and data.

All four checkpoint parameter metadata files have task embeddings [50,2048], task-stage embeddings [596,1024], stage prediction output [15], and FAST embeddings [1024,2048]. These agree with the local model's structural defaults. Tensor shapes cannot establish stage-table ordering, delta-action settings, normalization selection, correlated-noise settings, or training-run task remapping.

The local model uses a task-dependent stage-count table and cumulative stage offsets. The inference wrapper maintains stage history; its stage semantics must match the training configuration before launch.

Bundled assets exist under IliaLarchenko/behavior_224_rgb in all four checkpoints. Norm stats are byte-identical across checkpoints, SHA256 ccd14a0210fc59b2d2726ba599cc0c4b81347395dd60d2a15b334b28ed15a80b. State/action statistics have width 32, per-timestamp action statistics are [30,32], and the full Cholesky matrix is [960,960]. These arrays contain finite numeric values. The separate spatial correlation matrix contains 530 null entries; the local inference loader uses the full Cholesky matrix instead, so this alone is not evidence requiring a fix.

Bundled FAST metadata specifies vocab 1024, scale 10, horizon 30, and dimensions 0:6,7:23. FAST files are preserved and hashed. Runtime loading and model inference have not been validated.

## Unresolved prerequisite

The exact training-run configuration/code provenance is absent from the accessible checkpoint repository and local project. The GitHub remote has only the existing backup and evaluation branches; no 15-task training configuration was found. Selecting the local defaults as the checkpoint's actual training settings would be an unverified substitution. The repository also has four checkpoint steps without a designated selection.

Required to proceed: a recoverable location for the actual training-run config/code or a training manifest establishing its task mapping and conditioning/settings, plus the intended checkpoint step. No validation or scored rollout has been launched while these prerequisites remain unresolved.

Raw downloaded evidence, asset hashes, and structured checks are in `audit/`. Existing completed results found locally cover the earlier checkpoint's tasks 27,31,32; none of the requested 15 tasks were found completed with this checkpoint.

## Authorized deployment (supersedes historical metadata blocker)

The user explicitly authorized checkpoint_40000 with the verified local pi_behavior_b1k_fast inference implementation and bundled checkpoint assets. Exact historical training configuration is **UNRECOVERABLE**; it is not represented as verified or reconstructed. The deployed settings are the actual local configuration, not invented historical hyperparameters. No training or retraining is performed.

See manifest.json and checkpoint-sha256.json for the pinned revision, requested mapping, deployment profile, and immutable source/data/weight hashes. Generated Python bytecode is excluded from asset equality checks because local imports regenerate it.

One shared server holds one checkpoint for all requested tasks. Its adapter reuses the existing verified 2026 proprioception conversion. Added assertions/logging check the actual normalized model task/stage pair, normalized state, model outputs, and every emitted action without changing numeric outputs. The stock evaluator is invoked unchanged, one instance/process at a time, with exclusive output directories. Scored runs use default task timeouts. Initial validation uses task 5 public index 10 (instance 311), capped at 300 steps, stored separately and excluded from scored public indices 0–9. Scored evaluations wait for artifact review.
