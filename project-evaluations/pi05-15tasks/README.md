# Shared checkpoint evaluation: 15 tasks

Checkpoint: `estjtmi/behavior-2026-pi05-15tasks`, revision `479fd5240a5150d69903506d01c84cd77d25bbd2`, `checkpoint_40000`.

Historical training configuration: **UNRECOVERABLE**. The user explicitly authorized the verified local `pi_behavior_b1k_fast` inference implementation with bundled checkpoint assets. This audit makes no claim to reconstruct historical hyperparameters. No training is performed.

Tasks run in order: 5,7,11,13,14,15,16,17,18,21,24,25,29,30,45. All use the same trained weights and one shared policy server. Dataset indices, official evaluator IDs, and local embedding rows agree. See `manifest.json`, `checkpoint-sha256.json`, and `deployed-source-sha256-attempt02.json`.

A single completed clean validation is required first: task 5, public index 10 (instance 311), 300-step cap, excluded from scoring. Official evaluations use public indices 0–9 (instances 301–310), one rollout each, one environment at a time, default task timeouts, seed 0, and the official 224x224 RGB wrapper. The stock evaluator writes all result JSON and videos directly.

Live attempt: `/workspace/evaluation-results/pi05-15tasks-40000/attempt-02`.

- Progress: `events.jsonl` and `logs/orchestrator.log`.
- Shared server: `logs/policy.log` and `logs/policy-audit.jsonl`.
- Validation: `validation/task-5-public-index-10/`.
- Scored instances: `task-<id>-<name>/instance-<index>/`.
- Each instance contains the exact command, evaluator log, original JSON, and MP4.
- The orchestrator refuses occupied ports/existing processes, reserves output paths exclusively, skips existing result JSONs, performs no automatic failed-rollout retries, and stops on missing/invalid artifacts or a failed evaluator.

Execution runs in tmux window `behavior-eval:pi05-15tasks`. Do not launch another server/evaluator or rerun the launch command while it is active. The orchestrator keeps scored runs gated until the validation artifacts are reviewed.

The incomplete first attempt is preserved at the parent output directory. See `VALIDATION_DIAGNOSTIC_FIX.md`: the audit guard initially rejected deliberate negative-infinity masks on unavailable stages. The corrected guard checks valid stages and masks separately. No model/evaluator change was needed.
