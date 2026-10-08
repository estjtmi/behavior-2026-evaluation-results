# Validation diagnostic correction

Attempt 1 stopped before emitting any action to the evaluator because the new audit wrapper demanded all 15 stage logits be finite. Task 5 has 12 stages. The existing model intentionally applies `jnp.where(valid_mask, subtask_logits, -jnp.inf)` at src/b1k/models/pi_behavior.py:1022. The preceding action-finiteness assertion passed; the invalid-stage sentinel check failed.

Smallest correction: restrict finite-logit checks to the task's valid stages and assert negative infinity in the remaining masked stages. No model computation, normalization, tokenizer, action, score, or evaluator was changed. This was an audit-wrapper false positive, not a checkpoint incompatibility.

The evaluator exited without a result JSON even though its process returned zero; the orchestrator correctly rejected missing artifacts and stopped the server. Attempt 1 is preserved at /workspace/evaluation-results/pi05-15tasks-40000/validation/task-5-public-index-10, with original logs in the parent logs directory. No completed validation or scored instance exists from this attempt.

A second attempt will use the separate /workspace/evaluation-results/pi05-15tasks-40000/attempt-02 directory. This is a retry of an incomplete validation, not a rerun of a completed result. The required outcome remains one completed clean validation rollout.
