# Continued preflight

No policy server or evaluator was running at the initial check. No rollout, training, or production-code modification was performed.

The updated evaluation plan requires one clean validation rollout before the first task only; later validation is conditional on a new issue. The stock evaluator resolves requested public indices 0–9 to instance IDs 301–310. A saved plan lists all 15 task names. Inventory hashes preserve the 32 existing result JSON files; none match the requested task names.

Additional provenance searches covered all tmux scrollback, workspace and temporary training/config artifacts, GitHub refs and the backup commit that records previous training work, and the Hugging Face train-state metadata. No configuration or log for this specific 15-task run was found. The Hugging Face main revision remains 479fd5240a5150d69903506d01c84cd77d25bbd2. The newest checkpoint is checkpoint_40000, uploaded in that revision. Train-state metadata has step, params, optimizer state, and model_def=None; it does not recover runtime configuration.

The successful final check is runtime-checks-03.json / runtime-checks-03.log:

- All 75 checkpoint tensor paths and shapes match all 75 local abstract model state tensors for each of the four checkpoints. No missing or extra tensors, or shape mismatches.
- Each checkpoint's bundled norm stats load through the actual normalization loader. State/action statistics, per-timestamp statistics, and the full correlation Cholesky matrix are finite. Correlation regularization using the local configuration yields a finite Cholesky factor.
- Each bundled FAST tokenizer loads locally, encodes a synthetic [1,30,22] action tensor, and decodes it without errors; token IDs stay inside the 1024-token vocabulary. Maximum absolute quantization error is 0.090735 in this synthetic probe. This is an asset check, not policy inference.
- The installed Transformers processor loader omits keyword-only horizon/dimension settings, leaving them initially None. Encoding sets cached dimensions to [30,22], which decoding uses successfully. The production inference pipeline skips FAST action tokenization. No production fix is warranted for this finding.
- All 15 local task/stage packing checks pass, including first/final stage and reset. Local model stage offsets and counts are recorded in the JSON.

The first diagnostic failed because it required the tokenizer's optional constructor dimensions to be populated immediately; the actual encode/decode path works. The second diagnostic's parameter inventory incorrectly filtered VariableState objects with isinstance(nnx.Param); the corrected final check compares all state tensor values. These were audit-harness issues, not checkpoint incompatibilities. Earlier diagnostic logs and reports remain preserved.

Unresolved: structural compatibility does not establish the actual training run's mapping/configuration. Delta-action convention, normalization selection, correlated-noise beta, and stage-table semantics are not encoded by parameter tensor shapes. No saved manifest ties this checkpoint to those local settings. Under the explicit no-guessing requirement, evaluation remains unlaunched pending that evidence. A question requesting the run config/log location is pending.
