#!/bin/bash
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 OMNIGIBSON_DATA_PATH=/workspace/BEHAVIOR-2026/datasets
export PYTHONHASHSEED=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 TOKENIZERS_PARALLELISM=false
R=/workspace/behavior-1k-solution
O=/workspace/evaluation-results/ckpt3-10k-tasks-31-27/wash_a_baseball_cap/instance-batch
mkdir -p "$O"
/workspace/policy-env/bin/python "$R/evaluations/ckpt3-10k-tasks-31-27/serve_checkpoint.py" --checkpoint /workspace/checkpoints/ckpt3-10k --task-id 32 --port 8000 > "$O/policy.log" 2>&1 &
policy_pid=$!
trap 'kill "$policy_pid" 2>/dev/null || true' EXIT
ready=false
for i in $(seq 1 120); do
 kill -0 "$policy_pid"
 if curl -fsS http://127.0.0.1:8001/healthz >/dev/null 2>&1; then ready=true; break; fi
 sleep 5
done
[ "$ready" = true ]
/workspace/behavior2026-env/bin/python "$R/evaluations/task32-rgb/run_instances.py" --task-name wash_a_baseball_cap --mode public_test --instance-indices 0 --num-envs 1 --num-rollouts 1 --host 127.0.0.1 --port 8000 --env-wrapper omnigibson.eval.wrappers.DefaultWrapper --output-dir "$O" --write-video > "$O/evaluator.log" 2>&1
echo "Task 32 full evaluation completed"
python /workspace/eval-prep/publish-results.py
/workspace/policy-env/bin/python /workspace/eval-prep/upload-videos.py
