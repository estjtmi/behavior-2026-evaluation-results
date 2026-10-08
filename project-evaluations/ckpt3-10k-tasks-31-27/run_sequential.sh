#!/bin/bash
set -euo pipefail
REPO=/workspace/behavior-1k-solution
ARTIFACTS="$REPO/evaluations/ckpt3-10k-tasks-31-27"
OUTPUT=/workspace/evaluation-results/ckpt3-10k-tasks-31-27
mkdir -p "$OUTPUT/logs"
export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_DATA_PATH=/workspace/BEHAVIOR-2026/datasets
export PYTHONHASHSEED=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export TOKENIZERS_PARALLELISM=false
policy_pid=''
cleanup() { if [ -n "$policy_pid" ]; then kill "$policy_pid" 2>/dev/null || true; wait "$policy_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
cd "$REPO"
for item in 31:clean_boxing_gloves 27:sorting_household_items; do
  task_id=${item%%:*}
  task_name=${item#*:}
  /workspace/policy-env/bin/python "$ARTIFACTS/serve_checkpoint.py" --checkpoint /workspace/checkpoints/ckpt3-10k --task-id "$task_id" > "$OUTPUT/logs/policy-$task_id.log" 2>&1 &
  policy_pid=$!
  ready=false
  for attempt in $(seq 1 180); do
    if ! kill -0 "$policy_pid" 2>/dev/null; then echo "Policy startup failed: task $task_id"; exit 1; fi
    if curl --silent --fail http://127.0.0.1:8000/healthz > /dev/null; then ready=true; break; fi
    sleep 5
  done
  if [ "$ready" != true ]; then echo 'Policy startup timed out'; exit 1; fi
  for instance in $(seq 0 9); do
    echo "$(date -u +%FT%TZ) Starting task=$task_id name=$task_name public_index=$instance"
    /workspace/behavior2026-env/bin/python -m omnigibson.eval.eval \
      --task-name "$task_name" --mode public_test --instance-indices "$instance" \
      --num-envs 1 --num-rollouts 1 --host 127.0.0.1 --port 8000 \
      --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
      --output-dir "$OUTPUT/$task_name/instance-$instance" --write-video \
      > "$OUTPUT/logs/eval-$task_id-$instance.log" 2>&1
    echo "$(date -u +%FT%TZ) Completed task=$task_id public_index=$instance"
  done
  cleanup
  policy_pid=''
done
echo "$(date -u +%FT%TZ) All requested evaluations completed"
