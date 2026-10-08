#!/bin/bash
set -euo pipefail
echo "$(date -u +%FT%TZ) Benchmark queued after tasks 31 and 27 and uploads"
while true; do
 state=$(python -c 'import json; print(json.load(open("/workspace/eval-prep/status.json"))["status"])')
 if [ "$state" = failed ]; then echo 'Current evaluation failed; benchmark will not start'; exit 1; fi
 if [ "$state" = completed ] && ! pgrep -f '^bash /workspace/eval-prep/supervise.sh$' >/dev/null; then break; fi
 sleep 30
done
python /workspace/eval-prep/publish-results.py
/workspace/policy-env/bin/python /workspace/eval-prep/upload-videos.py
echo "$(date -u +%FT%TZ) Starting benchmark; no wall-clock time limit"
bash /workspace/eval-prep/benchmark-task32.sh
echo "$(date -u +%FT%TZ) Benchmark complete; see /workspace/evaluation-results/task32-speed-benchmark/timing.json"

echo "$(date -u +%FT%TZ) Starting authorized full task-32 evaluation"
bash /workspace/eval-prep/evaluate-task32.sh
