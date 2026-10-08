#!/bin/bash
set -Eeuo pipefail
export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_DATA_PATH=/workspace/BEHAVIOR-2026/datasets
status() {
  /usr/bin/python3 - "$1" "${2:-}" <<'PY'
import json,sys,datetime,pathlib
pathlib.Path('/workspace/eval-prep/status.json').write_text(json.dumps({'status':sys.argv[1],'detail':sys.argv[2],'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
PY
}
on_failure() {
  code=$?
  status failed "Supervisor stopped at line $1 with exit $code; inspect supervisor.log and evaluation logs"
  /usr/bin/python3 /workspace/eval-prep/publish-results.py || true
  exit "$code"
}
trap 'on_failure $LINENO' ERR
status waiting_for_installation
while kill -0 8176 2>/dev/null; do sleep 15; done
status checking_dependencies
uv pip install --python /workspace/behavior2026-env/bin/python 'numpy<2' 'warp-lang==1.12.0' 'websockets==15.0.1'
/workspace/behavior2026-env/bin/python - <<'PY'
import isaacsim
import omnigibson
from omnigibson.utils.asset_utils import download_key
from omnigibson.eval.policies import WebsocketPolicy
download_key()
print('2026 simulator imports and asset key ready',flush=True)
PY
/workspace/policy-env/bin/python - <<'PY'
from b1k.training import config
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper
print('Policy imports ready',flush=True)
PY
status evaluating 'Task 31 followed by task 27, public indices 0-9 each'
bash /workspace/behavior-1k-solution/evaluations/ckpt3-10k-tasks-31-27/run_sequential.sh
status completed
/usr/bin/python3 /workspace/eval-prep/publish-results.py
/workspace/policy-env/bin/python /workspace/eval-prep/upload-videos.py
