#!/bin/bash
set -euo pipefail
cd /workspace/behavior-1k-solution
export UV_HTTP_TIMEOUT=300 GIT_LFS_SKIP_SMUDGE=1
uv pip install --python /workspace/policy-env/bin/python -e ./openpi/packages/openpi-client -e ./openpi -e . 'lerobot @ git+https://github.com/huggingface/lerobot@577cd10974b84bea1f06b6472eb9e5e74e07f77a' 'ml-dtypes==0.4.1' 'tensorstore==0.1.74'
uv pip install --python /workspace/policy-env/bin/python -e ./BEHAVIOR-1K/bddl -e './BEHAVIOR-1K/OmniGibson[eval]'
