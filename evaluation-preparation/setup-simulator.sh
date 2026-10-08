#!/bin/bash
set -euo pipefail
cd /workspace/behavior-1k-solution
export UV_HTTP_TIMEOUT=300 GIT_LFS_SKIP_SMUDGE=1 OMNI_KIT_ACCEPT_EULA=YES
uv pip install --python /workspace/behavior-env/bin/python 'torch==2.6.0' 'torchvision==0.21.0' 'torchaudio==2.6.0' --index-url https://download.pytorch.org/whl/cu124
uv pip install --python /workspace/behavior-env/bin/python -e ./BEHAVIOR-1K/bddl -e './BEHAVIOR-1K/OmniGibson[eval]' -e ./BEHAVIOR-1K/joylo 'numpy<2'
uv pip install --python /workspace/behavior-env/bin/python 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
