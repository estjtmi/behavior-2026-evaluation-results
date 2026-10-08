#!/bin/bash
set -euo pipefail
export UV_HTTP_TIMEOUT=300 OMNI_KIT_ACCEPT_EULA=YES
uv pip install --python /workspace/behavior2026-env/bin/python 'torch==2.7.0' 'torchvision==0.22.0' 'torchaudio==2.7.0' 'torchcodec==0.5' --index-url https://download.pytorch.org/whl/cu126
uv pip install --python /workspace/behavior2026-env/bin/python -e /workspace/BEHAVIOR-2026/bddl3 -e '/workspace/BEHAVIOR-2026/OmniGibson[eval]' -e /workspace/BEHAVIOR-2026/joylo
uv pip install --python /workspace/behavior2026-env/bin/python 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
uv pip install --python /workspace/behavior2026-env/bin/python 'numpy<2' 'warp-lang==1.12.0'
