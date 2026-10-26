#!/usr/bin/env bash
set -euo pipefail
# Run within the mandated base. VLLM_REF must be a reviewed commit or release.
: "${VLLM_REF:?Set VLLM_REF to the vLLM revision validated on your AMD environment}"
python3 -c 'import torch; assert torch.version.hip, "ROCm PyTorch required"'
python3 -c 'import torch; print("torch==" + torch.__version__)' > /tmp/match4-torch-constraints.txt
export PIP_CONSTRAINT=/tmp/match4-torch-constraints.txt
export VLLM_TARGET_DEVICE=rocm
export MAX_JOBS="${MAX_JOBS:-2}"
git clone https://github.com/vllm-project/vllm.git /tmp/match4-vllm
cd /tmp/match4-vllm
git checkout "$VLLM_REF"
if [ -f tools/use_existing_torch.py ]; then
  python3 tools/use_existing_torch.py
else
  python3 use_existing_torch.py
fi
python3 -m pip install -r requirements/rocm.txt
python3 setup.py develop
python3 -c 'import torch, vllm; assert torch.version.hip; print(torch.__version__, vllm.__version__)'
