#!/usr/bin/env bash
# 将官方 lingbot-vla-v2 训练/推理依赖装入共享的 robotwin 环境，保留已有 PyTorch 不替换
# （官方 tools/create_train_env.sh 会强制 torch==2.8.0）。
set -euo pipefail
export PYTHONNOUSERSITE=1
export PIP_NO_INPUT=1

ENV_NAME="robotwin"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../lingbot-vla-v2" && pwd)"
[[ -f "${REPO_ROOT}/tools/create_train_env.sh" ]] || {
  echo "lingbot-vla-v2/ missing (it is vendored in this repo; check your checkout)" >&2
  exit 1
}

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

TORCH_BEFORE="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
[[ -n "${TORCH_BEFORE}" ]] || {
  echo "no torch in env ${ENV_NAME}; install torch first" >&2
  exit 1
}
echo "[lingbotvla-env] keep existing torch ${TORCH_BEFORE}"

export EXPECTED_TORCH="${TORCH_BEFORE}"
assert_torch_stack() {
  python - <<'PY'
import os
import torch

expected = os.environ["EXPECTED_TORCH"].split("+", 1)[0]
actual = torch.__version__.split("+", 1)[0]
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert actual == expected, f"torch version changed: expected {expected}, got {torch.__version__}"
assert torch.cuda.is_available(), "torch CUDA is not available"
PY
}

install_python_reqs() {
  local req_file="$1"
  local filtered
  filtered="$(mktemp)"
  grep -vE '^(torch|torchvision|torchaudio|torchdata|torchcodec|triton)==' "${req_file}" > "${filtered}"
  python -m pip install -r "${filtered}"
  rm -f "${filtered}"
  assert_torch_stack
}

python -m pip install -U pip setuptools wheel
assert_torch_stack

# --- 以下步骤与官方 tools/create_train_env.sh 对齐（torch 系 pin 已过滤） ---
install_python_reqs "${REPO_ROOT}/requirements.txt"

python -m pip install numpydantic==1.9.0 --no-deps
# torchdata 在过滤名单里（防连坐降 torch），单独按官方 pin 补装
python -m pip install --no-deps torchdata==0.11.0
assert_torch_stack

# [2026-09-25 运维] flash-attn 2.8.3 无 torch2.10 预编译 wheel、宿主只有 CUDA13 nvcc,代码里为可选导入,先跳过

python -m pip install --no-deps \
  "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
assert_torch_stack

python -m pip install -e "${REPO_ROOT}" --no-deps
assert_torch_stack

python -m pip install -r "${REPO_ROOT}/requirements-depth.txt"
assert_torch_stack
install_python_reqs "${REPO_ROOT}/requirements.txt"
python -m pip install numpydantic==1.9.0 --no-deps
assert_torch_stack

python - <<PY
import site
from pathlib import Path

site_packages = Path(site.getsitepackages()[0])
pth = site_packages / "stablevla_local_depth.pth"
pth.write_text("${REPO_ROOT}/lingbotvla/models/vla/vision_models/morgbd_clean/3rd/utils3d\n")
print("wrote", pth)
PY
python -m pip install -e "${REPO_ROOT}/lingbotvla/models/vla/vision_models/lingbot-depth" --no-deps
python -m pip install -e "${REPO_ROOT}/lingbotvla/models/vla/vision_models/MoGe"
assert_torch_stack

python - <<'PY'
import cv2
import accelerate
import mlflow
import trimesh
import moge
import mdm
import utils3d

print("depth imports ok")
PY
python -m pip install huggingface_hub==0.34.0
python -m pip check || {
  echo "[WARN] pip check reported dependency metadata issues." >&2
}

echo "Environment ready: ${ENV_NAME} (torch kept at ${TORCH_BEFORE})"
