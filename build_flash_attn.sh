#!/bin/bash
# 2026-09-25 运维:flash-attn 2.8.3 无 torch2.10 预编译 wheel,用 env 内 conda 装的 CUDA 12.8 源码编译(只编 A100 sm_80)
source /root/miniconda3/etc/profile.d/conda.sh; conda activate robotwin
# 必须用系统 gcc 11:conda 激活会把 CC/CXX 指向 env 里的 gcc 14,编出的 .so 要 CXXABI_1.3.15,运行时系统 libstdc++ 没有
export CC=/usr/bin/gcc CXX=/usr/bin/g++ NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++"
export CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=32 NVCC_THREADS=2
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ PIP_TRUSTED_HOST=mirrors.aliyun.com
nice -n 10 python -m pip install --no-build-isolation --no-cache-dir --force-reinstall --no-deps -v flash-attn==2.8.3
python -c "import torch, flash_attn; print(\"flash_attn\", flash_attn.__version__, \"torch\", torch.__version__)"
