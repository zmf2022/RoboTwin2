#!/bin/bash
# 用法: bash run_eval_4gpu.sh <hf_ckpt> [每卡客户端数，默认 8]
source /root/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
export LD_PRELOAD=/root/miniconda3/envs/robotwin/lib/libstdc++.so.6
export QWEN3VL_PATH=/data/zhangmingfa/Qwen3-VL-4B-Instruct/Qwen3-VL-4B-Instruct
cd /data/zhangmingfa/RoboTwin2
bash scripts/eval.sh "$1" 4 "${2:-8}"
