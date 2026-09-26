#!/bin/bash
# 2026-09-25 由运维按 zhangmingfa 的 bash_history 整理（原命令未改，仅固化成脚本）
source /root/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
# torchcodec 用的 conda-forge ffmpeg 依赖新版 libstdc++;在「已激活 robotwin 的终端」里 conda activate 是空操作,env 变量不会生效,故这里显式设置
export LD_PRELOAD=/root/miniconda3/envs/robotwin/lib/libstdc++.so.6
cd /data/zhangmingfa/RoboTwin2/lingbot-vla-v2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=4
MBS=${MBS:-4}
bash ../scripts/train.sh ../scripts/robotwin_local.yaml \
  --data.train_path $PWD/../data/training_data/RoboTwin_lerobot_v30 \
  --train.output_dir output_full/ \
  --train.max_steps 50000 --train.save_steps 5000 \
  --train.micro_batch_size $MBS \
  --train.gradient_accumulation_steps $((256 / (MBS * NGPU))) \
  --train.global_batch_size 256 \
  --train.enable_gradient_checkpointing false \
  --data.image_augment true --data.num_workers 8 --data.prefetch_factor 2
