#!/bin/bash
# 第二阶段：从 soup 20-30k 起训，random_aug_stage2.yaml，优化器和学习率从头开始。
# v3（当前默认）= v2 + mask_padded_actions：episode 末尾补齐的动作（重复最后一帧）不计损失，否则模型在像结尾的姿态
#   学成原地不动（本机 A/B clean +2.8、randomized +5.1）；2500 步、每 500 步存一个、5 个全部保留：v2（5000 步）的
#   randomized 3k > 4k > 5k，越训越差，峰值可能更早。
# 输出 output_stage2_v3/：output_stage2/、output_stage2_v2/ 是前两版的 checkpoint，用同一目录会从它断点续训。
# 第一版第二阶段（random_aug_s2.yaml）randomized 退化的根因：teacher_clean 让深度/视频教师看干净画面，学生必须从表征里
# 抹掉贴图干扰物，只能学"是否贴图"；训练中没被抹掉的物体全是任务物体，真实杂物不是贴图 -> 被当成目标去抓。
# v2 起：关 teacher_clean；干扰物贴到桌面后部和腕部、按来源任务均匀抽、轮廓模糊；去掉物体/机械臂周围残留的白边；纹理压暗、模糊。
# 冒烟（必须先跑，几分钟）: STEPS=20 SAVE=20 OUT=output_stage2_v3_smoke/ bash run_train_stage2_4gpu.sh
#   日志里应有 random_aug: prob=0.5 arm_masks=True textures=5640 files + procedural, distractors=3060（不应再出现 teacher_clean），
#   和 EMA decay=0.99 every=10，VLA_Loss 约 0.01-0.03，<OUT>/lingbotvla_cli.yaml 里 mask_padded_actions: true，
#   checkpoint 存盘成功，global_step_20 下有 hf_ckpt 和 ema_hf_ckpt
# EMA（openpi 的 0.99）只额外保存一份滑动平均权重（<ckpt>/ema、ema_hf_ckpt），不改变原始训练；EMA=0 关闭
# 正式: setsid nohup bash run_train_stage2_4gpu.sh > lingbot-vla-v2/train_stage2_v3_$(date +%m%d_%H%M).log 2>&1 < /dev/null &
# 中断后用同一命令重跑即从 $OUT 断点续训
# 额外参数原样传给训练，如状态历史：OUT=output_stage2_v4/ bash run_train_stage2_4gpu.sh --train.state_history_frames 10
source /root/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
export LD_PRELOAD=/root/miniconda3/envs/robotwin/lib/libstdc++.so.6
cd /data/zhangmingfa/RoboTwin2/lingbot-vla-v2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=4
MBS=${MBS:-8}                  # OOM 改 4
INIT=${INIT:-$PWD/output_full/checkpoints/soup_20-30k/hf_ckpt}   # 第一阶段 20k+30k 等权平均
STEPS=${STEPS:-2500}           # 约 9.3 s/步，2500 步约 6.5 小时
SAVE=${SAVE:-500}
KEEP=${KEEP:-5}                 # 保留的 checkpoint 数（默认 5 个存点全部保留，早期点要能评测）；全参每个约 126G（hf 24 + model 25 + 优化器约 28 + ema 25 + ema_hf 24），5 个约 630G，存盘时短暂多一个
EMA=${EMA:-0.99}
OUT=${OUT:-output_stage2_v3/}      # 不要用 output_stage2/、output_stage2_v2/（前两版）
bash ../scripts/train.sh ../scripts/robotwin_local.yaml \
  --model.model_path $INIT \
  --data.train_path $PWD/../data/training_data/RoboTwin_lerobot_v30 \
  --data.random_aug_config ../scripts/random_aug/random_aug_stage2.yaml \
  --data.image_augment true \
  --train.output_dir $OUT \
  --train.lr 5.0e-5 --train.lr_min 1.0e-5 --train.lr_warmup_ratio 0.02 \
  --train.ema_decay $EMA --train.ema_every 10 \
  --train.mask_padded_actions true \
  --train.max_steps $STEPS --train.save_steps $SAVE --train.save_total_limit $KEEP \
  --train.micro_batch_size $MBS \
  --train.gradient_accumulation_steps $((256 / (MBS * NGPU))) \
  --train.global_batch_size 256 \
  --train.enable_gradient_checkpointing false \
  --data.num_workers 8 --data.prefetch_factor 2 "$@"
