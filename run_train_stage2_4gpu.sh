#!/bin/bash
# 第二阶段：从 soup 20-30k 起训，random_aug_stage2.yaml，优化器和学习率从头开始。
# 输出 output_stage2_v2/：output_stage2/ 里是第一版的 checkpoint，用同一目录会从它断点续训。
# 第一版第二阶段（random_aug_s2.yaml）randomized 退化的根因：teacher_clean 让深度/视频教师看干净画面，学生必须从表征里
# 抹掉贴图干扰物，只能学"是否贴图"；训练中没被抹掉的物体全是任务物体，真实杂物不是贴图 -> 被当成目标去抓。
# 本版：关 teacher_clean；干扰物贴到桌面后部和腕部、按来源任务均匀抽、轮廓模糊；去掉物体/机械臂周围残留的白边；纹理压暗、模糊。
# 冒烟（必须先跑，几分钟）: STEPS=20 SAVE=20 OUT=output_stage2_v2_smoke/ bash run_train_stage2_4gpu.sh
#   日志里应有 random_aug: prob=0.5 arm_masks=True textures=5640 files + procedural, distractors=3060（不应再出现 teacher_clean），
#   VLA_Loss 约 0.01-0.03，checkpoint 存盘成功
# 正式: setsid nohup bash run_train_stage2_4gpu.sh > lingbot-vla-v2/train_stage2_v2_$(date +%m%d_%H%M).log 2>&1 < /dev/null &
# 中断后用同一命令重跑即从 $OUT 断点续训
source /root/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
export LD_PRELOAD=/root/miniconda3/envs/robotwin/lib/libstdc++.so.6
cd /data/zhangmingfa/RoboTwin2/lingbot-vla-v2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=4
MBS=${MBS:-8}                  # OOM 改 4
INIT=${INIT:-$PWD/output_full/checkpoints/soup_20-30k/hf_ckpt}   # 第一阶段 20k+30k 等权平均
STEPS=${STEPS:-5000}           # 约 9.3 s/步，5000 步约 13 小时
SAVE=${SAVE:-1000}
KEEP=${KEEP:-5}                 # 保留的 checkpoint 数；全参每个约 77G（hf 24 + model 25 + 优化器约 28），5 个约 400G
OUT=${OUT:-output_stage2_v2/}      # 不要用 output_stage2/（第一版）
bash ../scripts/train.sh ../scripts/robotwin_local.yaml \
  --model.model_path $INIT \
  --data.train_path $PWD/../data/training_data/RoboTwin_lerobot_v30 \
  --data.random_aug_config ../scripts/random_aug/random_aug_stage2.yaml \
  --data.image_augment true \
  --train.output_dir $OUT \
  --train.lr 5.0e-5 --train.lr_min 1.0e-5 --train.lr_warmup_ratio 0.02 \
  --train.max_steps $STEPS --train.save_steps $SAVE --train.save_total_limit $KEEP \
  --train.micro_batch_size $MBS \
  --train.gradient_accumulation_steps $((256 / (MBS * NGPU))) \
  --train.global_batch_size 256 \
  --train.enable_gradient_checkpointing false \
  --data.num_workers 8 --data.prefetch_factor 2
