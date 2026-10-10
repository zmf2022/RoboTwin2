#!/bin/bash
# 合规版从基础模型一次训完（6 卡 A100 80G）：只用组委会未禁止的方案（10/09 禁止项：纹理、贴图、改几何、重光照、生成式编辑、改指令）。
# - 相对关节动作：robotwin_rel（手臂关节 = 动作 - 当前状态，夹爪绝对）+ norm stats robotwin_clean_rel.json
# - 补齐动作不计损失（mask_padded_actions，本机 A/B clean +2.8、randomized +5.1）
# - 去掉静止帧（idle_threshold 1e-4，本机 A/B 约 +2-3）
# - 增强 random_aug_onepass_c.yaml：只有 PatchWAM 外观增强的亮度/颜色部分；上游 image_augment 关掉（与之重复）
# - state 丢弃关（A/B 无收益）；优化器 muon（第一阶段用的，dist_muon 无提速且去掉专家学习率）
# - EMA 0.99（只额外存 ema/、ema_hf_ckpt/）；学习率 1e-4 -> 1e-5 余弦，前 2% 预热
# 6 卡：MBS 8 × 累积 5 × 6 卡 = GBS 240（4 卡版 256），约 5.8 s/步，3 万步约 2 天；每 5000 步存一个，6 个全部保留（每个约 126G，共约 760G）。
# 6 卡下语言模型的 2D 权重第 0 维不能被 6 整除（2560、9728、4096、1024），需要 muon.py 按 torch.chunk 分片重建的修复，旧代码会把更新写错位。
# 冒烟（必须先跑，几分钟）: STEPS=20 SAVE=20 OUT=output_onepass_c_smoke/ bash run_train_onepass_c_6gpu.sh
#   日志里应有 non-idle filter ... 431174 of 548893 frames kept、random_aug: prob=0.0 ... distractors=0, paraphrased instructions=0、
#   EMA decay=0.99 every=10；从基础模型起训，前几步 VLA_Loss 明显高于 0.1（0.01-0.03 说明误加载了训过的权重）；
#   <OUT>/lingbotvla_cli.yaml 里 data_name: robotwin_rel、optimizer: muon、state_dropout_prob: 0.0、mask_padded_actions: true；
#   global_step_20 下有 hf_ckpt 和 ema_hf_ckpt。用完删掉 output_onepass_c_smoke/
# 正式: setsid nohup bash run_train_onepass_c_6gpu.sh > lingbot-vla-v2/train_onepass_c_$(date +%m%d_%H%M).log 2>&1 < /dev/null &
# 中断后用同一命令重跑即从 $OUT 断点续训；额外参数原样传给训练
source /root/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
export LD_PRELOAD=/root/miniconda3/envs/robotwin/lib/libstdc++.so.6
cd /data/zhangmingfa/RoboTwin2/lingbot-vla-v2
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
MBS=${MBS:-8}                  # OOM 改 4（ACCUM 相应改 10）
ACCUM=${ACCUM:-5}
STEPS=${STEPS:-30000}
SAVE=${SAVE:-5000}
KEEP=${KEEP:-6}
EMA=${EMA:-0.99}
OUT=${OUT:-output_onepass_c/}
bash ../scripts/train.sh ../scripts/robotwin_local.yaml \
  --data.train_path $PWD/../data/training_data/RoboTwin_lerobot_v30 \
  --data.data_name robotwin_rel --data.norm_stats_file assets/norm_stats/robotwin_clean_rel.json \
  --data.state_dropout_prob 0 --data.idle_threshold 1e-4 \
  --data.random_aug_config ../scripts/random_aug/random_aug_onepass_c.yaml \
  --data.image_augment false \
  --train.output_dir $OUT \
  --train.optimizer muon \
  --train.lr 1.0e-4 --train.lr_min 1.0e-5 --train.lr_warmup_ratio 0.02 \
  --train.ema_decay $EMA --train.ema_every 10 \
  --train.mask_padded_actions true \
  --train.max_steps $STEPS --train.save_steps $SAVE --train.save_total_limit $KEEP \
  --train.micro_batch_size $MBS \
  --train.gradient_accumulation_steps $ACCUM \
  --train.global_batch_size $((MBS * ACCUM * NGPU)) \
  --train.enable_gradient_checkpointing false \
  --data.num_workers 8 --data.prefetch_factor 2 "$@"
