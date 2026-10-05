#!/bin/bash
# 从基础模型一次训完（不再分第一、第二阶段），全部功能打开：
# - 相对关节动作：robotwin_rel（手臂关节 = 动作 - 当前状态，夹爪绝对）+ norm stats robotwin_clean_rel.json（openpi π0.5、GR00T N1.7）
# - state 丢弃 0.2：数据层把归一化后的 state 置零，模型层把 state token 嵌入置零（默认跟随数据层），只在训练时（GR00T N1.7 微调）
# - 补齐动作不计损失（mask_padded_actions，v3）
# - 去掉静止帧（idle_threshold 1e-4，PatchWAM：21.4% 的帧 14 个关节都不动）
# - 增强 random_aug_onepass.yaml：v3 场景增强（纹理、干扰物、指令改写）+ PatchWAM 外观增强（代替光照）；上游 image_augment 关掉（和外观增强重复，实测开关对画面分布几乎没影响，省约 3%）
# - EMA 0.99（只额外存 ema/、ema_hf_ckpt/）
# - 优化器 dist_muon（上游分布式 Muon，设置照 configs/vla/robotwin/robotwin_dist_muon.yaml）：正交化分到各卡并和通信重叠；
#   和原 muon 的区别：MoE 专家不再用约 2.8 倍的单独学习率（dist_muon 只支持一个 Muon 参数组）、Q/K/V 按注意力头正交化、
#   第 35 层 MLP 和 o_proj 用 AdamW。OPT=muon 切回原优化器（专家学习率照旧）
# 学习率 1e-4 -> 1e-5 余弦，前 2% 预热；3 万步约 3.2 天（约 9.3 s/步）；每 5000 步存一个，6 个全部保留（每个约 126G，共约 760G）。
# 冒烟（可选，几分钟）: STEPS=20 SAVE=20 OUT=output_onepass_smoke/ bash run_train_onepass_4gpu.sh
#   日志里应有 non-idle filter ... 431174 of 548893 frames kept、state dropout: normalised state p=0.2, state token embedding p=0.2、
#   random_aug: prob=0.5 ...、EMA decay=0.99 every=10；从基础模型起训，前几步 VLA_Loss 明显高于 0.1（0.01-0.03 说明误加载了训过的权重）；
#   [dist_muon] matrix_params=...；<OUT>/lingbotvla_cli.yaml 里 data_name: robotwin_rel、optimizer: dist_muon；global_step_20 下有 hf_ckpt 和 ema_hf_ckpt
# 正式: setsid nohup bash run_train_onepass_4gpu.sh > lingbot-vla-v2/train_onepass_$(date +%m%d_%H%M).log 2>&1 < /dev/null &
# 中断后用同一命令重跑即从 $OUT 断点续训；额外参数原样传给训练，如 --train.freeze_vision_encoder true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate robotwin
export LD_PRELOAD=/root/miniconda3/envs/robotwin/lib/libstdc++.so.6
cd /data/zhangmingfa/RoboTwin2/lingbot-vla-v2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=4
MBS=${MBS:-8}                  # OOM 改 4
STEPS=${STEPS:-30000}
SAVE=${SAVE:-5000}
KEEP=${KEEP:-6}
EMA=${EMA:-0.99}
OUT=${OUT:-output_onepass/}
OPT=${OPT:-dist_muon}
OPT_ARGS=()
if [ "$OPT" = dist_muon ]; then
  L35=qwenvl.model.language_model.layers.35
  OPT_ARGS=(--train.optimizer dist_muon --train.use_moe_expert_lr false
            --train.dist_muon_attn_per_head true --train.dist_muon_layers_per_bucket 2
            --train.muon_exclude_name_patterns $L35.mlp.down_proj.weight $L35.mlp.gate_proj.weight
                                               $L35.mlp.up_proj.weight $L35.self_attn.o_proj.weight)
fi
bash ../scripts/train.sh ../scripts/robotwin_local.yaml \
  --data.train_path $PWD/../data/training_data/RoboTwin_lerobot_v30 \
  --data.data_name robotwin_rel --data.norm_stats_file assets/norm_stats/robotwin_clean_rel.json \
  --data.state_dropout_prob 0.2 --data.idle_threshold 1e-4 \
  --data.random_aug_config ../scripts/random_aug/random_aug_onepass.yaml \
  --data.image_augment false \
  --train.output_dir $OUT \
  --train.lr 1.0e-4 --train.lr_min 1.0e-5 --train.lr_warmup_ratio 0.02 \
  --train.ema_decay $EMA --train.ema_every 10 \
  --train.mask_padded_actions true \
  --train.max_steps $STEPS --train.save_steps $SAVE --train.save_total_limit $KEEP \
  --train.micro_batch_size $MBS \
  --train.gradient_accumulation_steps $((256 / (MBS * NGPU))) \
  --train.global_batch_size 256 \
  --train.enable_gradient_checkpointing false \
  --data.num_workers 8 --data.prefetch_factor 2 "${OPT_ARGS[@]}" "$@"
