# clean 随机化增强（第二阶段）

只用 clean 训练数据，dataloader 里在线把一半样本变成"随机化"样本，提升 randomized 成功率。具体做法：
- 桌面、墙面换纹理，保留原有阴影；
- 在桌面空位贴上其他任务的物体，作为干扰物；
- 光照变化：色温、gamma、局部明暗、极端光；
- 头部相机小幅缩放平移，对应桌高 ±3cm；
- 指令改写。

机械臂和任务物体保持不动，动作标签不变。当前帧和 future 帧共用同一组增强参数。

实现：`lingbot-vla-v2/lingbotvla/data/vla_data/random_aug.py`；脚本与配置：`scripts/random_aug/`。

## 1. 生成素材（一次性，仓库根目录）

```bash
conda activate robotwin
# 机械臂 mask：按 clean 数据记录的关节角在 SAPIEN 里渲染每帧机械臂（白色机械臂与白桌面靠颜色分不开）
python scripts/random_aug/render_arm_masks.py --workers 4
python scripts/random_aug/render_arm_masks.py --verify 8 --verify_out arm_overlay.png   # 可选：叠加检查对齐
# 头部相机空场景图 + 干扰物库（约 1 分钟，依赖上一步剔除机械臂碎片）
python scripts/random_aug/build_assets.py --preview distractors.png
# 指令改写（Qwen3-VL-4B-Instruct，GPU 约 10 分钟，可断点续跑）
python scripts/random_aug/gen_instruction_paraphrases.py --device cuda
```

| 产物 | 位置 | 入库 |
|---|---|---|
| `cam_high_ref.png`、`instruction_paraphrases.json` | `lingbot-vla-v2/assets/random_aug/` | 是 |
| `arm_masks.npy`（3.9G）、`arm_masks_index.npz`、`distractors.npz` | `data/random_aug/` | 否，换机器直接拷：`rsync -a data/random_aug/ <host>:<仓库>/data/random_aug/` |

- `render_arm_masks.py`：每个 worker 约 1.4G 显存、约 70 帧/s，共 54.9 万帧。和训练共用 GPU 时用 `--workers 1`（约 2.2 小时，期间训练会慢约 25%）。
- 指令改写只读训练指令（`meta/tasks.parquet`，2411 条）。改动了左右、both arms、颜色或数字的改写会被丢弃。没有 GPU 时加 `--device cpu`，约 8 小时。

## 2. 预览（不需要 GPU）

```bash
python scripts/random_aug/preview.py --out random_aug_preview.png --num 6
```

每个样本占两行，上面是原图，下面是增强后。四列依次是头部、左腕、右腕、头部 future 帧；终端会打印改写前后的指令。调参改 `scripts/random_aug/random_aug.yaml`（注释即说明），未写的参数取 `random_aug.py` 里的 `DEFAULTS`。

## 3. 第二阶段训练

从第一阶段最优的 `hf_ckpt` 初始化（选 (clean+random)/2 最高的；可以先按 §3.1 做权重插值再初始化；LoRA 的 `hf_ckpt` 已合并，也可以用），优化器和学习率从头开始。配方与 A100 第一阶段一致（`run_train_4gpu.sh`）：

```bash
cd /data/zhangmingfa/RoboTwin2/lingbot-vla-v2
source /root/miniconda3/etc/profile.d/conda.sh && conda activate robotwin
export LD_PRELOAD=/root/miniconda3/envs/robotwin/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=4
MBS=8   # OOM 改 4
setsid nohup bash ../scripts/train.sh ../scripts/robotwin_local.yaml \
  --model.model_path $PWD/output_full/checkpoints/global_step_<N>/hf_ckpt \
  --data.train_path $PWD/../data/training_data/RoboTwin_lerobot_v30 \
  --data.random_aug_config ../scripts/random_aug/random_aug.yaml \
  --data.image_augment true \
  --train.output_dir output_stage2/ \
  --train.lr 5.0e-5 \
  --train.lr_min 1.0e-5 \
  --train.lr_warmup_ratio 0.02 \
  --train.max_steps 10000 \
  --train.save_steps 5000 \
  --train.save_total_limit 2 \
  --train.micro_batch_size $MBS \
  --train.gradient_accumulation_steps $((256 / (MBS * NGPU))) \
  --train.global_batch_size 256 \
  --train.enable_gradient_checkpointing false \
  --data.num_workers 8 \
  --data.prefetch_factor 2 > train_stage2_$(date +%m%d_%H%M).log 2>&1 < /dev/null &
```

- `prob: 0.5`：一半样本保持 clean 原样，相当于 clean 与随机化 1:1 混训，用来保住 clean 分。
- 4×A100 约 10 s/步，10000 步约 28 小时。第 5000 步跑快速子集，第 10000 步跑全量（A100 每卡 6 个客户端），用 `scripts/compare_results.py` 选点。若 clean 掉了 2 个点以上：把 `prob` 降到 0.3，或把 lr 降到 3e-5。若 randomized 涨得不够：适当提高 `distractor.num`，或配上 `background.texture_dir`。
- dataloader 开销：每样本约 56 ms，不开增强约 18 ms；机械臂 mask 用 mmap 读，worker 多不占内存。
- 先用 `--train.max_steps 20` 冒烟测试，确认能正常加载 checkpoint 和素材。
- 训练 LoRA 或从头训练时加上同一个 `--data.random_aug_config` 即可启用；推理和评测都不需要改动。

### 3.1 权重插值 / 平均

```bash
B=/mnt/datadisk/models/lingbot-vla/lingbot-vla-v2-6b
E=lingbot-vla-v2/output_full
# WiSE-FT：0.7×微调 + 0.3×基座，α 扫 0.5～0.9
python scripts/interpolate_weights.py $E/checkpoints/global_step_<N>/hf_ckpt:0.7 $B:0.3 --out $E/checkpoints/wise0.7_<N>/hf_ckpt
# 多个 checkpoint 等权平均（soup）
python scripts/interpolate_weights.py $E/checkpoints/global_step_{<N1>,<N2>,<N3>}/hf_ckpt --out $E/checkpoints/soup_<N1>-<N3>/hf_ckpt
# 对比多次评测：总分排序 + 最优 run 的逐任务掉点
python scripts/compare_results.py eval_result/output_full_*
```

- 输出放在 `<实验目录>/checkpoints/<名字>/hf_ckpt`，`eval.sh` 会自动用该实验的配置，评测命令不变。约 1 分钟，输出 24G。
- `--keys REGEX` 只插值匹配的参数，其余取第一个输入。例如只插值 VLM、动作专家保持微调值：`--keys 'qwenvl_with_expert\.qwenvl\.'`。
- 第二阶段结束后同样可在第一、二阶段之间扫 α。

## 4. 合规

- 只用 clean 训练数据（50 任务 × 50 条）。不读取 randomized 数据、RoboTwin 的 `assets/background_texture`，也不读取 unseen 指令模板。
- 机械臂 mask 是在仿真里按 clean 关节角只渲染机器人本体，不含物体，也没有开随机化，只作为分割用，不进训练图像。它不等于"重放轨迹渲染随机化场景"，但仍建议在钉钉群里确认。
- 默认只用程序生成的纹理。如果要用公开纹理库（如 DTD，`https://www.robots.ox.ac.uk/~vgg/data/dtd/`），解压后把 `background.texture_dir` 指向 `dtd/images`。

## 5. 已知问题

- 官方 v3.0 数据约 25% 的 episode 中段缺 1–2 帧，训练前用 `scripts/fix_v30_videos.py` 修复（见主指南 3.1），修复后沿用原名 `RoboTwin_lerobot_v30`。素材按修复后的数据生成，训练也必须用修复后的数据。
- 腕部相机里的白色物体可能被部分当成背景贴上纹理；红色物体在桌面上的反光可能留下小色斑。
