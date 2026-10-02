# clean 随机化增强（第二阶段）

只用 clean 训练数据，dataloader 里在线把一半样本变成"随机化"样本，提升 randomized 成功率。具体做法：
- 桌面、墙面换纹理，保留原有阴影；
- 在桌面空位贴上其他任务的物体，作为干扰物；
- 光照变化：色温、gamma、局部明暗、极端光；
- 头部相机小幅缩放平移，对应桌高 ±3cm；
- 指令改写。

机械臂和任务物体保持不动，动作标签不变。当前帧和 future 帧共用同一组增强参数。

实现：`lingbot-vla-v2/lingbotvla/data/vla_data/random_aug.py`；脚本与配置：`scripts/random_aug/`。两份配置：

| 配置 | 用途 |
|---|---|
| `random_aug.yaml` | 从基座起训（如 LoRA 第一阶段） |
| `random_aug_s2.yaml` | 第二阶段，从全参 checkpoint 起训：关几何、腕部换背景减半、DTD 纹理、教师看干净画面（§3.2） |

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
# DTD 公开纹理库（625MB，直连约 0.3 MB/s；断了重跑同一命令续传）
mkdir -p data/dtd && curl -L -C - -o data/dtd/dtd-r1.0.1.tar.gz https://thor.robots.ox.ac.uk/dtd/dtd-r1.0.1.tar.gz && tar -xzf data/dtd/dtd-r1.0.1.tar.gz -C data/dtd
```

| 产物 | 位置 | 入库 |
|---|---|---|
| `cam_high_ref.png`、`instruction_paraphrases.json` | `lingbot-vla-v2/assets/random_aug/` | 是 |
| `arm_masks.npy`（3.9G）、`arm_masks_index.npz`、`distractors.npz` | `data/random_aug/` | 否，换机器直接拷：`rsync -a data/random_aug/ <host>:<仓库>/data/random_aug/` |
| DTD 纹理（5640 张，614M） | `data/dtd/dtd/images/` | 否，`rsync -aR data/dtd/dtd/images <host>:<仓库>/` |

- `render_arm_masks.py`：每个 worker 约 1.4G 显存、约 70 帧/s，共 54.9 万帧。和训练共用 GPU 时用 `--workers 1`（约 2.2 小时，期间训练会慢约 25%）。
- 指令改写只读训练指令（`meta/tasks.parquet`，2411 条）。改动了左右、both arms、颜色或数字的改写会被丢弃。没有 GPU 时加 `--device cpu`，约 8 小时。

## 2. 预览（不需要 GPU）

```bash
python scripts/random_aug/preview.py --out random_aug_preview.png --num 6
```

每个样本占两行，上面是原图，下面是增强后。四列依次是头部、左腕、右腕、头部 future 帧；终端会打印改写前后的指令。调参改 `scripts/random_aug/random_aug.yaml`（注释即说明），未写的参数取 `random_aug.py` 里的 `DEFAULTS`。

## 3. 第二阶段训练

> **已验证无效（10/02）**：按本节从 30k 训的第二阶段 clean 升、randomized 降（A100 10k 前 308 局 26.0，同样种子本机 30k 36.9；5k 和 10k 都在 adjust_bottle、move_pillbottle_pad、blocks_ranking_rgb 上崩）。不要再按本节训练，保留作记录；原因分析见 `赛题要求与冲榜方案.md`「全量结果」。

从第一阶段最优的 `hf_ckpt` 初始化（选 (clean+random)/2 最高的；可以先按 §3.1 做权重插值再初始化；LoRA 的 `hf_ckpt` 已合并，也可以用），优化器和学习率从头开始。A100 全参第一阶段子集 clean / randomized：20k 79.0/37.0、30k 81.5/36.5、40k 77.5/32.0，起点用 30k（40k 训练 loss 仍在降、评测变差，见 `赛题要求与冲榜方案.md` §七）。

A100 全参（配方同 `run_train_4gpu.sh`）：

```bash
cd /data/zhangmingfa/RoboTwin2/lingbot-vla-v2
source /root/miniconda3/etc/profile.d/conda.sh && conda activate robotwin
export LD_PRELOAD=/root/miniconda3/envs/robotwin/lib/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=4
MBS=8   # OOM 改 4
setsid nohup bash ../scripts/train.sh ../scripts/robotwin_local.yaml \
  --model.model_path $PWD/output_full/checkpoints/global_step_30000/hf_ckpt \
  --data.train_path $PWD/../data/training_data/RoboTwin_lerobot_v30 \
  --data.random_aug_config ../scripts/random_aug/random_aug_s2.yaml \
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

本机单卡 LoRA（72G，MBS 8 约 27.5 s/步，10000 步约 3.2 天；用 `systemd-run --user` 防止随桌面应用退出）：

```bash
cd lingbot-vla-v2 && MBS=8 && systemd-run --user --unit=lora-s2 --collect --same-dir --setenv=PATH="$PATH" bash -c "eval \"\$(conda shell.bash hook)\"; conda activate robotwin; exec bash ../scripts/train.sh ../scripts/robotwin_local.yaml --model.model_path $PWD/output_full/checkpoints/global_step_30000/hf_ckpt --data.train_path $PWD/../data/training_data/RoboTwin_lerobot_v30 --train.lora_rank 32 --train.lora_alpha 64 --data.random_aug_config ../scripts/random_aug/random_aug_s2.yaml --data.image_augment true --train.output_dir output_lora_s2/ --train.lr 5.0e-5 --train.lr_min 1.0e-5 --train.lr_warmup_ratio 0.02 --train.max_steps 10000 --train.save_steps 5000 --train.save_total_limit 2 --train.micro_batch_size $MBS --train.gradient_accumulation_steps $((256 / MBS)) --train.global_batch_size 256 --train.enable_gradient_checkpointing false --data.num_workers 4 --data.prefetch_factor 1 > train_lora_s2_$(date +%m%d_%H%M).log 2>&1"
```

- `prob: 0.5`：一半样本保持 clean 原样，相当于 clean 与随机化 1:1 混训，用来保住 clean 分。
- 4×A100 约 11–12 s/步（第一阶段后期实测），10000 步约 31–36 小时。第 5000 步跑快速子集，第 10000 步跑全量（A100 每卡 6 个客户端），用 `scripts/compare_results.py` 选点；第二阶段又在同样 2500 条轨迹上多训约 5 轮，5000 步可能优于 10000 步。若 clean 掉了 2 个点以上：把 `prob` 降到 0.3，或把 lr 降到 3e-5。若 randomized 涨得不够：适当提高 `distractor.num`。
- 启动检查：
  - 日志有 `random_aug: prob=0.5 teacher_clean=True arm_masks=True textures=5640 files + procedural, distractors=3060, paraphrased instructions=2364`。
  - 前几步 `VLA_Loss` 约 0.011–0.03（30k 的训练 loss 约 0.011）；接近 0.36 说明没加载上 checkpoint（从基座起训第 1 步的值）。
  - `Depth_Loss` 约 0.65、`FutureVideo_Loss` 约 0.06，高于 30k 时的 0.25 / 0.023，是 teacher_clean 的预期（学生看随机化画面、对齐干净场景）；从基座起训第 1 步是 3.4 / 0.77。训练中应逐步下降。
  - 开头的 `Error detected in IndexPutBackward0` 是 torch.compile 加 graph break 时的警告，第一阶段也有，不影响训练。
- 监控：tfevents 里 `moe_summary/has_dead_expert`、`min_load_ratio`。第一阶段 30k 时为 11% / 0.027，40k 恶化到 22–24% / 0.010；第二阶段持续变差时考虑 `--train.bias_update_speed 0.00025`（未验证）。
- dataloader 开销：每样本约 66 ms（DTD + 教师干净画面），不开增强约 18 ms；机械臂 mask 用 mmap 读，worker 多不占内存。
- 先用 `--train.max_steps 20` 冒烟测试，确认能正常加载 checkpoint 和素材。
- 训练 LoRA 或从头训练时加上同一个 `--data.random_aug_config` 即可启用；推理和评测都不需要改动。

### 3.2 第二阶段配置（`random_aug_s2.yaml`）

| 改动 | 原因 |
|---|---|
| `geometry.prob: 0` | 画面缩放平移而动作不变，标签不一致，会削弱已学会的抓取精度；也会与 teacher_clean 的逐 patch 目标错位 |
| `background.wrist_prob: 0.8 → 0.4` | 腕部只靠颜色分背景（饱和度 <0.15、亮度 >0.6），贴着画面边缘的白/灰物体（碗、锅、铃）补洞失败，会被换成纹理 |
| `background.texture_dir`：DTD | 程序纹理只有条纹、棋盘格、噪声；randomized 评测的背景是照片类纹理 |
| `teacher_clean: true` | 原来深度（MoRGBD）/ 视频（DINO）教师看的是增强后的画面；改为看增强前（也不做 `image_augment`）的画面，学生从随机化画面学干净场景的深度和特征 |

- teacher_clean 实现在 `FeatureTransform`（`lingbotvla/data/vla_data/utils.py`）：增强前浅拷贝 `image` / `future_image`，再生成教师输入 `pil_images` / `future_pil_images`。`pil_images` 只给教师和训练可视化用，学生输入不变；默认关闭，`random_aug.yaml` 行为不变。
- 目标逐 patch 对齐，teacher_clean 要配 `geometry.prob: 0`，否则启动时会打警告。

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
- 纹理来自程序生成和公开纹理库 DTD（`https://www.robots.ox.ac.uk/~vgg/data/dtd/`，`random_aug_s2.yaml`），不用 RoboTwin 自带纹理。

## 5. 已知问题

- 官方 v3.0 数据约 25% 的 episode 中段缺 1–2 帧，训练前用 `scripts/fix_v30_videos.py` 修复（见主指南 3.1），修复后沿用原名 `RoboTwin_lerobot_v30`。素材按修复后的数据生成，训练也必须用修复后的数据。
- 腕部相机里贴着画面边缘的白/灰物体可能被当成背景贴上纹理（第二阶段把 `wrist_prob` 降到 0.4 缓解，未根治）；红色物体在桌面上的反光可能留下小色斑。
- 干扰物只贴头部相机，randomized 评测的腕部画面也会出现杂物。
