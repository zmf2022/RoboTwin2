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
| `random_aug_s4.yaml` | 第二阶段，从全参 checkpoint 起训（§3.2） |

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

从第一阶段 soup 20k+30k 起训（子集 clean / randomized 81.5 / 42.0，好于单个 checkpoint），`random_aug_s4.yaml`，5000 步，优化器和学习率从头开始（5e-5 → 1e-5）。A100 命令见 `A100多卡训练指南.md`「第二阶段」（`run_train_s4_4gpu.sh`）。

- `prob: 0.5`：一半样本保持 clean 原样，相当于 clean 与随机化 1:1 混训，用来保住 clean 分。
- 启动检查：日志有 `random_aug: prob=0.5 arm_masks=True textures=5640 files + procedural, distractors=3060, paraphrased instructions=2364`；前几步 `VLA_Loss` 约 0.01–0.03，接近 0.36 说明没加载上 checkpoint。开头的 `Error detected in IndexPutBackward0` 是 torch.compile 的警告，不影响训练。
- dataloader 开销约 40 ms/样本；机械臂 mask 用 mmap 读，worker 多不占内存。
- 推理和评测都不需要改动。

### 3.2 第二阶段配置（`random_aug_s4.yaml`）

第一版（`random_aug_s2.yaml`，已删）从 30k 训 5k/10k：clean 升 5.9，randomized 降（A100 10k 27.2）。根因定位（8 个崩溃任务 × 20 局，clean 只加一项，s2 5k / 30k 配对）：只加杂物 s2 不掉，只加纹理小降，两者一起大面积崩（adjust_bottle 6、blocks_ranking_rgb 2、place_object_basket 2 /20）；视频里第一个动作块就伸手去抓后部的真实杂物。评测时只给任务物体画一圈 2 px 白边，s2 纹理 + 杂物从 51 升到 131/160（不加白边 30k 108、soup 121），改给杂物画则降到 24/160：s2 靠换纹理后残留的白边认任务物体（贴图干扰物没有白边）。

| 改动 | 原因 |
|---|---|
| 去掉 teacher_clean（深度/视频教师看干净画面） | 学生必须从表征里抹掉贴图干扰物，只能学"是否贴图"；训练里没被抹掉的物体全是任务物体，真实杂物不是贴图，就被当成目标。没有 teacher_clean 的 `random_aug.yaml`（LoRA）在同样的任务上不崩 |
| `distractor`：`prob 0.95`、`num [3, 8]`、`top_margin 0.02`、`wrist_prob 0.6`、`task_balance` | 真实杂物 98% 局都有、每画面多个、靠墙的桌面后部也有、腕部初始视角正对后部杂物；s2 只在约 35% 样本贴 1–4 个、只贴头部、不贴后部；库里积木占 1/4，按来源任务均匀抽 |
| `mask.rim_px 2`、`arm_dilate_px 0` | 换纹理后物体和机械臂周围残留一圈白桌面，真实画面没有 |
| `background.brightness [0.15, 0.55]`、`blur [1, 3]`、`prob 0.98`、`wrist_prob 0.8` | 真实 randomized 头部画面平均亮度 0.33（s2 增强后 0.51）、纹理较平滑；腕部基本都是纹理 |
| `geometry.prob 0` | 画面缩放平移而动作不变，标签不一致，会削弱抓取精度 |

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
- 纹理来自程序生成和公开纹理库 DTD（`https://www.robots.ox.ac.uk/~vgg/data/dtd/`，`random_aug_s4.yaml`），不用 RoboTwin 自带纹理。

## 5. 已知问题

- 官方 v3.0 数据约 25% 的 episode 中段缺 1–2 帧，训练前用 `scripts/fix_v30_videos.py` 修复（见主指南 3.1），修复后沿用原名 `RoboTwin_lerobot_v30`。素材按修复后的数据生成，训练也必须用修复后的数据。
- 腕部相机只靠颜色分背景，白/灰物体（铃、盘、锅）可能被当成背景贴上纹理；红色物体在桌面上的反光可能留下小色斑。
- 腕部干扰物是独立贴的，和头部画面里的位置不对应。
