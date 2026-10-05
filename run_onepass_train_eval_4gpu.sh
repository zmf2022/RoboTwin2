#!/bin/bash
# 一次训完 + 边训边评（run_train_onepass_4gpu.sh）：每存一个点（默认每 5000 步）就停训练，用 4 卡评这个点的 EMA
# （50 任务 × 10，约 1.3 小时），评完从这个点断点续训；最后一个点（30000）评 50 × 20。结果在 eval_result/onepass/。
# 每次评完就能决定继续还是回退：停掉本脚本即可，存点和评测结果都留着。
# 用法: cd /data/zhangmingfa/RoboTwin2 && setsid nohup bash run_onepass_train_eval_4gpu.sh > onepass_$(date +%m%d_%H%M).log 2>&1 < /dev/null &
# 停止: pkill -f "[r]un_onepass_train_eval_4gpu.sh"; pkill -f "[t]rain_lingbotvla.py"; pkill -f "[s]cripts/eval.sh"
# 重跑本脚本：跳过已评完的点，从最新存点续训。训练每次被停时训练日志里有 SIGTERM，属正常。
# 额外参数原样传给训练（每次续训都带上），如 --train.freeze_vision_encoder true；重跑时要带同样的参数。
# 存点在 ema_hf_ckpt 出现时已完整：DCP 存盘是同步的，HF 导出先写临时目录再整体改名，ema_hf_ckpt 在 hf_ckpt 之后。
cd /data/zhangmingfa/RoboTwin2
C=lingbot-vla-v2/output_onepass/checkpoints
STEPS=${STEPS:-30000}
SAVE=${SAVE:-5000}
TRAIN_LOG=lingbot-vla-v2/train_onepass_$(date +%m%d_%H%M).log
training() { pgrep -f "[t]rain_lingbotvla.py" >/dev/null; }
for s in $(seq $SAVE $SAVE $STEPS); do
  if [ ! -d $C/global_step_$s/ema_hf_ckpt ]; then
    if training; then echo "$(date '+%F %T') 已有训练进程在跑，退出"; exit 1; fi
    echo "$(date '+%F %T') 训练到 $s 步（日志 $TRAIN_LOG）"
    STEPS=$STEPS SAVE=$SAVE bash run_train_onepass_4gpu.sh "$@" >> $TRAIN_LOG 2>&1 &
    pid=$!
    until [ -d $C/global_step_$s/ema_hf_ckpt ] || ! kill -0 $pid 2>/dev/null; do sleep 60; done
    if [ ! -d $C/global_step_$s/ema_hf_ckpt ]; then echo "$(date '+%F %T') 训练在 $s 步前退出，见 $TRAIN_LOG"; exit 1; fi
    if [ $s -lt $STEPS ]; then
      echo "$(date '+%F %T') $s 步已存好，暂停训练去评测"
      pkill -f "[t]rain_lingbotvla.py"; sleep 60; pkill -9 -f "[t]rain_lingbotvla.py"
    fi
    wait $pid
    sleep 30
  fi
  if grep -q TOTAL eval_result/onepass/output_onepass_${s}_ema_*/demo_randomized/stats.txt 2>/dev/null; then continue; fi
  n=10; [ $s -eq $STEPS ] && n=20
  echo "$(date '+%F %T') 评测 $s 步 EMA，50 任务 × $n"
  OUTPUT_BASE=$PWD/eval_result/onepass TEST_NUM=$n RAND_CLIENTS_PER_GPU=6 bash run_eval_4gpu.sh $C/global_step_$s/ema_hf_ckpt 8
  for d in eval_result/onepass/output_onepass_${s}_ema_*; do
    echo "$(date '+%F %T') $d $(for tc in demo_clean demo_randomized; do grep -a TOTAL $d/$tc/stats.txt 2>/dev/null | awk -v t=$tc '{printf "%s %s %s  ", t, $2, $3}'; done)"
  done
done
echo "$(date '+%F %T') 全部完成"
