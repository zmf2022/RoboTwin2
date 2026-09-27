#!/usr/bin/env bash
# Full LingBot-VLA 2.0 evaluation on this repo: 50 tasks x {demo_clean, demo_randomized} x 100 episodes, FP32.
# Usage: bash scripts/eval.sh <model_dir> [num_gpus] [clients_per_gpu]
#   one inference server per GPU; clients_per_gpu sim clients share it (sim is CPU-bound, ~1 core each)
#   model_dir: .../checkpoints/global_step_N/hf_ckpt (fine-tuned) or a flat model dir (e.g. the base model)
# Env overrides: CLI_YAML, OUTPUT_BASE, QWEN3VL_PATH, TASK_CONFIGS, TASKS, TEST_NUM, USE_LENGTH, NUM_STEPS, VIDEO, CONDA_ENV
#   non-default USE_LENGTH / NUM_STEPS (denoising steps, model default 10) are appended to the run name
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="$(realpath "${1:?model dir}")"
NUM_GPUS="${2:-$(nvidia-smi -L | wc -l)}"
CLIENTS_PER_GPU="${3:-1}"
OUTPUT_BASE="${OUTPUT_BASE:-$ROOT/eval_result}"
TASK_CONFIGS="${TASK_CONFIGS:-demo_clean demo_randomized}"
USE_LENGTH="${USE_LENGTH:-50}"
NUM_STEPS="${NUM_STEPS:-}"
VIDEO="${VIDEO:-1}"
TEST_NUM="${TEST_NUM:-100}"
export QWEN3VL_PATH="${QWEN3VL_PATH:-/mnt/datadisk/models/lingbot-vla/Qwen3-VL-4B-Instruct}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
MAX_RETRIES=3

ALL_TASKS="lift_pot hanging_mug stack_bowls_three scan_object handover_block click_bell put_object_cabinet
open_microwave stack_blocks_three place_shoe adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
click_alarmclock dump_bin_bigbin grab_roller handover_mic move_can_pot move_pillbottle_pad move_playingcard_away
place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad
place_object_basket place_object_scale place_object_stand place_phone_stand move_stapler_pad open_laptop
pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet
place_burger_fries place_can_basket press_stapler rotate_qrcode shake_bottle_horizontally shake_bottle
stack_blocks_two stack_bowls_two stamp_seal turn_switch put_bottles_dustbin"
read -r -a TASK_LIST <<< "$(echo ${TASKS:-$ALL_TASKS})"

[[ -f "$MODEL_PATH/config.json" ]] || { echo "no config.json in $MODEL_PATH" >&2; exit 1; }
# training arg dump: <exp>/lingbotvla_cli.yaml for train outputs, else the base-model dump in scripts/
EXP_DIR="$(dirname "$(dirname "$(dirname "$MODEL_PATH")")")"
if [[ -n "${CLI_YAML:-}" ]]; then
  NAME="$(basename "$MODEL_PATH")"
elif [[ -f "$EXP_DIR/lingbotvla_cli.yaml" ]]; then
  CLI_YAML="$EXP_DIR/lingbotvla_cli.yaml"
  STEP="$(basename "$(dirname "$MODEL_PATH")")"
  NAME="$(basename "$EXP_DIR")_${STEP#global_step_}"
else
  CLI_YAML="$ROOT/scripts/lingbotvla_cli_base.yaml"
  NAME="$(basename "$MODEL_PATH")"
fi
[[ $USE_LENGTH != 50 ]] && NAME+="_len$USE_LENGTH"
[[ -n $NUM_STEPS ]] && NAME+="_ns$NUM_STEPS"
export LINGBOT_CLI_YAML="$(realpath "$CLI_YAML")"
echo "model: $MODEL_PATH"
echo "config: $LINGBOT_CLI_YAML"

set +u  # conda (de)activate.d scripts reference unset vars
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-robotwin}"
set -u

NUM_SLOTS=$((NUM_GPUS * CLIENTS_PER_GPU))
for ((g = 0; g < NUM_GPUS; g++)); do
  if (exec 3<>/dev/tcp/127.0.0.1/$((9330 + g))) 2>/dev/null; then
    echo "port $((9330 + g)) already in use (stale inference server?)" >&2; exit 1
  fi
done

RUN_DIR="$OUTPUT_BASE/${NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/inference_logs"
echo "run dir: $RUN_DIR"

declare -a SERVER_PID SLOT_PID SLOT_TASK
cleanup() {
  for p in "${SLOT_PID[@]}" "${SERVER_PID[@]}"; do [[ -n "$p" && "$p" != 0 ]] && kill -- -"$p" 2>/dev/null; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# ---- inference servers (one per GPU, resident for all settings) ----
for ((g = 0; g < NUM_GPUS; g++)); do
  (CUDA_VISIBLE_DEVICES=$g exec setsid python "$ROOT/scripts/lingbot_policy_server.py" \
    --model_path "$MODEL_PATH" --use_length "$USE_LENGTH" --use_bf16 False --use_fp32 True --use_compile True \
    ${NUM_STEPS:+--num_steps "$NUM_STEPS"} --port $((9330 + g))) > "$RUN_DIR/inference_logs/gpu$g.log" 2>&1 &
  SERVER_PID[$g]=$!
done
for ((s = 0; s < NUM_SLOTS; s++)); do SLOT_PID[$s]=0; done

launch() {  # slot task_config task
  local s=$1 tc=$2 t=$3
  CUDA_VISIBLE_DEVICES=$((s % NUM_GPUS)) setsid python -u "$ROOT/scripts/eval_lingbotvla_client.py" \
    --task_name "$t" --task_config "$tc" --port $((9330 + s % NUM_GPUS)) \
    --output_dir "$RUN_DIR/$tc/eval_results" --video "$VIDEO" --test_num "$TEST_NUM" >> "$RUN_DIR/$tc/eval_logs/$t.log" 2>&1 &
  SLOT_PID[$s]=$!
  SLOT_TASK[$s]="$t"
}

for TC in $TASK_CONFIGS; do
  mkdir -p "$RUN_DIR/$TC/eval_logs" "$RUN_DIR/$TC/eval_results"
  declare -A RETRY=()
  queue=("${TASK_LIST[@]}")
  while :; do
    for ((g = 0; g < NUM_GPUS; g++)); do
      if ! kill -0 "${SERVER_PID[$g]}" 2>/dev/null; then
        echo "inference server on GPU $g exited, see $RUN_DIR/inference_logs/gpu$g.log" >&2; exit 1
      fi
    done
    for ((s = 0; s < NUM_SLOTS; s++)); do
      if [[ ${SLOT_PID[$s]} != 0 ]] && ! kill -0 "${SLOT_PID[$s]}" 2>/dev/null; then
        wait "${SLOT_PID[$s]}"; rc=$?
        t=${SLOT_TASK[$s]}
        if [[ $rc != 0 || ! -f "$RUN_DIR/$TC/eval_results/$t/result.json" ]]; then
          RETRY[$t]=$(( ${RETRY[$t]:-0} + 1 ))
          if (( RETRY[$t] < MAX_RETRIES )); then
            echo "[$TC] $t failed (rc=$rc), retry ${RETRY[$t]}"; queue=("$t" "${queue[@]}")
          else
            echo "[$TC] $t failed $MAX_RETRIES times, skipped"
          fi
        else
          echo "[$TC] $t done: $(grep -a 'Success rate' "$RUN_DIR/$TC/eval_logs/$t.log" | tail -1)"
        fi
        SLOT_PID[$s]=0
      fi
      if [[ ${SLOT_PID[$s]} == 0 && ${#queue[@]} -gt 0 ]]; then
        launch "$s" "$TC" "${queue[0]}"; queue=("${queue[@]:1}")
      fi
    done
    busy=0; for ((s = 0; s < NUM_SLOTS; s++)); do [[ ${SLOT_PID[$s]} != 0 ]] && busy=1; done
    [[ $busy == 0 && ${#queue[@]} == 0 ]] && break
    sleep 5
  done
  unset RETRY
done

# ---- summary: stats.txt per setting + results.json in the competition template format ----
python - "$RUN_DIR" "$TASK_CONFIGS" "${TASK_LIST[@]}" <<'EOF'
import json, sys
from pathlib import Path
run, tcs, tasks = Path(sys.argv[1]), sys.argv[2].split(), sys.argv[3:]
key = {"demo_clean": "clean", "demo_randomized": "randomized"}
out = {"schema_version": 1, "team_id": "replace_with_your_team_id", "results": {}}
for tc in tcs:
    rows, S, A = [], 0, 0
    res = out["results"].setdefault(key.get(tc, tc), {})
    for t in tasks:
        f = run / tc / "eval_results" / t / "result.json"
        r = json.loads(f.read_text()) if f.exists() else {"attempts": 0, "successes": 0}
        res[t] = {"attempts": r["attempts"], "successes": r["successes"]}
        S += r["successes"]; A += r["attempts"]
        rows.append(f"{t:<28}{r['successes']:>4}/{r['attempts']:<4}" + ("" if f.exists() else "  MISSING"))
    rows.append(f"{'TOTAL':<28}{S:>4}/{A:<4}  {S / A * 100 if A else 0:.2f}%")
    (run / tc / "stats.txt").write_text("\n".join(rows) + "\n")
    print(f"== {tc} ==\n" + "\n".join(rows))
(run / "results.json").write_text(json.dumps(out, indent=2))
print(f"results.json -> {run / 'results.json'}")
EOF
