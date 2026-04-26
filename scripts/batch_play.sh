#!/usr/bin/env bash
# Runs play.py for multiple W&B runs.
# play.py saves exports to logs/rsl_rl/temp/exported/ automatically.
#
# Usage (from repo root):
#   bash scripts/batch_play.sh
#
# RUNS format: "wandb_path|task|name"
# Task examples: Tracking-Flat-G1-Wo-State-Estimation-v0
#                Tracking-Flat-T1-Wo-State-Estimation-v0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -x /isaac-sim/python.sh ]]; then
  py_cmd=/isaac-sim/python.sh
else
  py_cmd=python
fi

# format: "wandb_path|task|name"
RUNS=(
  # "do-you-still-need-mocap/videos_t1/ik454x51|Tracking-Flat-T1-Wo-State-Estimation-v0"
  # "do-you-still-need-mocap/G1_vimos/zrey8ttz|Tracking-Flat-G1-Wo-State-Estimation-v0|macarena"
  # "do-you-still-need-mocap/G1_vimos/7zbf3r8l|Tracking-Flat-G1-Wo-State-Estimation-v0|boxe"
  # "do-you-still-need-mocap/G1_vimos/z6aehhwy|Tracking-Flat-G1-Wo-State-Estimation-v0|aceno"
  # "do-you-still-need-mocap/videos_t1_engemulti/vsoju11x|Tracking-Flat-T1-Wo-State-Estimation-v0|jump"
  # "do-you-still-need-mocap/videos_t1_engemulti/a4iqoltn|Tracking-Flat-T1-Wo-State-Estimation-v0|jumping_puddle"
  # "do-you-still-need-mocap/videos_t1_engemulti/ko8m1kxg|Tracking-Flat-T1-Wo-State-Estimation-v0|like_jennie"
  # "do-you-still-need-mocap/videos_t1_engemulti/c5u0gpcc|Tracking-Flat-T1-Wo-State-Estimation-v0|one_foot_balance"
  # "do-you-still-need-mocap/videos_t1_engemulti/7u08oshc|Tracking-Flat-T1-Wo-State-Estimation-v0|paradinha"
  # "do-you-still-need-mocap/videos_t1_engemulti/mifbo87x|Tracking-Flat-T1-Wo-State-Estimation-v0|pedalada"


  # "do-you-still-need-mocap/AIBrasil_t1/1kp6np2j|Tracking-Flat-T1-Wo-State-Estimation-v0|trivela_side"
  # "do-you-still-need-mocap/AIBrasil_t1/jxpvcwfv|Tracking-Flat-T1-Wo-State-Estimation-v0|paradinha_altura"
  "do-you-still-need-mocap/AIBrasil_t1/7xhp5chq|Tracking-Flat-T1-Wo-State-Estimation-v0|chuta_para"
)

for entry in "${RUNS[@]}"; do
  IFS='|' read -r run task name <<< "$entry"

  echo "========== play: $run (task: $task, export: $name) =========="
  $py_cmd scripts/rsl_rl/play.py \
    --task="$task" \
    --num_envs 1 \
    --wandb_path "$run" \
    --headless \
    --video \
    --video_length 1

  # infer robot prefix and export extension from task
  if [[ "$task" == *T1* ]]; then
    robot="booster_t1"; ext="pt"
  else
    robot="unitree_g1"; ext="onnx"
  fi

  export_dir="$REPO_ROOT/logs/rsl_rl/temp/exported"
  run_id="${run##*/}"  # last segment of wandb path, e.g. zrey8ttz
  latest=$(ls "$export_dir"/${run_id}_*."$ext" 2>/dev/null | head -1 || true)
  if [[ -n "$latest" ]]; then
    target="$export_dir/${robot}_${name}.${ext}"
    mv "$latest" "$target"
    echo "  → saved as $target"
  else
    echo "  [WARN] no export found matching ${run_id}_*.${ext} in $export_dir"
  fi
done

echo "========== batch_play done =========="
