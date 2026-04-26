#!/usr/bin/env bash
# Batch export: all runs in logs/rsl_rl/ → latest checkpoint per run
#   booster_t1_* → TorchScript JIT (.pt)
#   unitree_g1_* → ONNX (.onnx)
#
# Usage:
#   ./scripts/rsl_rl/batch_export.sh [LOGS_ROOT]
#   LOGS_ROOT defaults to ./logs/rsl_rl

set -euo pipefail

# WBT_ROOT: whole_body_tracking dir (where play.py lives)
WBT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LOGS_ROOT="${1:-${LOGS_ROOT:-}}"
if [[ -z "$LOGS_ROOT" ]]; then
  # default: logs/rsl_rl relative to WBT_ROOT, else one level up
  if [[ -d "$WBT_ROOT/logs/rsl_rl" ]]; then
    LOGS_ROOT="$WBT_ROOT/logs/rsl_rl"
  else
    LOGS_ROOT="$(cd "$WBT_ROOT/.." && pwd)/logs/rsl_rl"
  fi
fi
LOGS_ROOT="$(cd "$LOGS_ROOT" && pwd)"

# play.py uses os.path.abspath("logs/rsl_rl/...") relative to CWD,
# so we must run it from the directory that contains logs/rsl_rl/
PLAY_CWD="$(cd "$LOGS_ROOT/../.." && pwd)"
PLAY_SCRIPT="$WBT_ROOT/scripts/rsl_rl/play.py"
ARTIFACTS_ROOT="$PLAY_CWD/artifacts"

if [[ ! -d "$LOGS_ROOT" ]]; then
  echo "error: logs directory not found: $LOGS_ROOT" >&2; exit 1
fi

echo "[INFO] logs root : $LOGS_ROOT"
echo "[INFO] play CWD  : $PLAY_CWD"

if [[ -x /isaac-sim/python.sh ]]; then
  py_cmd=(/isaac-sim/python.sh)
elif [[ -n "${PYTHON:-}" ]]; then
  py_cmd=("$PYTHON")
else
  py_cmd=(python)
fi

TASK_T1="Tracking-Flat-T1-Wo-State-Estimation-v0"
TASK_G1="Tracking-Flat-G1-Wo-State-Estimation-v0"

exported=0
skipped=0

for motion_dir in "$LOGS_ROOT"/*/; do
  [[ -d "$motion_dir" ]] || continue
  motion_name="$(basename "$motion_dir")"

  if [[ "$motion_name" == booster_t1_* ]]; then
    task="$TASK_T1"
  elif [[ "$motion_name" == unitree_g1_* ]]; then
    task="$TASK_G1"
  else
    echo "[SKIP] Unknown robot prefix: $motion_name"
    (( skipped++ )) || true
    continue
  fi

  for run_dir in "$motion_dir"*/; do
    [[ -d "$run_dir" ]] || continue
    run_name="$(basename "$run_dir")"

    latest_ckpt=$(ls "$run_dir"model_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)
    if [[ -z "$latest_ckpt" ]]; then
      echo "[SKIP] No checkpoints in $run_dir"
      (( skipped++ )) || true
      continue
    fi
    ckpt_file="$(basename "$latest_ckpt")"

    # validate checkpoint: check for zip End-of-Central-Directory signature
    if ! python3 -c "
import sys
with open('$latest_ckpt','rb') as f:
    f.seek(0,2); n=f.tell()
    f.seek(-min(65536,n),2)
    sys.exit(0 if b'PK\x05\x06' in f.read() else 1)
" 2>/dev/null; then
      echo "[SKIP] Corrupted checkpoint (missing zip EOCD): $latest_ckpt"
      (( skipped++ )) || true
      continue
    fi

    # find motion.npz: try versioned dirs (v0, v1, v2, ...) newest first
    motion_file=""
    for artifact_dir in $(ls -d "$ARTIFACTS_ROOT/${motion_name}":* 2>/dev/null | sort -t: -k2 -rn); do
      if [[ -f "$artifact_dir/motion.npz" ]]; then
        motion_file="$artifact_dir/motion.npz"
        break
      fi
    done
    if [[ -z "$motion_file" ]]; then
      echo "[SKIP] No motion.npz found in $ARTIFACTS_ROOT/${motion_name}:*"
      (( skipped++ )) || true
      continue
    fi

    echo "========== export: $motion_name / $run_name / $ckpt_file =========="
    echo "  motion_file: $motion_file"
    (cd "$PLAY_CWD" && "${py_cmd[@]}" "$PLAY_SCRIPT" \
      --task "$task" \
      --headless \
      --num_envs 1 \
      --experiment_name "$motion_name" \
      --load_run "$run_name" \
      --checkpoint "$ckpt_file" \
      --motion_file "$motion_file")

    (( exported++ )) || true
  done
done

echo "========== batch export finished: $exported exported, $skipped skipped =========="
