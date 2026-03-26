#!/usr/bin/env bash
# Batch pipeline: (1) all CSVs → csv_to_npz / W&B registry, (2) train each motion in order.
#
# Usage:
#   export WANDB_ENTITY=your-organization
#   ./scripts/batch_csv_train.sh [OPTIONS] [CSV_DIR]
#
# CSV_DIR defaults to $CSV_DIR env, else ./motions under repo root.
#
# Options (CLI overrides env for that run where noted):
#   --max-iterations N   → train.py --max_iterations N (overrides MAX_ITERATIONS env)
#   --num-envs N         → train.py --num_envs N (overrides NUM_ENVS env)
#   --task NAME          → train task (overrides TASK env)
#   --log-project NAME   → W&B log project (overrides LOG_PROJECT env)
#   --input-fps N        → csv_to_npz --input_fps (overrides INPUT_FPS env)
#   -h, --help           → show this summary
#
# Environment (still supported): INPUT_FPS, TASK, LOG_PROJECT, NUM_ENVS, MAX_ITERATIONS,
# TRAIN_EXTRA_ARGS, CSV_DIR, WANDB_ENTITY, PYTHON

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
}

# --- defaults from environment ---
INPUT_FPS="${INPUT_FPS:-30}"
TASK="${TASK:-Tracking-Flat-G1-v0}"
LOG_PROJECT="${LOG_PROJECT:-whole_body_tracking_batch}"
NUM_ENVS="${NUM_ENVS:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
ENV_CSV_DIR="${CSV_DIR:-}"

positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-iterations|--max_iterations)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --num-envs|--num_envs)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      NUM_ENVS="$2"
      shift 2
      ;;
    --task)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      TASK="$2"
      shift 2
      ;;
    --log-project|--log_project)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      LOG_PROJECT="$2"
      shift 2
      ;;
    --input-fps|--input_fps)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      INPUT_FPS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do positional+=("$1"); shift; done
      break
      ;;
    -*)
      echo "error: unknown option: $1 (try --help)" >&2
      exit 1
      ;;
    *)
      positional+=("$1")
      shift
      ;;
  esac
done

if [[ ${#positional[@]} -gt 1 ]]; then
  echo "error: at most one CSV_DIR argument allowed; got: ${positional[*]}" >&2
  exit 1
fi

if [[ ${#positional[@]} -eq 1 ]]; then
  CSV_DIR="${positional[0]}"
else
  CSV_DIR="${ENV_CSV_DIR:-$REPO_ROOT/motions}"
fi

if [[ -z "${WANDB_ENTITY:-}" ]]; then
  echo "error: set WANDB_ENTITY to your W&B organization (see README_DOCKER.md)" >&2
  exit 1
fi

if [[ ! -d "$CSV_DIR" ]]; then
  echo "error: CSV directory not found: $CSV_DIR" >&2
  exit 1
fi

shopt -s nullglob
mapfile -t csv_files < <(find "$CSV_DIR" -maxdepth 1 -type f -name '*.csv' | sort)
shopt -u nullglob

if [[ ${#csv_files[@]} -eq 0 ]]; then
  echo "error: no .csv files in $CSV_DIR" >&2
  exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
  py_cmd=("$PYTHON")
elif [[ -x /isaac-sim/python.sh ]]; then
  py_cmd=(/isaac-sim/python.sh)
else
  py_cmd=(python)
fi

train_cmd=("${py_cmd[@]}" scripts/rsl_rl/train.py --task="$TASK" --headless --logger wandb --log_project_name "$LOG_PROJECT")
if [[ -n "$NUM_ENVS" ]]; then
  train_cmd+=(--num_envs "$NUM_ENVS")
fi
if [[ -n "$MAX_ITERATIONS" ]]; then
  train_cmd+=(--max_iterations "$MAX_ITERATIONS")
fi
# shellcheck disable=SC2206
extra_split=($TRAIN_EXTRA_ARGS)
train_cmd+=("${extra_split[@]}")

motions=()
for csv in "${csv_files[@]}"; do
  motions+=("$(basename "$csv" .csv)")
done

echo "========== Phase 1: CSV → NPZ (W&B registry), ${#motions[@]} motion(s) =========="
for csv in "${csv_files[@]}"; do
  motion="$(basename "$csv" .csv)"
  echo "---------- csv_to_npz: $motion ----------"
  "${py_cmd[@]}" scripts/csv_to_npz.py --input_file "$csv" --input_fps "$INPUT_FPS" --output_name "$motion" --headless
done

echo "========== Phase 2: training, ${#motions[@]} motion(s) =========="
for motion in "${motions[@]}"; do
  registry="${WANDB_ENTITY}-org/wandb-registry-motions/${motion}"
  echo "---------- train: $motion (registry: $registry) ----------"
  "${train_cmd[@]}" --registry_name "$registry" --experiment_name "$motion" --run_name "$motion"
done

echo "========== batch finished =========="
