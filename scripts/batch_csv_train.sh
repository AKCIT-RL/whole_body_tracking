#!/usr/bin/env bash
# Batch pipeline: (1) all CSVs → csv_to_npz / W&B, (2) train each motion in order.
#
# Usage:
#   export WANDB_ENTITY=gabrielruotolo-federal-univesity-of-goias
#   ./scripts/batch_csv_train.sh [OPTIONS] [CSV_DIR]
#
# CSV_DIR defaults to $CSV_DIR env, else ./motions under repo root.
#
# Options:
#   --max-iterations N      → train.py --max_iterations N
#   --num-envs N            → train.py --num_envs N
#   --task NAME             → train task (default: Tracking-Flat-T1-Wo-State-Estimation-v0)
#   --log-project NAME      → W&B log project (default: Booster_t1)
#   --registry-collection S → W&B registry collection (default: Booster_t1)
#   --input-fps N           → csv_to_npz --input_fps (default: 30)
#   --robot NAME            → csv_to_npz --robot (default: booster_t1)
#   --wandb-project NAME    → csv_to_npz --wandb_project (default: Booster_t1)
#   --video                 → train.py --video (this script always passes --video for phase 2)
#   --video-interval N      → train.py --video_interval N (default: from train.py, e.g. 2000)
#   --video-length N        → train.py --video_length N (default: from train.py, e.g. 200)
#   --skip-npz              → skip Phase 1 (csv_to_npz), go straight to training
#   -h, --help              → show this summary
#
# Environment (still supported): INPUT_FPS, TASK, LOG_PROJECT, NUM_ENVS, MAX_ITERATIONS,
# VIDEO_INTERVAL, VIDEO_LENGTH, TRAIN_EXTRA_ARGS, NPZ_EXTRA_ARGS, CSV_DIR, WANDB_ENTITY, PYTHON

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
}

# --- defaults from environment ---
INPUT_FPS="${INPUT_FPS:-30}"
TASK="${TASK:-Tracking-Flat-T1-Wo-State-Estimation-v0}"
LOG_PROJECT="${LOG_PROJECT:-Booster_t1}"
REGISTRY_COLLECTION="${REGISTRY_COLLECTION:-Booster_t1}"
ROBOT="${ROBOT:-booster_t1}"
WANDB_PROJECT_NPZ="${WANDB_PROJECT_NPZ:-Booster_t1}"
NUM_ENVS="${NUM_ENVS:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
NPZ_EXTRA_ARGS="${NPZ_EXTRA_ARGS:-}"
ENV_CSV_DIR="${CSV_DIR:-}"
ADD_VIDEO=0
VIDEO_INTERVAL="${VIDEO_INTERVAL:-}"
VIDEO_LENGTH="${VIDEO_LENGTH:-}"
SKIP_NPZ=0

positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-iterations|--max_iterations)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      MAX_ITERATIONS="$2"; shift 2 ;;
    --num-envs|--num_envs)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      NUM_ENVS="$2"; shift 2 ;;
    --task)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      TASK="$2"; shift 2 ;;
    --log-project|--log_project)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      LOG_PROJECT="$2"; shift 2 ;;
    --registry-collection|--registry_collection)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      REGISTRY_COLLECTION="$2"; shift 2 ;;
    --input-fps|--input_fps)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      INPUT_FPS="$2"; shift 2 ;;
    --robot)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      ROBOT="$2"; shift 2 ;;
    --wandb-project|--wandb_project)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      WANDB_PROJECT_NPZ="$2"; shift 2 ;;
    --video)
      ADD_VIDEO=1; shift ;;
    --video-interval|--video_interval)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      VIDEO_INTERVAL="$2"; shift 2 ;;
    --video-length|--video_length)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      VIDEO_LENGTH="$2"; shift 2 ;;
    --skip-npz|--skip_npz)
      SKIP_NPZ=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do positional+=("$1"); shift; done
      break ;;
    -*)
      echo "error: unknown option: $1 (try --help)" >&2; exit 1 ;;
    *)
      positional+=("$1"); shift ;;
  esac
done

# When log + NPZ projects match a non-default name, use it as the W&B model-registry collection
# for --registry_name (avoids uploading to G1_training but training against Booster_t1).
if [[ "${REGISTRY_COLLECTION}" == "Booster_t1" && "${LOG_PROJECT}" == "${WANDB_PROJECT_NPZ}" && "${LOG_PROJECT}" != "Booster_t1" ]]; then
  REGISTRY_COLLECTION="${LOG_PROJECT}"
fi

if [[ ${#positional[@]} -gt 1 ]]; then
  echo "error: at most one CSV_DIR argument allowed; got: ${positional[*]}" >&2; exit 1
fi

if [[ ${#positional[@]} -eq 1 ]]; then
  CSV_DIR="${positional[0]}"
else
  CSV_DIR="${ENV_CSV_DIR:-$REPO_ROOT/motions}"
fi

if [[ -z "${WANDB_ENTITY:-}" ]]; then
  echo "error: set WANDB_ENTITY to your W&B entity (e.g. gabrielruotolo-federal-univesity-of-goias)" >&2; exit 1
fi

if [[ ! -d "$CSV_DIR" ]]; then
  echo "error: CSV directory not found: $CSV_DIR" >&2; exit 1
fi

# Bind-mounting the repo over /workspace/whole_body_tracking hides URDFs baked into the Docker image;
# download Unitree G1 assets into source/.../assets if missing.
ensure_unitree_g1_assets() {
  [[ "${ROBOT}" == unitree_g1 ]] || return 0
  local assets_dir="$REPO_ROOT/source/whole_body_tracking/whole_body_tracking/assets"
  local urdf="$assets_dir/unitree_description/urdf/g1/main.urdf"
  [[ -f "$urdf" ]] && return 0
  echo "========== Fetching Unitree G1 URDF (not found at $urdf) =========="
  mkdir -p "$assets_dir"
  local tgz
  tgz="$(mktemp /tmp/unitree_desc.XXXXXX.tar.gz)"
  curl -fsSL -o "$tgz" https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
  tar -xzf "$tgz" -C "$assets_dir"
  rm -f "$tgz"
  [[ -f "$urdf" ]] || { echo "error: expected URDF after extract: $urdf" >&2; exit 1; }
}
ensure_unitree_g1_assets

mapfile -t csv_files < <(find "$CSV_DIR" -maxdepth 1 -type f -name "${ROBOT}_*.csv" | sort)

if [[ ${#csv_files[@]} -eq 0 ]]; then
  echo "error: no ${ROBOT}_*.csv files in $CSV_DIR" >&2; exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
  py_cmd=("$PYTHON")
elif [[ -x /isaac-sim/python.sh ]]; then
  py_cmd=(/isaac-sim/python.sh)
else
  py_cmd=(python)
fi

train_cmd=("${py_cmd[@]}" scripts/rsl_rl/train.py
  --task="$TASK"
  --headless
  --video
  --logger wandb
  --log_project_name "$LOG_PROJECT"
)
[[ -n "$NUM_ENVS" ]]       && train_cmd+=(--num_envs "$NUM_ENVS")
[[ -n "$MAX_ITERATIONS" ]] && train_cmd+=(--max_iterations "$MAX_ITERATIONS")
[[ -n "$VIDEO_INTERVAL" ]] && train_cmd+=(--video_interval "$VIDEO_INTERVAL")
[[ -n "$VIDEO_LENGTH" ]]   && train_cmd+=(--video_length "$VIDEO_LENGTH")
# shellcheck disable=SC2206
[[ -n "$TRAIN_EXTRA_ARGS" ]] && train_cmd+=($TRAIN_EXTRA_ARGS)

motions=()
for csv in "${csv_files[@]}"; do
  motions+=("$(basename "$csv" .csv)")
done

if [[ "$SKIP_NPZ" -eq 1 ]]; then
  echo "========== Phase 1: skipped (--skip-npz) =========="
else
  echo "========== Phase 1: CSV → NPZ (W&B), ${#motions[@]} motion(s) =========="
  for csv in "${csv_files[@]}"; do
    motion="$(basename "$csv" .csv)"
    echo "---------- csv_to_npz: $motion ----------"
    npz_cmd=("${py_cmd[@]}" scripts/csv_to_npz.py
      --input_file "$csv"
      --input_fps "$INPUT_FPS"
      --output_name "$motion"
      --robot "$ROBOT"
      --wandb_project "$WANDB_PROJECT_NPZ"
      --headless
    )
    # shellcheck disable=SC2206
    [[ -n "$NPZ_EXTRA_ARGS" ]] && npz_cmd+=($NPZ_EXTRA_ARGS)
    "${npz_cmd[@]}"
  done
fi

echo "========== Phase 2: training, ${#motions[@]} motion(s) =========="
for motion in "${motions[@]}"; do
  registry="${WANDB_ENTITY}/${REGISTRY_COLLECTION}/${motion}"
  echo "---------- train: $motion (registry: $registry) ----------"
  "${train_cmd[@]}" \
    --registry_name "$registry" \
    --experiment_name "$motion" \
    --run_name "$motion"
done

echo "========== batch finished =========="
