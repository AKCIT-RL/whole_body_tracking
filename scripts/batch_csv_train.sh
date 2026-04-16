#!/usr/bin/env bash
# Batch pipeline: (1) all CSVs → csv_to_npz / W&B, (2) train each motion in order.
#
# Usage:
#   export WANDB_ENTITY=gabrielruotolo-federal-univesity-of-goias
#   ./scripts/batch_csv_train.sh [OPTIONS] [CSV_DIR]
#   ./scripts/batch_csv_train.sh <num_instances> <cur_instance> [OPTIONS] [CSV_DIR]
#
# CSV_DIR defaults to $CSV_DIR env, else ./motions under repo root.
#
# Instance sharding (split work across machines / GPUs; same CSV_DIR and options on each):
#   Leading positional form (like a multi-experiment launcher):
#     ./scripts/batch_csv_train.sh 4 1 --max-iterations 10000 /path/to/csv_dir
#   Named form (do not combine with the leading pair above):
#     ./scripts/batch_csv_train.sh --num-shards 4 --shard-index 1 ... DIR
#   Or via env: BATCH_NUM_SHARDS=4 BATCH_SHARD_INDEX=1 (overridden by CLI flags if set).
#   Motions are assigned by sorted CSV list index: idx % num_instances == cur_instance.
#
# Options:
#   --max-iterations N      → train.py --max_iterations N
#   --num-envs N            → train.py --num_envs N
#   --task NAME             → train task (default: Tracking-Flat-T1-Wo-State-Estimation-v0)
#   --log-project NAME      → W&B log project (default: Booster_t1)
#   --registry-collection S → used only with --use-wandb-registry (default: Booster_t1)
#   --use-wandb-registry    → train loads motion as entity/<registry-collection>/<motion>
#                            (W&B org registry path; often fails on personal/team entities with wandb 0.24+)
#   --input-fps N           → csv_to_npz --input_fps (default: 30)
#   --csv-type TYPE         → csv_to_npz --csv-type (default: native; xsens = xyzw→wxyz for base quat)
#   --robot NAME            → csv_to_npz --robot (default: booster_t1)
#   --wandb-project NAME    → csv_to_npz --wandb_project (default: Booster_t1); also default train artifact project
#   --wandb-project-name N  → alias of --wandb-project
#   --wandb-entity ENTITY   → sets W&B entity for this run (same as export WANDB_ENTITY=...)
#   --video                 → add --video to train.py
#   --video-interval N      → train.py --video_interval (default: train.py default if omitted)
#   --video-length N        → train.py --video_length (default: train.py default if omitted)
#   --skip-npz              → skip Phase 1 (csv_to_npz), go straight to training
#   --skip-existing         → skip Phase 1 per motion if motions/<motion>.npz exists locally;
#                            skip Phase 2 per motion if logs/rsl_rl/<motion>/ has any model_*.pt
#   --num-shards N          → split work across N parallel invocations (default: 1 = all motions)
#   --shard-index I         → run only motions with index % N == I (0 <= I < N)
#   -h, --help              → show this summary
#
# Environment (still supported): INPUT_FPS, CSV_TYPE, TASK, LOG_PROJECT, NUM_ENVS, MAX_ITERATIONS,
# TRAIN_EXTRA_ARGS, NPZ_EXTRA_ARGS, CSV_DIR, WANDB_ENTITY, PYTHON,
# BATCH_NUM_SHARDS, BATCH_SHARD_INDEX

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'
}

# Optional leading positional: num_instances cur_instance (must not mix with --num-shards/--shard-index).
INST_LEADING=0
INSTANCE_NUM_SHARDS=""
INSTANCE_SHARD_INDEX=""
if [[ $# -ge 2 ]] && [[ "$1" =~ ^[0-9]+$ ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
  if (( "$1" >= 1 && "$2" >= 0 && "$2" < "$1" )); then
    INSTANCE_NUM_SHARDS="$1"
    INSTANCE_SHARD_INDEX="$2"
    INST_LEADING=1
    shift 2
  fi
fi

# --- defaults from environment ---
INPUT_FPS="${INPUT_FPS:-30}"
CSV_TYPE="${CSV_TYPE:-native}"
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
SKIP_NPZ=0
SKIP_EXISTING=0
USE_WANDB_REGISTRY=0
VIDEO_INTERVAL=""
VIDEO_LENGTH=""
NUM_SHARDS_CLI=""
SHARD_INDEX_CLI=""

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
    --csv-type|--csv_type)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      CSV_TYPE="$2"; shift 2 ;;
    --robot)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      ROBOT="$2"; shift 2 ;;
    --wandb-project|--wandb_project|--wandb-project-name)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      WANDB_PROJECT_NPZ="$2"; shift 2 ;;
    --wandb-entity)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      export WANDB_ENTITY="$2"; shift 2 ;;
    --video)
      ADD_VIDEO=1; shift ;;
    --video-interval|--video_interval)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      VIDEO_INTERVAL="$2"; shift 2 ;;
    --video-length|--video_length)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      VIDEO_LENGTH="$2"; shift 2 ;;
    --use-wandb-registry|--use_wandb_registry)
      USE_WANDB_REGISTRY=1; shift ;;
    --skip-npz|--skip_npz)
      SKIP_NPZ=1; shift ;;
    --skip-existing|--skip_existing)
      SKIP_EXISTING=1; shift ;;
    --num-shards|--num_shards)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      NUM_SHARDS_CLI="$2"; shift 2 ;;
    --shard-index|--shard_index)
      [[ -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 1; }
      SHARD_INDEX_CLI="$2"; shift 2 ;;
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

if [[ "$INST_LEADING" -eq 1 ]] && [[ -n "$NUM_SHARDS_CLI" || -n "$SHARD_INDEX_CLI" ]]; then
  echo "error: use either leading <num_instances> <cur_instance> or --num-shards/--shard-index, not both" >&2
  exit 1
fi

if [[ "$INST_LEADING" -eq 1 ]]; then
  NUM_SHARDS="$INSTANCE_NUM_SHARDS"
  SHARD_INDEX="$INSTANCE_SHARD_INDEX"
else
  NUM_SHARDS="${NUM_SHARDS_CLI:-${BATCH_NUM_SHARDS:-1}}"
  SHARD_INDEX="${SHARD_INDEX_CLI:-${BATCH_SHARD_INDEX:-0}}"
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
  echo "error: set WANDB_ENTITY to your W&B entity (e.g. gabrielruotolo-federal-univesity-of-goias), or pass --wandb-entity" >&2
  exit 1
fi

if [[ ! -d "$CSV_DIR" ]]; then
  echo "error: CSV directory not found: $CSV_DIR" >&2; exit 1
fi

if [[ "$CSV_TYPE" != "native" && "$CSV_TYPE" != "xsens" ]]; then
  echo "error: --csv-type / CSV_TYPE must be native or xsens, got: $CSV_TYPE" >&2
  exit 1
fi

if ! [[ "$NUM_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --num-shards (or BATCH_NUM_SHARDS / leading num_instances) must be a positive integer, got: $NUM_SHARDS" >&2
  exit 1
fi
if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]]; then
  echo "error: --shard-index (or BATCH_SHARD_INDEX / leading cur_instance) must be a non-negative integer, got: $SHARD_INDEX" >&2
  exit 1
fi
if (( SHARD_INDEX >= NUM_SHARDS )); then
  echo "error: shard-index must satisfy 0 <= index < num-shards (num-shards=$NUM_SHARDS, shard-index=$SHARD_INDEX)" >&2
  exit 1
fi

mapfile -t csv_files_all < <(find "$CSV_DIR" -maxdepth 1 -type f -name "${ROBOT}_*.csv" | sort)

if [[ ${#csv_files_all[@]} -eq 0 ]]; then
  echo "error: no ${ROBOT}_*.csv files in $CSV_DIR" >&2; exit 1
fi

csv_files=()
idx=0
for csv in "${csv_files_all[@]}"; do
  if (( idx % NUM_SHARDS == SHARD_INDEX )); then
    csv_files+=("$csv")
  fi
  ((++idx))
done

total_motions=${#csv_files_all[@]}
assigned=${#csv_files[@]}
echo "[INFO] shard ${SHARD_INDEX}/${NUM_SHARDS}: ${assigned} motion(s) of ${total_motions} total (idx % ${NUM_SHARDS} == ${SHARD_INDEX})"
if [[ "$assigned" -gt 0 ]]; then
  for csv in "${csv_files[@]}"; do
    echo "[INFO]   - $(basename "$csv" .csv)"
  done
fi

if [[ ${#csv_files[@]} -eq 0 ]]; then
  echo "[INFO] no motions assigned to this shard; exiting."
  exit 0
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
  --logger wandb
  --log_project_name "$LOG_PROJECT"
)
if [[ "$ADD_VIDEO" -eq 1 ]]; then
  train_cmd+=(--video)
  [[ -n "$VIDEO_INTERVAL" ]] && train_cmd+=(--video_interval "$VIDEO_INTERVAL")
  [[ -n "$VIDEO_LENGTH" ]]   && train_cmd+=(--video_length "$VIDEO_LENGTH")
fi
[[ -n "$NUM_ENVS" ]]       && train_cmd+=(--num_envs "$NUM_ENVS")
[[ -n "$MAX_ITERATIONS" ]] && train_cmd+=(--max_iterations "$MAX_ITERATIONS")
# shellcheck disable=SC2206
[[ -n "$TRAIN_EXTRA_ARGS" ]] && train_cmd+=($TRAIN_EXTRA_ARGS)

motions=()
for csv in "${csv_files[@]}"; do
  motions+=("$(basename "$csv" .csv)")
done

has_train_checkpoint() {
  local motion="$1"
  local root="$REPO_ROOT/logs/rsl_rl/$motion"
  local f
  [[ -d "$root" ]] || return 1
  f=$(find "$root" -type f -name 'model_*.pt' -print -quit 2>/dev/null)
  [[ -n "$f" ]]
}

has_local_npz() {
  local motion="$1"
  [[ -f "$REPO_ROOT/motions/${motion}.npz" ]]
}

if [[ "$SKIP_NPZ" -eq 1 ]]; then
  echo "========== Phase 1: skipped (--skip-npz) =========="
else
  echo "========== Phase 1: CSV → NPZ (W&B), ${#motions[@]} motion(s) =========="
  for csv in "${csv_files[@]}"; do
    motion="$(basename "$csv" .csv)"
    if [[ "$SKIP_EXISTING" -eq 1 ]] && has_local_npz "$motion"; then
      echo "---------- csv_to_npz: $motion (skipped: $REPO_ROOT/motions/${motion}.npz exists) ----------"
      continue
    fi
    echo "---------- csv_to_npz: $motion ----------"
    npz_cmd=("${py_cmd[@]}" scripts/csv_to_npz.py
      --input_file "$csv"
      --input_fps "$INPUT_FPS"
      --output_name "$motion"
      --robot "$ROBOT"
      --wandb_project "$WANDB_PROJECT_NPZ"
      --csv-type "$CSV_TYPE"
      --headless
    )
    # NPZ_EXTRA_ARGS may add flags after this line (e.g. extra Isaac args); duplicate --csv-type there wins rightmost.
    # shellcheck disable=SC2206
    [[ -n "$NPZ_EXTRA_ARGS" ]] && npz_cmd+=($NPZ_EXTRA_ARGS)
    "${npz_cmd[@]}"
  done
fi

echo "========== Phase 2: training, ${#motions[@]} motion(s) =========="
for motion in "${motions[@]}"; do
  if [[ "$SKIP_EXISTING" -eq 1 ]] && has_train_checkpoint "$motion"; then
    echo "---------- train: $motion (skipped: checkpoint under logs/rsl_rl/$motion/) ----------"
    continue
  fi
  if [[ "$USE_WANDB_REGISTRY" -eq 1 ]]; then
    registry="${WANDB_ENTITY}/${REGISTRY_COLLECTION}/${motion}"
    echo "---------- train: $motion (W&B registry path: $registry) ----------"
  else
    registry="${WANDB_ENTITY}/${WANDB_PROJECT_NPZ}/${motion}"
    echo "---------- train: $motion (W&B project artifact: $registry) ----------"
  fi
  "${train_cmd[@]}" \
    --registry_name "$registry" \
    --experiment_name "$motion" \
    --run_name "$motion"
done

echo "========== batch finished =========="
