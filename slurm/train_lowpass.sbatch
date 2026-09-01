#!/bin/bash
# ============================================================================
#  Train one BeyondMimic G1 tracking policy per low-pass motion, on an
#  OVX/L40S Slurm cluster, inside the Apptainer image (isaaclab_wbt.sif).
#
#  One array task per motion:  CSV -> NPZ -> W&B registry  (skipped if present)
#                              then  rsl_rl/train.py       (auto-resumes)
#
#  Usage (from this dir, image already built — see README):
#     sbatch --partition=<part> --export=ALL,WANDB_ENTITY=<org> train_lowpass.sbatch
#     sbatch --partition=<part> --export=ALL,WANDB_ENTITY=<org> --array=14 train_lowpass.sbatch
#     sbatch --partition=<part> --export=ALL,WANDB_ENTITY=<org>,NUM_ENVS=2048 train_lowpass.sbatch
#
#  Override any VAR below with  --export=ALL,VAR=value  (isaac-arenas pattern).
# ============================================================================
#SBATCH --job-name=lp-g1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-20
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=logs/lp-g1-%A_%a.out

set -euo pipefail

# ---- config (all overridable via --export=ALL,VAR=...) --------------------
RAID="${RAID:-/raid/$USER}"
SIF="${SIF:-$RAID/containers/isaaclab_wbt.sif}"
RUN_DIR="${RUN_DIR:-$RAID/wbt_lowpass}"                 # CWD of the job: logs/ artifacts/ motions/ wandb/ .env
CACHE="${CACHE:-$RUN_DIR/.isaac-cache}"                 # Isaac Sim / pip / torch caches
CSV_DIR="${CSV_DIR:-$RAID/lowpass_csv}"                 # the 21 low-pass CSVs (30 Hz), scp'd here
ENV_FILE="${ENV_FILE:-$RUN_DIR/.env}"                   # WANDB_API_KEY etc. (chmod 600)

WANDB_ENTITY="${WANDB_ENTITY:?set --export=ALL,WANDB_ENTITY=<your W&B org>}"
NPZ_PROJECT="${NPZ_PROJECT:-LowPass_G1}"                # csv_to_npz project == registry collection
TRAIN_PROJECT="${TRAIN_PROJECT:-LowPass_G1_Training}"
PREFIX="${PREFIX:-g1_lowpass_}"

TASK="${TASK:-Tracking-Flat-G1-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"                            # 2048 if the L40S (48 GB) OOMs
MAX_ITER="${MAX_ITER:-10000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                            # e.g. --video

# ---- motion for this array index --------------------------------------
MOTIONS=(aceno boxe bunny_hop cr7 drunk get_Down_get_Up jump jumping_puddle
         like_jennie macarena one_foot_balance paradinha pedalada polichinelo
         run side_walking square stealth trivela trote what_is_love)
M="${MOTIONS[$SLURM_ARRAY_TASK_ID]}"
NAME="${PREFIX}${M}"
CSV="$CSV_DIR/$M.csv"
CHOME="/home/$USER"                                     # writable cache-backed HOME in the container

echo "=== task $SLURM_ARRAY_TASK_ID  motion=$M  ->  $NAME  |  node $(hostname)  restart#${SLURM_RESTART_COUNT:-0}  $(date)"
[ -f "$SIF" ] || { echo "ERROR: image not found: $SIF  — build it first (see README)"; exit 1; }
[ -f "$CSV" ] || { echo "ERROR: CSV not found: $CSV  — rsync lowpass_mocap/output/csv/ to \$CSV_DIR"; exit 1; }

mkdir -p "$RUN_DIR"/{logs,artifacts,motions,wandb} "$RAID/tmp" \
         "$CACHE"/{kit,kit-data,ov,pip,glcache,computecache,logs,data,documents,home}

# ---- requeue 300 s before the walltime limit / on maintenance drain -----
trap 'echo ">>> SIGUSR1 $(date): requeue"; scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; exit 0' SIGUSR1

# ---- helper: run <python-args...> inside the image --------------------
run() {
  local envfile=()
  [ -f "$ENV_FILE" ] && envfile=(--env-file "$ENV_FILE")
  apptainer exec --nv --no-home --cleanenv --writable-tmpfs --pwd "$RUN_DIR" \
    -B "$RUN_DIR:$RUN_DIR" -B "$CSV_DIR:$CSV_DIR:ro" -B "$RAID/tmp:/tmp" \
    -B "$CACHE/kit:/isaac-sim/kit/cache"        -B "$CACHE/kit-data:/isaac-sim/kit/data" \
    -B "$CACHE/ov:/root/.cache/ov"              -B "$CACHE/pip:/root/.cache/pip" \
    -B "$CACHE/glcache:/root/.cache/nvidia/GLCache" -B "$CACHE/computecache:/root/.nv/ComputeCache" \
    -B "$CACHE/logs:/root/.nvidia-omniverse/logs"   -B "$CACHE/data:/root/.local/share/ov/data" \
    -B "$CACHE/documents:/root/Documents"       -B "$CACHE/home:$CHOME" \
    "${envfile[@]}" \
    --env HOME="$CHOME" --env TMPDIR=/tmp --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y \
    --env WANDB_ENTITY="$WANDB_ENTITY" --env WANDB_DIR="$RUN_DIR/wandb" \
    --env WANDB_ARTIFACT_DIR="$RUN_DIR/artifacts" \
    "$SIF" bash -lc "export HOME='$CHOME'; cd '$RUN_DIR' && exec /isaac-sim/python.sh $*"
}

# ---- phase 1: CSV -> NPZ -> W&B registry (skip if already uploaded) ----
if run - <<PY
import sys, wandb
try:
    wandb.Api(timeout=30).artifact("$WANDB_ENTITY/$NPZ_PROJECT/$NAME:latest"); sys.exit(0)
except Exception:
    sys.exit(1)
PY
then
    echo "[npz] $NAME already in the registry — skipping"
else
    echo "[npz] building $NAME"
    run /opt/wbt/scripts/csv_to_npz.py --input_file "$CSV" --input_fps 30 \
        --output_name "$NAME" --robot unitree_g1 --wandb_project "$NPZ_PROJECT" --headless
fi

# ---- phase 2: train (resume from the latest checkpoint of this motion) -
EXP="$RUN_DIR/logs/rsl_rl/$NAME"
RESUME=()
last=$(ls -d "$EXP"/*/ 2>/dev/null | sort | tail -n1 || true)
if [ -n "$last" ] && compgen -G "${last}model_*.pt" >/dev/null; then
    RESUME=(--resume True --load_run "$(basename "$last")")
    echo "[train] resuming from ${last}"
fi
export WANDB_RUN_ID="lp_${M}" WANDB_RESUME=allow          # same W&B run across requeues

rc=0
run /opt/wbt/scripts/rsl_rl/train.py \
    --task "$TASK" \
    --registry_name "$WANDB_ENTITY/$NPZ_PROJECT/$NAME" \
    --experiment_name "$NAME" --run_name "$NAME" \
    --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITER" \
    --headless --logger wandb --log_project_name "$TRAIN_PROJECT" \
    "${RESUME[@]}" ${EXTRA_ARGS} &
wait $! || rc=$?
echo "=== done  motion=$M  rc=$rc  $(date)"
exit $rc
