#!/usr/bin/env bash
# Replay the latest checkpoint of every run under an experiment log folder (same layout as training).
# Usage (from repo root or any cwd): ./play_logs.sh [LOG_DIR]
# LOG_DIR: e.g. logs/rsl_rl/g1_flat, g1_flat (implies logs/rsl_rl/g1_flat), or an absolute path to .../logs/rsl_rl/<experiment_name>
# Does not modify play.py — passes --load_run / --checkpoint / --experiment_name per run.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Isaac Lab saves file logs under ${TMPDIR:-/tmp}/isaaclab/logs (see isaaclab.utils.logger).
# PermissionError happens if that folder exists but is not writable (e.g. root-owned under /tmp).
_base_tmp="${TMPDIR:-/tmp}"
_isaaclab_logs="${_base_tmp}/isaaclab/logs"
if ! mkdir -p "$_isaaclab_logs" 2>/dev/null || [[ ! -w "$_isaaclab_logs" ]]; then
  export TMPDIR="${REPO_ROOT}/.isaaclab_play_tmp"
  mkdir -p "$TMPDIR"
fi

# --- same style as train.sh; adjust as needed ---
task="XsensTracking-Flat-G1-v0"
video_length=400
npz_dir="/home/lucas_olives/Documents/g1_mocap/npz"
# Optional: force the same --motion_file for every run (overrides per-run yaml).
motion_name=""
motion_file_override=""
if [[ -n "$motion_name" ]]; then
  motion_file_override="${npz_dir}/${motion_name}.npz"
fi
# Or set motion_file_override="/path/to/motion.npz" directly.

# motion_file used at play time: by default read from each run's params/env.yaml (saved during train).
motion_file_from_run_yaml() {
  local yaml="${1%/}/params/env.yaml"
  [[ -f "$yaml" ]] || return 1
  local v
  v="$(sed -n 's/^[[:space:]]*motion_file:[[:space:]]*//p' "$yaml" | head -n1 | tr -d '\r' | sed 's/[[:space:]]*$//')"
  [[ -n "$v" ]] || return 1
  if [[ ${#v} -ge 2 && ${v:0:1} == '"' && ${v: -1} == '"' ]]; then
    v="${v:1:${#v}-2}"
  elif [[ ${#v} -ge 2 && ${v:0:1} == "'" && ${v: -1} == "'" ]]; then
    v="${v:1:${#v}-2}"
  fi
  printf '%s' "$v"
}

# First argument: experiment log directory (contains one folder per run).
log_dir_arg="${1:-logs/rsl_rl/g1_flat}"

resolve_exp_log() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s' "$p"
  elif [[ "$p" == logs/rsl_rl/* ]]; then
    printf '%s' "${REPO_ROOT}/${p}"
  else
    printf '%s' "${REPO_ROOT}/logs/rsl_rl/${p}"
  fi
}

EXP_LOG="$(resolve_exp_log "$log_dir_arg")"
if [[ ! -d "$EXP_LOG" ]]; then
  echo "ERROR: experiment log directory not found: $EXP_LOG" >&2
  exit 1
fi

experiment_name="$(basename "$EXP_LOG")"
echo "[INFO] Runs under: $EXP_LOG (experiment_name=$experiment_name)"

failed=()
shopt -s nullglob
for run_dir in "$EXP_LOG"/*/; do
  [[ -d "$run_dir" ]] || continue
  run_name="$(basename "$run_dir")"

  latest_ckpt=""
  # Pick highest model_NNNN.pt (version sort)
  while IFS= read -r -d '' f; do
    latest_ckpt="$f"
  done < <(find "$run_dir" -maxdepth 1 -type f -name 'model_*.pt' -print0 | sort -z -V)

  if [[ -z "$latest_ckpt" || ! -f "$latest_ckpt" ]]; then
    echo "[SKIP] No model_*.pt in $run_name"
    continue
  fi

  ckpt_base="$(basename "$latest_ckpt")"
  echo "[RUN] $run_name  <--  $ckpt_base"

  motion_f=""
  if [[ -n "$motion_file_override" ]]; then
    motion_f="$motion_file_override"
  elif motion_f="$(motion_file_from_run_yaml "$run_dir")"; then
    :
  else
    echo "[SKIP] No motion_file in ${run_name}/params/env.yaml (set motion_file_override in play_logs.sh)" >&2
    continue
  fi
  if [[ ! -f "$motion_f" ]]; then
    echo "[SKIP] motion_file missing on disk: $motion_f ($run_name)" >&2
    continue
  fi

  cmd=(python scripts/rsl_rl/play.py
    --task="$task"
    --headless
    --video --video_length "$video_length"
    --experiment_name "$experiment_name"
    --load_run "$run_name"
    --checkpoint "$ckpt_base"
    --motion_file "$motion_f"
  )

  if ! "${cmd[@]}"; then
    echo "[FAIL] play.py exited with error for run: $run_name" >&2
    failed+=("$run_name")
  fi
done
shopt -u nullglob

if ((${#failed[@]} > 0)); then
  echo "[INFO] Failed runs (${#failed[@]}): ${failed[*]}" >&2
  exit 1
fi
echo "[INFO] All runs finished OK."
