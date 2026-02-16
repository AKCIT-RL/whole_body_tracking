#!/usr/bin/env bash
# Train policy with local motion file. Change motion_name to use a different .npz from npz_dir.

motion_name="bebeto-001"
project_name="xsens_g1"
run_name="bebeto-001_no_time_out_10s_ep_24steps_per_env"
npz_dir="/home/lucas_olives/Documents/g1_mocap/npz"

python scripts/rsl_rl/train.py \
  --task=XsensTracking-Flat-G1-Low-Freq-v0 \
  --motion_file "${npz_dir}/${motion_name}.npz" \
  --headless --logger wandb --log_project_name "${project_name}" --run_name "${run_name}" \
  --video --video_length 400 --video_interval 300
