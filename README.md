# BeyondMimic Motion Tracking Code

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

## Overview

BeyondMimic is a versatile humanoid control framework that provides highly dynamic motion tracking with the
state-of-the-art motion quality on real-world deployment and steerable test-time control with guided diffusion-based
controllers.

This repo covers the motion tracking training in BeyondMimic. **You should be able to
train any sim-to-real-ready motion in the LAFAN1 dataset, without tuning any parameters**.

For sim-to-sim and sim-to-real deployment, please refer to
the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

### Alternative Implementations

- There is an alternative reproduction of BeyondMimic in [mjlab](https://github.com/mujocolab/mjlab), a new Isaac Lab-style manager API powered by MuJoCo-Warp for RL and robotics research. See the implementation [here](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py).

## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:HybridRobotics/whole_body_tracking.git

# Option 2: HTTPS
git clone https://github.com/HybridRobotics/whole_body_tracking.git
```

- Pull the robot description files from GCS

```bash
# Enter the repository
cd whole_body_tracking
# Rename all occurrences of whole_body_tracking (in files/directories) to your_fancy_extension_name
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```

## Motion Tracking

### Motion Preprocessing & Registry Setup

In order to manage the large set of motions we used in this work, we leverage the WandB registry to store and load
reference motions automatically.
Note: The reference motion should be retargeted and use generalized coordinates only.

- Gather the reference motion datasets (please follow the original licenses), we use the same convention as .csv of
  Unitree's dataset

    - Unitree-retargeted LAFAN1 Dataset is available
      on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    - Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
    - Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
    - Balance motions are from [HuB](https://hub-robot.github.io/)


- Log in to your WandB account; access Registry under Core on the left. Create a new registry collection with the name "
  Motions" and artifact type "All Types".


- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body
  acceleration) via forward kinematics:

```bash
# Unitree G1 (default)
python scripts/csv_to_npz.py \
  --input_file {motion_name}.csv --input_fps 30 \
  --output_name {motion_name} \
  --wandb_project {artifact_project_name} \
  --headless

# Booster T1
python scripts/csv_to_npz.py \
  --input_file {motion_t1}.csv --input_fps 30 \
  --output_name {motion_name} \
  --robot booster_t1 \
  --wandb_project {artifact_project_name} \
  --headless
```

`--wandb_project` controls the wandb project where the NPZ artifact is uploaded (default: `csv_to_npz`). The
project is created automatically on first upload. The artifact path used in training will be
`{your-entity}/{artifact_project_name}/{motion_name}`.

- Test the upload by replaying the motion in Isaac Sim:

```bash
# From wandb registry
python scripts/replay_npz.py \
  --registry_name {your-entity}/{artifact_project_name}/{motion_name} \
  --robot booster_t1

# From local file
python scripts/replay_npz.py \
  --motion_file /path/to/motion.npz \
  --robot booster_t1
```

- Debugging
    - Make sure to export `WANDB_ENTITY` to your organization name, not your personal username.
    - If `/tmp` is not accessible, modify `csv_to_npz.py` to a temporary folder of your choice.

### Policy Training

Available tasks:

| Task ID | Robot | Notes |
|---------|-------|-------|
| `Tracking-Flat-G1-v0` | Unitree G1 | Full obs (requires state estimation) |
| `Tracking-Flat-G1-Wo-State-Estimation-v0` | Unitree G1 | No `motion_anchor_pos_b` / `base_lin_vel` |
| `Tracking-Flat-G1-Low-Freq-v0` | Unitree G1 | Half control frequency |
| `Tracking-Flat-T1-Wo-State-Estimation-v0` | Booster T1 | No `motion_anchor_pos_b` / `base_lin_vel` — **deploy-ready** |

> **T1 note:** The Booster T1 hardware does not have an absolute position or linear velocity estimator.
> Always use `Tracking-Flat-T1-Wo-State-Estimation-v0` for T1 — a full-obs variant is not registered.

- Train policy by the following command:

```bash
# Unitree G1
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --registry_name {your-entity}/{artifact_project_name}/{motion_name} \
  --headless --logger wandb \
  --log_project_name {training_project_name} \
  --run_name {run_name}

# Booster T1
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-T1-Wo-State-Estimation-v0 \
  --registry_name {your-entity}/{artifact_project_name}/{motion_name} \
  --headless --logger wandb \
  --log_project_name {training_project_name} \
  --run_name {run_name}
```

**wandb flags explained:**

| Flag | Where it goes | Set by |
|------|--------------|--------|
| `--registry_name` | Where to **download** the NPZ artifact from | Upload step (`csv_to_npz.py --wandb_project`) |
| `--log_project_name` | wandb **project** where training charts/metrics are logged | You choose at train time |
| `--run_name` | Name of the individual **run** inside that project | You choose at train time |

- `--log_project_name` and `--registry_name` are **completely independent**. Not passing `--log_project_name` does NOT default to the artifact project — wandb will use its own default (typically `uncategorized`). Always pass it explicitly.
- Both projects are created automatically on first use — no need to create them on wandb.ai beforehand.
- `--run_name` is optional. If omitted, wandb generates a random name (e.g. `golden-sunset-42`).

Visually:
```
BoosterT1_Training/      ← --log_project_name
├── chute_v1_run1        ← --run_name
├── chute_v1_run2
└── chute_pedro_v1
```

Example end-to-end:
```bash
# 1. Upload motion to project "BoosterT1_Motions"
python scripts/csv_to_npz.py ... --wandb_project BoosterT1_Motions --output_name chute_v1

# 2. Train and log charts to project "BoosterT1_Training"
python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-T1-Wo-State-Estimation-v0 \
  --registry_name {your-entity}/BoosterT1_Motions/chute_v1 \
  --logger wandb \
  --log_project_name BoosterT1_Training \
  --run_name chute_v1_run1 \
  --headless
```

Add `--max_iterations N` to override the number of PPO iterations (default: 30000 for T1/G1, set in `agents/rsl_rl_ppo_cfg.py`).

### Policy Evaluation

- Play the trained policy by the following command:

```bash
# Unitree G1
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}

# Booster T1
python scripts/rsl_rl/play.py --task=Tracking-Flat-T1-Wo-State-Estimation-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The WandB run path can be located in the run overview. It follows the format {your_organization}/{project_name}/ along
with a unique 8-character identifier. Note that run_name is different from run_path.

## Code Structure

Below is an overview of the code structure for this repository:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains the atomic functions to define the MDP for BeyondMimic. Below is a breakdown of the functions:

    - **`commands.py`**
      Command library to compute relevant variables from the reference motion, current robot state, and error
      computations. This includes pose and velocity error calculation, initial state randomization, and adaptive
      sampling.

    - **`rewards.py`**
      Implements the DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection.

    - **`terminations.py`**
      Implements early terminations and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains the environment (MDP) hyperparameters configuration for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  Contains the PPO hyperparameters for the G1 tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/t1/`**
  Robot-specific configuration for the Booster T1 (23 DOF). Mirrors the G1 structure:
  - `flat_env_cfg.py` — `T1FlatEnvCfg` and `T1FlatWoStateEstimationEnvCfg`
  - `agents/rsl_rl_ppo_cfg.py` — `T1FlatPPORunnerCfg`

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale
  calculation. Robots supported: `g1.py` (Unitree G1), `t1.py` (Booster T1).

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies, and evaluating trained policies.

This structure is designed to ensure modularity and ease of navigation for developers expanding the project.
