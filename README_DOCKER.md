# Whole Body Tracking — Docker Guide

This guide explains how to run BeyondMimic (`whole_body_tracking`) inside a Docker container using the pre-built NVIDIA Isaac Lab 2.1.0 image.

## Host Prerequisites

### 1. Docker and Docker Compose

- **Docker Engine** 26.0.0 or newer
- **Docker Compose** 2.25.0 or newer

Install using the [official Docker documentation](https://docs.docker.com/engine/install/).

### 2. NVIDIA Container Toolkit

Required for GPU execution. Follow the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### 3. NGC Account and Login

The base image `nvcr.io/nvidia/isaac-lab:2.1.0` requires NGC authentication:

1. Create an account on the [NVIDIA Developer Program](https://developer.nvidia.com/)
2. Generate an NGC API key in [NGC Setup](https://ngc.nvidia.com/setup/api-key)
3. Log in to Docker:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: your NGC API key
```

### 4. System Requirements

- **OS:** Ubuntu 22.04 (Linux x64)
- **RAM:** 32 GB or more
- **GPU VRAM:** 16 GB or more
- **NVIDIA driver:** 535.129.03 or newer recommended

### 5. Repository Location

Because of snap/Docker limitations on some setups, keep the repository under `/home` on the host.

## Building the Image

```bash
cd whole_body_tracking
docker compose build
```

The build pulls the NGC base image (~12 GB) and installs `whole_body_tracking`. The first run may take several minutes.

## Running the Container

### Interactive mode (bash)

```bash
docker compose run --rm whole-body-tracking
```

This opens a shell inside the container. The working directory is `/workspace/whole_body_tracking`.

### Usage Examples

Inside the container:

**1. Convert motion CSV to NPZ and upload to the W&B Registry:**

```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
```

**2. Test motion replay from the registry:**

```bash
python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
```

**3. Train a policy:**

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
  --registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
  --headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

**4. Evaluate a trained policy:**

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

## Weights & Biases

To use the W&B registry and logging:

1. Inside the container: `wandb login`
2. Set `WANDB_ENTITY` to your organization name (not your personal username):

```bash
export WANDB_ENTITY=your-organization
```

## Volumes and Cache

`docker-compose.yaml` mounts named volumes for Isaac Sim caches to reduce load time on later runs:

- `isaac-cache-kit` — compiled shaders and kit resources
- `isaac-cache-pip` — Python packages
- `isaac-logs` — Omniverse logs
- `./logs` — `whole_body_tracking` logs (bind mount on the host)

## Run a Command Directly (no shell)

```bash
docker compose run --rm whole-body-tracking python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --registry_name org/motions/motion --headless
```

## EULA

By running this container you agree to the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement).

## Troubleshooting

- **GPU errors:** Ensure the NVIDIA Container Toolkit is installed and `nvidia-smi` works on the host.
- **NGC login errors:** Confirm `docker login nvcr.io` completed successfully.
- **Inaccessible `/tmp`:** If the container cannot use `/tmp`, edit `scripts/csv_to_npz.py` around lines 319 and 326 to use another temporary directory.
