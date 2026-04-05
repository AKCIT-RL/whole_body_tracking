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

**Workstation (default image, amd64):**

- **OS:** Ubuntu 22.04 (Linux x64)
- **RAM:** 32 GB or more
- **GPU VRAM:** 16 GB or more
- **NVIDIA driver:** 535.129.03 or newer recommended

**DGX Spark (aarch64):** use the Spark profile (see [DGX Spark (Docker-only)](#dgx-spark-docker-only)). Driver and VRAM must match NVIDIA guidance for your Grace Blackwell system and the Isaac Lab tag you choose.

### 5. Repository Location

Because of snap/Docker limitations on some setups, keep the repository under `/home` on the host.

## Building the Image

**Option A — Docker Compose (amd64 default):**

```bash
cd whole_body_tracking
docker compose build
```

**Option B — Build script (explicit `linux/amd64`, same result as A):**

```bash
cd whole_body_tracking
bash docker/build.sh
```

The build pulls the NGC base image (~12 GB) and installs `whole_body_tracking`. The first run may take several minutes.

Override the Isaac Lab base (e.g. testing a newer NGC tag):

```bash
docker compose build --build-arg ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
# or
bash docker/build.sh --build-arg ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
```

## DGX Spark (Docker-only)

Everything runs **inside the container**; the host only needs Docker, NVIDIA Container Toolkit, and a compatible driver. This mirrors the **profile + dedicated Dockerfile** pattern used in [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (`docker/build.sh --profile=spark`), but the base image **must** be **`nvcr.io/nvidia/isaac-lab:*`** (Isaac Sim + Lab), not a plain CUDA image.

### 1. Verify `linux/arm64` on NGC

Pick a tag from the [Isaac Lab NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-lab/tags). Then:

```bash
bash scripts/deployment/spark/check_ngc_manifest.sh nvcr.io/nvidia/isaac-lab:2.3.2
```

If arm64 is missing, **do not** expect `docker pull` / `docker build` to succeed on Spark until NVIDIA publishes a suitable multi-arch or arm64 tag.

### 2. Build (native aarch64 — no `--platform linux/amd64`)

```bash
cd whole_body_tracking
bash docker/build.sh --profile=spark
```

Optional: pin another NGC tag:

```bash
bash docker/build.sh --profile=spark --build-arg ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
```

Or with Compose:

```bash
docker compose -f docker-compose.spark.yaml build
# or
export ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
docker compose -f docker-compose.spark.yaml build
```

The Spark Dockerfile defaults to `isaac-lab:2.3.2` (often referenced in current Isaac Lab docs); the amd64 workflow still defaults to **2.1.0** for parity with the upstream BeyondMimic README. If you move Spark to **2.3.x**, re-run training/sanity checks — API differences are possible.

### 3. Run (same runtime flags style as GR00T deployment README)

**Interactive shell via Compose:**

```bash
docker compose -f docker-compose.spark.yaml run --rm whole-body-tracking-spark
```

**Equivalent `docker run` (mount live repo + Hugging Face cache if needed):**

```bash
docker run --rm --runtime nvidia --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=16g \
  --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v "$(pwd)":/workspace/whole_body_tracking \
  -w /workspace/whole_body_tracking \
  whole-body-tracking-spark:latest
```

Inside the container, prefer **`/isaac-sim/python.sh`** for scripts if `python` is not on `PATH` (same as the amd64 image). Example:

```bash
/isaac-sim/python.sh scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --headless --help
```

### 4. Spark-specific troubleshooting

- **`no matching manifest for linux/arm64`:** NGC tag is amd64-only — try another tag or wait for NVIDIA arm64 support; confirm with `check_ngc_manifest.sh`.
- **Version skew:** Spark default base may differ from 2.1.0; align `ISAAC_LAB_IMAGE` across team and document the tag you validated.
- **Isaac / GPU errors on GB100:** Driver and container tag must be supported by that Isaac Sim release; check NVIDIA release notes for your stack.

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

`docker-compose.spark.yaml` uses a separate set of named volumes (`isaac-spark-*`) so amd64 and Spark caches do not collide on a machine that runs both profiles.

## Run a Command Directly (no shell)

```bash
docker compose run --rm whole-body-tracking python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 --registry_name org/motions/motion --headless
```

## Batch pipeline (multiple CSVs → registry → train)

[`scripts/batch_csv_train.sh`](scripts/batch_csv_train.sh) runs in two phases: first **all** `*.csv` in a directory through `csv_to_npz.py` (W&B motions registry), then **all** corresponding `train.py` runs in the same order, each with its own `--experiment_name` and `--run_name` (CSV basename without `.csv`).

**Prerequisites:** `wandb login` and `export WANDB_ENTITY=your-organization` (see [Weights & Biases](#weights--biases)).

**Environment variables (optional):** `INPUT_FPS` (default `30`), `TASK` (default `Tracking-Flat-G1-v0`), `LOG_PROJECT`, `NUM_ENVS`, `MAX_ITERATIONS`, `TRAIN_EXTRA_ARGS` (space-separated extra flags for `train.py`), `CSV_DIR` if you do not pass a directory on the command line. The batch script uses `/isaac-sim/python.sh` when present (Isaac Lab image); override with `PYTHON=/path/to/python` if needed.

**Training length (`max_iterations`):** The default comes from the task agent config (e.g. `G1FlatPPORunnerCfg.max_iterations` in the source tree). Override per run in either way:

- **Batch script:** `--max-iterations N` CLI flag, or `export MAX_ITERATIONS=N` (passed through as `train.py --max_iterations N` in phase 2 only).
- **Direct `train.py`:** `--max_iterations N` (see [`scripts/rsl_rl/train.py`](scripts/rsl_rl/train.py)).

Mount a host folder of CSVs into the container (example: read-only under `/workspace/motions`):

```bash
docker compose run --rm \
  -e WANDB_ENTITY=your-organization \
  -v /path/on/host/motions:/workspace/motions:ro \
  whole-body-tracking \
  ./scripts/batch_csv_train.sh /workspace/motions
```

With a training iteration cap (pick one):

```bash
# Via environment
docker compose run --rm \
  -e WANDB_ENTITY=your-organization \
  -e MAX_ITERATIONS=10000 \
  -v /path/on/host/motions:/workspace/motions:ro \
  whole-body-tracking \
  ./scripts/batch_csv_train.sh /workspace/motions
```

```bash
# Via batch script flag (overrides MAX_ITERATIONS for that run if both are set)
docker compose run --rm \
  -e WANDB_ENTITY=your-organization \
  -v /path/on/host/motions:/workspace/motions:ro \
  whole-body-tracking \
  ./scripts/batch_csv_train.sh --max-iterations 10000 /workspace/motions
```

The image `ENTRYPOINT` is already `/bin/bash`, so do **not** prefix the command with `bash` (e.g. `bash -lc '...'` becomes `/bin/bash bash -lc '...'` and Bash tries to run the `bash` binary as a script, which yields `cannot execute binary file`).

If you omit the directory argument, the script uses `./motions` under the repo root inside the container.

**Batch CLI flags (optional):** `--max-iterations N`, `--num-envs N`, `--task NAME`, `--log-project NAME`, `--input-fps N`, `-h` / `--help`. Any single remaining argument is `CSV_DIR` (same as the first positional before).

## EULA

By running this container you agree to the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement).

## Troubleshooting

- **GPU errors:** Ensure the NVIDIA Container Toolkit is installed and `nvidia-smi` works on the host.
- **NGC login errors:** Confirm `docker login nvcr.io` completed successfully.
- **`/usr/bin/bash: cannot execute binary file`:** You likely ran `bash -lc '...'` after the service name. Use `-e WANDB_ENTITY=...` and pass `./scripts/batch_csv_train.sh ...` directly (see [Batch pipeline](#batch-pipeline-multiple-csvs--registry--train)), or use `-c '...'` as the only extra args to Bash (not `bash -c`).
- **`python: command not found` in `batch_csv_train.sh`:** The image often has no `python` on `PATH` in non-interactive runs. The script selects `/isaac-sim/python.sh` automatically; upgrade to the latest `scripts/batch_csv_train.sh` from this repo or set `PYTHON=/isaac-sim/python.sh` if you run commands manually.
- **Volume mount:** Keep `:ro` on the same line as `-v host/path:/container/path:ro`. A line break between the path and `:ro` can break the command. Use a bind mount to a real host directory (e.g. `/home/you/motions`) if you need your CSV files from disk; a named volume `motions:` starts empty unless you populate it another way.
- **`csv_to_npz` hangs after W&B:** The script skips `sim.stop()` by default (it often blocks in headless Docker). Teardown still runs `clear_all_callbacks`, `SimulationContext.clear_instance()`, then `simulation_app.close()`. To force `sim.stop()`, set `CSV_TO_NPZ_CALL_SIM_STOP=1`.
- **Inaccessible `/tmp`:** If the container cannot use `/tmp`, edit `scripts/csv_to_npz.py` around lines 319 and 326 to use another temporary directory.
