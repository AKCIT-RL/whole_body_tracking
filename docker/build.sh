#!/usr/bin/env bash
# Build whole_body_tracking images (pattern similar to Isaac-GR00T docker/build.sh).
#
# default — linux/amd64 workstation image (matches BeyondMimic Isaac Lab 2.1.0 pin)
#   bash docker/build.sh
#   bash docker/build.sh --profile=default
#
# spark — native platform build for DGX Spark (aarch64). Do NOT pass --platform linux/amd64.
#   bash docker/build.sh --profile=spark
#   bash docker/build.sh --profile=spark --build-arg ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
#
# Extra docker build flags are forwarded (e.g. --no-cache).
set -euo pipefail

export DOCKER_BUILDKIT=1
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"

profile="default"
docker_args=()
for arg in "$@"; do
  case $arg in
    --profile=*)
      profile="${arg#--profile=}"
      ;;
    *)
      docker_args+=("$arg")
      ;;
  esac
done

cd "$ROOT"

if [ "$profile" = "spark" ]; then
  image_name="whole-body-tracking-spark"
  docker build "${docker_args[@]}" \
    --network host \
    -f "$ROOT/scripts/deployment/spark/Dockerfile" \
    -t "${image_name}:latest" \
    "$ROOT"
  echo "Image ${image_name}:latest built successfully."
else
  image_name="whole-body-tracking"
  docker build "${docker_args[@]}" \
    --platform linux/amd64 \
    --network host \
    -f "$ROOT/Dockerfile" \
    -t "${image_name}:latest" \
    "$ROOT"
  echo "Image ${image_name}:latest built successfully."
fi
