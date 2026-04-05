#!/usr/bin/env bash
# Check whether an NGC Isaac Lab image lists a linux/arm64 variant (DGX Spark / SBSA).
# Usage: bash scripts/deployment/spark/check_ngc_manifest.sh [IMAGE]
# Example: bash scripts/deployment/spark/check_ngc_manifest.sh nvcr.io/nvidia/isaac-lab:2.3.2
#
# For the full JSON: docker manifest inspect IMAGE > manifest.json
set -euo pipefail

IMAGE="${1:-nvcr.io/nvidia/isaac-lab:2.3.2}"

echo "Manifest inspect for: $IMAGE"
echo "(Requires: docker login nvcr.io)"
echo ""

JSON=$(docker manifest inspect "$IMAGE")

if echo "$JSON" | grep -q '"architecture"[[:space:]]*:[[:space:]]*"arm64"'; then
  echo "OK: manifest JSON contains architecture arm64."
else
  echo "WARNING: no '\"architecture\": \"arm64\"' found in manifest output."
  echo "This image may be amd64-only. Spark builds need linux/arm64 (or a listed arm64 digest)."
fi

if echo "$JSON" | grep -q '"os"[[:space:]]*:[[:space:]]*"linux"'; then
  echo "Info: linux OS entries present in manifest."
fi
