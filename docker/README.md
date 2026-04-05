# Docker build helper — whole_body_tracking

## Profiles

| Profile | Image tag | Platform | Dockerfile |
|--------|-----------|----------|------------|
| `default` | `whole-body-tracking:latest` | `linux/amd64` | [`../Dockerfile`](../Dockerfile) |
| `spark` | `whole-body-tracking-spark:latest` | native (aarch64 on DGX Spark) | [`../scripts/deployment/spark/Dockerfile`](../scripts/deployment/spark/Dockerfile) |

From the repository root:

```bash
bash docker/build.sh
bash docker/build.sh --profile=spark
```

Before building for Spark, confirm the Isaac Lab NGC tag exposes `linux/arm64`:

```bash
bash scripts/deployment/spark/check_ngc_manifest.sh nvcr.io/nvidia/isaac-lab:2.3.2
```

Full runbook: [`../README_DOCKER.md`](../README_DOCKER.md) (section **DGX Spark**).
