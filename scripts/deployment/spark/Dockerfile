# DGX Spark (aarch64): same layers as repo-root Dockerfile, different default base tag.
# You MUST use an NGC Isaac Lab tag that publishes linux/arm64 — verify with:
#   bash scripts/deployment/spark/check_ngc_manifest.sh nvcr.io/nvidia/isaac-lab:<tag>
#
# Override at build time:
#   docker build --build-arg ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2 -f scripts/deployment/spark/Dockerfile .
ARG ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.2
FROM ${ISAAC_LAB_IMAGE}

WORKDIR /workspace/whole_body_tracking

COPY source/whole_body_tracking ./source/whole_body_tracking

RUN curl -L -o unitree_description.tar.gz \
    https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz

RUN /isaac-sim/python.sh -m pip install -e source/whole_body_tracking

WORKDIR /workspace/whole_body_tracking
COPY . .

ENTRYPOINT ["/bin/bash"]
