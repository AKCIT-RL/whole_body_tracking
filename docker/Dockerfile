# Base image is configurable so x64 and Spark (aarch64) can share this Dockerfile.
# For Spark, pass an image tag that includes linux/arm64 support.
ARG ISAAC_LAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.1.0
FROM ${ISAAC_LAB_IMAGE}

WORKDIR /workspace/whole_body_tracking

COPY source/whole_body_tracking ./source/whole_body_tracking

# Download Unitree G1 robot description
RUN curl -L -o unitree_description.tar.gz \
    https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz

# Install whole_body_tracking extension (uses Isaac Sim Python: /isaac-sim/python.sh)
RUN /isaac-sim/python.sh -m pip install -e source/whole_body_tracking

WORKDIR /workspace/whole_body_tracking
COPY . .

# Interactive shell by default
ENTRYPOINT ["/bin/bash"]