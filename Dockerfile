# Base: Isaac Lab 2.1.0 (Isaac Sim 4.5.0, Python 3.10)
# We override the isaaclab packages with our patched fork (v2.3.2 + Isaac Sim 4.5 fixes).
FROM nvcr.io/nvidia/isaac-lab:2.1.0

WORKDIR /workspace/whole_body_tracking

# Fix PyTorch: base image ships cu130 which requires CUDA 13.0 drivers.
# Driver 570.x supports max CUDA 12.8 — reinstall with cu128 build.
RUN /isaac-sim/python.sh -m pip install \
    torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install patched Isaac Lab fork (v2.3.2 + headless URDF fix, pickle IO, ONNX exporter fix)
RUN git clone --depth 1 -b whole_body_tracking \
        https://github.com/AKCIT-RL/IsaacLab.git /opt/IsaacLab && \
    /isaac-sim/python.sh -m pip install -e /opt/IsaacLab/source/isaaclab --no-deps && \
    /isaac-sim/python.sh -m pip install -e /opt/IsaacLab/source/isaaclab_assets --no-deps && \
    /isaac-sim/python.sh -m pip install -e /opt/IsaacLab/source/isaaclab_tasks --no-deps && \
    /isaac-sim/python.sh -m pip install -e /opt/IsaacLab/source/isaaclab_mimic --no-deps && \
    /isaac-sim/python.sh -m pip install -e /opt/IsaacLab/source/isaaclab_rl --no-deps && \
    /isaac-sim/python.sh -m pip install rsl-rl-lib==5.0.1

COPY source/whole_body_tracking ./source/whole_body_tracking

# Download Unitree G1 robot description
RUN curl -L -o unitree_description.tar.gz \
    https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
    tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
    rm unitree_description.tar.gz

# Download Booster robot assets (URDFs, meshes) and install Python helper
RUN git clone --depth 1 https://github.com/BoosterRobotics/booster_assets /workspace/booster_assets && \
    /isaac-sim/python.sh -m pip install -e /workspace/booster_assets

# Install whole_body_tracking extension
RUN /isaac-sim/python.sh -m pip install -e source/whole_body_tracking

WORKDIR /workspace/whole_body_tracking
COPY . .

# Interactive shell by default
ENTRYPOINT ["/bin/bash"]
