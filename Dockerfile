# Base: Isaac Lab 2.1.0 (includes Isaac Sim 4.5.0, Python 3.10)
FROM nvcr.io/nvidia/isaac-lab:2.1.0

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
