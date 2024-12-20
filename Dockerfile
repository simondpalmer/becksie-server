# Use specific version of NVIDIA CUDA image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Remove any third-party apt sources to avoid issues with expiring keys
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]

# Set working directory
WORKDIR /

# Update and upgrade the system packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends \
    sudo ca-certificates git wget curl bash libgl1 libx11-6 software-properties-common ffmpeg build-essential gpg && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Add the deadsnakes PPA and install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install python3.10-dev python3.10-venv python3-pip -y --no-install-recommends && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Download and install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# # Set up NVIDIA repository
# RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
#     curl -s -L https://nvidia.github.io/libnvidia-container/stable/ubuntu20.04/$(dpkg --print-architecture)/nvidia-container-toolkit.list | \
#     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
#     tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
#     apt-get update && \
#     apt-get install -y nvidia-container-toolkit && \
#     apt-get clean -y && \
#     rm -rf /var/lib/apt/lists/*

# Ensure NVIDIA runtime is available if the base image does not have it
RUN ln -s /usr/local/cuda/bin/nvidia-smi /usr/bin/nvidia-smi

# Upgrade pip, setuptools, and wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Copy and run script to fetch models
COPY builder/fetch_models.py /fetch_models.py
RUN python /fetch_models.py && \
    rm /fetch_models.py

# Copy source code into image
COPY src .

# Set default command
CMD ["python", "-u", "/rp_handler.py"]