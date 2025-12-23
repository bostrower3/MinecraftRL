FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# --------------------
# System dependencies
# --------------------
RUN apt update && apt install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    openjdk-8-jdk \
    git \
    xvfb \
    ffmpeg \
    libgl1 \
    libx11-6 \
    libxext6 \
    libxi6 \
    build-essential \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# --------------------
# Python tooling
# --------------------
RUN pip install --upgrade pip setuptools wheel

# --------------------
# PyTorch (CUDA 11.8)
# --------------------
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# --------------------
# MineRL
# --------------------
RUN pip install git+https://github.com/minerllabs/minerl

WORKDIR /workspace
