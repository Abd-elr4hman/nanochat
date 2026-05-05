#!/bin/bash

# One-time environment setup for running nanochat with the Mamba backbone.
# Installs uv, a Python 3.11 venv, torch + cu124, the CUDA toolkit (for nvcc),
# and mamba-ssm with its causal-conv1d extra. Builds the Triton/CUDA kernels
# from source against the installed torch.
#
# Tested on Ubuntu 22.04 inside WSL2 with a host NVIDIA driver supporting
# CUDA 12.x. Requires sudo (apt steps) and a working GPU passthrough
# (verify with `nvidia-smi` before running).
#
# Usage: bash runs/setup_mamba.sh

set -e

# -----------------------------------------------------------------------------
# System packages: C++ toolchain, Python dev headers, CUDA compiler (nvcc)
# build-essential pulls in gcc/g++/c++/make. python3.11-dev provides Python.h
# for compiling C extensions. nvidia-cuda-toolkit provides nvcc.

sudo apt update
sudo apt install -y build-essential python3.11-dev nvidia-cuda-toolkit

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

[ -d ".venv" ] || uv venv --python 3.11
source .venv/bin/activate

# -----------------------------------------------------------------------------
# PyTorch with CUDA 12.4 runtime libraries (bundled in the wheel)

uv pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# -----------------------------------------------------------------------------
# Build dependencies that mamba-ssm's pyproject.toml does not declare.
# Required because we pass --no-build-isolation below.

uv pip install setuptools wheel

# -----------------------------------------------------------------------------
# Build mamba-ssm + causal-conv1d from source against the installed torch.
# MAX_JOBS=1 keeps the parallel nvcc compilations from OOM'ing on small
# machines (each fused kernel build can use 4-8 GB of RAM). Slow (~30-60min)
# but reliable. If you have plenty of memory, bump it: MAX_JOBS=4 bash ...

MAX_JOBS="${MAX_JOBS:-1}" uv pip install "mamba-ssm[causal-conv1d]" --no-build-isolation

echo "Setup complete!"
