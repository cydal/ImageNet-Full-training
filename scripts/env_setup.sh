#!/bin/bash
# Environment setup script
# Installs system dependencies and Python packages

set -e

echo "=========================================="
echo "Setting up ImageNet training environment"
echo "=========================================="

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    wget \
    curl \
    libjpeg-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch and related packages
echo "Installing Python packages..."
pip install -r requirements.txt

# Optional: Install NVIDIA DALI for faster data loading
read -p "Install NVIDIA DALI for faster data loading? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing NVIDIA DALI..."
    pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
fi

echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
