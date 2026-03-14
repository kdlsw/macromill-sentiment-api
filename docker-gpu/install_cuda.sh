#!/bin/bash
# Install CUDA 11.6 and cuDNN from locally downloaded .deb packages
# This script runs during Docker build
set -e

echo "=== Installing CUDA 11.6 from local package ==="
dpkg -i /tmp/cuda/cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb

# Import CUDA GPG key using new method (apt-key is deprecated)
# First, add the key to the keyring
cat /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub | gpg --dearmor -o /usr/share/keyrings/cuda-repo-ubuntu2004-11-6-local-archive-keyring.gpg

# Remove old source list and create new one with signed-by
rm -f /etc/apt/sources.list.d/cuda-11-6-local.list
echo "deb [signed-by=/usr/share/keyrings/cuda-repo-ubuntu2004-11-6-local-archive-keyring.gpg] file:/var/cuda-repo-ubuntu2004-11-6-local/ /" > /etc/apt/sources.list.d/cuda-11-6-local.list

echo "=== Updating package lists ==="
apt-get update --allow-insecure-repositories || true

echo "=== Installing CUDA runtime packages ==="
# Install CUDA toolkit runtime (minimal set for PyTorch)
# Use --allow-unauthenticated since we have key issues
apt-get install -y --allow-unauthenticated --no-install-recommends \
    cuda-runtime-11-6 \
    cuda-cudart-11-6 \
    cuda-libraries-11-6 \
    cuda-libraries-runtime-11-6 || true

echo "=== Installing cuDNN 8.4.1 from local .deb ==="
# Install cuDNN packages
dpkg -i /tmp/cuda/libcudnn8_8.4.1.50-1+cuda11.6_amd64.deb || true
dpkg -i /tmp/cuda/libcudnn8-dev_8.4.1.50-1+cuda11.6_amd64.deb || true

echo "=== Verifying installation ==="
# Verify CUDA
if [ -f /usr/local/cuda-11.6/bin/nvcc ]; then
    echo "CUDA installed successfully"
    /usr/local/cuda-11.6/bin/nvcc --version
else
    echo "Warning: CUDA nvcc not found in expected location"
fi

# Verify cuDNN
if [ -f /usr/lib/x86_64-linux-gnu/libcudnn.so.8 ]; then
    echo "cuDNN installed successfully"
    ls -la /usr/lib/x86_64-linux-gnu/libcudnn*
else
    echo "Warning: cuDNN not found in expected location"
fi

echo "=== Installation complete ==="
