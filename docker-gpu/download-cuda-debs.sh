#!/bin/bash
# Download CUDA 11.6 and cuDNN .deb packages for offline Docker installation.
# Run this on your host machine BEFORE building Docker, so Docker build does not
# need to download large CUDA/cuDNN packages.
#
# Default cuDNN version: 8.4.1.50-1+cuda11.6 (newer of the two you provided).
#
# Usage:
#   cd docker-gpu
#   ./download-cuda-debs.sh
#
# Optional:
#   CUDNN_VER=8.4.0.27-1 ./download-cuda-debs.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="$SCRIPT_DIR/cuda-packages"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

CUDA_DEB_NAME="cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb"
CUDA_DEB_URL="https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/${CUDA_DEB_NAME}"

# Pick default cuDNN version unless overridden.
CUDNN_VER="${CUDNN_VER:-8.4.1.50-1}"
CUDNN_DEB_NAME="libcudnn8_${CUDNN_VER}+cuda11.6_amd64.deb"
CUDNN_DEV_DEB_NAME="libcudnn8-dev_${CUDNN_VER}+cuda11.6_amd64.deb"

echo "=== Downloading CUDA 11.6 packages ==="

echo "Downloading ${CUDA_DEB_NAME}..."
curl -fSL -o "${CUDA_DEB_NAME}" "${CUDA_DEB_URL}"

echo ""
echo "=== Downloading cuDNN (matching CUDA 11.6) ==="
echo "Using cuDNN version: ${CUDNN_VER}"
echo "Downloading ${CUDNN_DEB_NAME}..."
curl -fSL -o "${CUDNN_DEB_NAME}" \
    "https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/${CUDNN_DEB_NAME}"

echo "Downloading ${CUDNN_DEV_DEB_NAME}..."
curl -fSL -o "${CUDNN_DEV_DEB_NAME}" \
    "https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2004/x86_64/${CUDNN_DEV_DEB_NAME}"

echo ""
echo "Other cuDNN versions you mentioned (not downloaded by default):"
echo "  - 8.4.0.27-1+cuda11.6"
echo "  - 8.4.1.50-1+cuda11.6"
echo "To pick the older one:"
echo "  CUDNN_VER=8.4.0.27-1 ./download-cuda-debs.sh"

echo ""
echo "=== Verifying downloads ==="
ls -lh *.deb

echo ""
echo "=== Download complete ==="
echo "Packages saved to: $DOWNLOAD_DIR"
echo ""
echo "Now build the Docker image with:"
echo "  cd $SCRIPT_DIR"
echo "  docker build -t macromill/sentiment-api-gpu:latest ."

