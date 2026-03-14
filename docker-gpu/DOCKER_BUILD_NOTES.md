# Docker Build Notes

This document records the issues encountered and solutions found during the Docker image build process for the Macromill Sentiment Analysis API.

## Issues and Solutions

### 1. apt-get Failing Inside Container - Proxy Configuration

**Problem**: When building the Docker image, apt-get inside the container failed to connect to Debian repositories. The error showed it was trying to use the host's proxy (127.0.0.1:8118) which wasn't available inside the container.

```
Err:3 http://deb.debian.org/debian-security/trixie-security InRelease
  Unable to connect to 127.0.0.1:8118
```

**Solution**: Added ENV instructions to clear proxy environment variables before apt-get in both build stages:

```dockerfile
# In running both builder and final stages:
ENV http_proxy=
ENV https_proxy=
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
```

---

### 2. COPY Commands Failing - Build Context Issues

**Problem**: Docker COPY commands failed because files were not in the build context. The Dockerfile was in `/home/ubun/macromill/docker/` but tried to reference files from parent directory (`../src/`, `../requirements.txt`).

```
COPY failed: forbidden path outside the build context: ../requirements.txt ()
```

**Solution**: 
- Created a separate build context in the docker directory
- Copied all necessary files into the docker directory:
  - `cp /home/ubun/macromill/requirements.txt /home/ubun/macromill/docker/`
  - `cp /home/ubun/macromill/setup.py /home/ubun/macromill/docker/`
  - `cp -r /home/ubun/macromill/src /home/ubun/macromill/docker/`
  - `cp -r /home/ubun/macromill/work/artifacts /home/ubun/macromill/docker/`
  - `cp "/home/ubun/macromill/work/IMDB Dataset.csv" /home/ubun/macromill/docker/`

---

### 3. PyTorch Version Not Available on PyPI

**Problem**: The requirements.txt specified `torch==1.13.1+cu116` which is a CUDA 11.6 specific build. This version is not available on standard PyPI and needs to be downloaded from PyTorch's own index.

```
ERROR: Could not find a version that satisfies the requirement torch==1.13.1+cu116
ERROR: No matching distribution found for torch==1.13.1+cu116
```

**Solution**: Created a separate requirements file for CPU-only Docker builds using:
```dockerfile
# Use CPU version for Docker build (no GPU needed for inference)
torch>=2.0.0
transformers>=4.35.0
tokenizers>=0.14.0
accelerate>=0.20.0
datasets>=2.0.0
```

**Note**: The original requirements.txt in the main project still uses the CUDA-specific versions for GPU training. This docker-specific requirements.txt should be updated when building the GPU image later.

---

### 4. Filename with Space Causing COPY Failure

**Problem**: The file "IMDB Dataset.csv" with a space in the name caused Docker COPY instruction to fail:

```
failed to process "\"IMDB": unexpected end of statement while looking for matching double-quote
```

**Solution**: Renamed the file to remove the space:
```bash
mv "/home/ubun/macromill/docker/IMDB Dataset.csv" /home/ubun/macromill/docker/imdb_dataset.csv
```

Updated Dockerfile COPY instruction:
```dockerfile
COPY imdb_dataset.csv /home/ubun/macromill/work/imdb_dataset.csv
```

---

### 5. Module Import Error - Wrong PYTHONPATH

**Problem**: The API failed to start with:
```
ModuleNotFoundError: No module named 'macromill_sentiment'
```

The uvicorn command was trying to import `macromill_sentiment.api.main:app` but the module wasn't found.

**Solution**: Fixed the PYTHONPATH in Dockerfile to point to the correct location:

```dockerfile
# Wrong:
ENV PYTHONPATH=/app:$PYTHONPATH

# Correct:
ENV PYTHONPATH=/app/src:$PYTHONPATH
```

This is because the source code structure is:
```
/app/src/macromill_sentiment/api/main.py
```

---

## Summary of Key Takeaways

1. **Proxy issues**: Always clear proxy env vars inside Docker containers when building, especially for apt-get
2. **Build context**: All files needed for Docker build must be within the build context directory - Docker cannot access files outside
3. **PyPI availability**: CUDA-specific PyTorch versions aren't on standard PyPI - use CPU versions for Docker or specify correct index URL
4. **Filenames**: Avoid spaces in filenames when using Docker COPY - use underscores or hyphens
5. **PYTHONPATH**: Must match the actual module structure - check where `__init__.py` files are located
6. **Separate requirements**: Keep CPU and GPU requirements separate to avoid conflicts

---

## GPU Build Specific Notes (March 2026)

### 6. PyTorch CUDA Version Compatibility

**Problem**: Tried to use PyTorch 2.0.1+cu116 but it doesn't exist in the PyTorch wheel index.

```
ERROR: Could not find a version that satisfies the requirement torch==2.0.1+cu116
ERROR: No matching distribution found for torch==2.0.1+cu116
```

**Solution**: Use PyTorch 1.13.1+cu116 which is available:

```dockerfile
RUN pip install --no-cache-dir torch==1.13.1+cu116 --index-url https://download.pytorch.org/whl/cu116
```

**Note**: CUDA 11.6 support was deprecated in newer PyTorch versions. The last version supporting cu116 is 1.13.x.

---

### 7. torchvision Not Available for CUDA 11.6

**Problem**: torchvision with CUDA 11.6 support is very limited. The index only shows versions 0.1.6 and 0.2.0.

```
ERROR: Could not find a version that satisfies the requirement torchvision==0.14.1+cu116
```

**Solution**: Since the codebase doesn't use torchvision (only torch), we removed it from the GPU Dockerfile. This significantly reduces image size.

---

### 8. .dockerignore Excluding CUDA Packages

**Problem**: The docker build failed because `cuda-packages/` directory wasn't being copied to the build context.

```
COPY failed: file not found in build context or excluded by .dockerignore: stat cuda-packages/: file does not exist
```

**Solution**: Removed `cuda-packages/` from `.dockerignore`:

```dockerignore
# Before:
cuda-packages/

# After: (removed this line)
```

---

### 9. apt-key Command Deprecated

**Problem**: The `apt-key` command used in install_cuda.sh is deprecated in newer Debian/Ubuntu versions.

```
/tmp/cuda/install_cuda.sh: line 10: apt-key: command not found
```

**Solution**: Updated to use the new keyring method:

```bash
# Old (deprecated):
apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub

# New:
cat /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub | gpg --dearmor -o /usr/share/keyrings/cuda-repo-ubuntu2004-11-6-local-archive-keyring.gpg
```

And use signed-by in sources.list:

```bash
echo "deb [signed-by=/usr/share/keyrings/cuda-repo-ubuntu2004-11-6-local-archive-keyring.gpg] file:/var/cuda-repo-ubuntu2004-11-6-local/ /" > /etc/apt/sources.list.d/cuda-11-6-local.list
```

---

### 10. CUDA Repository Package Availability

**Problem**: Even after fixing GPG keys, the CUDA runtime packages couldn't be installed because the packages aren't available for Debian Trixie (the base image).

```
E: Unable to locate package cuda-libraries-runtime-11-6
```

**Result**: Only cuDNN was successfully installed from the local .deb packages. The CUDA toolkit libraries come pre-bundled with PyTorch's CUDA runtime.

**Note**: This is acceptable because:
- PyTorch with CUDA support includes the necessary CUDA runtime libraries
- cuDNN (the deep neural network library) was successfully installed
- The container can still utilize GPU for training via PyTorch

---

## Final GPU Image Summary

- **Base Image**: python:3.11-slim
- **PyTorch**: 1.13.1+cu116 (with CUDA 11.6 support)
- **cuDNN**: 8.4.1 (installed from local .deb)
- **Image Size**: ~17.4GB
- **Key Dependencies**: transformers, accelerate, fastapi, uvicorn, scikit-learn

---

### 11. PyTorch Version Upgraded by pip - CUDA Incompatibility

**Problem**: The GPU Docker container was built with PyTorch 1.13.1+cu116 (compatible with CUDA 11.6), but after installing other packages (like `transformers`, `accelerate`), pip automatically upgraded PyTorch to version 2.10.0 (which requires CUDA 12.8). This caused the GPU to not be usable:

```python
# Inside container:
>>> import torch
>>> torch.cuda.is_available()
False
>>> torch.version.cuda
'12.8'  # Expected 11.6!
```

The error message was:
```
RuntimeError: The NVIDIA driver on your system is too old (found version 11060). 
Please update your GPU driver... Alternatively, go to pytorch.org to install a PyTorch 
version that has been compiled with your version of the CUDA driver.
```

**Root Cause**: The host machine has:
- NVIDIA Driver version 510.73.05
- CUDA Version 11.6

When pip installs packages, it resolves dependencies and may upgrade PyTorch to the latest version (2.10.0+cu128), which is incompatible with the older CUDA driver.

**Solution**: Pin exact versions of all packages to match the host machine's working configuration:

```dockerfile
# Install PyTorch with CUDA 11.6 support (must match host CUDA version)
RUN pip install --no-cache-dir torch==1.13.1+cu116 torchvision==0.14.1+cu116 --index-url https://download.pytorch.org/whl/cu116

# Install remaining requirements with EXACT versions from host machine
RUN pip install --no-cache-dir \
    transformers==4.35.0 \
    tokenizers==0.14.1 \
    accelerate==1.13.0 \
    tqdm>=4.66.0 \
    pandas>=2.0.0 \
    numpy>=1.26.0 \
    scikit-learn>=1.7.0 \
    joblib>=1.5.0 \
    fastapi==0.135.1 \
    uvicorn==0.41.0 \
    pydantic>=2.6.0

# Re-install PyTorch to ensure correct version (in case pip upgraded it)
RUN pip install --no-cache-dir --force-reinstall torch==1.13.1+cu116 torchvision==0.14.1+cu116 --index-url https://download.pytorch.org/whl/cu116
```

**Host Machine Working Configuration**:
| Package | Version |
|---------|---------|
| Python | 3.10.4 |
| PyTorch | 1.13.1+cu116 |
| CUDA | 11.6 |
| transformers | 4.35.0 |
| accelerate | 1.13.0 |
| tokenizers | 0.14.1 |
| fastapi | 0.135.1 |
| uvicorn | 0.41.0 |

**Key Takeaway**: Always use exact versions (`==`) for packages that have CUDA dependencies to prevent pip from upgrading them. The `--force-reinstall` at the end ensures PyTorch stays at the correct version.

**Mirror Configuration**:
- Regular Python packages: Tsinghua/Tuna mirror (`pypi.tuna.tsinghua.edu.cn`)
- PyTorch CUDA wheels: SJTU mirror (`mirror.sjtu.edu.cn/pytorch-wheels/cu116`) - faster than official pytorch.org for China region
