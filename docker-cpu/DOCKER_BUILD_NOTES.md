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
