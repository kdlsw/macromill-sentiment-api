#!/bin/bash
# Run Docker container for Macromill Sentiment Analysis API

set -e

IMAGE_NAME="macromill/sentiment-api"
PORT="${1:-8000}"

echo "Running container on port ${PORT}..."

docker run -p ${PORT}:8000 \
    --name macromill-sentiment-api \
    -e PYTHONUNBUFFERED=1 \
    --restart unless-stopped \
    ${IMAGE_NAME}:latest
