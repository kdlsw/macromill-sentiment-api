#!/bin/bash
# Build Docker image for Macromill Sentiment Analysis API

set -e

IMAGE_NAME="macromill/sentiment-api"
TAG="${1:-latest}"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"

cd "$(dirname "$0")/.."

# Build the image
docker build -t "${IMAGE_NAME}:${TAG}" -f docker/Dockerfile .

echo ""
echo "Build complete!"
echo ""
echo "To run the container:"
echo "  docker run -p 8000:8000 ${IMAGE_NAME}:${TAG}"
echo ""
echo "Or use docker-compose:"
echo "  cd docker && docker-compose up -d"
echo ""
