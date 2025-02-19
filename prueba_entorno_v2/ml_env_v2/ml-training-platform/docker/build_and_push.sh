#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define image name and tag
IMAGE_NAME="yagoutad/ml-training-env"
IMAGE_TAG="latest"

# Build the image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ${DIR}

# Push the image
echo "Pushing Docker image..."
docker push ${IMAGE_NAME}:${IMAGE_TAG}
