#!/usr/bin/env bash
set -euo pipefail

IMAGE="dockerish999/ishan_posttraining"
TAG="${1:-latest}"

echo "[docker_push.sh] Building $IMAGE:$TAG..."
docker buildx build --platform linux/amd64 -t "$IMAGE:$TAG" --push .

echo "[docker_push.sh] Done: $IMAGE:$TAG"
