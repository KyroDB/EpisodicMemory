#!/bin/bash
# Container image build script for EpisodicMemory
#  Automated build process
#
# Usage:
#   ./scripts/build.sh [OPTIONS]
#
# Options:
#   --prod: Build production image (default)
#   --dev: Build development image
#   --tag TAG: Specify image tag (default: latest)
#   --push: Push to registry after build
#   --platform PLATFORM: Target platform (default: linux/amd64)
#   --registry REGISTRY: Container registry (default: docker.io)
#   --no-cache: Build without cache

set -euo pipefail

# Default values
IMAGE_NAME="episodic-memory"
IMAGE_TAG="latest"
BUILD_TYPE="prod"
PUSH=false
USE_CACHE=true
PLATFORM="linux/amd64"
REGISTRY="docker.io"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prod)
            BUILD_TYPE="prod"
            shift
            ;;
        --dev)
            BUILD_TYPE="dev"
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prod              Build production image (default)"
            echo "  --dev               Build development image"
            echo "  --tag TAG           Specify image tag (default: latest)"
            echo "  --push              Push to registry after build"
            echo "  --platform PLATFORM Target platform (default: linux/amd64)"
            echo "  --registry REGISTRY Container registry (default: docker.io)"
            echo "  --no-cache          Build without cache"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Determine Dockerfile
if [ "$BUILD_TYPE" = "dev" ]; then
    DOCKERFILE="Dockerfile.dev"
    IMAGE_TAG="${IMAGE_TAG}-dev"
else
    DOCKERFILE="Dockerfile"
fi

# Full image name
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=========================================="
echo "Building EpisodicMemory Container Image"
echo "=========================================="
echo "Build type: $BUILD_TYPE"
echo "Dockerfile: $DOCKERFILE"
echo "Platform: $PLATFORM"
echo "Image name: $FULL_IMAGE_NAME"
echo "Use cache: $USE_CACHE"
echo "Push after build: $PUSH"
echo "=========================================="
echo ""

# Build arguments
BUILD_ARGS=()
BUILD_ARGS+=("--file" "$DOCKERFILE")
BUILD_ARGS+=("--platform" "$PLATFORM")
BUILD_ARGS+=("--tag" "$FULL_IMAGE_NAME")

if [ "$USE_CACHE" = false ]; then
    BUILD_ARGS+=("--no-cache")
fi

# Build image
echo "Building image..."
docker build "${BUILD_ARGS[@]}" .

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo ""
echo "Build successful!"
echo ""

# Show image details
echo "Image details:"
docker images | head -1
docker images | grep "$IMAGE_NAME" | grep "$IMAGE_TAG"
echo ""

# Push to registry if requested
if [ "$PUSH" = true ]; then
    echo "Pushing image to registry..."
    docker push "$FULL_IMAGE_NAME"

    if [ $? -ne 0 ]; then
        echo "ERROR: Push failed"
        exit 1
    fi

    echo "Push successful!"
fi

echo "=========================================="
echo "Build complete!"
echo "Image: $FULL_IMAGE_NAME"
echo "=========================================="
