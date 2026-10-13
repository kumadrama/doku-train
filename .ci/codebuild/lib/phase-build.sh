#!/usr/bin/env bash
# CodeBuild build orchestration (must be invoked with bash, not /bin/sh).
set -euo pipefail

LIB="$(cd "$(dirname "$0")" && pwd)"
ROOT="${CODEBUILD_SRC_DIR:?CODEBUILD_SRC_DIR is required}"

test -n "${SERVICE_DIR:-}"
test -n "${ECR_REPOSITORY:-}"
test -n "${AWS_DEFAULT_REGION:-}"
test -f "${ROOT}/image_tag.env"

# shellcheck disable=SC1091
. "${ROOT}/image_tag.env"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

cd "${ROOT}/${SERVICE_DIR}"
./build/build.sh
./build/docker-build.sh "${IMAGE_URI}"
PUSH_ARGS=("${IMAGE_URI}")
if [[ -n "${IMAGE_TAG_ACTOR:-}" ]]; then
  PUSH_ARGS+=("${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG_ACTOR}")
fi
bash "${LIB}/push-ecr-images.sh" \
  --out "${ROOT}/image_uri.txt" \
  "${PUSH_ARGS[@]}"
