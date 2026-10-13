#!/usr/bin/env bash
# CodeBuild pre_build orchestration (must be invoked with bash, not /bin/sh).
set -euo pipefail

LIB="$(cd "$(dirname "$0")" && pwd)"
ROOT="${CODEBUILD_SRC_DIR:?CODEBUILD_SRC_DIR is required}"

test -n "${SERVICE_DIR:-}"
test -n "${ECR_REPOSITORY:-}"
test -n "${AWS_DEFAULT_REGION:-}"

bash "${LIB}/ecr-login.sh" --region "${AWS_DEFAULT_REGION}"
bash "${LIB}/github-mod-auth.sh"
bash "${LIB}/allocate-image-tags.sh" \
  --ecr-repository "${ECR_REPOSITORY}" \
  --channel "${BUILD_CHANNEL:-main}" \
  --repo-root "${ROOT}" \
  --region "${AWS_DEFAULT_REGION}" \
  --out "${ROOT}/image_tag.env" \
  --service-dir "${SERVICE_DIR}" \
  --source-ref "${SOURCE_REF:-}"

SVC="${ROOT}/${SERVICE_DIR}"
test -f "${SVC}/build/build.sh"
test -f "${SVC}/build/Dockerfile"
test -x "${SVC}/build/build.sh" || chmod +x "${SVC}/build/build.sh"
test -x "${SVC}/build/docker-build.sh" || chmod +x "${SVC}/build/docker-build.sh"
