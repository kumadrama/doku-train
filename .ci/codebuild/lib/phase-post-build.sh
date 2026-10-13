#!/usr/bin/env bash
# CodeBuild post_build orchestration (must be invoked with bash, not /bin/sh).
set -euo pipefail

ROOT="${CODEBUILD_SRC_DIR:?CODEBUILD_SRC_DIR is required}"

# CodeBuild always runs post_build; avoid cascading noise after a failed phase.
if [[ "${CODEBUILD_BUILD_SUCCEEDING:-1}" != "1" ]]; then
  echo "post_build: skipped (earlier phase failed)" >&2
  exit 0
fi

echo "Pushed (channel=${BUILD_CHANNEL:-main}):"
cat "${ROOT}/image_uri.txt"
