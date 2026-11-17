#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker build --platform linux/amd64 -f build/Dockerfile -t "${1:?image URI required}" .
