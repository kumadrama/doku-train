#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
base_image=$(python3 -c 'import json; print(json.load(open("release.json"))["pipeline_image"])')
docker build --platform linux/amd64 --build-arg "BASE_IMAGE=$base_image" -f build/Dockerfile -t "${1:?image URI required}" .
