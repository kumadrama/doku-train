#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
runtime_image=$(python3 -c 'import json; print(json.load(open("runtime.json"))["runtime_image"])')
revision=$(git rev-parse HEAD)
docker build --platform linux/amd64 --build-arg "RUNTIME_IMAGE=$runtime_image" --label "org.opencontainers.image.revision=$revision" -t "${1:?image URI required}" build
docker run --rm --network none --entrypoint python "$1" -c 'from recommend_tf.platforms.model_management.worker import run; import boto3,botocore; assert boto3.__version__ == botocore.__version__ == "1.43.120"; print("managed-worker-imports-ok")'
