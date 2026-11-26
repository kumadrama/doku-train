#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
base_image=$(python3 -c 'import json; print(json.load(open("release.json"))["managed_runtime_base_image"])')
revision=$(git rev-parse HEAD)
docker build --platform linux/amd64 --build-arg "BASE_IMAGE=$base_image" --label "org.opencontainers.image.revision=$revision" -f build/Dockerfile -t "${1:?image URI required}" build/output
docker run --rm --network none --entrypoint python "$1" -c 'import boto3,botocore,tensorflow; from recommend_tf.platforms.model_management.worker import run; from recommend_tf.platforms.model_management.api import handler; from recommend_tf.train.cloud_train import main; assert boto3.__version__ == botocore.__version__ == "1.43.120"; print("managed-runtime-imports-ok")'
