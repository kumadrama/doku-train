#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -m pip install --disable-pip-version-check 'uv==0.11.18'
python3 -m recommend_tf.platforms.sagemaker.build_context --kind pipeline --output tf-pipeline/build/output
