#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
import json,re,subprocess
r=json.load(open('release.json'))
assert re.fullmatch(r'[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[^@]+@sha256:[a-f0-9]{64}', r['pipeline_image'])
revision=r['source_revision']
assert re.fullmatch(r'[a-f0-9]{40}',revision)
if subprocess.run(['git','cat-file','-e',revision],capture_output=True).returncode:
    subprocess.run(['git','fetch','origin',revision],check=True)
# Changing runtime source requires a newly tested immutable runtime artifact.
subprocess.run(['git','diff','--exit-code',revision,'HEAD','--',':(top)recommend_tf',':(top,exclude)recommend_tf/train/tests/**'],check=True)
PY
