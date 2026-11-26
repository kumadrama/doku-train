#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c 'import json,re; value=json.load(open("runtime.json")); assert re.fullmatch(r".+@sha256:[a-f0-9]{64}", value["runtime_image"]); print("pinned shared runtime verified")'
