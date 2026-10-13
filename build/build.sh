#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
python3 - "$root" <<'PY'
import pathlib
import shutil
import subprocess
import sys
root = pathlib.Path(sys.argv[1])
out = root / 'output'
if out.is_symlink():
    raise SystemExit('refusing symlink build output')
if out.exists():
    shutil.rmtree(out)
out.mkdir()
for name in ('pyproject.toml', 'uv.lock', 'README.md'):
    shutil.copyfile(root / name, out / name)
for path in (root / 'recommend').rglob('*'):
    if path.is_file() and path.suffix in ('.py', '.yml', '.yaml'):
        dest = out / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True).stdout
(out / 'build-revision.txt').write_text(revision)
PY
