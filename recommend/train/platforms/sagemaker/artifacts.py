from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

REQUIRED = [
    "checkpoint.pt",
    "metrics.json",
    "fitted-feature-state/state.json",
    "lineage.json",
    "resolved-config.json",
    "status.json",
    "test-metrics.json",
    "test-evidence.json",
    "input/record.json",
    "input/manifest.json",
    "input/contracts/features.json",
    "input/contracts/labels.json",
]


def archive_run(run: Path, target: Path, submission: Path) -> None:
    if json.loads((run / "status.json").read_text()).get("status") != "complete":
        raise ValueError("training incomplete")
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise FileExistsError("archive destination is not empty")
    for name in REQUIRED:
        source = run / name
        if not source.is_file():
            raise ValueError(f"missing artifact {name}")
    for name in REQUIRED:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / name, dest)
    shutil.copyfile(submission, target / "submission.json")
    files = {
        str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in target.rglob("*")
        if p.is_file()
    }
    (target / "artifact-manifest.json").write_text(
        json.dumps(
            {"version": "doku-sagemaker-artifacts/v1", "status": "complete", "files": files},
            sort_keys=True,
        )
    )


def verify_archive(root: Path) -> None:
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    if (
        manifest.get("version") != "doku-sagemaker-artifacts/v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("invalid artifact manifest")
    if not set([*REQUIRED, "submission.json"]) <= set(manifest["files"]):
        raise ValueError("incomplete artifact manifest")
    for name, sha in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
            raise ValueError("unsafe or missing artifact")
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("artifact checksum mismatch")
