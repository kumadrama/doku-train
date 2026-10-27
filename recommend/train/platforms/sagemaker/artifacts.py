from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from recommend.train.comm import timing

REQUIRED = [
    "metrics.json",
    "fitted-feature-state/state.json",
    "lineage.json",
    "resolved-config.json",
    "status.json",
    "input/record.json",
    "input/manifest.json",
    "input/contracts/features.json",
    "input/contracts/labels.json",
]


def required_files(package: dict[str, Any]) -> list[str]:
    model = package["config"]["model_type"]
    checkpoint = {"rerank": "checkpoint.pt", "rerank_tree": "checkpoint.xgb.zip"}[model]
    stages = package["runtime"].get("stages", ["train", "test"])
    return [
        *REQUIRED,
        checkpoint,
        *(
            ["tree-buckets.json"]
            if package["config"].get("tree_binning", {}).get("enabled")
            else []
        ),
        *(["test-metrics.json", "test-evidence.json"] if "test" in stages else []),
    ]


def validate_timing_files(
    root: Path, package: dict[str, Any], names: set[str], *, archived: bool
) -> None:
    if not package["config"].get("profiling", {}).get("enabled"):
        return
    required_phases = {"train"}
    if "test" in package["runtime"].get("stages", ["train", "test"]):
        required_phases.add("test")
    if archived:
        required_phases.add("archive")
    found = set()
    for name in names:
        if not name.startswith("timing/") or not name.endswith("-summary.json"):
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
            raise ValueError("unsafe timing artifact")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe or missing timing summary")
        summary = json.loads(path.read_text())
        phase = summary.get("phase")
        invocation = summary.get("invocation_id", "")
        expected = f"timing/{phase}-{invocation}"
        if (
            summary.get("version") != "doku-timing/v1"
            or name != expected + "-summary.json"
            or expected + ".jsonl" not in names
        ):
            raise ValueError("incomplete timing invocation")
        if summary.get("status") == "complete":
            found.add(phase)
    if not required_phases <= found:
        raise ValueError("missing completed timing phases")


def _archive_run(run: Path, target: Path, submission: Path) -> dict[str, str]:
    if json.loads((run / "status.json").read_text()).get("status") != "complete":
        raise ValueError("training incomplete")
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise FileExistsError("archive destination is not empty")
    for name in required_files(json.loads(submission.read_text())):
        source = run / name
        if not source.is_file():
            raise ValueError(f"missing artifact {name}")
    for name in required_files(json.loads(submission.read_text())):
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(run / name, dest)
    shutil.copyfile(submission, target / "submission.json")
    return {
        str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in target.rglob("*")
        if p.is_file() and p.name != "artifact-manifest.json"
    }


def write_manifest(target: Path, files: dict[str, str]) -> None:
    partial = target / "artifact-manifest.partial"
    partial.write_text(
        json.dumps(
            {"version": "doku-sagemaker-artifacts/v1", "status": "complete", "files": files},
            sort_keys=True,
        )
    )

    partial.replace(target / "artifact-manifest.json")


def archive_run(run: Path, target: Path, submission: Path) -> None:
    package = json.loads(submission.read_text())
    config = package["config"]
    timing_names = {str(p.relative_to(run)) for p in (run / "timing").glob("*") if p.is_file()}
    validate_timing_files(run, package, timing_names, archived=False)
    with timing.session(config, run, "archive") as timer, timer.span("archive", report=True):
        files = _archive_run(run, target, submission)
    if timer.enabled:
        # Publish only finalized diagnostics after archive timing is closed.
        destination = target / "timing"
        destination.mkdir(exist_ok=True)
        for source in (run / "timing").iterdir():
            if source.suffix in {".json", ".jsonl"}:
                if source.is_symlink() or not source.is_file():
                    raise ValueError("unsafe timing artifact")
                dest = destination / source.name
                shutil.copyfile(source, dest)
                files[str(dest.relative_to(target))] = hashlib.sha256(dest.read_bytes()).hexdigest()
    validate_timing_files(target, package, set(files), archived=True)
    write_manifest(target, files)


def verify_archive(root: Path) -> None:
    manifest = json.loads((root / "artifact-manifest.json").read_text())
    if (
        manifest.get("version") != "doku-sagemaker-artifacts/v1"
        or manifest.get("status") != "complete"
    ):
        raise ValueError("invalid artifact manifest")
    required = required_files(json.loads((root / "submission.json").read_text()))
    if not set([*required, "submission.json"]) <= set(manifest["files"]):
        raise ValueError("incomplete artifact manifest")
    for name, sha in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
            raise ValueError("unsafe or missing artifact")
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("artifact checksum mismatch")

    validate_timing_files(
        root,
        json.loads((root / "submission.json").read_text()),
        set(manifest["files"]),
        archived=True,
    )
