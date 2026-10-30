"""Sealed successful run evidence, independent of the training runtime."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path


def checksum(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def files(root):
    result = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("artifact symlinks are forbidden")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = checksum(path)
    return result


def seal_run(root, status_name):
    root = Path(root)
    evidence = {
        name: checksum(root / name)
        for name in (status_name, "experiment-config.json", "resolved-config.json")
    }
    for folder in ("input", "training"):
        evidence.update({f"{folder}/{name}": value for name, value in files(root / folder).items()})
    (root / "COMMITTED.json").write_text(
        json.dumps({"version": "doku-tf-run/v1", "files": evidence}, sort_keys=True)
    )


def verify_run(root, evidence=None):
    root = Path(root).resolve()
    record = evidence if evidence is not None else json.loads((root / "COMMITTED.json").read_text())
    if record.get("version") != "doku-tf-run/v1":
        raise ValueError("unsupported run evidence")
    for name, expected in record["files"].items():
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or checksum(path) != expected
        ):
            raise ValueError("run artifact checksum/path mismatch")
    return record


@contextmanager
def reserve_output(path):
    """Reserve a create-only destination before export; clean only our empty placeholder."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield
    finally:
        try:
            path.rmdir()
        except OSError:
            # A successfully committed nonempty package must be retained.
            pass
