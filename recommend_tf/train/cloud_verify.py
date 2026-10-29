"""SageMaker synthetic acceptance harness. No external data or AWS calls."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def main():
    output = Path(os.environ.get("DOKU_VERIFY_OUTPUT", "/opt/ml/output/data"))
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    lock_hash = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    expected = os.environ.get("DOKU_LOCK_SHA256")
    if expected and lock_hash != expected:
        raise ValueError("Lock digest mismatch")
    import pyarrow as pa

    sources = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "train").rglob("*.py"))
    }
    source_hash = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if os.environ.get("DOKU_SOURCE_SHA256") and source_hash != os.environ["DOKU_SOURCE_SHA256"]:
        raise ValueError("Source digest mismatch")
    import tensorflow as tf

    identity = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "tensorflow": tf.__version__,
        "pyarrow": pa.__version__,
        "lock_sha256": lock_hash,
        "source_digest": source_hash,
    }
    print("VERIFY_IDENTITY " + json.dumps(identity), flush=True)
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("Cloud acceptance requires native Linux amd64")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(root / "train/tests"),
            "-q",
            "-p",
            "no:cacheprovider",
            "--disable-warnings",
            "--tb=short",
            "--junitxml=" + str(output / "pytest.xml"),
            "--basetemp=" + str(output / "synthetic-work"),
        ],
        check=False,
    )
    report = {
        **identity,
        "exit_code": result.returncode,
        "seconds": time.monotonic() - started,
        "synthetic_only": True,
        "passed": result.returncode == 0,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2))
    print("VERIFY_RESULT " + json.dumps(report), flush=True)
    if result.returncode:
        Path("/opt/ml/output/failure").write_text(
            "Synthetic migration acceptance failed; see pytest.xml"
        )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
