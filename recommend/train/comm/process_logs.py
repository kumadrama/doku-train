"""Preserve captured subprocess behavior while forwarding opted-in timing events."""

from __future__ import annotations

import subprocess
import tempfile
from typing import Any


def run_captured(args: list[str], config: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    profiling = config.get("profiling", {})
    if not (profiling.get("enabled") and profiling.get("print_logs")):
        return subprocess.run(args, text=True, capture_output=True, check=False)
    # Separate stderr avoids pipe deadlocks; only diagnostics are forwarded, never samples.
    with tempfile.TemporaryFile(mode="w+t") as errors:
        with subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=errors, text=True, bufsize=1
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                if line.startswith("DOKU_TIMING "):
                    print(line, end="", flush=True)
            returncode = proc.wait()
        errors.seek(0)
        stderr = errors.read()[-2000:]
    return subprocess.CompletedProcess(args, returncode, "", stderr)
