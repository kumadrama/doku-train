"""The deployed ready/recovery ZIP must import without training dependencies."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def test_ready_package_imports_without_site_packages(tmp_path):
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "package_tf_ready", root / "scripts/package_tf_ready.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "function.zip"
    module.package(archive)
    extracted = tmp_path / "lambda"
    with ZipFile(archive) as bundle:
        bundle.extractall(extracted)
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import ready_event; from recommend_tf.platforms.sagemaker import recover_daily",
        ],
        cwd=extracted,
        check=True,
        capture_output=True,
    )
