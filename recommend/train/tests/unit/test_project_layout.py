from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


def test_finder_compatible_namespace_is_importable() -> None:
    for module in (
        "recommend.train.comm",
        "recommend.train.models",
        "recommend.train.offline",
        "recommend.train.tools",
    ):
        assert importlib.import_module(module) is not None


def test_torch_uses_explicit_cpu_index() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert project["tool"]["uv"]["sources"]["torch"] == [
        {"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"}
    ]
    assert project["tool"]["uv"]["index"][0]["explicit"] is True


def test_forbidden_runtime_dependencies_are_absent() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = "\n".join(project["project"]["dependencies"]).lower()
    for forbidden in ("onnx", "tensorflow", "cuda", "nccl", "mpi"):
        assert forbidden not in dependencies
