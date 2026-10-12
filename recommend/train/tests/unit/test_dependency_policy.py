from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

FORBIDDEN = ("onnx", "tensorflow", "nvidia", "cuda", "nccl", "mpi")
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)aws_secret_access_key\s*=\s*['\"][^'\"]+['\"]"),
    re.compile(r"(?i)password\s*=\s*['\"][^'\"]+['\"]"),
)


def test_project_and_lock_have_no_gpu_or_inference_runtime_packages() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = "\n".join(project["project"]["dependencies"]).casefold()
    lock = Path("uv.lock").read_text().casefold()
    for forbidden in FORBIDDEN:
        assert forbidden not in dependencies
        assert f'name = "{forbidden}' not in lock


def test_make_check_validates_dependency_lock() -> None:
    makefile = Path("Makefile").read_text()
    assert "uv lock --check" in makefile
    assert "check: lockcheck format lint typecheck test" in makefile


def test_production_python_has_no_forbidden_platform_or_exporter_coupling() -> None:
    production_files = [
        path for path in Path("recommend/train").rglob("*.py") if "tests" not in path.parts
    ]
    for path in production_files:
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                assert not module.startswith(("tensorflow", "doku_offline", "finder")), path
                if "comm" in path.parts:
                    assert not module.startswith("recommend.train.models"), path
        for pattern in SECRET_PATTERNS:
            assert pattern.search(source) is None, path


def test_model_exporter_is_protocol_only() -> None:
    path = Path("recommend/train/comm/model_exporter.py")
    tree = ast.parse(path.read_text())
    exporter_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.casefold().endswith("exporter")
    ]
    assert len(exporter_classes) == 1
    assert exporter_classes[0].name == "ModelExporter"
    assert any(
        isinstance(base, ast.Name) and base.id == "Protocol" for base in exporter_classes[0].bases
    )
