"""Prepare a clean, hash-locked Docker context; never build or push implicitly."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def prepare(output, *, kind, config=None):
    source = Path(__file__).resolve().parents[2]
    output = Path(output).resolve()
    if output.is_relative_to(source):
        raise ValueError("build context must be outside source tree")
    output.mkdir(parents=True, exist_ok=False)
    package = output / "recommend_tf"
    package.mkdir()
    include = ["inference"] if kind == "inference" else ["train", "platforms"]
    for directory in include:
        for path in (source / directory).rglob("*"):
            if not path.is_file() or not (
                path.suffix == ".py" or path.name.endswith(".schema.yml")
            ):
                continue
            if "__pycache__" in path.parts or (kind == "inference" and "tests" in path.parts):
                continue
            target = package / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    if kind != "inference":
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copyfile(source / name, package / name)
        subprocess.run(
            [
                "uv",
                "export",
                "--project",
                str(source),
                "--frozen",
                "--no-dev",
                "--no-emit-project",
                "--format",
                "requirements-txt",
                "--output-file",
                str(output / "requirements.txt"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    if kind == "train":
        if config is None:
            raise ValueError("training image requires explicit source config")
        raw = Path(config).read_bytes()
        json.loads(raw)
        (output / "config.json").write_bytes(raw)
    dockerfile = source / ("inference/Dockerfile" if kind == "inference" else "Dockerfile." + kind)
    shutil.copyfile(dockerfile, output / "Dockerfile")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["train", "pipeline", "inference"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    prepare(args.output, kind=args.kind, config=args.config)


if __name__ == "__main__":
    main()
