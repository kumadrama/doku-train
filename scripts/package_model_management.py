"""Build a deterministic controller ZIP; no config, upload, or deployment side effects."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def package(output, sdk_directory):
    root = Path(__file__).resolve().parents[1]
    source_names = [
        "recommend_tf/platforms/sagemaker/executor.py",
        "recommend_tf/platforms/sagemaker/idea_appconfig.py",
        "recommend_tf/platforms/sagemaker/publish_model.py",
        "recommend_tf/platforms/sagemaker/ready_event.py",
        "recommend_tf/platforms/sagemaker/operation_identity.py",
        "recommend_tf/platforms/sagemaker/runtime_binding.py",
        "recommend_tf/inference/official.py",
        "recommend_tf/inference/artifact_loader.py",
        "recommend_tf/inference/request_adapter.py",
        "recommend_tf/train/comm/artifact_integrity.py",
        "recommend_tf/train/comm/early_stop.py",
        "recommend_tf/train/comm/export_tasks.py",
        "recommend_tf/train/comm/learning_rate.py",
        "recommend_tf/train/comm/model_identity.py",
        "recommend_tf/train/comm/datasvr/daily_window.py",
        "recommend_tf/train/models/rerank/model_params.py",
    ]
    source_names += [
        p.relative_to(root).as_posix()
        for p in (root / "recommend_tf/platforms/model_management").glob("*.py")
        if p.name != "worker.py"
    ]
    entries = {name: (root / name).read_bytes() for name in source_names}
    for name in list(entries):
        for parent in Path(name).parents:
            if parent != Path("."):
                entries.setdefault((parent / "__init__.py").as_posix(), b"")
    # Lambda's bundled SDK may lack the deployment CAS fields. Package the repository-locked SDK.
    sdk = Path(sdk_directory)
    for project in ("boto3", "botocore"):
        metadata = sdk / (project + "-1.43.120.dist-info/METADATA")
        if not metadata.is_file():
            raise ValueError("SDK vendor directory must contain locked " + project + "==1.43.120")
    for path in sdk.rglob("*"):
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        ):
            name = path.relative_to(sdk).as_posix()
            if name in entries or path.is_symlink():
                raise ValueError("SDK package path collision/symlink")
            entries[name] = path.read_bytes()
    inventory = {name: hashlib.sha256(body).hexdigest() for name, body in entries.items()}
    entries["management-source.json"] = json.dumps(inventory, sort_keys=True).encode()
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for name, body in sorted(entries.items()):
            info = ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, body)
    return hashlib.sha256(Path(output).read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sdk-directory", required=True)
    args = parser.parse_args()
    print(package(args.output, args.sdk_directory))
