"""Build the existing S3 bootstrap bundle and verify the deployment source pin."""

import argparse
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path


def source_archive(root):
    root = Path(root)
    members = {}
    source = root / "recommend_tf"
    paths = [source / "__init__.py"]
    for directory in ("train", "platforms", "inference"):
        paths.extend((source / directory).rglob("*"))
    for path in sorted(paths):
        relative = path.relative_to(root)
        if not path.is_file() or "tests" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.suffix == ".py" or path.name.endswith(".schema.yml"):
            members[relative.as_posix()] = path.read_bytes()
    if "recommend_tf/platforms/sagemaker/idea_publication.py" not in members:
        raise ValueError("Idea publication source is missing")
    stream = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0, compresslevel=9) as zipped,
        tarfile.open(fileobj=zipped, mode="w|", format=tarfile.USTAR_FORMAT) as archive,
    ):
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(body))
    return stream.getvalue(), members


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin", action="store_true", help="update the local manifest source pin")
    args = parser.parse_args()
    service = Path(__file__).resolve().parents[1]
    body, members = source_archive(service.parent)
    digest = hashlib.sha256(body).hexdigest()
    manifest_path = service / "deploy/prod/task-definition.yml"
    manifest = json.loads(manifest_path.read_text())
    environment = manifest["containerDefinitions"][0]["environment"]
    values = {item["name"]: item["value"] for item in environment}
    key = values["ARTIFACT_KEY"].rsplit("/", 2)[0] + "/" + digest + "/worker.tar.gz"
    if args.pin:
        for item in environment:
            if item["name"] == "ARTIFACT_SHA":
                item["value"] = digest
            elif item["name"] == "ARTIFACT_KEY":
                item["value"] = key
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    elif values["ARTIFACT_SHA"] != digest or values["ARTIFACT_KEY"] != key:
        raise ValueError("runtime changed: regenerate and review the Idea worker source pin")
    output = service / "build/output"
    if output.exists():
        shutil.rmtree(output)
    for name, content in members.items():
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (output / "worker.tar.gz").write_bytes(body)
    print(json.dumps({"source_sha256": digest, "files": len(members), "bytes": len(body)}))


if __name__ == "__main__":
    main()
