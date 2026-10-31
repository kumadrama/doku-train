"""Deterministic ready/recovery Lambda package. Does not upload or deploy."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def package(output, release=None, config=None):
    root = Path(__file__).resolve().parents[1]
    names = [
        "recommend_tf/platforms/runtime_policy.py",
        "recommend_tf/platforms/sagemaker/ready_event.py",
        "recommend_tf/platforms/sagemaker/recover_daily.py",
        "recommend_tf/platforms/sagemaker/publish_model.py",
        "recommend_tf/train/comm/artifact_integrity.py",
    ]
    entries = {name: (root / name).read_bytes() for name in names}
    if release is not None:
        manifest = json.loads(Path(release).read_text())
        raw = Path(config).read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["config_sha256"]:
            raise ValueError("release/config checksum mismatch")
        entries["recommend_tf/platforms/sagemaker/ready_release.json"] = json.dumps(
            {
                "schema_version": "doku-tf-ready-release/v1",
                "release": manifest,
                "config_json": raw.decode(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    # Preserve the existing Lambda handler while using proper package imports.
    entries["ready_event.py"] = (
        b"from recommend_tf.platforms.sagemaker.ready_event import handler\n"
    )
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, body in sorted(entries.items()):
            info = ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    package(args.output, args.release, args.config)
