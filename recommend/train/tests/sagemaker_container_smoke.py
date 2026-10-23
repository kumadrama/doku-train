"""Offline container acceptance: synthetic S3 objects, real CPU train/test and events."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from recommend.train.comm.datasvr.published_dataset import PublishedDataset, digest
from recommend.train.comm.training_pipeline import source_identity
from recommend.train.platforms.sagemaker.artifacts import verify_archive
from recommend.train.platforms.sagemaker.config import Binding, Runtime
from recommend.train.platforms.sagemaker.entrypoint import run_package
from recommend.train.platforms.sagemaker.submit import canonical, prepare
from recommend.train.tests.experiment_fixture import make_experiment


class SyntheticS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, io.BytesIO]:
        return {"Body": io.BytesIO(self.objects[Bucket, Key])}

    def put_object(self, **kwargs: object) -> None:
        assert kwargs["ServerSideEncryption"] == "aws:kms"
        assert kwargs["IfNoneMatch"] == "*"
        name = str(kwargs["Bucket"]), str(kwargs["Key"])
        assert name not in self.objects
        body = kwargs["Body"]
        assert isinstance(body, bytes)
        self.objects[name] = body


def main() -> None:
    import argparse
    import platform
    import subprocess
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["dnn", "deepfm", "xgboost"])
    args = parser.parse_args()
    if args.model is None:
        for model in ("dnn", "deepfm", "xgboost"):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "recommend.train.tests.sagemaker_container_smoke",
                    "--model",
                    model,
                ],
                check=True,
                timeout=300,
            )
        return

    assert platform.machine() == "x86_64", "Acceptance must run on amd64"
    source = source_identity()
    with tempfile.TemporaryDirectory() as directory:
        for model in (args.model,):
            root = Path(directory) / model
            config = make_experiment(root / "fixture", model)
            if model == "xgboost":
                config["execution"]["intra_op_threads"] = 1
            local = root / "fixture/dataset"
            manifest = json.loads((local / "manifest.json").read_text())
            s3 = SyntheticS3()
            shards = {}
            for i, part in enumerate(manifest["partitions"]):
                uri = f"s3://synthetic/input/shard-{i}.parquet"
                shards[uri] = Path(part["uri"].removeprefix("file://"))
                part["uri"] = uri
            manifest["content_sha256"] = digest(
                {k: v for k, v in manifest.items() if k != "content_sha256"}
            )
            s3.objects["synthetic", "input/manifest.json"] = canonical(manifest)
            for name in ("features", "labels"):
                s3.objects["synthetic", f"input/contracts/{name}.json"] = (
                    local / f"contracts/{name}.json"
                ).read_bytes()
            config["dataset"]["manifest_uri"] = "s3://synthetic/input/manifest.json"
            binding = Binding(
                version="doku-sagemaker-binding/v1",
                environment="test",
                region="us-east-2",
                account_id="123456789012",
                role_arn="arn:aws:iam::123456789012:role/test",
                subnet_ids=["subnet-test"],
                security_group_ids=["sg-test"],
                input_prefixes=["s3://synthetic/input/"],
                experiment_uri="s3://synthetic/experiments/",
                kms_key_arn="arn:aws:kms:us-east-2:123456789012:key/test",
            )
            runtime = Runtime(
                version="doku-sagemaker-runtime/v1",
                environment="test",
                binding="synthetic.json",
                image_uri="123456789012.dkr.ecr.us-east-2.amazonaws.com/test@sha256:" + "a" * 64,
                source_revision=source["revision"],
                source_sha256=source["working_source_sha256"],
                lock_sha256=source["lock_sha256"],
                loss_interval_steps=1,
                stages=["train"] if model == "xgboost" else ["train", "test"],
            )
            prepare(config, runtime, binding, model, s3, root / "prepared")
            package = root / "prepared/package.json"
            with patch.object(
                PublishedDataset, "_local_path", lambda self, uri, mapping=shards: mapping.get(uri)
            ):
                run_package(
                    package,
                    root / "ml",
                    s3,
                    expected_sha=hashlib.sha256(package.read_bytes()).hexdigest(),
                )
            verify_archive(root / "ml/model")
            files = list((root / "ml/work/events").rglob("events.out.tfevents.*"))
            assert files
            tags = set()
            for file in files:
                events = EventAccumulator(str(file)).Reload()
                tags.update(events.Tags()["scalars"])
            if model == "xgboost":
                assert any(tag.startswith("validation/") for tag in tags)
                assert not (root / "ml/model/test-metrics.json").exists()
                assert (root / "ml/model/checkpoint.xgb.zip").exists()
            else:
                assert any(tag.startswith("train/") for tag in tags)
                assert any(tag.startswith("test/") for tag in tags)
            print(f"CONTAINER_SMOKE_PASS model={model} machine={platform.machine()}", flush=True)


if __name__ == "__main__":
    main()
