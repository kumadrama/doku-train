"""Upstream identity and window checks for the deployment-only ready adapter."""

import io
import json
from datetime import date

import pytest

from recommend_tf.platforms.sagemaker.ready_event import build_ready, cutoff_from_task, sha


def settings():
    return dict(
        cluster_arn="cluster",
        upstream_definition="arn:task-definition/samples",
        container="samples",
        environment="prod",
        bucket="bucket",
        daily_prefix="daily",
        training_days=28,
        validation_days=1,
    )


def test_only_completed_daily_projection():
    options = settings()
    task = dict(
        clusterArn="cluster",
        lastStatus="STOPPED",
        taskDefinitionArn="arn:task-definition/samples:3",
        containers=[dict(exitCode=0)],
        overrides=dict(
            containerOverrides=[
                dict(
                    name="samples",
                    command=[
                        "fyp-rank-samples",
                        "join",
                        "--environment",
                        "prod",
                        "--mode",
                        "daily",
                        "--project-training",
                        "--date",
                        "2026-10-29",
                    ],
                )
            ]
        ),
    )
    assert cutoff_from_task(task, options) == date(2026, 10, 29)
    task["overrides"]["containerOverrides"][0]["command"] = [
        "fyp-rank-samples",
        "join",
        "--mode",
        "hourly",
    ]
    assert cutoff_from_task(task, options) is None
    task["containers"][0]["exitCode"] = 1
    with pytest.raises(ValueError, match="did not succeed"):
        cutoff_from_task(task, options)


def test_window_and_manifest_hash():
    objects = {}
    for n in range(1, 30):
        day = f"2026-10-{n:02d}"
        base = f"daily/date={day}"
        manifest = json.dumps(
            dict(source_markers={str(i): "hash" for i in range(24)}, projection_sha256="projection")
        ).encode()
        objects[base + "/run_id=a/manifest.json"] = manifest
        objects[base + "/_SUCCESS"] = json.dumps(
            dict(
                pointer_version="doku-train-day/v1",
                date=day,
                manifest_uri=f"s3://bucket/{base}/run_id=a/manifest.json",
                manifest_sha256=sha(manifest),
                projection_sha256="projection",
            )
        ).encode()
        for name in ("features", "labels"):
            objects[base + f"/run_id=a/contracts/{name}.json"] = b"{}"

    class Store:
        def get_object(self, *, Bucket, Key):
            assert Bucket == "bucket"
            return dict(Body=io.BytesIO(objects[Key]), VersionId="version")

    result = build_ready(Store(), settings(), date(2026, 10, 29))
    assert len(result["sources"]) == 29
    assert result["sources"][0]["date"] == "2026-10-01"
    assert result["sources"][-1]["date"] == "2026-10-29"
    objects["daily/date=2026-10-29/run_id=a/manifest.json"] = b"{}"
    with pytest.raises(ValueError, match="evidence differs"):
        build_ready(Store(), settings(), date(2026, 10, 29))
