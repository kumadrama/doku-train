"""Synthetic published-manifest fixture, never a substitute for production evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from recommend.train.comm.datasvr.published_dataset import digest
from recommend.train.tests.synthetic_config import synthetic_config


def make_experiment(root: Path, model: str = "dnn", rows_per_day: int = 32) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    config = synthetic_config(model)
    dataset = root / "dataset"
    dataset.mkdir(exist_ok=True)
    (dataset / "contracts").mkdir(exist_ok=True)
    lists = {
        "u_skipped__drama_id",
        "u_drama_interests__id",
        "u_drama_interests__weight",
        "i_genre_v2_ids",
        "u_onboarding_genre_ids",
    }
    features = []
    for f in config["features"]:
        dtype = (
            "list<float64>"
            if f.endswith("__weight")
            else "list<int64>"
            if f in lists
            else "string"
            if f == "ctx_platform"
            else "int64"
            if f == "i_drama_id"
            else "float64"
        )
        features.append({"name": f, "dtype": dtype})
    labels = [
        "completion",
        "non_fast_swipe",
        "immersive_click",
        "watch_duration_ms",
        "video_finish_seen",
    ]
    (dataset / "contracts/features.json").write_text(
        json.dumps({"feature_spec_version": "fyp-rank-features/v4", "features": features})
    )
    (dataset / "contracts/labels.json").write_text(
        json.dumps({"label_definition_versions": ["fixed-5m-playback-stream-v4"], "labels": labels})
    )
    parts = []
    for day in range(8):
        start = datetime(2026, 10, 1, tzinfo=UTC) + timedelta(days=day)
        rows = []
        for i in range(rows_per_day):
            row = {f: float(i % 7) for f in config["features"]}
            row.update(
                i_drama_id=i % 9,
                ctx_platform="ios" if i % 2 else "android",
                u_skipped__drama_id=[i % 9, (i + 1) % 9],
                u_drama_interests__id=[i % 9, (i + 2) % 9],
                u_drama_interests__weight=[0.8, 0.2],
                i_genre_v2_ids=[i % 3],
                u_onboarding_genre_ids=[i % 3, (i + 1) % 3],
                completion=i % 2,
                non_fast_swipe=int(i % 3 != 0),
                immersive_click=int(i % 4 == 0),
                watch_duration_ms=float((i * 7) % 150) * 1000,
                video_finish_seen=1,
                impression_time=start + timedelta(seconds=i),
                request_time_ms=int(start.timestamp() * 1000) + (i - 1) * 1000,
            )
            if i % 11 == 0:
                row["watch_duration_ms"] = None
            rows.append(row)
        path = dataset / f"day-{day}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        parts.append(
            {
                "event_time_from": start.isoformat().replace("+00:00", "Z"),
                "event_time_to": (start + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
                "rows": len(rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "uri": path.as_uri(),
            }
        )
    manifest = {
        "manifest_version": "doku-train-dataset-manifest/v1",
        "dataset_id": "fixture/rerank",
        "dataset_version": "fixture-v1",
        "feature_schema_version": "fyp-rank-features/v4",
        "label_definition_version": "fixed-5m-playback-stream-v4",
        "as_of_ms": int((start + timedelta(days=1)).timestamp() * 1000),
        "partitions": parts,
        "row_count": sum(p["rows"] for p in parts),
    }
    manifest["content_sha256"] = digest(manifest)
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    config["dataset"].update(
        manifest_uri=(dataset / "manifest.json").as_uri(),
        version="fixture-v1",
        manifest_sha256=manifest["content_sha256"],
        allowed_prefixes=[dataset.as_uri() + "/"],
    )
    config["execution"]["cache_dir"] = str(root / "cache")
    config["output"] = {"root": str(root / "runs"), "run_id": model + "-smoke"}
    config["batch_size"] = 64
    config["execution"]["intra_op_threads"] = 4
    if model == "xgboost":
        config["model_params"].update(
            num_boost_round=3, early_stopping_rounds=2, max_depth=2, min_child_weight=1
        )
    else:
        config.update(max_epochs=2, early_stopping_patience=2)
    (root / f"{model}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return config
