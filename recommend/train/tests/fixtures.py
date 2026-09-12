from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from recommend.train.comm.datasvr.dataset_manifest import calculate_manifest_sha256


@dataclass(frozen=True, slots=True)
class TrainingFixture:
    manifest_uri: str
    config_path: Path
    output_uri: str


def build_training_fixture(tmp_path: Path) -> TrainingFixture:
    start = datetime(2026, 1, 1, 12, tzinfo=UTC)
    feature_schema = tmp_path / "contracts" / "features.json"
    label_definition = tmp_path / "contracts" / "labels.json"
    feature_schema.parent.mkdir(parents=True)
    feature_schema.write_bytes(b"{}")
    label_definition.write_bytes(b"{}")
    shards: list[dict[str, object]] = []
    for offset in range(28):
        event_time = start + timedelta(days=offset)
        event_time_ms = int(event_time.timestamp() * 1000)
        table = pa.table(
            {
                "event_time_ms": [event_time_ms] * 4,
                "watch_7d": [0.0, 1.0, 2.0, 3.0],
                "watch_7d__window_start_ms": [event_time_ms - 7 * 86_400_000] * 4,
                "watch_7d__window_end_ms": [event_time_ms] * 4,
                "country": ["US", "JP", "US", "JP"],
                "effective_watch": [0.0, 1.0, 0.0, 1.0],
                "effective_watch_valid": [True] * 4,
                "non_fast_swipe": [1.0, 0.0, 1.0, 0.0],
                "non_fast_swipe_valid": [True] * 4,
                "immersive_click": [0.0, 0.0, 1.0, 1.0],
                "immersive_click_valid": [True] * 4,
                "watch_duration_ms": [90.0, 100.0, 90.0, 100.0],
                "content_duration_ms": [100.0] * 4,
            }
        )
        shard_path = tmp_path / "shards" / f"day={event_time.date()}" / "part.parquet"
        shard_path.parent.mkdir(parents=True)
        pq.write_table(table, shard_path)
        body = shard_path.read_bytes()
        shards.append(
            {
                "uri": shard_path.as_uri(),
                "size_bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "row_count": 4,
                "event_date": event_time.date().isoformat(),
                "min_event_time_ms": event_time_ms,
                "max_event_time_ms": event_time_ms,
            }
        )
    manifest_path = tmp_path / "manifest.json"
    payload: dict[str, object] = {
        "manifest_version": "1",
        "dataset_id": "rerank-fixture",
        "dataset_version": "2026-01-28",
        "feature_schema_version": "features-v1",
        "feature_schema": {
            "uri": feature_schema.as_uri(),
            "size_bytes": 2,
            "sha256": hashlib.sha256(b"{}").hexdigest(),
        },
        "label_definition_version": "labels-v1",
        "label_definition": {
            "uri": label_definition.as_uri(),
            "size_bytes": 2,
            "sha256": hashlib.sha256(b"{}").hexdigest(),
        },
        "as_of_ms": int((start + timedelta(days=27, hours=25)).timestamp() * 1000),
        "label_maturity_hours": 24,
        "shards": shards,
        "row_count": 112,
    }
    payload["content_sha256"] = calculate_manifest_sha256(payload)
    import json

    manifest_path.write_text(json.dumps(payload, sort_keys=True))

    config_path = tmp_path / "config.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_name": "rerank",
                "checkpoint_dir": str(tmp_path / "runs"),
                "seed": 7,
                "batch_size": 16,
                "max_epochs": 1,
                "early_stopping_patience": 1,
                "bad_row_policy": "fail",
                "execution": {
                    "strategy": "single_process_cpu",
                    "intra_op_threads": 1,
                    "inter_op_threads": 1,
                    "reader_workers": 0,
                    "prefetch_batches": 1,
                    "memory_limit_bytes": 2_000_000_000,
                },
                "split": {"train_days": 27, "validation_days": 1, "refit_days": 28},
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                },
                "metrics": {"minimum_valid_rows": 4, "auc_warning_floor": 0.5},
                "model_params": {
                    "numeric_features": ["watch_7d"],
                    "categorical_features": ["country"],
                    "categorical_cardinalities": [4],
                    "embedding_dim": 2,
                    "shared_hidden_dims": [4],
                    "tower_hidden_dim": 2,
                    "activation": "relu",
                    "dropout": 0.0,
                    "loss_weights": {
                        "effective_watch": 1.0,
                        "completion": 1.0,
                        "non_fast_swipe": 1.0,
                        "immersive_click": 1.0,
                    },
                },
            },
            sort_keys=True,
        )
    )
    return TrainingFixture(
        manifest_uri=manifest_path.as_uri(),
        config_path=config_path,
        output_uri=(tmp_path / "outputs").as_uri(),
    )
