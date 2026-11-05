import hashlib
import json
import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from recommend_tf.train.comm.datasvr.daily_contracts import digest
from recommend_tf.train.comm.datasvr.source_manifest import resolve_source
from recommend_tf.train.comm.training_input import train
from recommend_tf.train.models.rerank.model_params import parameters
from recommend_tf.train.tests.test_mixed_integration import mixed_config


def fixture(tmp_path):
    cfg = mixed_config(tmp_path)
    for source in cfg["data_sources"]:
        path = Path(unquote(urlsplit(source["manifest_uri"]).path))
        original = json.loads(path.read_text())
        for part in original["partitions"]:
            day = part["event_time_from"][:10]
            root = path.parent / f"date={day}/run_id=abc"
            root.mkdir(parents=True)
            shutil.copytree(path.parent / "contracts", root / "contracts")
            manifest = dict(original, partitions=[part], row_count=part["rows"])
            manifest.pop("content_sha256")
            manifest["content_sha256"] = digest(manifest)
            raw = json.dumps(manifest).encode()
            target = root / "manifest.json"
            target.write_bytes(raw)
            (root.parent / "_SUCCESS").write_text(
                json.dumps(
                    {
                        "manifest_uri": target.as_uri(),
                        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
            )
        source["manifest_uri"] = path.parent.as_uri() + "/date={date}/_SUCCESS"
    return cfg


def test_one_template_per_source_trains_and_replay_ignores_changed_pointers(tmp_path):
    cfg = fixture(tmp_path)
    first = train(cfg, tmp_path / "first", model_parameters=parameters)
    for pointer in tmp_path.glob("*/dataset/date=*/_SUCCESS"):
        pointer.write_text("broken")
    again = train(
        cfg, tmp_path / "again", model_parameters=parameters, frozen_input=tmp_path / "first/input"
    )
    assert first[0]["train_loss"] == again[0]["train_loss"]
    assert first[0]["validation_metrics"] == again[0]["validation_metrics"]
    record = json.loads((tmp_path / "again/input/sources.json").read_text())
    assert all(len(r["resolved_manifest_uris"]) == 8 for r in record["sources"])


def test_bad_pointer_digest_rejected_and_missing_eval_day_skipped(tmp_path):
    cfg = fixture(tmp_path)
    source = cfg["data_sources"][0]
    pointer = tmp_path / "fyp/dataset/date=2026-10-08/_SUCCESS"
    pointer.unlink()
    uris, evidence = resolve_source(source, cfg["dataset_config"], ("train", "validation", "test"))
    assert len(uris) == 7
    assert evidence[-1]["status"] == "skipped"
    pointer = tmp_path / "fyp/dataset/date=2026-10-01/_SUCCESS"
    body = json.loads(pointer.read_text())
    body["manifest_sha256"] = "0" * 64
    pointer.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="checksum"):
        resolve_source(source, cfg["dataset_config"], ("train", "validation", "test"))
