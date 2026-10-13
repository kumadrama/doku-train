import hashlib
import json
from datetime import date, timedelta

import pytest

from recommend.train.comm.datasvr.daily_input import resolve_days
from recommend.train.comm.datasvr.published_dataset import digest


def fixture():
    objects = {}
    prefix = "s3://lake/train/daily/environment=dte/"
    spec = {"feature_spec_version": "v5", "features": [], "labels": []}
    labels = {"label_definition_versions": ["labels-v1"]}
    projection = digest(spec)
    for n in range(2):
        day = date(2026, 10, 1) + timedelta(days=n)
        run = prefix + f"date={day}/run_id=abc/"
        m = {
            "manifest_version": "doku-train-dataset-manifest/v1",
            "dataset_id": "train",
            "dataset_version": "abc",
            "feature_schema_version": "v5",
            "label_definition_version": "labels-v1",
            "label_config_version": "l1",
            "projection_sha256": projection,
            "row_count": 2,
            "as_of_ms": 1790985600000,
            "partitions": [
                {
                    "uri": run + "part.parquet",
                    "sha256": "a" * 64,
                    "rows": 2,
                    "event_time_from": f"{day}T00:00:00Z",
                    "event_time_to": f"{day + timedelta(days=1)}T00:00:00Z",
                }
            ],
        }
        m["content_sha256"] = digest(m)
        body = json.dumps(m).encode()
        objects[run + "manifest.json"] = body
        objects[run + "contracts/features.json"] = json.dumps(spec).encode()
        objects[run + "contracts/labels.json"] = json.dumps(labels).encode()
        objects[prefix + f"date={day}/_SUCCESS"] = json.dumps(
            {
                "pointer_version": "doku-train-day/v1",
                "date": str(day),
                "manifest_uri": run + "manifest.json",
                "manifest_sha256": hashlib.sha256(body).hexdigest(),
                "projection_sha256": projection,
            }
        ).encode()
    options = {
        "daily_prefix": prefix,
        "projection_sha256": projection,
        "feature_schema_version": "v5",
        "label_definition_version": "labels-v1",
    }
    return objects, options


def test_freezes_exact_runs_and_rejects_projection_mix():
    objects, options = fixture()
    days = [date(2026, 10, 1), date(2026, 10, 2)]
    m, _contracts = resolve_days(options, days, objects.__getitem__)
    assert m["row_count"] == 4 and len(m["source_manifests"]) == 2
    objects[options["daily_prefix"] + "date=2026-10-01/_SUCCESS"] = b"changed"
    assert m["partitions"][0]["uri"].endswith("/run_id=abc/part.parquet")
    objects, options = fixture()
    options["projection_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="projection"):
        resolve_days(options, days, objects.__getitem__)


def test_pointer_changed_during_resolution_fails():
    objects, options = fixture()
    calls = {}

    def read(uri):
        calls[uri] = calls.get(uri, 0) + 1
        if uri.endswith("_SUCCESS") and calls[uri] > 1:
            return b"new"
        return objects[uri]

    with pytest.raises(ValueError, match="changed"):
        resolve_days(options, [date(2026, 10, 1)], read)


def test_training_reader_pins_daily_input_and_ignores_later_backfill(tmp_path):
    import shutil
    from pathlib import Path

    from recommend.train.comm.datasvr.published_dataset import PublishedDataset
    from recommend.train.comm.model_params_validator import validate_config
    from recommend.train.tests.experiment_fixture import make_experiment

    config = make_experiment(tmp_path)
    original = Path(config["dataset"]["manifest_uri"].removeprefix("file://"))
    manifest = json.loads(original.read_bytes())
    features = (original.parent / "contracts/features.json").read_bytes()
    projection = digest(json.loads(features))
    root = tmp_path / "daily"
    for part in manifest["partitions"]:
        day = part["event_time_from"][:10]
        run = root / f"date={day}" / "run_id=abc"
        (run / "contracts").mkdir(parents=True)
        for name in ["features", "labels"]:
            shutil.copyfile(
                original.parent / f"contracts/{name}.json", run / f"contracts/{name}.json"
            )
        shutil.copyfile(Path(part["uri"].removeprefix("file://")), run / "part.parquet")
        m = dict(
            manifest,
            partitions=[dict(part, uri="file://" + str(run / "part.parquet"))],
            row_count=part["rows"],
            projection_sha256=projection,
            label_config_version="labels-v1",
        )
        m.pop("content_sha256")
        m["content_sha256"] = digest(m)
        raw = json.dumps(m).encode()
        (run / "manifest.json").write_bytes(raw)
        (run.parent / "_SUCCESS").write_text(
            json.dumps(
                {
                    "pointer_version": "doku-train-day/v1",
                    "date": day,
                    "manifest_uri": "file://" + str(run / "manifest.json"),
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    "projection_sha256": projection,
                }
            )
        )
    for name in ["manifest_uri", "manifest_sha256", "version"]:
        config["dataset"].pop(name)
    config["dataset"].update(daily_prefix=root.as_uri(), projection_sha256=projection)
    config["dataset"]["allowed_prefixes"].append(root.as_uri() + "/")
    validate_config(config)
    data = PublishedDataset(config)
    frozen = validate_config(data.config)
    assert "daily_prefix" not in frozen["dataset"]
    assert Path(frozen["dataset"]["manifest_uri"].removeprefix("file://")).is_file()
    for pointer in root.glob("*/_SUCCESS"):
        pointer.write_bytes(b"new invalid current run")
    restored = PublishedDataset(frozen)
    assert restored.manifest == data.manifest
    assert list(restored.rows("train", {"completion"})) == list(data.rows("train", {"completion"}))
