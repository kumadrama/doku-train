import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pyarrow.parquet as pq

from recommend_tf.train.models.rerank.data_utils import migrate_experiment
from recommend_tf.train.tests.experiment_fixture import make_experiment


def test_local_benchmark_end_to_end(tmp_path):
    source = make_experiment(tmp_path / "fixture", "deepfm", rows_per_day=4)
    source["max_epochs"] = 1
    manifest = json.loads((tmp_path / "fixture/dataset/manifest.json").read_text())
    schema = pq.read_schema(unquote(urlsplit(manifest["partitions"][0]["uri"]).path))
    dtypes = {
        field.name: "str"
        if str(field.type) == "string"
        else "int64"
        if "int64" in str(field.type)
        else "float32"
        for field in schema
        if field.name != "impression_time"
    }
    config = migrate_experiment(
        source,
        dtypes,
        {"optimizer": "adamw", "learning_rate": 0.004, "optimizer_args": {"weight_decay": 0.0001}},
        16,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "benchmark"
    command = [
        sys.executable,
        "-m",
        "recommend_tf.train.benchmark",
        "--configs",
        str(path),
        "--output",
        str(output),
        "--local-root",
        str(tmp_path / "fixture/dataset"),
        "--repeats",
        "1",
        "--profile",
    ]
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[3])
    report = json.loads((output / "benchmark.json").read_text())
    assert report["summaries"]["1"]["completed"] == 1
    run = report["runs"][0]
    assert run["process_wall_seconds"] > run["execution_seconds"] > 0
    assert run["process_cpu_seconds"] > 0
    summaries = list(output.glob("repeat-*/timing/*summary.json"))
    assert len(summaries) == 1
    stages = json.loads(summaries[0].read_text())["stages"]
    assert "experiment/test" in stages
    assert "experiment/prepare_and_train/prepare" in stages
    # Re-running is create-only and must not overwrite existing evidence.
    assert subprocess.run(command, cwd=Path(__file__).resolve().parents[3]).returncode != 0


def test_failed_process_is_not_a_completed_benchmark(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    from recommend_tf.train import benchmark

    config = tmp_path / "config.json"
    config.write_text("{}")
    output = tmp_path / "failed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--configs",
            str(config),
            "--output",
            str(output),
            "--local-root",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(benchmark.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=42))
    with pytest.raises(RuntimeError, match="Benchmark failed"):
        benchmark.main()
    report = json.loads((output / "benchmark.json").read_text())
    assert report["summaries"]["1"]["completed"] == 0
    assert report["summaries"]["1"]["median_seconds"] is None
    assert report["runs"][0]["returncode"] == 42
