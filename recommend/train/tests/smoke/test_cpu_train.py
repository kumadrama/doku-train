from __future__ import annotations

import json
from pathlib import Path

import boto3

from recommend.train.tests.fixtures import build_training_fixture
from recommend.train.train_rerank import execute_train


def test_full_cpu_training_writes_and_reloads_exact_four_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = build_training_fixture(tmp_path)
    monkeypatch.setattr(
        boto3,
        "client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local smoke must not construct an AWS client")
        ),
    )
    result = execute_train(
        manifest_uri=fixture.manifest_uri,
        config_path=fixture.config_path,
        output_prefix=fixture.output_uri,
        run_id="smoke-run",
    )
    assert result.checkpoint_reloaded is True
    assert result.selected_epoch == 1
    output_root = tmp_path / "outputs" / "smoke-run"
    assert sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    ) == sorted(
        [
            "checkpoint.pt",
            "metrics.json",
            "fitted-feature-state/state.json",
            "lineage.json",
        ]
    )
    metrics = json.loads((output_root / "metrics.json").read_bytes())
    lineage = json.loads((output_root / "lineage.json").read_bytes())
    assert set(metrics["targets"]) == {
        "effective_watch",
        "completion",
        "non_fast_swipe",
        "immersive_click",
    }
    assert metrics["resource_report"]["row_count"] == 112
    assert metrics["resource_report"]["peak_rss_bytes"] > 0
    assert lineage["run_id"] == "smoke-run"
