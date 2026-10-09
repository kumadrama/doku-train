from __future__ import annotations

import json
from pathlib import Path

import boto3

from recommend.train.tests.fixtures import build_xgboost_training_fixture
from recommend.train.train_rerank_xgboost import execute_xgboost_train


def test_full_xgboost_training_writes_and_reloads_exact_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = build_xgboost_training_fixture(tmp_path)
    monkeypatch.setattr(
        boto3,
        "client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local smoke must not construct an AWS client")
        ),
    )
    result = execute_xgboost_train(
        manifest_uri=fixture.manifest_uri,
        config_path=fixture.config_path,
        output_prefix=fixture.output_uri,
        run_id="xgboost-smoke-run",
    )
    assert result.checkpoint_reloaded is True
    assert set(result.selected_rounds) == {
        "effective_watch",
        "completion",
        "non_fast_swipe",
        "immersive_click",
    }
    output_root = tmp_path / "xgboost-outputs" / "xgboost-smoke-run"
    assert sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    ) == sorted(
        [
            "checkpoint.xgb.zip",
            "metrics.json",
            "fitted-feature-state/state.json",
            "lineage.json",
        ]
    )
    metrics = json.loads((output_root / "metrics.json").read_bytes())
    lineage = json.loads((output_root / "lineage.json").read_bytes())
    assert set(metrics["targets"]) == set(result.selected_rounds)
    assert metrics["resource_report"]["row_count"] == 112
    assert lineage["model_name"] == "rerank_train"
    assert lineage["run_id"] == "xgboost-smoke-run"
