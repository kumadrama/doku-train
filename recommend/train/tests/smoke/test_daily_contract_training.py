import json
from pathlib import Path

from recommend.train.eval_rerank import execute_test
from recommend.train.run import daily_run
from recommend.train.tests.unit.test_daily_contract_evolution import evolve


def test_mixed_contract_daily_training_and_frozen_test(tmp_path):
    config, daily = evolve(tmp_path)
    result = daily_run(
        config,
        daily_root=daily.root,
        frozen_root=(tmp_path / "frozen").as_uri(),
        cutoff="2026-10-08",
        artifact_root=(tmp_path / "artifacts").as_uri(),
        export_root=tmp_path / "exports",
    )
    assert result["status"] == "published"
    run = Path(result["run_dir"])
    lineage = json.loads((run / "lineage.json").read_text())
    assert lineage["feature_schema_version"].startswith("merged-")
    assert lineage["daily_sources"][0]["defaulted_features"] == [
        "i_episode_dur_s",
        "u_skipped__drama_id",
    ]
    assert "labels-v6" in lineage["label_definition_version"]
    checkpoint = (run / "checkpoint.xgb.zip").read_bytes()
    for root in [Path(daily.root.removeprefix("file://")), tmp_path / "frozen"]:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix != ".parquet":
                path.unlink()
    execute_test(run)
    assert checkpoint == (run / "checkpoint.xgb.zip").read_bytes()
