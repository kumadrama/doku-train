"""Offline validation diagnostics; no training, publication, or test-set access."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from recommend.train.comm.datasvr.published_dataset import PublishedDataset
from recommend.train.comm.expressions import dependencies, evaluate
from recommend.train.comm.feature_plan import FeaturePlan, prepare_arrays
from recommend.train.models.rerank_tree.runner import TreeRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frequency-boundaries", type=int, nargs=2, default=[100, 1000])
    args = parser.parse_args()
    config = json.loads((args.run / "resolved-config.json").read_text())
    config["execution"]["cache_dir"] = str(args.cache)
    dataset = PublishedDataset(config, frozen_input=args.run / "input")
    plan = FeaturePlan(config, dataset.dtypes)
    state = json.loads((args.run / "fitted-feature-state/state.json").read_text())
    plan.restore(state)
    frequency = Counter()
    filter_columns = dependencies(config["item_filter"])
    for row in dataset.rows("train", filter_columns | {"i_drama_id"}):
        if evaluate(config["item_filter"], row) and row["i_drama_id"] is not None:
            frequency[row["i_drama_id"]] += 1
    buckets = []
    episode_hits = []
    extra_interest = []
    low, high = args.frequency_boundaries
    extra_columns = {
        "i_drama_id",
        "i_episode_id",
        "u_watched__episode_id",
        "u_finished__episode_id",
        "u_skipped__episode_id",
        "u_drama_interests__id",
        "u_drama_interests__weight",
    }
    for source in dataset.rows("validation", filter_columns | extra_columns):
        if not evaluate(config["item_filter"], source):
            continue
        row = source
        ident = row["i_drama_id"]
        episode = source["i_episode_id"]
        episode_hits.append(
            episode is not None
            and any(
                episode in (source[f] or [])
                for f in [
                    "u_watched__episode_id",
                    "u_finished__episode_id",
                    "u_skipped__episode_id",
                ]
            )
        )
        merged = Counter()
        for item, weight in zip(
            source["u_drama_interests__id"] or [],
            source["u_drama_interests__weight"] or [],
            strict=True,
        ):
            merged[item] += weight
        ordered = sorted(merged, key=lambda item: (-merged[item], item))
        extra_interest.append(ident in ordered[100:200])
        n = frequency[ident]
        buckets.append(
            "missing"
            if ident is None
            else "unseen"
            if n == 0
            else "low"
            if n < low
            else "medium"
            if n < high
            else "high"
        )
    groups = np.asarray(buckets)
    runner = TreeRunner(config, state)
    runner.load((args.run / "checkpoint.xgb.zip").read_bytes())
    for booster in runner.boosters.values():
        booster.set_param({"nthread": config["execution"]["intra_op_threads"], "device": "cpu"})
    with tempfile.TemporaryDirectory() as temp:
        arrays = prepare_arrays(
            plan,
            dataset.rows("validation", plan.columns),
            Path(temp) / "arrays",
            len(groups),
            config["execution"]["cache_limit_bytes"],
        )
        result = {}
        slices = {
            group: groups == group for group in ["missing", "unseen", "low", "medium", "high"]
        }
        slices["all"] = np.ones(len(groups), dtype=bool)
        slices.update(
            episode_hit=np.asarray(episode_hits),
            episode_no_hit=~np.asarray(episode_hits),
            interest_added_top200=np.asarray(extra_interest),
        )
        for group, mask in slices.items():
            result[group] = {
                "rows": int(mask.sum()),
                "metrics": runner.evaluate({k: v[mask] for k, v in arrays.items()})
                if mask.any()
                else None,
            }
    for group in result.values():
        for task_result in (group["metrics"] or {}).values():
            rate = task_result["metrics"].get("positive_rate")
            if rate and rate["value"] is not None:
                n = rate["valid_rows"]
                positives = round(rate["value"] * n)
                task_result["binary_counts"] = {"positive": positives, "negative": n - positives}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"split": "validation", "frequency_boundaries": [low, high], "groups": result}, indent=2
        )
    )


if __name__ == "__main__":
    main()
