"""Audit only train/validation histories; never output sample or identifier values."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from recommend.train.comm.datasvr.published_dataset import PublishedDataset
from recommend.train.comm.expressions import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.baseline_run / "resolved-config.json").read_text())
    dataset = PublishedDataset(config, frozen_input=args.baseline_run / "input")
    fields = [
        "u_skipped__drama_id",
        "u_finished__drama_id",
        "u_immersive_enter__drama_id",
        "u_watched__episode_id",
        "u_finished__episode_id",
        "u_skipped__episode_id",
        "u_drama_interests__id",
    ]
    columns = set(fields) | {"i_drama_id", "i_episode_id", "video_finish_seen"}
    report = {}
    for split in ["train", "validation"]:
        counts = Counter()
        lengths = {f: Counter() for f in fields}
        for part in dataset.partitions(split):
            day = part["event_time_from"][:10]
            for row in dataset._rows_in_shard(part, dataset._shard(part), columns):
                if not evaluate(config["item_filter"], row):
                    continue
                counts[f"{day}:rows"] += 1
                for f in fields:
                    value = row[f]
                    counts[f"{day}:{f}:null"] += value is None
                    seq = value or []
                    lengths[f][len(seq)] += 1
                    counts[f"{day}:{f}:empty"] += value == []
                    candidate = row["i_episode_id" if "episode" in f else "i_drama_id"]
                    counts[f"{day}:{f}:hit"] += candidate is not None and candidate in seq
                    counts[f"{day}:{f}:over100"] += len(seq) > 100
                    counts[f"{day}:{f}:lost_tail100_hit"] += (
                        candidate in seq and candidate not in seq[-100:]
                    )
        report[split] = {
            "counts": dict(counts),
            "length_histograms": {f: dict(v) for f, v in lengths.items()},
        }
        print(f"{split} audit complete", flush=True)
    report["run_e04"] = any(
        n > 0
        for split in ["train", "validation"]
        for k, n in report[split]["counts"].items()
        if k.endswith(":over100") and any(f in k for f in fields[:3])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print("run_e04:", report["run_e04"])


if __name__ == "__main__":
    main()
