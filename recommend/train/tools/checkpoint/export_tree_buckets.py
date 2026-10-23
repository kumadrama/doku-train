"""Export numeric buckets from a completed tree training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommend.train.models.rerank_tree.thresholds import export_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = args.run
    if json.loads((run / "status.json").read_text()).get("status") != "complete":
        raise ValueError("tree bucket export requires successful training")
    artifact = export_thresholds(
        (run / "checkpoint.xgb.zip").read_bytes(),
        json.loads((run / "resolved-config.json").read_text()),
        json.loads((run / "lineage.json").read_text()),
    )
    with args.output.open("x") as stream:
        json.dump(artifact, stream, sort_keys=True, allow_nan=False, indent=2)


if __name__ == "__main__":
    main()
