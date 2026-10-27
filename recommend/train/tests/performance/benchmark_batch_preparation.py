"""Opt-in local synthetic benchmark, no cloud or model training.

python -m recommend.train.tests.performance.benchmark_batch_preparation --output runs/batch-bench
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from recommend.train.comm.batch_feature_plan import BatchFeaturePlan
from recommend.train.comm.datasvr.published_dataset import PublishedDataset
from recommend.train.comm.feature_plan import FeaturePlan
from recommend.train.tests.experiment_fixture import make_experiment


def worker(root: Path, mode: str, trial: str) -> dict:
    config = json.loads((root / "config.json").read_text())
    setup_started = time.perf_counter()
    data = PublishedDataset(config)
    plan = (FeaturePlan if mode == "old" else BatchFeaturePlan)(config, data.dtypes)
    elapsed = {"setup": time.perf_counter() - setup_started}
    start = time.perf_counter()
    plan.fit_dataset(data)
    elapsed["fit"] = time.perf_counter() - start
    fitted = plan.state()
    state_hash = hashlib.sha256(json.dumps(fitted, sort_keys=True).encode()).hexdigest()
    signatures = {}
    # Full preparation including source validation, decoding and memmap writes.
    for split in ["train", "validation", "test"]:
        start = time.perf_counter()
        arrays = plan.prepare_dataset(data, split, root / trial / split)
        elapsed[split] = time.perf_counter() - start
        signatures[split] = {
            k: {
                "shape": v.shape,
                "dtype": str(v.dtype),
                "sha256": hashlib.sha256(v.tobytes()).hexdigest(),
            }
            for k, v in arrays.items()
        }
    return {
        "mode": mode,
        "seconds": elapsed,
        "total_seconds": sum(elapsed.values()),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "fitted_state_sha256": state_hash,
        "arrays": signatures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["old", "batch"])
    parser.add_argument("--trial", default="worker")
    parser.add_argument("--rows-per-day", type=int, default=1024)
    parser.add_argument("--extra-transforms", type=int, default=64)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        print(json.dumps(worker(args.output, args.worker, args.trial)))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    config = make_experiment(args.output / "fixture", "deepfm", args.rows_per_day)
    config["batch_size"] = 2048
    # Independent synthetic workload: many derived categorical columns over one
    # fixture count column; explicitly not production E1's data distribution.
    buckets = []
    for i in range(args.extra_transforms):
        name = f"synthetic_bucket_{i}"
        buckets.append(name)
        config["feature_transformations"].append(
            {
                "expr": name + " = cast(clip(round(log2(maximum(u_expose_count_7d_all"
                ' + 1, 1))), 0, 31), "int64")'
            }
        )
        config["embed_name"][name] = name
    if buckets:
        config["input_features"]["synthetic_buckets"] = {"features": buckets, "combiner": "concat"}
    (args.output / "config.json").write_text(json.dumps(config))
    records = []
    reference = None
    for repetition in range(4):  # Warm-up once per implementation, then three measured runs.
        for mode in ["old", "batch"] if repetition % 2 == 0 else ["batch", "old"]:
            command = [
                sys.executable,
                "-m",
                __name__.replace(
                    "__main__", "recommend.train.tests.performance.benchmark_batch_preparation"
                ),
                "--output",
                str(args.output),
                "--worker",
                mode,
                "--trial",
                f"{repetition}-{mode}",
            ]
            result = json.loads(
                subprocess.run(command, check=True, text=True, capture_output=True).stdout
            )
            signature = (result["fitted_state_sha256"], result["arrays"])
            if reference is None:
                reference = signature
            assert signature == reference, "old/batch preparation mismatch"
            result["warmup"] = repetition == 0
            records.append(result)
            print(mode, repetition, round(result["total_seconds"], 3), flush=True)
    summary = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rows": args.rows_per_day * 8,
        "batch_size": config["batch_size"],
        "extra_transforms": args.extra_transforms,
        "arrays_and_state_equal": True,
        "medians": {},
        "records": records,
    }
    for mode in ["old", "batch"]:
        measured = [r for r in records if r["mode"] == mode and not r["warmup"]]
        summary["medians"][mode] = {
            "total_seconds": statistics.median(r["total_seconds"] for r in measured),
            "peak_rss_bytes": statistics.median(r["peak_rss_bytes"] for r in measured),
            "stages": {
                k: statistics.median(r["seconds"][k] for r in measured)
                for k in ["setup", "fit", "train", "validation", "test"]
            },
        }
    summary["speedup"] = (
        summary["medians"]["old"]["total_seconds"] / summary["medians"]["batch"]["total_seconds"]
    )
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
