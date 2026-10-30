"""Reproducible synthetic SavedModel benchmark, not an Endpoint capacity acceptance."""

import argparse
import json
import platform
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

from recommend_tf.train.comm.datasvr.daily_dataset import freeze
from recommend_tf.train.daily_train_model import execute as daily
from recommend_tf.train.export_model import execute as export
from recommend_tf.train.export_model import synthetic_batches
from recommend_tf.train.tests.test_daily_publication import fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--candidates", nargs="+", type=int, default=[1, 32, 128])
    args = parser.parse_args()
    if args.iterations <= 0 or any(n <= 0 for n in args.candidates):
        parser.error("positive counts required")
    with tempfile.TemporaryDirectory(prefix="tf-export-benchmark-") as directory:
        root = Path(directory)
        config, ready = fixture(root)
        resolved = freeze(config, ready, root / "frozen")
        daily(resolved, root / "run", cutoff=ready["cutoff"], frozen_input=root / "frozen/input")
        outputs = {
            "completion": "probability",
            "fast_swipe": "probability",
            "immersive_click": "probability",
            "watch_seconds": "seconds",
        }
        export(root / "run", root / "package", outputs=outputs)
        vocabs = json.loads((root / "run/training/feature-state.json").read_text())
        contract = json.loads((root / "run/input/contracts/features.json").read_text())
        values = synthetic_batches(resolved, vocabs, contract)[-1]
        results = []
        options = tf.ConfigProto(
            device_count={"GPU": 0}, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
        )
        with tf.Graph().as_default(), tf.Session(config=options) as session:
            signature = tf.saved_model.loader.load(
                session, ["serve"], str(root / "package/saved_model")
            ).signature_def["serving_default"]
            graph = tf.get_default_graph()
            fetch = {k: graph.get_tensor_by_name(v.name) for k, v in signature.outputs.items()}
            for size in args.candidates:
                feed = {}
                for key, value in signature.inputs.items():
                    group, field = key.split("__", 1)
                    feed[graph.get_tensor_by_name(value.name)] = np.repeat(
                        values[group][field][:1], size, axis=0
                    )
                for _ in range(10):
                    session.run(fetch, feed)
                elapsed = []
                for _ in range(args.iterations):
                    started = time.perf_counter()
                    session.run(fetch, feed)
                    elapsed.append(time.perf_counter() - started)
                results.append(
                    {
                        "batch_candidates": size,
                        "calls": args.iterations,
                        "candidates_per_second": size * args.iterations / sum(elapsed),
                        "p50_ms": float(np.percentile(elapsed, 50)) * 1000,
                        "p95_ms": float(np.percentile(elapsed, 95)) * 1000,
                    }
                )
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {
            "kind": "synthetic_savedmodel_only",
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "tensorflow": tf.__version__,
            "threads": 1,
            "fixture_training_rows": 24,
            "peak_process_rss_bytes_including_training_export": peak
            * (1 if platform.system() == "Darwin" else 1024),
            "results": results,
        }
        Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
