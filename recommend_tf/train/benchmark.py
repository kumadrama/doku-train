"""Local, isolated-process timing runs; never submits or publishes cloud jobs."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


def worker(config_path, output, local_root, profile):
    # Import costs are included in the outer process measurement, not stage totals.
    import resource

    import psutil

    from .comm import timing
    from .experiment import execute

    if any((output / name).exists() for name in ("run", "measurement.json", "timing")):
        raise FileExistsError(output)
    config = json.loads(config_path.read_text())
    process = psutil.Process()
    before = process.cpu_times()
    started = time.perf_counter()
    status = "failed"
    samples = []
    stop = threading.Event()

    def sample():
        process.cpu_percent()
        while not stop.wait(1):
            try:
                io = process.io_counters() if hasattr(process, "io_counters") else None
                samples.append(
                    {
                        "elapsed_seconds": time.perf_counter() - started,
                        "process_cpu_percent": process.cpu_percent(),
                        "rss_bytes": process.memory_info().rss,
                        "host_cpu_percent": psutil.cpu_percent(),
                        "read_bytes": io.read_bytes if io else None,
                        "write_bytes": io.write_bytes if io else None,
                    }
                )
            except (psutil.Error, OSError):
                samples.append({"error": "resource_sample_unavailable"})

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        with timing.session({"profiling": {"enabled": profile}}, output, "benchmark"):
            result = execute(config, output / "run", local_root=local_root)
            status = result["status"]
    finally:
        execution_seconds = time.perf_counter() - started
        stop.set()
        monitor.join()
        after = process.cpu_times()
        report = {
            "version": "doku-tf-benchmark/v1",
            "status": status,
            "profile": profile,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "execution_seconds": execution_seconds,
            "resource_samples": samples,
            "resource_semantics": "1s samples; process CPU may exceed 100%; host CPU is shared",
            "process_cpu_seconds": after.user + after.system - before.user - before.system,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "cache_state": "uncontrolled; no cache eviction performed",
            "scope": "execute including preparation, training, test and artifact seal; no publish",
        }
        (output / "measurement.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.worker:
        if len(args.configs) != 1:
            parser.error("worker requires exactly one config")
        worker(args.configs[0], args.output, args.local_root, args.profile)
        return
    configs = [p.resolve(strict=True) for p in args.configs]
    local_root = args.local_root.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        for repeat in range(args.repeats):
            # Alternate order to reduce systematic warm-up bias; each run is a fresh process.
            order = list(enumerate(configs))
            if repeat % 2:
                order.reverse()
            for index, config in order:
                destination = args.output / f"repeat-{repeat + 1:02d}-config-{index + 1:02d}"
                destination.mkdir()
                frozen = destination / "config.json"
                frozen.write_bytes(config.read_bytes())
                command = [
                    sys.executable,
                    "-m",
                    "recommend_tf.train.benchmark",
                    "--worker",
                    "--configs",
                    str(frozen.resolve()),
                    "--output",
                    str(destination.resolve()),
                    "--local-root",
                    str(local_root),
                ]
                if args.profile:
                    command.append("--profile")
                started = time.perf_counter()
                with (destination / "process.log").open("w") as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=False
                    )
                elapsed = time.perf_counter() - started
                path = destination / "measurement.json"
                record = json.loads(path.read_text()) if path.exists() else {"status": "failed"}
                record.update(
                    config_index=index + 1,
                    repeat=repeat + 1,
                    process_wall_seconds=elapsed,
                    returncode=result.returncode,
                )
                results.append(record)
                if result.returncode:
                    raise RuntimeError(f"Benchmark failed; inspect {destination / 'process.log'}")
    finally:
        summaries = {}
        for index in range(1, len(configs) + 1):
            values = [
                r["execution_seconds"]
                for r in results
                if r["config_index"] == index
                and r["status"] == "completed"
                and r["returncode"] == 0
            ]
            summaries[index] = {
                "completed": len(values),
                "median_seconds": statistics.median(values) if values else None,
                "range_seconds": [min(values), max(values)] if values else None,
            }
        (args.output / "benchmark.json").write_text(
            json.dumps(
                {
                    "runs": results,
                    "summaries": summaries,
                    "warning": "Cache uncontrolled; not a cold-cache acceptance or semantic parity verdict.",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
