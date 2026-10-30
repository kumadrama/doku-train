"""Manifest-first composition root; normalized Finder model config is explicit."""

import json
import time
from pathlib import Path

from .datasvr.prepared import digest, prepare
from .datasvr.published_dataset import PublishedDataset
from .early_stop import validate_early_stop
from .learning_rate import validate_schedules
from .runtime_budget import MemoryBudget
from .training_pipeline import run, source_identity


def train(
    config,
    root,
    local_root=None,
    on_epoch=None,
    *,
    model_parameters,
    splits=("train", "validation", "test"),
    frozen_input=None,
    on_created=None,
):
    validate_early_stop(config)
    validate_schedules(config["model"], config["max_epochs"])
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    if on_created:
        on_created()
    budget = MemoryBudget(config["dataset_config"]["execution"]["memory_limit_bytes"])
    started = time.monotonic()
    try:
        budget.start()
        dataset = PublishedDataset(
            config["dataset_config"], local_root=local_root, frozen_input=frozen_input
        )
        dataset.freeze(root / "input")
        params = model_parameters(config["model"])
        identity = digest(
            {
                "manifest": dataset.manifest,
                "source": source_identity(),
                "model": config["model"],
                "defaults": config["dataset_config"]["feature_default_values"],
                "histories": config["histories"],
                "item_filter": config["item_filter"],
            }
        )
        # Explicit raw dtype mapping is checked against source columns by reader;
        # no hidden production source or current/latest pointer.
        columns = set(config["dtypes"])
        for split in splits:
            print(json.dumps({"event": "prepare_split", "split": split}), flush=True)

            def blocks():
                for cols in dataset.iter_batches(split, columns):
                    # Reader already applied event-time splitting; omit audit-only columns.
                    names = sorted(columns)
                    yield {name: cols[name] for name in names}

            prepare(
                blocks(),
                root / "prepared" / split,
                config["dataset_config"]["feature_default_values"],
                params,
                config["histories"],
                config["item_filter"],
                identity,
                config["dataset_config"]["execution"]["cache_limit_bytes"],
            )
        preparation_seconds = time.monotonic() - started
        runtime = {
            k: v
            for k, v in config.items()
            if k not in ("dataset_config", "histories", "item_filter", "online_config")
        }
        runtime["prepared_identity"] = identity
        (root / "resolved-config.json").write_text(json.dumps(runtime, sort_keys=True))
        metrics = run(runtime, root / "prepared", root / "training", on_epoch=on_epoch)
        (root / "run-resources.json").write_text(
            json.dumps(
                {
                    "prepare_seconds": preparation_seconds,
                    "total_seconds": time.monotonic() - started,
                    "sampled_peak_rss_bytes": budget.peak,
                }
            )
        )
        return metrics
    finally:
        budget.close()
