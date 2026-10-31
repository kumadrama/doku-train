"""Two-stage daily training; no serving or deployment side effects."""

import argparse
import json
import time
from pathlib import Path

from .comm.artifact_integrity import seal_run
from .comm.datasvr.daily_dataset import resolve_config
from .comm.datasvr.published_dataset import PublishedDataset
from .comm.epoch_metrics import write_selected_metrics
from .comm.model_identity import model_name
from .comm.training_input import train
from .models.rerank.model_params import parameters


def execute(config, root, *, cutoff, frozen_input, publish=None, on_completed=None):
    model_name(config, required=True)
    config = resolve_config(config, cutoff)
    dataset = PublishedDataset(config["dataset_config"], frozen_input=Path(frozen_input))
    for split in ("train", "validation"):
        dataset.partitions(split)
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)

    owned = False

    def created():
        nonlocal owned
        owned = True

    def sync(record=None):
        (root / "experiment-config.json").write_text(json.dumps(config, sort_keys=True))
        if publish:
            publish(root)

    try:
        train(
            config,
            root,
            on_epoch=sync,
            on_created=created,
            model_parameters=parameters,
            splits=("train", "validation"),
            frozen_input=Path(frozen_input),
        )
        selection = json.loads((root / "training/selection.json").read_text())
        status = {
            "status": "completed" if "checkpoint" in selection else "trained_without_selection",
            "test": "not_requested",
            "training_completed_at_ms": time.time_ns() // 1_000_000,
            "model_name": config["model_name"],
            "cutoff": cutoff,
            "selection": selection,
        }
        try:
            write_selected_metrics(root, config)
            status["monitoring"] = "ready"
        except Exception as error:
            # Original selection/epoch evidence remains sealed for independent recovery.
            status["monitoring"] = "failed"
            status["monitoring_error_type"] = type(error).__name__
        (root / "daily-status.json").write_text(json.dumps(status, sort_keys=True))
        (root / "experiment-config.json").write_text(json.dumps(config, sort_keys=True))
        if on_completed is not None:
            try:
                on_completed(root)
            except Exception as error:
                status["monitoring"] = "failed"
                status["monitoring_error_type"] = type(error).__name__
                (root / "daily-status.json").write_text(json.dumps(status, sort_keys=True))
        seal_run(root, "daily-status.json")
        sync()
        return status
    except Exception as error:
        if owned:
            (root / "COMMITTED.json").unlink(missing_ok=True)
            (root / "daily-status.json").write_text(
                json.dumps({"status": "failed", "error_type": type(error).__name__})
            )
        raise


def main():
    parser = argparse.ArgumentParser()
    for option in ("config", "output", "cutoff", "frozen-input"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    execute(
        json.loads(Path(args.config).read_text()),
        args.output,
        cutoff=args.cutoff,
        frozen_input=args.frozen_input,
    )


if __name__ == "__main__":
    main()
