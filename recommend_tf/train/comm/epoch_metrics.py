"""Durable epoch records and TensorBoard scalars, including partial runs."""

import json
import math
from pathlib import Path


def save_epoch(output, record):
    import tensorflow.compat.v1 as tf

    output = Path(output)
    epoch = record["epoch"]
    folder = output / "epoch-metrics"
    folder.mkdir(exist_ok=True)
    target = folder / f"{epoch:04d}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, allow_nan=False, indent=2))
    temporary.replace(target)
    scalars = {}
    for split in ["train", "validation"]:
        for task, value in record[split + "_loss"].items():
            scalars[f"{split}/{task}/loss"] = value
    for task, metrics in record["validation_metrics"].items():
        for metric, result in metrics.items():
            scalars[f"validation/{task}/{metric}"] = result["value"]
    for stage, value in record["stage_seconds"].items():
        scalars["timing/" + stage] = value
    for name, value in record.get("train_objective_batch_mean", {}).items():
        scalars["train/objective_batch_mean/" + name] = value
    for key in ("learning_rate", "sparse_learning_rate"):
        if key in record:
            scalars["train/" + key] = record[key]
    if "early_stop" in record:
        scalars["validation/early_stop_score"] = record["early_stop"]["score"]
        scalars["train/early_stop_wait"] = record["early_stop"]["wait"]
    # One closed event file per epoch: upload never races a live event writer.
    writer = tf.summary.FileWriter(
        str(output / "tensorboard"), filename_suffix=f".epoch-{epoch:04d}"
    )
    try:
        summary = tf.Summary(
            value=[
                tf.Summary.Value(tag=tag, simple_value=float(value))
                for tag, value in scalars.items()
                if value is not None and math.isfinite(value)
            ]
        )
        writer.add_summary(summary, global_step=epoch)
        writer.flush()
    finally:
        writer.close()


def read_selected_metrics(root, config):
    """Read the existing selector's decision; never rank epochs again."""
    root = Path(root)
    selection = json.loads((root / "training/selection.json").read_text())
    if "checkpoint" not in selection:
        return {
            "schema_version": "doku-selected-epoch/v1",
            "status": "unavailable",
            "selection": selection,
        }
    epoch = selection["epoch"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError("invalid selected epoch")
    expected = f"epoch-{epoch:04d}"
    if (
        Path(selection["checkpoint"]).name != expected
        or ".." in Path(selection["checkpoint"]).parts
    ):
        raise ValueError("selected checkpoint path differs")
    checkpoint = json.loads((root / "training" / expected / "checkpoint.json").read_text())
    record = json.loads((root / "training/epoch-metrics" / f"{epoch:04d}.json").read_text())
    if checkpoint["epoch"] != epoch or record["epoch"] != epoch:
        raise ValueError("selected epoch evidence differs")
    metrics = {}
    for task, options in config["model"]["tasks"].items():
        metrics[task] = {}
        for name, option in options.get("metrics", {}).items():
            result = record["validation_metrics"][task][name]
            value = result["value"]
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError("invalid selected metric")
            if value is None and not result.get("reason"):
                raise ValueError("unavailable metric requires a reason")
            metrics[task][name] = dict(result, func=option["func"])
    return {
        "schema_version": "doku-selected-epoch/v1",
        "status": "selected",
        "selection": selection,
        "checkpoint_identity": checkpoint["identity"],
        "evaluation_split": "validation",
        "metrics": metrics,
    }


def write_selected_metrics(root, config):
    record = read_selected_metrics(root, config)
    target = Path(root) / "training/selected-model-metrics.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, allow_nan=False))
    temporary.replace(target)
    return record
