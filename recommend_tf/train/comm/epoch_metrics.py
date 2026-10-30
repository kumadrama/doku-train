"""Durable epoch records and TensorBoard scalars, including partial runs."""

import json
import math
from pathlib import Path

import tensorflow.compat.v1 as tf


def save_epoch(output, record):
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
