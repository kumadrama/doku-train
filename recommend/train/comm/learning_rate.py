"""Stateless, one-based epoch learning-rate schedules."""

from typing import Any


def validate_schedule(optimizer: dict[str, Any], max_epochs: int) -> None:
    schedule = optimizer.get("learning_rate_schedule")
    if schedule is None:
        return
    points = schedule["points"]
    epochs = [p["epoch"] for p in points]
    if epochs[0] != 1 or epochs != sorted(set(epochs)) or epochs[-1] > max_epochs:
        raise ValueError(
            "learning-rate epochs must start at 1, strictly increase and fit max_epochs"
        )
    if points[0]["learning_rate"] != optimizer["learning_rate"]:
        raise ValueError("first scheduled learning rate must match optimizer.learning_rate")


def learning_rate_for_epoch(optimizer: dict[str, Any], epoch: int) -> float:
    if epoch < 1:
        raise ValueError("epoch must be one-based")
    schedule = optimizer.get("learning_rate_schedule")
    if schedule is None:
        return float(optimizer["learning_rate"])
    points = schedule["points"]
    left = points[0]
    for right in points[1:]:
        if epoch < right["epoch"]:
            if schedule["type"] == "piecewise_constant":
                return float(left["learning_rate"])
            fraction = (epoch - left["epoch"]) / (right["epoch"] - left["epoch"])
            return float(
                left["learning_rate"] + fraction * (right["learning_rate"] - left["learning_rate"])
            )
        left = right
    return float(left["learning_rate"])
