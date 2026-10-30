"""Validated, stateless learning rates for one-based epochs."""

import math


def validate_schedules(model, max_epochs=None):
    for prefix in ("", "sparse_"):
        key = prefix + "learning_rate_schedule"
        if key not in model:
            continue
        if prefix and not model.get("sparse_optimizer"):
            raise ValueError("sparse learning-rate schedule requires sparse_optimizer")
        schedule = model[key]
        if not isinstance(schedule, dict) or set(schedule) != {"type", "points"}:
            raise ValueError(key + " requires exactly type and points")
        if schedule["type"] not in ("piecewise_constant", "piecewise_linear"):
            raise ValueError("Unsupported learning-rate schedule type")
        points = schedule["points"]
        if not isinstance(points, list) or not points:
            raise ValueError("Learning-rate points must be a nonempty list")
        previous = 0
        for point in points:
            if not isinstance(point, dict) or set(point) != {"epoch", "learning_rate"}:
                raise ValueError("Each learning-rate point requires epoch and learning_rate")
            epoch, rate = point["epoch"], point["learning_rate"]
            if (
                type(epoch) is not int
                or epoch <= previous
                or (max_epochs is not None and epoch > max_epochs)
            ):
                raise ValueError("Learning-rate epochs must strictly increase within max_epochs")
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(rate)
                or rate <= 0
            ):
                raise ValueError("Scheduled learning rates must be finite and positive")
            previous = epoch
        if points[0]["epoch"] != 1 or points[0]["learning_rate"] != model.get(
            prefix + "learning_rate", 0.001
        ):
            raise ValueError("First learning-rate point must be epoch 1 and match initial rate")


def learning_rate_for_epoch(model, epoch, prefix=""):
    if type(epoch) is not int or epoch < 1:
        raise ValueError("epoch must be a one-based integer")
    schedule = model.get(prefix + "learning_rate_schedule")
    if schedule is None:
        return model.get(prefix + "learning_rate", 0.001)
    left = schedule["points"][0]
    for right in schedule["points"][1:]:
        if epoch < right["epoch"]:
            if schedule["type"] == "piecewise_constant":
                return left["learning_rate"]
            fraction = (epoch - left["epoch"]) / (right["epoch"] - left["epoch"])
            return left["learning_rate"] + fraction * (
                right["learning_rate"] - left["learning_rate"]
            )
        left = right
    return left["learning_rate"]
