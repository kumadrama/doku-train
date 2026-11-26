"""Pure date/window resolution, usable by admission without the dataset reader."""

import copy
from datetime import date, timedelta


def window(config, cutoff):
    options = config["online_config"]
    if set(options) - {"training_date", "validation_date", "first_day"} or any(
        type(options.get(k)) is not int or options[k] <= 0
        for k in ("training_date", "validation_date")
    ):
        raise ValueError("online_config requires positive integer day counts")
    end = date.fromisoformat(cutoff)
    validation = [end - timedelta(days=i) for i in reversed(range(options["validation_date"]))]
    training = [
        validation[0] - timedelta(days=i) for i in reversed(range(1, options["training_date"] + 1))
    ]
    if "first_day" in options:
        first = date.fromisoformat(options["first_day"])
        training = [d for d in training if d >= first]
        if not training:
            raise ValueError("window requires a training day before validation")
    return {
        "train": [d.isoformat() for d in training],
        "validation": [d.isoformat() for d in validation],
    }


def resolve_config(config, cutoff):
    result = copy.deepcopy(config)
    split = result["dataset_config"]["split"]
    for key in list(split):
        if key.startswith(("train_", "validation_", "test_")):
            del split[key]
    for name, days in window(config, cutoff).items():
        split.update({f"{name}_dates": days, f"{name}_from": days[0], f"{name}_to": days[-1]})
    return result
