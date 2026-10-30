"""Optional validation-only stopping and checkpoint selection, independent of TF."""

import math
from copy import deepcopy


def validate_early_stop(config):
    if "early_stop" not in config:
        return None
    options = config["early_stop"]
    required = {"patient", "metric", "weight"}
    if (
        not isinstance(options, dict)
        or not required <= options.keys()
        or options.keys() - (required | {"min_delta"})
    ):
        raise ValueError("early_stop requires patient, metric, weight; optional min_delta")
    if type(options["patient"]) is not int or options["patient"] < 1:
        raise ValueError("early_stop.patient must be a positive integer")
    metric = options["metric"]
    if metric not in ("auc", "logloss"):
        raise ValueError("early_stop.metric must be auc or logloss")
    delta = options.get("min_delta", 0.0)
    if not _finite(delta) or delta < 0:
        raise ValueError("early_stop.min_delta must be finite and nonnegative")
    weights = options["weight"]
    if not isinstance(weights, dict) or not weights:
        raise ValueError("early_stop.weight must be a nonempty task-to-weight mapping")
    tasks = config["model"]["tasks"]
    for task, weight in weights.items():
        if task not in tasks:
            raise ValueError("Unknown early_stop task: " + str(task))
        if not _finite(weight) or weight < 0:
            raise ValueError("early_stop weights must be finite and nonnegative")
        if weight > 0 and tasks[task].get("metrics", {}).get(metric, {}).get("func") != metric:
            raise ValueError(f"early_stop task {task} must declare metric {metric}")
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("early_stop weights must have a finite positive sum")
    return {
        "patient": options["patient"],
        "metric": metric,
        "mode": "max" if metric == "auc" else "min",
        "weight": {task: weight / total for task, weight in weights.items() if weight > 0},
        "min_delta": delta,
    }


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


class EarlyStop:
    def __init__(self, options, state=None):
        self.options = options
        self.state = (
            deepcopy(state)
            if state is not None
            else {
                "best": None,
                "reference_score": None,
                "wait": 0,
                "stopped": False,
            }
        )

    @property
    def best(self):
        return self.state["best"]

    @property
    def stopped(self):
        return self.state["stopped"]

    def _gain(self, score, reference):
        return score - reference if self.options["mode"] == "max" else reference - score

    def observe(self, metrics, epoch, checkpoint):
        if self.stopped:
            raise ValueError("Cannot advance an already stopped training run")
        metric = self.options["metric"]
        missing = {}
        terms = []
        for task, weight in self.options["weight"].items():
            result = metrics.get(task, {}).get(metric, {})
            value = result.get("value")
            if not _finite(value):
                missing[task] = result.get("reason") or "missing_or_nonfinite_metric"
            else:
                terms.append(weight * value)
        score = None if missing else sum(terms)
        if score is not None and not math.isfinite(score):
            missing["aggregate"] = "nonfinite_score"
            score = None
        if score is not None:
            # Save every genuine best checkpoint, even if below the patience reset threshold.
            if self.best is None or self._gain(score, self.best["score"]) > 0:
                self.state["best"] = {
                    "epoch": epoch,
                    "score": score,
                    "checkpoint": str(checkpoint),
                    "metric": metric,
                    "mode": self.options["mode"],
                    "weight": self.options["weight"],
                }
            reference = self.state["reference_score"]
            if reference is None or self._gain(score, reference) > self.options["min_delta"]:
                self.state["reference_score"] = score
                self.state["wait"] = 0
            else:
                self.state["wait"] += 1
            self.state["stopped"] = self.state["wait"] >= self.options["patient"]
        return {
            "metric": metric,
            "mode": self.options["mode"],
            "score": score,
            "unavailable": missing,
            "wait": self.state["wait"],
            "stopped": self.stopped,
            "best_epoch": self.best["epoch"] if self.best else None,
        }
