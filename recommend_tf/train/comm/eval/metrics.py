"""Bounded aggregation; exact AUC spills predictions to local SQLite."""

import math
import sqlite3

import numpy as np


class Metrics:
    def __init__(self, path, tasks):
        self.tasks = tasks
        self.db = sqlite3.connect(str(path))
        self.db.execute(
            "CREATE TABLE predictions(task TEXT, metric TEXT, score REAL, positive INTEGER)"
        )
        self.sums = {name: {} for name in tasks}
        self.counts = {name: {} for name in tasks}

    def add(self, name, prediction, values):
        if name not in self.tasks:
            return
        predictions = np.asarray(prediction, dtype=np.float64).reshape(-1)
        for metric, option in self.tasks[name].get("metrics", {}).items():
            entry = values[metric]
            label = entry["label"] if isinstance(entry, dict) else entry
            mask = entry.get("filter", True) if isinstance(entry, dict) else True
            mask = np.asarray(mask)
            if mask.dtype != np.bool_:
                raise ValueError("Metric filter must be boolean")
            keep = np.broadcast_to(mask, predictions.shape)
            p = predictions[keep]
            y = np.broadcast_to(np.asarray(label, dtype=np.float64), predictions.shape)[keep]
            if not np.isfinite(p).all() or not np.isfinite(y).all():
                raise ValueError("Non-finite evaluation prediction or metric label")
            self.counts[name][metric] = self.counts[name].get(metric, 0) + len(p)
            fn = option["func"]
            if fn == "auc":
                if not np.isin(y, [0, 1]).all():
                    raise ValueError("AUC requires binary labels")
                self.db.executemany(
                    "INSERT INTO predictions VALUES (?,?,?,?)",
                    ((name, metric, float(score), int(label)) for score, label in zip(p, y)),
                )
                continue
            if fn == "valid_rows":
                value = len(y)
            elif fn in ("mean", "positive_rate"):
                value = float(y.sum())
            elif fn == "mae":
                value = float(np.abs(p - y).sum())
            elif fn in ("mse", "rmse"):
                value = float(np.square(p - y).sum())
            elif fn == "logloss":
                clipped = np.clip(p, 1e-15, 1 - 1e-15)
                value = float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).sum())
            else:
                raise ValueError("Unsupported metric: " + fn)
            self.sums[name][metric] = self.sums[name].get(metric, 0.0) + value
        self.db.commit()

    def finish(self):
        result = {}
        for name, head in self.tasks.items():
            out = {}
            for metric, option in head.get("metrics", {}).items():
                count = self.counts[name].get(metric, 0)
                fn = option["func"]
                reason = None
                if not count:
                    value = None
                    reason = "empty"
                elif fn == "auc":
                    negatives = positives = 0
                    wins = 0.0
                    for score, pos, total in self.db.execute(
                        "SELECT score,SUM(positive),COUNT(*) FROM predictions WHERE task=? AND metric=? GROUP BY score ORDER BY score",
                        (name, metric),
                    ):
                        neg = total - pos
                        wins += pos * (negatives + 0.5 * neg)
                        negatives += neg
                        positives += pos
                    value = wins / (positives * negatives) if positives and negatives else None
                    if value is None:
                        reason = "single_class"
                elif fn == "valid_rows":
                    value = count
                else:
                    value = self.sums[name][metric] / count
                    if fn == "rmse":
                        value = math.sqrt(value)
                out[metric] = {"value": value, "reason": reason, "rows": count}
            result[name] = out
        return result

    def close(self):
        self.db.close()
