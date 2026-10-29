"""Bounded aggregation; exact AUC spills predictions to local SQLite."""

import math
import sqlite3

import numpy as np


class Metrics:
    def __init__(self, path, tasks):
        self.tasks = tasks
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE predictions(task TEXT, score REAL, positive INTEGER)")
        self.sums = {name: {} for name in tasks}
        self.counts = {name: 0 for name in tasks}

    def add(self, name, prediction, values):
        p = np.asarray(prediction, dtype=np.float64).reshape(-1)
        if not np.isfinite(p).all():
            raise ValueError("Non-finite evaluation prediction")
        self.counts[name] += len(p)
        auc_done = False
        for metric, option in self.tasks[name].get("metrics", {}).items():
            y = np.broadcast_to(np.asarray(values[metric], dtype=np.float64), p.shape)
            if not np.isfinite(y).all():
                raise ValueError("Non-finite metric label")
            fn = option["func"]
            if fn == "auc":
                if not np.isin(y, [0, 1]).all():
                    raise ValueError("AUC requires binary labels")
                if not auc_done:
                    self.db.executemany(
                        "INSERT INTO predictions VALUES (?,?,?)",
                        ((name, float(score), int(label)) for score, label in zip(p, y)),
                    )
                    auc_done = True
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
            count = self.counts[name]
            out = {}
            for metric, option in head.get("metrics", {}).items():
                fn = option["func"]
                reason = None
                if not count:
                    value = None
                    reason = "empty"
                elif fn == "auc":
                    negatives = positives = 0
                    wins = 0.0
                    for score, pos, total in self.db.execute(
                        "SELECT score,SUM(positive),COUNT(*) FROM predictions WHERE task=? GROUP BY score ORDER BY score",
                        (name,),
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
