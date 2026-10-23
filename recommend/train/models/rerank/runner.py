from __future__ import annotations

import copy
import io
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

from recommend.train.comm import metric_events
from recommend.train.comm.checkpoint_agent import CheckpointAgent
from recommend.train.comm.nn_utils import get_loss
from recommend.train.comm.runtime_budget import check_memory
from recommend.train.comm.task_metrics import metric_values
from recommend.train.models.rerank.rerank_model import ConfiguredRerank, predictions


def batches(
    arrays: dict[str, np.ndarray], size: int, order: np.ndarray | None = None
) -> Iterator[dict[str, torch.Tensor]]:
    count = len(next(iter(arrays.values())))
    for start in range(0, count, size):
        check_memory()
        index = slice(start, start + size) if order is None else order[start : start + size]
        yield {k: torch.from_numpy(np.array(v[index], copy=True)) for k, v in arrays.items()}


class NeuralRunner:
    checkpoint_name = "checkpoint.pt"

    def __init__(self, config: dict[str, Any], state: dict[str, Any]) -> None:
        torch.manual_seed(config["seed"])
        torch.set_num_threads(config["execution"]["intra_op_threads"])
        wanted = config["execution"]["inter_op_threads"]
        if torch.get_num_interop_threads() != wanted:
            torch.set_num_interop_threads(wanted)
        torch.use_deterministic_algorithms(True)
        self.config = config
        self.state = state
        self.model = ConfiguredRerank(config, state)
        opt = config["optimizer"]
        cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}[
            opt["name"]
        ]
        self.optimizer = cls(
            self.model.parameters(), lr=opt["learning_rate"], weight_decay=opt["weight_decay"]
        )

    def evaluate(self, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
        self.model.eval()
        pred: dict[str, list[np.ndarray]] = {n: [] for n in self.config["tasks"]}
        logits: dict[str, list[np.ndarray]] = {n: [] for n in pred}
        totals = {n: [0.0, 0] for n in pred}
        with torch.no_grad():
            for batch in batches(arrays, self.config["batch_size"]):
                outputs = self.model(batch)
                for name, t in self.config["tasks"].items():
                    mask = batch["mask:" + name][:, 0].bool()
                    n = int(mask.sum())
                    if n:
                        totals[name][0] += float(
                            get_loss(
                                t,
                                outputs[name][mask],
                                batch["label:" + name][:, 0][mask],
                                reduction="none",
                            ).sum()
                        )
                        totals[name][1] += n
                    logits[name].append(outputs[name].numpy())
                    pred[name].append(predictions(outputs[name], t).numpy())
        return {
            name: {
                "metrics": metric_values(
                    t, name, arrays, np.concatenate(pred[name]), logits=np.concatenate(logits[name])
                ),
                "loss": totals[name][0] / totals[name][1] if totals[name][1] else None,
            }
            for name, t in self.config["tasks"].items()
        }

    def fit(
        self,
        train: dict[str, np.ndarray],
        validation: dict[str, np.ndarray],
        checkpoint: CheckpointAgent,
        identity: str,
        resume: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        best = None
        best_state = None
        best_epoch = None
        stale = 0
        start = 0
        trace = []
        rng = np.random.default_rng(self.config["seed"])
        if resume:
            saved = resume
            if saved["identity"] != identity:
                raise ValueError("resume identity mismatch")
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            torch.set_rng_state(saved["torch_rng"])
            random.setstate(saved["python_rng"])
            rng.bit_generator.state = saved["numpy_rng"]
            start = saved["epoch"]
            best = saved["best"]
            best_state = saved["best_state"]
            best_epoch = saved["best_epoch"]
            stale = saved["stale"]
            trace = saved["trace"]
        step = int(resume.get("global_step", 0)) if resume else 0
        best_step = int(resume.get("best_step", 0)) if resume else 0
        count = len(next(iter(train.values())))
        for epoch in range(start, self.config["max_epochs"]):
            self.model.train()
            sums = {n: [0.0, 0] for n in self.config["tasks"]}
            for batch in batches(train, self.config["batch_size"], rng.permutation(count)):
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch)
                losses = []
                batch_losses: dict[str, float | None] = {n: None for n in self.config["tasks"]}
                for name, t in self.config["tasks"].items():
                    mask = batch["mask:" + name][:, 0].bool()
                    n = int(mask.sum())
                    if n:
                        per = get_loss(
                            t,
                            outputs[name][mask],
                            batch["label:" + name][:, 0][mask],
                            reduction="none",
                        )
                        losses.append(
                            get_loss(
                                t,
                                outputs[name][mask],
                                batch["label:" + name][:, 0][mask],
                                loss_weight_data=batch["weight:" + name][:, 0][mask],
                            )
                        )
                        batch_losses[name] = float(per.detach().mean())
                        sums[name][0] += float(per.detach().sum())
                        sums[name][1] += n
                if losses:
                    loss = torch.stack(losses).sum()
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite training loss")
                    loss.backward()  # type: ignore[no-untyped-call]
                    self.optimizer.step()
                    step += 1
                    metric_events.training_step(step, epoch + 1, batch_losses)
            metrics = self.evaluate(validation)
            metric_events.evaluation("validation", step, epoch + 1, metrics)
            terms = self.config["selection"]["task_losses"]
            score = (
                sum(metrics[n]["loss"] * w for n, w in terms.items())
                if all(metrics[n]["loss"] is not None for n in terms)
                else None
            )
            if score is not None and (best is None or score < best):
                best = score
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1
                best_step = step
                stale = 0
            elif score is not None:
                stale += 1
            trace.append(
                {
                    "epoch": epoch + 1,
                    "global_step": step,
                    "selection_score": score,
                    "validation": metrics,
                    "train_loss": {n: x[0] / x[1] if x[1] else None for n, x in sums.items()},
                }
            )
            state = {
                "global_step": step,
                "best_step": best_step,
                "identity": identity,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "numpy_rng": rng.bit_generator.state,
                "epoch": epoch + 1,
                "best": best,
                "best_state": best_state,
                "best_epoch": best_epoch,
                "stale": stale,
                "trace": trace,
                "feature_state": self.state,
            }
            checkpoint.save_resume(state)
            print(f"epoch={epoch + 1} validation_selection={score}", flush=True)
            if stale >= self.config["early_stopping_patience"]:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        buffer = io.BytesIO()
        torch.save(
            {
                "format": "rerank/v2",
                "identity": identity,
                "config": self.config,
                "feature_state": self.state,
                "model": self.model.state_dict(),
                "selected_epoch": best_epoch,
                "global_step": best_step if best_state is not None else step,
            },
            buffer,
        )
        checkpoint.save_bytes(self.checkpoint_name, buffer.getvalue())
        return {
            "selection": {
                "status": "selected" if best_epoch else "unavailable",
                "epoch": best_epoch,
                "global_step": best_step if best_state is not None else step,
            },
            "validation": self.evaluate(validation),
            "history": trace,
        }

    def load(self, payload: bytes) -> None:
        saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        self.model.load_state_dict(saved["model"])
