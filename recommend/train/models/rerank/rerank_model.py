"""Shared CPU network; FM interactions are entirely configured."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def layers(width: int, cfg: dict[str, Any]) -> tuple[nn.Sequential, int]:
    modules: list[nn.Module] = []
    for size, act in zip(cfg["dims"], cfg["acts"], strict=True):
        modules.append(nn.Linear(width, size))
        width = size
        if act:
            modules.append({"relu": nn.ReLU, "silu": nn.SiLU, "prelu": nn.PReLU}[act]())
        keep_probability = cfg.get("layer_dropout", 1.0)
        if keep_probability < 1.0:
            modules.append(nn.Dropout(p=1.0 - keep_probability))
    return nn.Sequential(*modules), width


class ConfiguredRerank(nn.Module):
    def __init__(self, config: dict[str, Any], state: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.state = state
        self.table_names = sorted(state["vocabs"])
        self.embeddings = nn.ModuleList()
        widths = {}
        for name in self.table_names:
            dim = config["embed_dim"].get(name, config["embed_dim"].get("*"))
            if dim is None:
                raise ValueError(f"missing embedding dimension: {name}")
            emb = nn.Embedding(len(state["vocabs"][name]) + 2, dim, padding_idx=0)
            args = config["embed_initializer_args"]
            nn.init.uniform_(emb.weight, args["minval"], args["maxval"])
            with torch.no_grad():
                emb.weight[0].zero_()
            self.embeddings.append(emb)
            widths[name] = dim
        self.group_dims = {}
        for name, g in config["input_features"].items():
            dims = [
                widths[state["tables"][f]] if f in state["tables"] else 1 for f in g["features"]
            ]
            if g.get("combiner") in {"mean", "weighted_mean"} and (
                len(dims) != 1 or g["features"][0] not in state["tables"]
            ):
                raise ValueError("pooling requires one embedding sequence")
            self.group_dims[name] = sum(dims)
        self.pairs = [pair for layer in config["fm_layers"] for pair in layer["features"]]
        for a, b in self.pairs:
            if self.group_dims[a] != self.group_dims[b]:
                raise ValueError("FM dimensions differ")
        self.shared, width = layers(
            sum(self.group_dims[n] for n in config["dnn_inputs"]), config["output_layer"]
        )
        self.task_names = sorted(config["tasks"])
        self.towers = nn.ModuleList()
        for name in self.task_names:
            t = config["tasks"][name]
            tower, size = layers(width + len(self.pairs), t)
            self.towers.append(nn.Sequential(tower, nn.Linear(size, t["num_outputs"])))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        groups = {}
        for name, g in self.config["input_features"].items():
            parts = []
            for f in g["features"]:
                if f not in self.state["tables"]:
                    continue
                table = self.state["tables"][f]
                v = batch[f].long()
                emb = self.embeddings[self.table_names.index(table)](v)
                valid = v != 0
                if g.get("combiner") == "weighted_mean":
                    weight = batch[g["weight_feature"]] * valid
                    denom = weight.sum(1, keepdim=True)
                    value = (emb * weight.unsqueeze(-1)).sum(1) / denom.clamp_min(
                        torch.finfo(emb.dtype).tiny
                    )
                else:
                    value = (emb * valid.unsqueeze(-1)).sum(1) / valid.sum(
                        1, keepdim=True
                    ).clamp_min(1)
                parts.append(value)
            if "numeric:" + name in batch:
                parts.append(batch["numeric:" + name].float())
            groups[name] = torch.cat(parts, dim=1)
        shared = self.shared(torch.cat([groups[n] for n in self.config["dnn_inputs"]], dim=1))
        if self.pairs:
            shared = torch.cat(
                [shared, *[(groups[a] * groups[b]).sum(1, keepdim=True) for a, b in self.pairs]],
                dim=1,
            )
        return {n: layer(shared) for n, layer in zip(self.task_names, self.towers, strict=True)}


def predictions(logits: torch.Tensor, task: dict[str, Any]) -> torch.Tensor:
    if task["predict_fn"] == "sigmoid":
        return torch.sigmoid(logits[:, 0])
    if task["predict_fn"] == "identity":
        return logits[:, 0]
    bins = (
        torch.arange(task["num_outputs"], device=logits.device, dtype=logits.dtype)
        * task["soft_label"]["scale"]
    )
    result: torch.Tensor = (torch.softmax(logits, dim=1) * bins).sum(1)
    return result
