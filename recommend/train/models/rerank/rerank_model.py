"""Shared CPU network; FM interactions are entirely configured."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from recommend.train.models.rerank.attention import history_attention
from recommend.train.models.rerank.model_params import group_plans


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
        self.group_plans = group_plans(config, state)
        self.table_names = sorted(state["vocabs"])
        self.embeddings = nn.ModuleList()
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
        self.group_dims = {name: plan.width for name, plan in self.group_plans.items()}
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

    def input_groups(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Combine fields without reducing positions, then aggregate each history group."""
        raw = {}
        masks: dict[str, torch.Tensor] = {}
        for name, options in self.config["input_features"].items():
            plan = self.group_plans[name]
            parts = []
            numeric_index = 0
            for field in options["features"]:
                if field in self.state["tables"]:
                    ids = batch[field].long()
                    table = self.state["tables"][field]
                    embedding = self.embeddings[self.table_names.index(table)](ids)
                    if plan.history:
                        valid = ids != 0
                        if name in masks and not torch.equal(masks[name], valid):
                            raise ValueError("history fields must have aligned positions and masks")
                        masks[name] = valid
                        parts.append(embedding)
                    else:
                        if ids.ndim != 2 or ids.shape[1] != 1:
                            raise ValueError("scalar field must have shape [batch, 1]")
                        parts.append(embedding[:, 0])
                else:
                    parts.append(
                        batch["numeric:" + name][:, numeric_index : numeric_index + 1].float()
                    )
                    numeric_index += 1
            combiner = options.get("combiner", "concat")
            if combiner == "concat":
                raw[name] = torch.cat(parts, dim=-1)
            else:
                fields = torch.stack(parts, dim=0)
                raw[name] = fields.sum(0) if combiner == "sum" else fields.mean(0)
        groups = {}
        for name, plan in self.group_plans.items():
            value = raw[name]
            if not plan.history or plan.pooling == "none":
                groups[name] = value
                continue
            valid = masks[name]
            if plan.pooling == "attention":
                assert plan.query is not None
                groups[name] = history_attention(
                    raw[plan.query], value, valid, self.config.get("attn_scale", 1.0)
                )
            elif plan.pooling == "weighted_mean":
                weights = batch[self.config["input_features"][name]["weight_feature"]] * valid
                groups[name] = (value * weights.unsqueeze(-1)).sum(1) / weights.sum(
                    1, keepdim=True
                ).clamp_min(torch.finfo(value.dtype).tiny)
            else:
                groups[name] = (value * valid.unsqueeze(-1)).sum(1) / valid.sum(
                    1, keepdim=True
                ).clamp_min(1)
        return groups

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        groups = self.input_groups(batch)
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
