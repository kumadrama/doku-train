"""Offline three-split training composition root."""

from .comm.training_input import train as _train
from .models.rerank.model_params import parameters


def train(config, root, local_root=None, on_epoch=None, *, on_created=None):
    return _train(
        config,
        root,
        local_root=local_root,
        on_epoch=on_epoch,
        model_parameters=parameters,
        on_created=on_created,
    )
