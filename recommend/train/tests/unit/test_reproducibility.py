from __future__ import annotations

import random

import numpy as np
import torch

from recommend.train.comm.checkpoint_agent import seed_everything


def test_seed_everything_repeats_python_numpy_and_torch_values() -> None:
    seed_everything(11)
    first = (random.random(), np.random.rand(3), torch.rand(3))
    seed_everything(11)
    assert random.random() == first[0]
    assert np.array_equal(np.random.rand(3), first[1])
    assert torch.equal(torch.rand(3), first[2])
