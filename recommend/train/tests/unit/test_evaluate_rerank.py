from __future__ import annotations

import torch

from recommend.train.comm.eval.eval_model_rerank import evaluate_rerank
from recommend.train.models.rerank.data_utils import RerankBatch
from recommend.train.models.rerank.model_params import TARGETS


class PerfectAdapter:
    def logits(self, batch: RerankBatch) -> dict[str, torch.Tensor]:
        return {target: batch.labels[target] * 4.0 - 2.0 for target in TARGETS}


def test_evaluate_rerank_aggregates_all_batches_and_targets() -> None:
    batches = [
        RerankBatch(
            numeric=torch.zeros(2, 2),
            categorical=torch.zeros(2, 0, dtype=torch.long),
            labels={target: torch.tensor([0.0, 1.0]) for target in TARGETS},
            masks={target: torch.tensor([True, True]) for target in TARGETS},
        ),
        RerankBatch(
            numeric=torch.zeros(2, 2),
            categorical=torch.zeros(2, 0, dtype=torch.long),
            labels={target: torch.tensor([1.0, 0.0]) for target in TARGETS},
            masks={target: torch.tensor([True, True]) for target in TARGETS},
        ),
    ]
    result = evaluate_rerank(
        PerfectAdapter(),
        batches,
        minimum_valid_rows=4,
        validation_slice_digest="slice-a",
    )
    assert tuple(metric.target for metric in result.targets) == TARGETS
    assert all(metric.valid_count == 4 for metric in result.targets)
    assert all(metric.auc == 1.0 for metric in result.targets)
