from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from recommend.train.comm.metrics_utils import MetricSet, MetricStatus


class GateSeverity(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    severity: GateSeverity
    message: str


def build_metric_gates(
    metrics: MetricSet,
    *,
    auc_warning_floor: float,
    previous: MetricSet | None = None,
    auc_drop_warning: float = 0.02,
) -> tuple[GateResult, ...]:
    results: list[GateResult] = []
    for metric in metrics.targets:
        if metric.status is MetricStatus.NOT_EVALUABLE:
            results.append(
                GateResult(
                    name=f"{metric.target}.evaluable",
                    passed=False,
                    severity=GateSeverity.BLOCK,
                    message="validation metric is not evaluable",
                )
            )
        elif metric.auc is not None and metric.auc < auc_warning_floor:
            results.append(
                GateResult(
                    name=f"{metric.target}.auc_floor",
                    passed=False,
                    severity=GateSeverity.WARN,
                    message=f"AUC {metric.auc:.6f} is below warning floor",
                )
            )
        else:
            results.append(
                GateResult(
                    name=f"{metric.target}.quality",
                    passed=True,
                    severity=GateSeverity.PASS,
                    message="metric is evaluable",
                )
            )
    if previous is not None:
        if previous.validation_slice_digest != metrics.validation_slice_digest:
            results.append(
                GateResult(
                    name="previous_run_comparison",
                    passed=False,
                    severity=GateSeverity.WARN,
                    message="previous run used a different validation slice",
                )
            )
        else:
            previous_by_target = previous.by_target()
            for metric in metrics.targets:
                old = previous_by_target.get(metric.target)
                if (
                    old is not None
                    and old.auc is not None
                    and metric.auc is not None
                    and old.auc - metric.auc > auc_drop_warning
                ):
                    results.append(
                        GateResult(
                            name=f"{metric.target}.auc_delta",
                            passed=False,
                            severity=GateSeverity.WARN,
                            message="AUC decrease exceeds warning threshold",
                        )
                    )
    return tuple(results)
