from __future__ import annotations

from recommend.train.comm.eval.gates import GateSeverity, build_metric_gates
from recommend.train.comm.metrics_utils import MetricSet, MetricStatus, TargetMetrics
from recommend.train.models.rerank.model_params import TARGETS


def metric_set(*, status: MetricStatus = MetricStatus.OK, auc: float | None = 0.7) -> MetricSet:
    return MetricSet(
        validation_slice_digest="slice-a",
        targets=tuple(
            TargetMetrics(
                target=target,
                metric_version="binary-v1",
                status=status,
                valid_count=10,
                positive_count=5,
                negative_count=5,
                loss=0.5,
                auc=auc,
            )
            for target in TARGETS
        ),
    )


def test_not_evaluable_metric_is_a_blocking_pre_refit_gate() -> None:
    gates = build_metric_gates(
        metric_set(status=MetricStatus.NOT_EVALUABLE, auc=None),
        auc_warning_floor=0.5,
    )
    assert all(gate.severity is GateSeverity.BLOCK for gate in gates)
    assert all(not gate.passed for gate in gates)


def test_auc_floor_is_warning_only() -> None:
    gates = build_metric_gates(metric_set(auc=0.4), auc_warning_floor=0.5)
    assert all(gate.severity is GateSeverity.WARN for gate in gates)
    assert all(not gate.passed for gate in gates)


def test_previous_run_requires_the_same_current_validation_slice() -> None:
    previous = MetricSet(
        validation_slice_digest="different-slice",
        targets=metric_set(auc=0.9).targets,
    )
    gates = build_metric_gates(metric_set(auc=0.7), auc_warning_floor=0.5, previous=previous)
    comparison = [gate for gate in gates if gate.name == "previous_run_comparison"]
    assert len(comparison) == 1
    assert comparison[0].severity is GateSeverity.WARN
    assert "different validation slice" in comparison[0].message


def test_gate_module_contains_no_export_gate() -> None:
    import inspect

    from recommend.train.comm.eval import gates

    source = inspect.getsource(gates).casefold()
    assert "onnx" not in source
    assert "export" not in source
