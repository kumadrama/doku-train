from __future__ import annotations

import math

import numpy as np
import pytest

from recommend.train.comm.batch_expressions import compile_batch_expression, objects
from recommend.train.comm.expressions import evaluate


@pytest.mark.parametrize(
    "expr",
    [
        "x + 1",
        "x / 1000",
        "-x",
        "not x",
        "x > 10",
        "0 < x <= 30",
        "not is_null(x)",
        "is_finite(x)",
        "fill_null(x, 3)",
        "where(x, 1 / x, 0)",
        "x and y",
        "x or y",
        "cast(x, 'float32')",
        "clip(x, 0, 300)",
        "round(x)",
        "tree(x)",
    ],
)
def test_column_plan_matches_scalar(expr):
    values = [None, 0, -0.0, -2, 0.5, 10, 30, 2**53 + 1, 2**70]
    cols = {"x": objects(values), "y": objects([None, False, True] * 3)}
    threshold = {"x": [0.5, 10.0]}
    old = [
        evaluate(expr, {k: v[i] for k, v in cols.items()}, tree_thresholds=threshold)
        for i in range(len(values))
    ]
    new = compile_batch_expression(expr)(cols, np.arange(len(values)), threshold)
    assert new.tolist() == old
    for a, b in zip(old, new, strict=True):
        if isinstance(a, float):
            assert a.hex() == float(b).hex()


@pytest.mark.parametrize("overflow", [False, True])
def test_log_buckets_at_boundaries_and_large_integers(overflow):
    expr = 'cast(clip(round(log2(maximum(x + 1, 1))), 0, 31), "int64")'
    values = [None, -2, -1, 0, 2**53 + 1, 2**63 - 1, 2**70]
    if overflow:
        values.append(10**400)
    for n in range(32):
        x = 2 ** (n + 0.5) - 1
        for _ in range(64):
            values += [x, np.nextafter(x, -math.inf).item()]
            x = np.nextafter(x, math.inf).item()
    rng = np.random.default_rng(22)
    values.extend(rng.integers(0, 2**50, 10000).tolist())
    got = compile_batch_expression(expr)({"x": objects(values)}, np.arange(len(values)))
    assert got.tolist() == [evaluate(expr, {"x": x}) for x in values]


@pytest.mark.parametrize(
    "expr,values",
    [
        ("1 / x", [0]),
        ("log2(x)", [-1]),
        ("x + x", [1e308]),
        ("False and (1 / x)", [0]),
        ("True or (1 / x)", [0]),
    ],
)
def test_same_failure_type(expr, values):
    with pytest.raises(Exception) as old:
        evaluate(expr, {"x": values[0]})
    with pytest.raises(type(old.value)):
        compile_batch_expression(expr)({"x": objects(values)}, np.arange(len(values)))


def test_inactive_rows_never_evaluate_invalid_values():
    expr = "where(x > 0, log2(x), 0)"
    cols = {"x": objects([-1, 0, 4, None])}
    assert compile_batch_expression(expr)(cols, np.arange(4)).tolist() == [0, 0, 2.0, 0]
    assert compile_batch_expression("1 / x")(cols, np.array([2])).tolist() == [0.25]
