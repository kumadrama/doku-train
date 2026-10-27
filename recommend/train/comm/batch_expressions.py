"""Compile validated expressions to reusable column operations, preserving scalar semantics."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np

from recommend.train.comm import timing
from recommend.train.comm.expressions import (
    _BINARY,
    _COMPARE,
    _FUNCTIONS,
    compile_expression,
    dependencies,
    evaluate,
)

Columns = dict[str, np.ndarray]
Step = Callable[[Columns, np.ndarray, Any], np.ndarray]


def objects(values: Any) -> np.ndarray:
    """Keep Python scalars and list cells (never expand ragged lists into dimensions)."""
    values = list(values)
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _function(fn: Any, children: list[Step], nullable: bool = True) -> Step:
    def scalar(*args: Any) -> Any:
        if nullable and any(x is None for x in args):
            return None
        return fn(*args)

    # Object kernels intentionally retain Python integer overflow and libm rules.
    # This removes per-row AST dispatch, not all Python scalar callbacks.
    operation = np.frompyfunc(scalar, len(children), 1)

    def run(columns: Columns, indices: np.ndarray, thresholds: Any) -> np.ndarray:
        with np.errstate(all="ignore"):
            return operation(*(child(columns, indices, thresholds) for child in children))  # type: ignore[no-any-return]

    return run


def _compile(node: ast.expr) -> Step:
    if isinstance(node, ast.Constant):
        return lambda cols, ids, thresholds: objects([node.value] * len(ids))
    if isinstance(node, ast.Name):
        return lambda cols, ids, thresholds: cols[node.id][ids]
    if isinstance(node, ast.BinOp):
        return _function(_BINARY[type(node.op)], [_compile(node.left), _compile(node.right)])
    if isinstance(node, ast.UnaryOp):
        fn = (
            (lambda x: not x)
            if isinstance(node.op, ast.Not)
            else ((lambda x: -x) if isinstance(node.op, ast.USub) else (lambda x: +x))
        )
        return _function(fn, [_compile(node.operand)])
    if isinstance(node, ast.Compare):
        ops = [_COMPARE[type(op)] for op in node.ops]

        def compare(*args: Any) -> Any:
            results = [
                None if a is None or b is None else op(a, b)
                for op, a, b in zip(ops, args, args[1:], strict=False)
            ]
            return False if False in results else None if None in results else True

        return _function(compare, [_compile(node.left), *map(_compile, node.comparators)], False)
    if isinstance(node, ast.BoolOp):

        def boolean(*values: Any) -> Any:
            if isinstance(node.op, ast.And):
                return (
                    False
                    if any(x is not None and not bool(x) for x in values)
                    else (None if None in values else True)
                )
            return (
                True
                if any(x is not None and bool(x) for x in values)
                else (None if None in values else False)
            )

        return _function(boolean, list(map(_compile, node.values)), False)
    if isinstance(node, ast.Call):
        assert isinstance(node.func, ast.Name)
        name = node.func.id
        if name == "where" and len(node.args) == 3:
            condition, yes, no = map(_compile, node.args)

            def select(cols: Columns, ids: np.ndarray, thresholds: Any) -> np.ndarray:
                mask = condition(cols, ids, thresholds).astype(bool)
                result = np.empty(len(ids), dtype=object)
                if mask.any():
                    result[mask] = yes(cols, ids[mask], thresholds)
                if (~mask).any():
                    result[~mask] = no(cols, ids[~mask], thresholds)
                return result

            return select
        if name not in {"where", "tree"}:
            return _function(
                _FUNCTIONS[name],
                list(map(_compile, node.args)),
                name not in {"is_null", "is_finite", "fill_null"},
            )
    # Unsupported batch operations retain the old interpreter on active rows only.
    source = ast.unparse(node)
    names = dependencies(source)

    def fallback(cols: Columns, ids: np.ndarray, thresholds: Any) -> np.ndarray:
        with timing.current().span("expression_fallback", count=len(ids), unit="rows"):
            return objects(
                evaluate(source, {k: cols[k][i] for k in names}, tree_thresholds=thresholds)
                for i in ids
            )

    return fallback


def _integer_log_bucket(source: str) -> Step | None:
    """Fuse only the exact bounded integer bucket idiom, not raw floating log output."""
    node = compile_expression(source)
    names = dependencies(source)
    if len(names) != 1:
        return None
    field = next(iter(names))
    template = f'cast(clip(round(log2(maximum({field} + 1, 1))), 0, 31), "int64")'
    if ast.dump(node) != ast.dump(compile_expression(template)):
        return None

    def run(cols: Columns, ids: np.ndarray, thresholds: Any) -> np.ndarray:
        values = cols[field][ids]
        output = np.empty(len(ids), dtype=object)
        output[:] = None
        positions = np.fromiter((x is not None for x in values), dtype=bool, count=len(ids))
        active = values[positions]
        # Object arithmetic preserves Python big integers and the order of +1 before
        # float conversion; raw floating output elsewhere still uses math.log2.
        try:
            arguments = np.fromiter(
                (max(x + 1, 1) for x in active), dtype=np.float64, count=len(active)
            )
        except (OverflowError, ValueError):
            return objects(evaluate(source, {field: x}) for x in values)
        logs = np.log2(arguments)
        if not np.isfinite(logs).all():
            return objects(evaluate(source, {field: x}) for x in values)
        buckets = np.clip(np.rint(logs), 0, 31).astype(np.int64)
        near = np.abs(logs - (np.floor(logs) + 0.5)) <= 8 * np.spacing(np.maximum(abs(logs), 1))
        # Keep conservative scalar correction at every rounding boundary.
        for i in np.flatnonzero(near):
            buckets[i] = evaluate(source, {field: active[i]})
        output[positions] = objects(buckets.tolist())
        return output

    return run


class BatchExpression:
    def __init__(self, source: str):
        self.step = _integer_log_bucket(source) or _compile(compile_expression(source))

    def __call__(self, columns: Columns, indices: np.ndarray, thresholds: Any = None) -> np.ndarray:
        if not len(indices):
            return np.empty(0, dtype=object)
        result = self.step(columns, indices, thresholds)
        if any(isinstance(x, float) and not math.isfinite(x) for x in result):
            raise ValueError("expression produced non-finite value")
        return result


@lru_cache(maxsize=2048)
def compile_batch_expression(source: str) -> BatchExpression:
    return BatchExpression(source)
