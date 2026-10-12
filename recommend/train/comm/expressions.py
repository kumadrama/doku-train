"""Small, flat expression language; never executes Python code."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from functools import lru_cache
from typing import Any

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_COMPARE = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _cast(value: Any, dtype: str) -> Any:
    if dtype not in {"float32", "float64", "int32", "int64", "bool"}:
        raise ValueError("unsupported cast dtype")
    return (
        bool(value) if dtype == "bool" else int(value) if dtype.startswith("int") else float(value)
    )


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "minimum": min,
    "maximum": max,
    "clip": lambda x, lo, hi: min(max(x, lo), hi),
    "log1p": math.log1p,
    "log2": math.log2,
    "sqrt": math.sqrt,
    "round": round,
    "cast": _cast,
    "is_null": lambda x: x is None,
    "is_finite": lambda x: x is not None and math.isfinite(x),
    "fill_null": lambda x, d: d if x is None else x,
}


@lru_cache(maxsize=2048)
def compile_expression(source: str) -> ast.expr:
    root = ast.parse(source, mode="eval").body
    for node in ast.walk(root):
        if isinstance(
            node,
            (
                ast.Expression,
                ast.Load,
                ast.Constant,
                ast.Name,
                ast.BinOp,
                ast.UnaryOp,
                ast.BoolOp,
                ast.Compare,
                ast.Call,
                ast.And,
                ast.Or,
                ast.Not,
                ast.USub,
                ast.UAdd,
                *_BINARY,
                *_COMPARE,
            ),
        ):
            continue
        raise ValueError(f"unsupported expression node: {type(node).__name__}")
    for node in ast.walk(root):
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in {*_FUNCTIONS, "where"}
            or node.keywords
        ):
            raise ValueError("unsupported expression function")
    return root


def dependencies(source: str) -> set[str]:
    root = compile_expression(source)
    return {n.id for n in ast.walk(root) if isinstance(n, ast.Name)} - {*_FUNCTIONS, "where"}


def evaluate(source: str, row: dict[str, Any]) -> Any:
    def visit(n: ast.expr) -> Any:
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            return row[n.id]
        if isinstance(n, ast.Call):
            assert isinstance(n.func, ast.Name)
            name = n.func.id
            if name == "where":
                if len(n.args) != 3:
                    raise ValueError("where requires three arguments")
                return visit(n.args[1] if visit(n.args[0]) else n.args[2])
            args = [visit(x) for x in n.args]
            if name not in {"is_null", "is_finite", "fill_null"} and any(x is None for x in args):
                return None
            return _FUNCTIONS[name](*args)
        if isinstance(n, ast.BinOp):
            a, b = visit(n.left), visit(n.right)
            return None if a is None or b is None else _BINARY[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp):
            a = visit(n.operand)
            if a is None:
                return None
            return not a if isinstance(n.op, ast.Not) else -a if isinstance(n.op, ast.USub) else +a
        if isinstance(n, ast.BoolOp):
            values = [visit(x) for x in n.values]
            if isinstance(n.op, ast.And):
                return (
                    False
                    if any(x is not None and not bool(x) for x in values)
                    else None
                    if None in values
                    else True
                )
            return (
                True
                if any(x is not None and bool(x) for x in values)
                else None
                if None in values
                else False
            )
        if isinstance(n, ast.Compare):
            a = visit(n.left)
            results = []
            for op, value in zip(n.ops, n.comparators, strict=True):
                b = visit(value)
                results.append(None if a is None or b is None else _COMPARE[type(op)](a, b))
                a = b
            return False if False in results else None if None in results else True
        raise ValueError("unsupported expression")

    result = visit(compile_expression(source))
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("expression produced non-finite value")
    return result


def transformations(config: dict[str, Any]) -> list[tuple[str, str]]:
    result = []
    for entry in config.get("feature_transformations", []):
        for field in entry.get("feature_list") or [""]:
            source = entry["expr"].replace("[*]", field)
            tree = ast.parse(source)
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
                raise ValueError("transformation must be one assignment")
            assignment = tree.body[0]
            if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
                raise ValueError("transformation target must be a field")
            expression = ast.unparse(assignment.value)
            compile_expression(expression)
            result.append((assignment.targets[0].id, expression))
    return result
