"""Finder graph expression evaluator, current-path operator subset."""

import ast
from collections.abc import Mapping
from functools import reduce, wraps

import tensorflow.compat.v1 as tf


def wrap_for_sparse(fn):

    @wraps(fn)
    def new_fn(x, *args, **kwargs):
        if isinstance(x, tf.SparseTensor):
            return tf.SparseTensor(x.indices, fn(x.values, *args, **kwargs), x.dense_shape)
        return fn(x, *args, **kwargs)

    return new_fn


def wrap_for_sparse2(fn):

    @wraps(fn)
    def new_fn(x, y, *args, **kwargs):
        if isinstance(x, tf.SparseTensor) and isinstance(y, tf.SparseTensor):
            return tf.SparseTensor(
                x.indices, fn(x.values, y.values, *args, **kwargs), x.dense_shape
            )
        elif isinstance(x, tf.SparseTensor):
            return tf.SparseTensor(x.indices, fn(x.values, y, *args, **kwargs), x.dense_shape)
        elif isinstance(y, tf.SparseTensor):
            return tf.SparseTensor(y.indices, fn(x, y.values, *args, **kwargs), y.dense_shape)
        else:
            return fn(x, y, *args, **kwargs)

    return new_fn


def eval_expr(expr, variables):

    def eval_inner(expr):
        if isinstance(expr, ast.Name):
            return variables[expr.id]
        elif isinstance(expr, ast.NameConstant):
            return expr.value
        elif isinstance(expr, ast.Attribute):
            value = eval_inner(expr.value)
            if isinstance(value, Mapping):
                return value[expr.attr]
            else:
                return getattr(value, expr.attr)
        elif isinstance(expr, ast.Num):
            return expr.n
        elif isinstance(expr, ast.Str):
            return expr.s
        elif isinstance(expr, ast.List):
            return list(map(eval_inner, expr.elts))
        elif isinstance(expr, ast.Tuple):
            return tuple(map(eval_inner, expr.elts))
        elif isinstance(expr, ast.Dict):
            return dict(zip(map(eval_inner, expr.keys), map(eval_inner, expr.values)))
        elif isinstance(expr, ast.BinOp):
            binops = {
                ast.Add: lambda x, y: x + y,
                ast.Sub: lambda x, y: x - y,
                ast.Mult: lambda x, y: x * y,
                ast.Div: lambda x, y: x / y,
                ast.FloorDiv: lambda x, y: x // y,
                ast.Mod: lambda x, y: x % y,
                ast.Pow: lambda x, y: x**y,
                ast.BitAnd: lambda x, y: x & y,
                ast.BitOr: lambda x, y: x | y,
                ast.BitXor: lambda x, y: x ^ y,
                ast.LShift: bitwise_left_shift,
                ast.RShift: bitwise_right_shift,
            }
            assert type(expr.op) in binops, f"unsupported op: {type(expr.op)}"
            opfn = wrap_for_sparse2(binops[type(expr.op)])
            return opfn(eval_inner(expr.left), eval_inner(expr.right))
        elif isinstance(expr, ast.BoolOp):
            boolops = {
                ast.And: tf.logical_and,
                ast.Or: tf.logical_or,
            }
            assert type(expr.op) in boolops, f"unsupported op: {type(expr.op)}"
            opfn = wrap_for_sparse2(boolops[type(expr.op)])
            return reduce(opfn, map(eval_inner, expr.values))
        elif isinstance(expr, ast.Compare):
            cmpops = {
                ast.Lt: tf.less,
                ast.Gt: tf.greater,
                ast.LtE: tf.less_equal,
                ast.GtE: tf.greater_equal,
                ast.Eq: tf.equal,
                ast.NotEq: tf.not_equal,
            }
            xs = list(map(eval_inner, [expr.left] + expr.comparators))
            res = []
            for op, x, y in zip(expr.ops, xs[:-1], xs[1:]):
                assert type(op) in cmpops, f"unsupported op: {type(op)}"
                opfn = wrap_for_sparse2(cmpops[type(op)])
                res.append(opfn(x, y))
            return reduce(wrap_for_sparse2(tf.logical_and), res)
        elif isinstance(expr, ast.UnaryOp):
            unaryops = {
                ast.USub: lambda x: -x,
                ast.UAdd: lambda x: +x,
                ast.Invert: lambda x: ~x,
                ast.Not: tf.logical_not,
            }
            assert type(expr.op) in unaryops, f"unsupported op: {type(expr.op)}"
            opfn = wrap_for_sparse(unaryops[type(expr.op)])
            return opfn(eval_inner(expr.operand))
        elif isinstance(expr, ast.Call):
            assert isinstance(expr.func, ast.Name), f"unsupported func type: {type(expr.func)}"
            assert expr.func.id in expr_funcs, f"unsupported func: {expr.func.id}"
            args = []
            for arg in expr.args:
                if isinstance(arg, ast.Starred):
                    args += eval_inner(arg.value)
                else:
                    args.append(eval_inner(arg))
            kwargs = {}
            for kw in expr.keywords:
                if kw.arg is None:
                    kwargs.update(eval_inner(kw.value))
                else:
                    kwargs[kw.arg] = eval_inner(kw.value)
            return expr_funcs[expr.func.id](*args, **kwargs)
        elif isinstance(expr, ast.IfExp):
            return tf.cond(
                eval_inner(expr.test),
                lambda: eval_inner(expr.body),
                lambda: eval_inner(expr.orelse),
            )
        elif isinstance(expr, ast.Subscript):
            value = eval_inner(expr.value)
            slice_spec = eval_inner(expr.slice)
            if isinstance(value, tf.SparseTensor):
                return sparse_slice_helper(value, slice_spec)
            return value.__getitem__(slice_spec)
        elif isinstance(expr, ast.Ellipsis):
            return Ellipsis
        elif isinstance(expr, ast.Slice):
            return slice(
                expr.lower and eval_inner(expr.lower),
                expr.upper and eval_inner(expr.upper),
                expr.step and eval_inner(expr.step),
            )
        elif isinstance(expr, ast.ExtSlice):
            return tuple(map(eval_inner, expr.dims))
        elif isinstance(expr, ast.Index):
            return eval_inner(expr.value)
        else:
            assert False, f"unsupported expr type: {type(expr)}"

    if isinstance(expr, str):
        ast_tree = ast.parse(expr)
        assert len(ast_tree.body) == 1
        expr = ast_tree.body[0]
        assert isinstance(expr, ast.Expr)
        expr = expr.value
    return eval_inner(expr)


def sparse_slice_helper(*args):
    raise ValueError("Sparse expression slicing is outside the migrated experiment path")


bitwise_left_shift = tf.bitwise.left_shift
bitwise_right_shift = tf.bitwise.right_shift
expr_funcs = {
    "float": lambda x: tf.cast(x, tf.float32),
    "int": lambda x: tf.cast(x, tf.int64),
    "cast": lambda x, dtype: tf.cast(x, tf.as_dtype(dtype)),
    "where": wrap_for_sparse2(tf.where_v2),
    "maximum": tf.maximum,
    "minimum": tf.minimum,
    "round": tf.round,
    "clip": tf.clip_by_value,
    "log2": lambda x: tf.math.log(tf.cast(x, tf.float32)) / tf.math.log(2.0),
}
