"""Finite CPU graph runtime; consumes frozen prepared input, not raw online data."""

import hashlib
import json
import platform
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

from ..models.rerank.model_params import parameters
from ..models.rerank.rerank_model import model_fn, transform_features
from .datasvr.prepared import (
    batches,
    column_batches,
    digest,
    input_columns,
    rows,
    tensor_batch,
    tensor_columns,
)
from .eval.metrics import Metrics
from .learning_rate import learning_rate_for_epoch, validate_schedules


def seed_graph(seed):
    """Seed Python-backed Keras initializers as well as NumPy and graph RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def source_identity():
    root = Path(__file__).resolve().parents[2]
    sources = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "train").rglob("*.py"))
        if "tests" not in p.parts
    }
    lock = root / "uv.lock"
    return {
        "source_digest": digest(sources),
        "lock_digest": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
    }


def placeholders(values):
    return {
        group: {
            key: tf.placeholder(
                tf.as_dtype(value.dtype), [None] * value.ndim, name=group + "/" + key
            )
            for key, value in fields.items()
        }
        for group, fields in values.items()
    }


def feed(place, values):
    return {
        tensor: values[group][key]
        for group, fields in place.items()
        for key, tensor in fields.items()
    }


def fit_vocabulary(prepared, identity, params, dtypes, batch_size):
    keys = {}
    types = {}
    source = column_batches(prepared, identity, input_columns(params, dtypes), batch_size)
    first = next(source)
    values = tensor_columns(first, params, dtypes)
    with tf.Graph().as_default():
        ph = placeholders(values)
        transformed = transform_features(ph, params)
        fetch = {}
        for name, group in params.input_features.items():
            if group["input_type"] != "embed":
                continue
            for field in group["features"]:
                table = params.embed_name[field]
                value = transformed[group["from"]][field]
                if group.get("history_pooling"):
                    length = transformed[group["from"]]["length"]
                    value = tf.boolean_mask(value, tf.sequence_mask(length, tf.shape(value)[1]))
                fetch[name + "/" + field] = value
                dtype = value.dtype.name
                if table in types and types[table] != dtype:
                    raise ValueError("Shared vocabulary dtype conflict")
                types[table] = dtype
                keys.setdefault(table, set())
        with tf.Session() as sess:
            import itertools

            for batch in itertools.chain([first], source):
                values = tensor_columns(batch, params, dtypes)
                for path, arr in sess.run(fetch, feed(ph, values)).items():
                    name, field = path.split("/", 1)
                    keys[params.embed_name[field]].update(np.asarray(arr).reshape(-1).tolist())

    def normalize(x):
        return x.decode() if isinstance(x, bytes) else x

    return {
        name: {"dtype": types[name], "keys": sorted(normalize(x) for x in values)}
        for name, values in keys.items()
    }


def run(config, prepared_root, output, resume=None, on_epoch=None):
    validate_schedules(config["model"], config["max_epochs"])
    tf.disable_eager_execution()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prepared_root = Path(prepared_root)
    ident = config["prepared_identity"]
    params = parameters(config["model"])
    dtypes = config["dtypes"]
    size = config["batch_size"]
    start = time.monotonic()
    print(json.dumps({"event": "fit_vocabulary_start"}), flush=True)
    vocabs = fit_vocabulary(prepared_root / "train", ident, params, dtypes, size)
    vocabulary_seconds = time.monotonic() - start
    print(
        json.dumps({"event": "fit_vocabulary_complete", "seconds": vocabulary_seconds}), flush=True
    )
    params = parameters(config["model"], vocabs)
    logical = digest({"config": config, "vocabs": vocabs, "source": source_identity()})
    (output / "feature-state.json").write_text(json.dumps(vocabs, sort_keys=True))
    first = next(
        batches(rows(prepared_root / "train", ident, columns=input_columns(params, dtypes)), size)
    )
    metrics = []
    start_epoch = 0
    best = None
    previous = Path(resume) if resume else None
    if previous:
        record = json.loads((previous / "checkpoint.json").read_text())
        if record["identity"] != logical:
            raise ValueError("Checkpoint identity mismatch")
        start_epoch = record["epoch"]
        best = record.get("best")
    for epoch in range(start_epoch, config["max_epochs"]):
        params.learning_rate = learning_rate_for_epoch(config["model"], epoch + 1)
        params.sparse_learning_rate = learning_rate_for_epoch(config["model"], epoch + 1, "sparse_")
        epoch_start = time.monotonic()
        # Rebuild each epoch so Finder stateful dropout has reproducible epoch-local RNG.
        # Trainable weights and optimizer slots are restored from the preceding checkpoint.
        with tf.Graph().as_default():
            seed_graph(config["seed"] + epoch)
            ph = placeholders(tensor_batch(first, params, dtypes))
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                train = model_fn(ph, None, tf.estimator.ModeKeys.TRAIN, params)
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                evaluation = model_fn(ph, None, tf.estimator.ModeKeys.EVAL, params)
            saver = tf.train.Saver()
            session = tf.ConfigProto(
                device_count={"GPU": 0},
                intra_op_parallelism_threads=config["intra_op_threads"],
                inter_op_parallelism_threads=config["inter_op_threads"],
            )
            with tf.Session(config=session) as sess:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                if previous:
                    saver.restore(sess, str(previous / "model"))
                train_start = time.monotonic()
                graph_seconds = train_start - epoch_start
                sums = {name: 0.0 for name in params.tasks}
                denominators = {name: 0.0 for name in params.tasks}
                objective_sums = {}
                objective_batches = 0
                for batch in batches(
                    rows(prepared_root / "train", ident, columns=input_columns(params, dtypes)),
                    size,
                    config["seed"] + epoch,
                    config["shuffle_buffer_size"],
                ):
                    result = sess.run(
                        {
                            "losses": train["losses"],
                            "step": train["train_op"],
                            "objective": train["objective_losses"],
                            "denominators": train["loss_denominators"],
                        },
                        feed(ph, tensor_batch(batch, params, dtypes)),
                    )
                    objective_batches += 1
                    for name, value in result["objective"].items():
                        objective_sums[name] = objective_sums.get(name, 0.0) + float(value)
                    if int(result["step"]) % 100 == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "train_step",
                                    "epoch": epoch + 1,
                                    "step": int(result["step"]),
                                }
                            ),
                            flush=True,
                        )
                    for name, value in result["losses"].items():
                        sums[name] += float(value) * result["denominators"][name]
                        denominators[name] += float(result["denominators"][name])

                validation_start = time.monotonic()
                train_seconds = validation_start - train_start
                valid = {name: 0.0 for name in params.tasks}
                valid_denominators = {name: 0.0 for name in params.tasks}
                metric_path = output / f"metrics-{epoch + 1:04d}.sqlite"
                accumulator = Metrics(metric_path, params.tasks)
                for batch in batches(
                    rows(
                        prepared_root / "validation", ident, columns=input_columns(params, dtypes)
                    ),
                    size,
                ):
                    result = sess.run(
                        {
                            "losses": evaluation["losses"],
                            "denominators": evaluation["loss_denominators"],
                            "predictions": evaluation["predictions"],
                            "metric_values": evaluation["metric_values"],
                        },
                        feed(ph, tensor_batch(batch, params, dtypes)),
                    )
                    for name, value in result["losses"].items():
                        valid[name] += float(value) * result["denominators"][name]
                        valid_denominators[name] += float(result["denominators"][name])
                        accumulator.add(
                            name, result["predictions"][name], result["metric_values"][name]
                        )
                calculated = accumulator.finish()
                accumulator.close()
                metric_path.unlink()
                record = {
                    "epoch": epoch + 1,
                    "train_objective_batch_mean": {
                        k: v / objective_batches for k, v in objective_sums.items()
                    },
                    "train_loss": {
                        k: v / denominators[k] if denominators[k] else None for k, v in sums.items()
                    },
                    "validation_loss": {
                        k: v / valid_denominators[k] if valid_denominators[k] else None
                        for k, v in valid.items()
                    },
                    "validation_metrics": calculated,
                    "stage_seconds": {
                        "graph_restore": graph_seconds,
                        "train_with_input": train_seconds,
                        "validation_with_input": time.monotonic() - validation_start,
                    },
                }
                record["learning_rate"] = params.learning_rate
                if params.sparse_optimizer:
                    record["sparse_learning_rate"] = params.sparse_learning_rate
                metrics.append(record)
                target = output / f"epoch-{epoch + 1:04d}"
                target.mkdir()
                saver.save(sess, str(target / "model"))
                selection = config.get("selection")
                if selection:
                    terms = selection["task_losses"]
                    if all(record["validation_loss"].get(k) is not None for k in terms):
                        score = sum(record["validation_loss"][k] * w for k, w in terms.items())
                        if best is None or (
                            score < best["score"]
                            if selection["mode"] == "min"
                            else score > best["score"]
                        ):
                            best = {
                                "epoch": epoch + 1,
                                "score": score,
                                "checkpoint": str(target.resolve()),
                            }
                            (output / "selection.json").write_text(json.dumps(best))
                    elif best is None:
                        (output / "selection.json").write_text(
                            json.dumps({"status": "unavailable", "reason": "missing_task_loss"})
                        )
                (target / "checkpoint.json").write_text(
                    json.dumps(
                        {"identity": logical, "epoch": epoch + 1, "prefix": "model", "best": best}
                    )
                )
                from .epoch_metrics import save_epoch

                save_epoch(output, record)
                print(
                    json.dumps({"event": "epoch_complete", **record}, allow_nan=False), flush=True
                )
                if on_epoch:
                    on_epoch(record)
                previous = target
    # Final success marker is written last, only after all requested epochs complete.
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "epochs": metrics,
                "seconds": time.monotonic() - start,
                "vocabulary_seconds": vocabulary_seconds,
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
            },
            allow_nan=False,
        )
    )
    (output / "lineage.json").write_text(
        json.dumps(
            {
                "identity": logical,
                "prepared_identity": ident,
                **source_identity(),
                "seed": config["seed"],
            }
        )
    )
    return metrics


def evaluate_checkpoint(config, prepared_root, checkpoint, output, split="test"):
    """Read-only checkpoint evaluation; never updates the training run."""
    if split not in ("validation", "test"):
        raise ValueError("Evaluation split must be validation or test")
    tf.disable_eager_execution()
    checkpoint = Path(checkpoint)
    output = Path(output)
    record = json.loads((checkpoint / "checkpoint.json").read_text())
    vocabs = json.loads((checkpoint.parent / "feature-state.json").read_text())
    logical = digest({"config": config, "vocabs": vocabs, "source": source_identity()})
    if record["identity"] != logical:
        raise ValueError("Checkpoint identity mismatch")
    params = parameters(config["model"], vocabs)
    ident = config["prepared_identity"]
    dtypes = config["dtypes"]
    size = config["batch_size"]
    prepared = Path(prepared_root) / split
    first = next(batches(rows(prepared, ident, columns=input_columns(params, dtypes)), size))
    output.mkdir(parents=True, exist_ok=False)
    sums = {name: 0.0 for name in params.tasks}
    denoms = {name: 0.0 for name in params.tasks}
    path = output / "metrics.sqlite"
    accumulator = Metrics(path, params.tasks)
    try:
        with tf.Graph().as_default():
            seed_graph(config["seed"])
            ph = placeholders(tensor_batch(first, params, dtypes))
            with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
                graph = model_fn(ph, None, tf.estimator.ModeKeys.EVAL, params)
            saver = tf.train.Saver()
            options = tf.ConfigProto(
                device_count={"GPU": 0},
                intra_op_parallelism_threads=config["intra_op_threads"],
                inter_op_parallelism_threads=config["inter_op_threads"],
            )
            with tf.Session(config=options) as sess:
                sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                saver.restore(sess, str(checkpoint / "model"))
                for batch in batches(
                    rows(prepared, ident, columns=input_columns(params, dtypes)), size
                ):
                    values = sess.run(
                        {
                            k: graph[k]
                            for k in ("losses", "loss_denominators", "predictions", "metric_values")
                        },
                        feed(ph, tensor_batch(batch, params, dtypes)),
                    )
                    for name, value in values["losses"].items():
                        count = values["loss_denominators"][name]
                        sums[name] += float(value) * count
                        denoms[name] += float(count)
                        accumulator.add(
                            name, values["predictions"][name], values["metric_values"][name]
                        )
        metrics = {
            "loss": {
                name: value / denoms[name] if denoms[name] else None for name, value in sums.items()
            },
            "metrics": accumulator.finish(),
            "checkpoint": str(checkpoint),
            "identity": logical,
            "split": split,
        }
        (output / "metrics.json").write_text(json.dumps(metrics, allow_nan=False))
        return metrics
    finally:
        accumulator.close()
        path.unlink(missing_ok=True)
