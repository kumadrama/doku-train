"""Independent-process CPU fixture used by reproducibility acceptance."""

import hashlib
import json
import sys
from pathlib import Path

import tensorflow.compat.v1 as tf

from recommend_tf.train.comm.datasvr.prepared import prepare
from recommend_tf.train.comm.training_pipeline import run, seed_graph
from recommend_tf.train.models.rerank.model_params import parameters
from recommend_tf.train.models.rerank.rerank_model import model_fn
from recommend_tf.train.tests.test_migration import config
from recommend_tf.train.tests.test_regularization import inputs, vocab


def checksum(arrays):
    return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}


def main():
    tf.disable_eager_execution()
    root = Path(sys.argv[1])
    seed = int(sys.argv[2])
    root.mkdir()
    model = config()
    model.update(optimizer="adamw", optimizer_args={"weight_decay": 0.0001})
    with tf.Graph().as_default(), tf.Session() as sess:
        seed_graph(seed)
        model_fn(inputs(), None, "train", parameters(model, vocab()))
        sess.run(tf.global_variables_initializer())
        initial = checksum(
            {v.name: a for v, a in zip(tf.global_variables(), sess.run(tf.global_variables()))}
        )
    if len(sys.argv) > 3:
        (root / "result.json").write_text(json.dumps({"initial": initial}))
        return
    params = parameters(model)
    examples = [
        dict(id=i % 3, history=[0, 1][: i % 3], value=float(i % 3), click=i % 2, scene=i % 2)
        for i in range(17)
    ]
    for split in ["train", "validation"]:
        prepare([examples], root / "input" / split, {}, params, {}, "True", "fixture", 1000000)
    cfg = dict(
        model=model,
        prepared_identity="fixture",
        dtypes=dict(id="int64", history="int64", value="float32", click="int64", scene="int64"),
        batch_size=3,
        seed=seed,
        max_epochs=2,
        selection={"mode": "min", "task_losses": {"a": 1.0}},
        shuffle_buffer_size=4,
        intra_op_threads=4,
        inter_op_threads=1,
    )
    history = run(cfg, root / "input", root / "output")
    resumed = run(cfg, root / "input", root / "resume", root / "output/epoch-0001")

    def logical(records):
        return [{k: v for k, v in r.items() if k != "stage_seconds"} for r in records]

    def checkpoint(path):
        reader = tf.train.load_checkpoint(str(path / "model"))
        return checksum({k: reader.get_tensor(k) for k in reader.get_variable_to_shape_map()})

    checkpoints = [checkpoint(root / "output" / f"epoch-{i:04d}") for i in [1, 2]]
    assert logical(resumed) == logical(history[1:])
    assert checkpoint(root / "resume/epoch-0002") == checkpoints[1]
    (root / "result.json").write_text(
        json.dumps(dict(initial=initial, history=logical(history), checkpoints=checkpoints))
    )


if __name__ == "__main__":
    main()
