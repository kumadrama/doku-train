# Finder CPU migration (Spec 024)

Independent Python 3.10 / TensorFlow 2.11 package. The old TF implementation is not imported.

From repository root:

```sh
uv sync --project recommend_tf
PYTHONPATH=. recommend_tf/.venv/bin/python -m pytest recommend_tf/train/tests -q
PYTHONPATH=. recommend_tf/.venv/bin/python -m recommend_tf.train.run --help
```

`train_rerank.train` accepts an explicit Doku dataset configuration and normalized Finder `model`
configuration. `run` is the lower-level prepared-input entry. No production source is defaulted.
The runnable synthetic example is `train/tests/test_migration.py`.

Input preparation stores basic fields and ragged histories; transforms, lookup and labels remain
in the graph. Use separate scalar/embed groups. Labels have no implicit validity mask: configure
default values and `loss_weight`. `loss_by_weight` preserves Finder's nonzero-count denominator.

Adam does not accept a Torch-style `weight_decay`. Finder AdamW requires an explicit coefficient;
its coefficient is not automatically multiplied by learning rate and sparse decay touches only
indexed rows. Dynamic embedding services are replaced with fixed train-only mappings.

`source-inventory.json` records exact extracted source functions. Local adapter functions are
identified in Spec 024; whole-file identity is not claimed for adapted model functions.

Current implementation and remaining acceptance work are recorded in Spec 024 tasks/verification.
Do not treat passing synthetic tests as production or throughput validation.


For a published Manifest, normalize the experiment with
`models.rerank.data_utils.migrate_experiment(experiment, raw_dtypes, optimizer_settings, shuffle_buffer_size)`.
The Finder optimizer settings must be explicit; no implicit Torch decay conversion is provided.
Pass the resulting JSON to `python -m recommend_tf.train.run --config ... --output ...`.
`--local-root` supports offline synthetic input. Prepared checkpoint evaluation uses
`--prepared ... --evaluate-checkpoint ... --split test --output ...` with the frozen resolved config.

The separate TF test suite is part of repository `make check` through `make tf-check`.
See Spec024 verification for host/platform limits; native Linux input/training acceptance passed on SageMaker; see Spec024 cloud-verification.md.


Manifest CLI now runs train → validation selection → selected-checkpoint test. Per-epoch JSON and TensorBoard are stored in training/epoch-metrics and training/tensorboard. cloud_train publishes completed epoch artifacts to an explicit run-scoped S3 URI; prepared input/cache are excluded. Dockerfile.train runs native acceptance with DOKU_MODE=verify before the first formal job.
