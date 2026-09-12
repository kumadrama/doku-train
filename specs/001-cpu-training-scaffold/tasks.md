# CPU Training Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `subagent-driven-development` or `executing-plans`
> when implementing this plan. This document is planning output only; do not create runtime code while it is
> under review.

**Goal:** Build a Finder-compatible, single-machine CPU pipeline that reads immutable S3 Parquet manifests,
trains and evaluates a four-target PyTorch reranker, refits on 28 mature days, and writes exactly four kinds of
training output: PyTorch Checkpoint, metrics, fitted feature state, and lineage.

**Architecture:** Keep Finder's `recommend/train` namespace and familiar filenames while replacing its
GPU/private-platform internals with strict Pydantic contracts, an injected object-store port, bounded PyArrow
readers, and a model-agnostic `TrainingPipeline`. The evaluation model trains on 27 days and validates on day 28;
hard-gate success selects an epoch count for a fresh 28-day production refit. `ModelExporter` is only a typed
extension point: Spec 001 does not implement or call an exporter and has no ONNX dependency.

**Tech Stack:** Python 3.12, uv, Pydantic 2, PyYAML, Boto3, PyArrow, NumPy, PyTorch CPU,
scikit-learn, Typer, pytest, ruff, mypy, psutil.

---

## Plan rules

- Execute tasks in dependency order and use TDD: failing test, minimal implementation, passing test, commit.
- Finder is a naming and responsibility reference only. Do not copy private implementation, binaries, internal
  endpoints, GPU/distributed behavior, or model parameters.
- Reuse dependency ranges and the CPU-only Torch index pattern already proven in
  `doku-mono/reco-offline/pyproject.toml`; do not add a cross-repository Python import.
- Do not add ONNX, ONNX Runtime, `onnxscript`, an exporter implementation, parity tests, or serving code.
- Commands run from the `doku-train` repository root.

## File map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml`, `uv.lock` | CPU-only dependencies and deterministic toolchain |
| `Makefile`, `.gitignore` | unified checks and exclusion of datasets/checkpoints/outputs |
| `recommend/train/comm/model_params.py` | strict common training configuration |
| `recommend/train/comm/model_params_validator.py` | model-agnostic config loading and schema snapshots |
| `recommend/train/comm/model_protocol.py` | model adapter/factory protocols consumed by the pipeline |
| `recommend/train/comm/model_exporter.py` | future export protocol only; no concrete implementation |
| `recommend/train/comm/datasvr/dataset_manifest.py` | Dataset/Feature/Label contracts and canonical digests |
| `recommend/train/comm/datasvr/storage.py` | URI parser, metadata, and `ObjectStore` protocol |
| `recommend/train/comm/datasvr/local_storage.py` | local fixture implementation of `ObjectStore` |
| `recommend/train/comm/datasvr/s3_storage.py` | bounded S3 reads and create-only writes |
| `recommend/train/comm/datasvr/parquet_dataset.py` | checksum-verified, bounded-memory batch iteration |
| `recommend/train/models/rerank/model_params.py` | strict rerank network/loss configuration |
| `recommend/train/models/rerank/data_utils.py` | temporal split, completion label, and tensor batch |
| `recommend/train/models/rerank/feature_utils.py` | fitted numeric/category state and deterministic transforms |
| `recommend/train/models/rerank/rerank_model.py` | shared-bottom four-tower PyTorch module and masked loss |
| `recommend/train/models/rerank/__init__.py` | rerank adapter/factory implementing the common protocol |
| `recommend/train/comm/feature_check.py` | point-in-time, OOV, missingness, and forbidden-ID checks |
| `recommend/train/comm/checkpoint_agent.py` | evaluation/production PyTorch checkpoint save and restore |
| `recommend/train/comm/metrics_utils.py` | versioned four-target loss/AUC metrics |
| `recommend/train/comm/eval/eval_model_rerank.py` | evaluation loop returning typed metric sets |
| `recommend/train/comm/eval/gates.py` | pre-refit hard gates and soft quality warnings |
| `recommend/train/comm/training_pipeline.py` | evaluation train, gate, and production refit state machine |
| `recommend/train/comm/training_output.py` | exact four-file, run-scoped, create-only training output |
| `recommend/train/comm/offline_evaluate.py` | explicit 24/2/2 and rolling backtest orchestration |
| `recommend/train/comm/resource_report.py` | capacity data embedded into `metrics.json` |
| `recommend/train/run.py` | root Typer composition entrypoint |
| `recommend/train/train_rerank.py` | `validate`, `train`, and `backtest` use cases |
| `recommend/train/tests/{unit,contract,smoke,performance}` | requirement-level automated evidence |

## Dependency graph

| Task | Depends on | Working result |
| --- | --- | --- |
| 1 | none | installable Finder-compatible CPU project |
| 2 | 1 | strict common/model config and checked schemas |
| 3 | 1 | immutable Manifest and local/S3 storage ports |
| 4 | 3 | bounded Parquet reader and deterministic temporal splits |
| 5 | 2, 3, 4 | point-in-time feature state and four-label batches |
| 6 | 2, 5 | four-target rerank model and masked loss |
| 7 | 2, 6 | role-separated checkpoints and reproducibility metadata |
| 8 | 5, 6 | metrics and pre-refit hard/soft gates |
| 9 | 4, 6, 7, 8 | model-agnostic 27/1 → gate → 28-day refit pipeline |
| 10 | 3, 7, 9 | exact four-file training output and exporter interface boundary |
| 11 | 2, 3, 5, 9, 10 | CLI and complete local CPU smoke path |
| 12 | 1–11 | security/dependency checks, capacity evidence, final traceability |

## Task 1: Bootstrap the Finder-compatible CPU project

**Requirements:** R001, NFR003, NFR004
**Depends on:** none

**File range:**

- Create `pyproject.toml`, `uv.lock`, `Makefile`, `.gitignore`.
- Create package markers under `recommend/`, `recommend/train/`, `comm/`, `models/`, `offline/`, and `tools/`.
- Create `recommend/train/tests/unit/test_project_layout.py`.

**Steps:**

- [x] Write a failing layout test that imports every Finder-compatible namespace and inspects the CPU-only Torch
  index.
- [x] Add Python 3.12 dependencies without `onnx`, `onnxruntime`, `onnxscript`, CUDA, NCCL, MPI, TensorFlow, or
  TensorFlow Serving packages.
- [x] Add `make format`, `lint`, `typecheck`, `unit`, `test`, and `check`; keep performance tests outside the
  default test target.
- [x] Ignore local data, runs, outputs, `*.pt`, caches, and credentials.
- [x] Run `uv lock`, `uv sync --group dev`, and the focused layout test.

**Completion conditions:** the package imports in Python 3.12; the lock resolves a CPU Torch build; a dependency
policy assertion proves the forbidden packages are absent; `make unit` passes.

**Commit boundary:** `build: bootstrap cpu training package`

## Task 2: Add strict common and rerank configuration

**Requirements:** R002, R006, NFR002
**Depends on:** Task 1

**File range:**

- Create `recommend/train/comm/model_params.py`, `model_params_validator.py`, `config.schema.yml`.
- Create `recommend/train/models/rerank/model_params.py`, `config.schema.yml`, `__init__.py`.
- Create `recommend/train/tests/unit/test_model_params.py`.

**Steps:**

- [x] Write failing tests for unknown fields, non-CPU execution strategies, invalid thread/memory budgets, invalid
  task weights, and missing caller-provided input/output.
- [x] Define frozen Pydantic models with `extra="forbid"`; the only execution strategy is
  `single_process_cpu`, and split values are 27/1/28.
- [x] Keep model architecture, optimizer, early stopping, metrics, and resource settings explicit. Do not add an
  export or ONNX config block.
- [x] Generate common and rerank JSON Schema snapshots and add a test that fails when source models and checked-in
  snapshots diverge.
- [x] Run `uv run pytest recommend/train/tests/unit/test_model_params.py -v`.

**Completion conditions:** invalid/unknown configuration fails before I/O; no production URI has a default;
schemas match the models; configuration contains no serving or exporter policy.

**Commit boundary:** `feat: add strict training configuration`

## Task 3: Define immutable Manifest and object-store ports

**Requirements:** R003, NFR002, NFR004
**Depends on:** Task 1

**File range:**

- Create `recommend/train/comm/datasvr/{dataset_manifest,storage,local_storage,s3_storage}.py`.
- Create unit/contract tests for URI parsing, canonical digests, immutable reads, and create-only writes.

**Steps:**

- [x] Write failing tests for unsupported Manifest versions, missing schema/label references, duplicate shards,
  total-row mismatch, digest mismatch, prefix escape, and existing output objects.
- [x] Define a versioned `DatasetManifest` with exact shard URI, event range, row count, byte count, and SHA-256.
- [x] Define the narrow `ObjectStore` protocol: bounded `get_bytes`, `head`, streaming `download_to`, and
  `put_bytes_if_absent`.
- [x] Implement local fixture storage and S3 storage with an injected Boto3 client and allowed bucket/prefix.
- [x] Prove no reader code calls `list_objects*`, resolves `latest`, or logs signed query parameters.

**Completion conditions:** input identity is fixed by canonical digest; storage adapters share one port; S3 access
is bounded and explicitly scoped; output overwrites are rejected.

**Commit boundary:** `feat: add immutable dataset contracts`

## Task 4: Stream Parquet and build deterministic temporal splits

**Requirements:** R004, NFR001, NFR002
**Depends on:** Task 3

**File range:**

- Create `recommend/train/comm/datasvr/parquet_dataset.py`.
- Create `recommend/train/models/rerank/data_utils.py`.
- Create Parquet fixture builders and reader/split contract tests.

**Steps:**

- [x] Write failing tests for deterministic shard ordering, bounded batch size, bad checksum, bad row count,
  duplicate/overlapping days, incomplete 28-day windows, and labels not mature at `as_of_ms`.
- [x] Stream one shard and one record batch at a time; verify file digest and row count without concatenating all
  28 days in memory.
- [x] Implement event-date split planning: days 1–27 evaluation train, day 28 validation, days 1–28 production
  refit.
- [x] Preserve all valid exposure rows and emit independent `*_valid` masks; `bad_row_policy` initially supports
  only `fail` with a counted error.
- [x] Implement canonical completion as positive only when duration is valid and the ratio is at least 0.95.

**Completion conditions:** all three batch factories are replayable and deterministic; no random cross-day split
exists; no negative sampling exists; invalid data fails with a stable classification.

**Commit boundary:** `feat: stream deterministic training splits`

## Task 5: Fit and freeze point-in-time feature state

**Requirements:** R005, R006, NFR002
**Depends on:** Tasks 2, 3, 4

**File range:**

- Create `recommend/train/models/rerank/feature_utils.py`.
- Create `recommend/train/comm/feature_check.py`.
- Create feature-state unit tests and point-in-time contract tests.

**Steps:**

- [x] Write failing tests distinguishing missing numeric values from real zero and missing categories from OOV.
- [x] Fit evaluation feature state only on days 1–27; freeze and reuse it for day 28 validation.
- [x] Refit production feature state from scratch on all 28 days.
- [x] Serialize numeric statistics, vocab/hash policy, hash seed, bucket counts, OOV index, and schema version into
  canonical JSON.
- [x] Reject raw `user_id` in the tensor signature and reject 7-day statistics whose feature cutoff exceeds the
  sample event time.

**Completion conditions:** validation never mutates feature state; production has a separate 28-day state;
missing/OOV semantics are stable; feature serialization round-trips deterministically.

**Commit boundary:** `feat: add fitted rerank feature state`

## Task 6: Implement the shared-bottom four-target reranker

**Requirements:** R006
**Depends on:** Tasks 2, 5

**File range:**

- Create `recommend/train/models/rerank/rerank_model.py` and any leaf modules below `modules/`.
- Complete the adapter in `recommend/train/models/rerank/__init__.py`.
- Create `recommend/train/tests/unit/test_rerank_model.py`.

**Steps:**

- [x] Write failing shape tests for the four stable logits and masked-loss tests for independently missing labels.
- [x] Implement numeric inputs plus categorical embeddings, a shared-bottom MLP, and four task towers.
- [x] Compute per-task mean BCE-with-logits only over valid rows; normalize configured weights across active tasks.
- [x] Reject batches with no active target instead of producing a fake zero loss.
- [x] Add an import-boundary test proving the model does not access storage, CLI, environment variables, or
  process launch APIs.

**Completion conditions:** the model is pure tensor code; outputs exactly the four canonical target names; default
loss weights are equal; the common package does not import the concrete reranker.

**Commit boundary:** `feat: add cpu multitask reranker`

## Task 7: Add role-separated checkpoints and seed control

**Requirements:** R007, R009, NFR002
**Depends on:** Tasks 2, 6

**File range:**

- Create `recommend/train/comm/checkpoint_agent.py`.
- Create `recommend/train/tests/unit/test_checkpoint_agent.py` and `test_reproducibility.py`.

**Steps:**

- [x] Write failing tests for deterministic Python/NumPy/Torch seeds, atomic replacement, format mismatch,
  config/dataset/model-lineage mismatch, and role mismatch.
- [x] Save evaluation and production checkpoints at distinct internal paths using explicit role values; never let
  an evaluation checkpoint become the delivered `checkpoint.pt`.
- [x] Save model, optimizer, epoch, step, early-stopping/training state, Python/NumPy/Torch RNG state, format
  version, config digest, dataset digest, and model-structure digest.
- [x] Validate all metadata before mutating a target model or optimizer during restore.
- [x] Add a byte-stream loader so the CLI can strictly reload the production checkpoint read back from object
  storage on CPU.

**Completion conditions:** mismatches leave the target untouched; paths are role-separated; a production
checkpoint round-trips from bytes in a clean CPU model instance.

**Commit boundary:** `feat: add strict training checkpoints`

## Task 8: Compute four-target metrics and classify gates

**Requirements:** R008, NFR003
**Depends on:** Tasks 5, 6

**File range:**

- Create `recommend/train/comm/metrics_utils.py`.
- Create `recommend/train/comm/eval/{eval_model_rerank,gates}.py`.
- Create metric/gate unit tests.

**Steps:**

- [x] Write failing tests for valid/positive/negative counts, loss, AUC, single-class targets, minimum sample
  count, and current-slice comparison.
- [x] Return `NOT_EVALUABLE` instead of numeric zero when AUC is undefined.
- [x] Model schema/integrity/maturity/numerical/Checkpoint/metric-evaluability failures as blocking gates before
  production refit.
- [x] Model AUC floors, candidate-vs-previous deltas, cohorts, and drift as warnings only; require the same current
  validation slice digest for comparisons.
- [x] Do not define an exportability, ONNX, parity, Artifact, or post-refit gate.

**Completion conditions:** all four tasks carry metric version and sample counts; blocking gates stop before refit;
AUC changes cannot block refit; there is no exporter-related gate.

**Commit boundary:** `feat: add rerank evaluation gates`

## Task 9: Orchestrate evaluation training, gates, and production refit

**Requirements:** R006, R007, R008, R009, NFR002
**Depends on:** Tasks 4, 6, 7, 8

**File range:**

- Create `recommend/train/comm/model_protocol.py` and `training_pipeline.py`.
- Complete `recommend/train/models/rerank/__init__.py`.
- Create `recommend/train/tests/unit/test_training_pipeline.py`.

**Steps:**

- [x] Write failing state-machine tests for fresh evaluation/production model instances, early-stopping epoch
  selection, a blocking gate, soft AUC warning, NaN/Inf, empty data, and memory budget exhaustion.
- [x] Train the evaluation model on days 1–27, restore its best role=`evaluation` checkpoint, and evaluate only on
  day 28.
- [x] Run only the Task 8 pre-refit gates. The constructor and `run()` signature must have no exporter,
  `pre_refit_validator`, export callback, or serving dependency.
- [x] After hard-gate success, reseed deterministically, construct a fresh production model, refit on all 28 days
  for exactly `selected_epoch`, and save role=`production` checkpoint.
- [x] Return typed metrics, gates, traces, production adapter, production checkpoint path, and selected epoch.
- [x] Add an import-boundary test proving `comm/training_pipeline.py` does not import concrete models,
  `model_exporter`, ONNX, or storage adapters.

**Completion conditions:** a blocking pre-refit gate creates only one model; a successful run creates two fresh
models and a production checkpoint; refit success is final and cannot be invalidated by an export step because no
export step exists.

**Commit boundary:** `feat: orchestrate cpu rerank training`

## Task 10: Persist training outputs and define the exporter boundary

**Requirements:** R009, NFR002, NFR004
**Depends on:** Tasks 3, 7, 9

**File range:**

- Create `recommend/train/comm/training_output.py`.
- Create `recommend/train/comm/model_exporter.py`.
- Create `recommend/train/tests/unit/test_model_exporter.py`.
- Create `recommend/train/tests/contract/test_training_output.py`.

**Contract to implement:**

```python
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class TrainingOutputUris:
    checkpoint: str
    metrics: str
    fitted_feature_state: str
    lineage: str


@dataclass(frozen=True, slots=True)
class ExportRequest:
    run_id: str
    training_outputs: TrainingOutputUris


@dataclass(frozen=True, slots=True)
class ExportResult:
    export_attempt_id: str
    status: Literal["SUCCEEDED", "FAILED"]
    artifact_uri: str | None
    error_category: str | None


class ModelExporter(Protocol):
    def export(self, request: ExportRequest) -> ExportResult: ...
```

No class in Spec 001 implements `ModelExporter`, and no production composition root accepts one.

**Steps:**

- [x] Write a failing contract test that expects exactly these object names under `<output>/<run_id>/`:
  `checkpoint.pt`, `metrics.json`, `fitted-feature-state/state.json`, `lineage.json`.
- [x] Implement a `TrainingOutputWriter` that accepts all four bodies together, rejects an incomplete/extra file
  set, uses `put_bytes_if_absent`, and returns `TrainingOutputUris` only after every write succeeds.
- [x] Canonicalize and validate `metrics.json`, feature state, and lineage before writes; the Checkpoint body must
  have already passed strict production-role reload.
- [x] Add tests proving any existing target rejects the run, no target is overwritten, and partial writes are never
  returned as a successful bundle.
- [x] Define the exporter dataclasses and Protocol in `model_exporter.py`; add static/import tests proving it has
  no implementation, no ONNX import, and no dependency from `TrainingPipeline`.
- [x] Document in module docstrings that a future exporter runs only after successful refit and has independent
  status; exporter failure cannot modify a Training Run.

**Completion conditions:** success returns exactly four URI fields; no manifest/signature/model-format file is
created; no exporter implementation or runtime dependency exists; overwrite and partial-write paths fail closed.

**Commit boundary:** `feat: persist cpu training outputs`

## Task 11: Wire the Finder-style CLI and full CPU smoke test

**Requirements:** R001, R003, R004, R005, R006, R007, R008, R009, R010, NFR003, NFR004
**Depends on:** Tasks 2, 3, 5, 9, 10

**File range:**

- Create `recommend/train/run.py`, `train_rerank.py`, `comm/errors.py`, `comm/offline_evaluate.py`.
- Create local 28-day synthetic fixture helpers.
- Create `recommend/train/tests/smoke/test_cpu_train.py` and CLI/failure tests.

**Steps:**

- [x] Write failing CLI tests for `rerank validate`, `rerank train`, and `rerank backtest`, including nonzero
  structured errors with `run_id`, category, stage, and redacted message.
- [x] In the composition root, parse typed config, validate the exact Manifest, construct storage/readers/model
  factory/CheckpointAgent/evaluator/gates/pipeline, and never shell out to another training command.
- [x] For `train`, execute validate → 27/1 evaluation → gates → 28-day refit. Strictly load the production
  Checkpoint bytes, hand them and the three JSON bodies to `TrainingOutputWriter`, then read `checkpoint.pt` back
  through `ObjectStore` and load it again on CPU.
- [x] Return `TrainingOutputUris`, `selected_epoch`, gate summary, and `checkpoint_reloaded=true`; do not return
  `manifest_uri`, output signature, ONNX metadata, or exporter state.
- [x] Add a deterministic pure-CPU smoke fixture small enough for CI that completes the full path and reads back
  all four outputs.
- [x] Implement `backtest` as an explicit 24/2/2 or rolling-fold use case that does not create production outputs.
- [x] Add failure tests for future feature time, immature labels, checksum mismatch, non-evaluable target,
  NaN/Inf, incompatible Checkpoint, output conflict, and sensitive log redaction.

**Completion conditions:** the smoke test performs
`validate → 27/1 train/evaluate → 28-day refit → write-training-output → checkpoint reload`; all four JSON/model
objects match the Run; no external cloud account or real user data is required.

**Commit boundary:** `feat: wire cpu rerank training cli`

## Task 12: Enforce quality policy and capture capacity evidence

**Requirements:** R001, NFR001, NFR002, NFR003, NFR004
**Depends on:** Tasks 1–11

**File range:**

- Create `recommend/train/comm/resource_report.py`.
- Create `recommend/train/tests/performance/test_capacity_probe.py`.
- Modify `Makefile`, dependency-policy tests, and `specs/001-cpu-training-scaffold/verification.md`.

**Steps:**

- [x] Add wall time, CPU model/count, memory, row/batch throughput, peak RSS, and per-stage duration collection;
  embed the report inside `metrics.json` so the output set remains exactly four files.
- [x] Add an opt-in performance test that streams representative generated shards under a configured RSS budget;
  label its result as evidence, not a production SLA.
- [x] Add static checks that reject CUDA/NCCL/MPI/TensorFlow/ONNX packages, concrete exporter classes,
  `list_objects`, raw secrets, and imports from `doku-offline` or Finder repositories.
- [x] Run `uv lock --check`, `make check`, the performance probe on the target machine when available, and
  `git diff --check`.
- [x] Populate verification evidence with revision, lock digest, fixture Manifest digest, config digest, seed,
  four training-output digests, selected epoch, gate results, Checkpoint reload result, and resource metrics.

**Completion conditions:** `make check` covers format/lint/type/unit/contract/smoke; dependency policy passes;
capacity evidence contains measured hardware context; verification traces every R/NFR and acceptance criterion to
an executable test or explicitly pending production benchmark.

**Commit boundary:** `test: verify cpu training scaffold`

## Execution handoff

Implementation must start in a fresh session with the approved plan and one of the required execution skills.
Before claiming completion, run the full verification commands and attach their actual output to
`verification.md`. Spec 001 is complete only when all four training documents agree that:

- production refit success is not conditional on exporter success;
- the first version writes only Checkpoint, metrics, fitted feature state, and lineage;
- `ModelExporter` exists only as an interface;
- ONNX Artifact, signature, parity, and serving compatibility remain future work.
