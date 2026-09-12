# CPU Training Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Finder-compatible, single-machine CPU pipeline that reads immutable S3 Parquet manifests, trains and evaluates a four-target PyTorch reranker, refits on 28 mature days, and commits a verified ONNX artifact.

**Architecture:** Keep Finder's `recommend/train` namespace and familiar filenames while replacing its GPU/private-platform internals with strict Pydantic contracts, an injected object-store port, bounded PyArrow readers, a model-agnostic `TrainingPipeline`, and manifest-last artifacts. The evaluation model trains on 27 days and validates on day 28; hard-gate success selects an epoch count for a fresh 28-day production refit.

**Tech Stack:** Python 3.12, uv, Pydantic 2, PyYAML, Boto3, PyArrow, NumPy, PyTorch CPU, scikit-learn, ONNX, ONNX Runtime CPU, Typer, pytest, ruff, mypy, psutil.

---

## Plan rules

- This document is an implementation plan only. Do not create runtime code while reviewing it.
- Execute tasks in dependency order and use TDD: failing test, minimal implementation, passing test, commit.
- Finder source is naming/structure reference only. Do not copy its private implementation, binaries, internal
  endpoints, or model parameters.
- Reuse the dependency ranges and CPU-only Torch index pattern already proven in
  `doku-mono/reco-offline/pyproject.toml`; do not add a cross-repository Python import.
- Commands run from the `doku-train` repository root.

## File map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml`, `uv.lock` | CPU-only dependencies and deterministic toolchain |
| `Makefile`, `.gitignore` | unified checks and exclusion of data/checkpoints/models |
| `recommend/train/comm/model_params.py` | strict common training configuration |
| `recommend/train/comm/model_params_validator.py` | model-agnostic config loading and schema snapshot checks |
| `recommend/train/comm/model_protocol.py` | model adapter/factory protocols consumed by the pipeline |
| `recommend/train/comm/datasvr/dataset_manifest.py` | Dataset/Feature/Label contract models and canonical digests |
| `recommend/train/comm/datasvr/storage.py` | URI parser, object metadata, and `ObjectStore` protocol |
| `recommend/train/comm/datasvr/local_storage.py` | local fixture implementation of `ObjectStore` |
| `recommend/train/comm/datasvr/s3_storage.py` | bounded S3 reads and immutable writes with injected client |
| `recommend/train/comm/datasvr/parquet_dataset.py` | checksum-verified, bounded-memory Parquet batch iteration |
| `recommend/train/models/rerank/model_params.py` | strict rerank network and loss configuration |
| `recommend/train/models/rerank/data_utils.py` | temporal split, canonical completion label, and tensor batch |
| `recommend/train/models/rerank/feature_utils.py` | fitted numeric/category state and deterministic transforms |
| `recommend/train/models/rerank/rerank_model.py` | shared-bottom four-tower PyTorch module and masked loss |
| `recommend/train/models/rerank/__init__.py` | rerank adapter/factory implementing the common protocol |
| `recommend/train/comm/feature_check.py` | point-in-time, OOV, missingness, and forbidden-ID validation |
| `recommend/train/comm/checkpoint_agent.py` | atomic PyTorch checkpoint save/strict restore |
| `recommend/train/comm/metrics_utils.py` | versioned four-target loss/AUC metrics |
| `recommend/train/comm/eval/eval_model_rerank.py` | evaluation loop returning typed metric sets |
| `recommend/train/comm/eval/gates.py` | hard-gate failures and soft quality warnings |
| `recommend/train/comm/training_pipeline.py` | evaluation train, gate, refit, export state machine |
| `recommend/train/tools/onnx/gen_onnx.py` | ONNX export, schema validation, and runtime parity |
| `recommend/train/comm/artifact_utils.py` | deterministic versioning and manifest-last artifact commit |
| `recommend/train/comm/offline_evaluate.py` | explicit 24/2/2 and rolling backtest orchestration |
| `recommend/train/run.py` | root Typer composition entrypoint |
| `recommend/train/train_rerank.py` | `validate`, `train`, and `backtest` use cases |
| `recommend/train/tests/{unit,contract,smoke,performance}` | requirement-level automated evidence |

## Dependency graph

| Task | Depends on | Working result |
| --- | --- | --- |
| 1 | none | installable Finder-compatible CPU project |
| 2 | 1 | strict common/model config and checked schemas |
| 3 | 1 | immutable Manifest and local/S3 storage ports |
| 4 | 3 | bounded Parquet reader and deterministic date splits |
| 5 | 2, 3, 4 | point-in-time feature state and four-label batches |
| 6 | 2, 5 | four-target rerank model and masked loss |
| 7 | 2, 6 | strict atomic checkpoints and reproducibility metadata |
| 8 | 5, 6 | metrics and hard/soft gates |
| 9 | 4, 6, 7, 8 | model-agnostic 27/1 → gate → 28-day refit pipeline |
| 10 | 3, 6, 8, 9 | verified ONNX artifact committed manifest-last |
| 11 | 2, 3, 9, 10 | CLI and full local CPU smoke path |
| 12 | 1–11 | security/dependency checks, capacity evidence, final traceability |

### Task 1: Bootstrap the Finder-compatible CPU project

**Requirements:** R001, NFR003, NFR004
**Depends on:** none

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `Makefile`
- Create: `recommend/__init__.py`
- Create: `recommend/train/__init__.py`
- Create: `recommend/train/comm/__init__.py`
- Create: `recommend/train/models/__init__.py`
- Create: `recommend/train/offline/__init__.py`
- Create: `recommend/train/tools/__init__.py`
- Create: `recommend/train/tests/unit/test_project_layout.py`
- Generate: `uv.lock`

**Completion conditions:** Python 3.12 installs only CPU runtime dependencies; the Finder namespace imports; `make
unit` passes; generated data, checkpoints and model files are ignored.

- [ ] **Step 1: Write the failing namespace and dependency-policy test**

```python
# recommend/train/tests/unit/test_project_layout.py
from __future__ import annotations

import importlib
import tomllib
from pathlib import Path


def test_finder_compatible_namespace_is_importable() -> None:
    for module in (
        "recommend.train.comm",
        "recommend.train.models",
        "recommend.train.offline",
        "recommend.train.tools",
    ):
        assert importlib.import_module(module) is not None


def test_torch_uses_explicit_cpu_index() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())
    assert project["tool"]["uv"]["sources"]["torch"] == [
        {"index": "pytorch-cpu", "marker": "sys_platform == 'linux'"}
    ]
    assert project["tool"]["uv"]["index"][0]["explicit"] is True
```

- [ ] **Step 2: Run the test and observe the missing project/package failure**

Run: `python3.12 -m pytest recommend/train/tests/unit/test_project_layout.py -v`
Expected: FAIL because `pyproject.toml` and the `recommend` package do not exist.

- [ ] **Step 3: Add the minimal package and CPU-only toolchain**

```toml
# pyproject.toml
[project]
name = "doku-train"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "boto3>=1.40,<2",
  "numpy>=2.5,<3",
  "onnx>=1.20,<2",
  "onnxruntime>=1.23,<2",
  "onnxscript>=0.5,<1",
  "pyarrow>=25,<26",
  "psutil>=7,<8",
  "pydantic>=2.11,<3",
  "pyyaml>=6,<7",
  "scikit-learn>=1.8,<2",
  "torch>=2.13,<3",
  "typer>=0.16,<1",
]

[dependency-groups]
dev = [
  "mypy>=1.17,<2",
  "pytest>=8.4,<9",
  "pytest-cov>=6.2,<7",
  "ruff>=0.12,<1",
]

[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["recommend"]

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.pytest.ini_options]
testpaths = ["recommend/train/tests"]
markers = ["performance: records capacity evidence without declaring an SLA"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]

[tool.mypy]
python_version = "3.12"
strict = true
packages = ["recommend"]
exclude = ["recommend/train/tests"]
ignore_missing_imports = true
```

```python
# recommend/__init__.py and each package __init__.py
"""Doku recommendation training package."""
```

```makefile
# Makefile
.PHONY: format lint typecheck unit test check
format:
	uv run ruff format --check .
lint:
	uv run ruff check .
typecheck:
	uv run mypy recommend
unit:
	uv run pytest recommend/train/tests/unit -q
test:
	uv run pytest recommend/train/tests -q -m "not performance"
check: format lint typecheck test
```

```gitignore
# .gitignore
.venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.py[cod]
data/
runs/
artifacts/
*.pt
*.onnx
```

- [ ] **Step 4: Lock dependencies and run the focused test**

Run: `uv lock && uv sync --group dev && make unit`
Expected: dependency lock succeeds and `test_project_layout.py` reports 2 passed.

- [ ] **Step 5: Commit the bootstrap**

```bash
git add pyproject.toml uv.lock Makefile .gitignore recommend
git commit -m "build: bootstrap cpu training package"
```

### Task 2: Add strict common and rerank configuration

**Requirements:** R002, R006, NFR002
**Depends on:** Task 1

**Files:**
- Create: `recommend/train/comm/model_params.py`
- Create: `recommend/train/comm/model_params_validator.py`
- Create: `recommend/train/comm/config.schema.yml`
- Create: `recommend/train/models/rerank/__init__.py`
- Create: `recommend/train/models/rerank/model_params.py`
- Create: `recommend/train/models/rerank/config.schema.yml`
- Create: `recommend/train/tests/unit/test_model_params.py`

**Completion conditions:** unknown keys and non-CPU strategies fail validation; required resource/split/model fields
are explicit; checked schema files exactly match the Pydantic-generated schemas.

- [ ] **Step 1: Write failing strict-config tests**

```python
# recommend/train/tests/unit/test_model_params.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from recommend.train.comm.model_params_validator import validate_config_payload
from recommend.train.models.rerank.model_params import RerankModelParams


def valid_payload() -> dict[str, object]:
    return {
        "model_name": "rerank",
        "checkpoint_dir": "runs/test",
        "seed": 7,
        "batch_size": 128,
        "max_epochs": 5,
        "early_stopping_patience": 2,
        "bad_row_policy": "fail",
        "execution": {
            "strategy": "single_process_cpu",
            "intra_op_threads": 8,
            "inter_op_threads": 1,
            "reader_workers": 0,
            "prefetch_batches": 1,
            "memory_limit_bytes": 8_589_934_592,
        },
        "split": {"train_days": 27, "validation_days": 1, "refit_days": 28},
        "optimizer": {"name": "adamw", "learning_rate": 0.001, "weight_decay": 0.0001},
        "metrics": {"minimum_valid_rows": 10, "auc_warning_floor": 0.5},
        "export": {"onnx_opset": 18, "absolute_tolerance": 1e-5},
        "model_params": {
            "numeric_input_dim": 6,
            "categorical_cardinalities": [8, 16],
            "embedding_dim": 4,
            "shared_hidden_dims": [32, 16],
            "tower_hidden_dim": 8,
            "activation": "relu",
            "dropout": 0.1,
            "loss_weights": {
                "effective_watch": 1.0,
                "completion": 1.0,
                "non_fast_swipe": 1.0,
                "immersive_click": 1.0,
            },
        },
    }


def test_unknown_common_key_is_rejected() -> None:
    payload = valid_payload() | {"cuda": True}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_config_payload(payload, RerankModelParams)


def test_only_single_process_cpu_is_accepted() -> None:
    payload = valid_payload()
    payload["execution"] = dict(payload["execution"], strategy="ddp")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="single_process_cpu"):
        validate_config_payload(payload, RerankModelParams)


def test_valid_payload_returns_typed_model_params() -> None:
    config = validate_config_payload(valid_payload(), RerankModelParams)
    assert config.common.split.refit_days == 28
    assert config.model.shared_hidden_dims == (32, 16)
    assert config.model.activation == "relu"
```

- [ ] **Step 2: Run the strict-config tests and verify import failure**

Run: `uv run pytest recommend/train/tests/unit/test_model_params.py -v`
Expected: FAIL because `model_params.py` and `model_params_validator.py` are absent.

- [ ] **Step 3: Implement strict common and model configuration**

```python
# recommend/train/comm/model_params.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionConfig(StrictModel):
    strategy: Literal["single_process_cpu"]
    intra_op_threads: int = Field(ge=1)
    inter_op_threads: int = Field(ge=1)
    reader_workers: Literal[0]
    prefetch_batches: Literal[1]
    memory_limit_bytes: int = Field(ge=1)


class SplitConfig(StrictModel):
    train_days: Literal[27] = 27
    validation_days: Literal[1] = 1
    refit_days: Literal[28] = 28


class OptimizerConfig(StrictModel):
    name: Literal["adamw"] = "adamw"
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)


class MetricsConfig(StrictModel):
    minimum_valid_rows: int = Field(ge=2)
    auc_warning_floor: float = Field(ge=0.0, le=1.0)


class ExportConfig(StrictModel):
    onnx_opset: Literal[18]
    absolute_tolerance: float = Field(gt=0.0, le=1e-2)


class TrainConfig(StrictModel):
    model_name: Literal["rerank"]
    checkpoint_dir: Path
    seed: int = Field(ge=0)
    batch_size: int = Field(ge=1)
    max_epochs: int = Field(ge=1)
    early_stopping_patience: int = Field(ge=1)
    bad_row_policy: Literal["fail"]
    execution: ExecutionConfig
    split: SplitConfig
    optimizer: OptimizerConfig
    metrics: MetricsConfig
    export: ExportConfig
    model_params: dict[str, object]
```

```python
# recommend/train/models/rerank/model_params.py
from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from recommend.train.comm.model_params import StrictModel

TARGETS = ("effective_watch", "completion", "non_fast_swipe", "immersive_click")


class RerankModelParams(StrictModel):
    numeric_input_dim: int = Field(ge=1)
    categorical_cardinalities: tuple[int, ...]
    embedding_dim: int = Field(ge=1, le=256)
    shared_hidden_dims: tuple[int, ...]
    tower_hidden_dim: int = Field(ge=1, le=4096)
    activation: Literal["relu", "silu"]
    dropout: float = Field(ge=0.0, lt=1.0)
    loss_weights: dict[str, float]

    @field_validator("categorical_cardinalities")
    @classmethod
    def validate_cardinalities(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(size < 2 for size in value):
            raise ValueError("categorical cardinality must reserve missing and OOV")
        return value

    @field_validator("shared_hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or len(value) > 5 or any(size < 1 or size > 4096 for size in value):
            raise ValueError("shared_hidden_dims must contain 1-5 values in [1, 4096]")
        return value

    @field_validator("loss_weights")
    @classmethod
    def validate_loss_weights(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != set(TARGETS) or any(weight <= 0.0 for weight in value.values()):
            raise ValueError("loss_weights must contain four positive target weights")
        return value
```

```python
# recommend/train/comm/model_params_validator.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import yaml
from pydantic import BaseModel

from recommend.train.comm.model_params import TrainConfig

ModelParamsT = TypeVar("ModelParamsT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ValidatedTrainConfig(Generic[ModelParamsT]):
    common: TrainConfig
    model: ModelParamsT


def validate_config_payload(
    payload: dict[str, object], model_type: type[ModelParamsT]
) -> ValidatedTrainConfig[ModelParamsT]:
    common = TrainConfig.model_validate(payload)
    model = model_type.model_validate(common.model_params)
    return ValidatedTrainConfig(common=common, model=model)


def write_schema(model_type: type[BaseModel], path: Path) -> None:
    path.write_text(yaml.safe_dump(model_type.model_json_schema(), sort_keys=True))


def schema_matches(model_type: type[BaseModel], path: Path) -> bool:
    return yaml.safe_load(path.read_text()) == model_type.model_json_schema()
```

- [ ] **Step 4: Generate schema snapshots and run the focused tests**

Run:
```bash
uv run python -c 'from pathlib import Path; from recommend.train.comm.model_params import TrainConfig; from recommend.train.comm.model_params_validator import write_schema; from recommend.train.models.rerank.model_params import RerankModelParams; write_schema(TrainConfig, Path("recommend/train/comm/config.schema.yml")); write_schema(RerankModelParams, Path("recommend/train/models/rerank/config.schema.yml"))'
uv run pytest recommend/train/tests/unit/test_model_params.py -v
```
Expected: schema files are generated and 3 tests pass.

- [ ] **Step 5: Commit strict configuration**

```bash
git add recommend/train/comm recommend/train/models/rerank recommend/train/tests/unit/test_model_params.py
git commit -m "feat: add strict training configuration"
```

### Task 3: Define immutable Manifest and object-store ports

**Requirements:** R003, NFR002, NFR003, NFR004
**Depends on:** Task 1

**Files:**
- Create: `recommend/train/comm/datasvr/__init__.py`
- Create: `recommend/train/comm/datasvr/dataset_manifest.py`
- Create: `recommend/train/comm/datasvr/storage.py`
- Create: `recommend/train/comm/datasvr/local_storage.py`
- Create: `recommend/train/comm/datasvr/s3_storage.py`
- Create: `recommend/train/tests/contract/test_dataset_manifest.py`
- Create: `recommend/train/tests/unit/test_s3_storage.py`

**Completion conditions:** canonical digest mismatch, aggregate row mismatch, non-UTC timestamps, non-canonical URI,
bucket/prefix escape and overwrites all fail closed; tests use injected clients and local paths only.

- [ ] **Step 1: Write failing Manifest and S3-boundary tests**

```python
# recommend/train/tests/contract/test_dataset_manifest.py
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from recommend.train.comm.datasvr.dataset_manifest import DatasetManifest, Partition


def test_manifest_rejects_row_count_or_digest_mismatch() -> None:
    start = datetime(2026, 8, 1, tzinfo=UTC)
    partition = Partition(
        uri="s3://doku-train/input/day=2026-08-01/part-000.parquet",
        event_date=date(2026, 8, 1),
        min_event_time=start,
        max_event_time=start + timedelta(hours=23),
        row_count=10,
        size_bytes=100,
        sha256="a" * 64,
    )
    payload = DatasetManifest.unsigned_payload(partitions=(partition,), row_count=10)
    payload["content_sha256"] = DatasetManifest.digest_payload(payload)
    assert DatasetManifest.model_validate(payload).row_count == 10
    payload["row_count"] = 11
    with pytest.raises(ValidationError, match="row_count"):
        DatasetManifest.model_validate(payload)
    payload["row_count"] = 10
    payload["content_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="content_sha256"):
        DatasetManifest.model_validate(payload)


def test_manifest_rejects_non_utc_or_immature_partitions() -> None:
    local_offset = datetime.fromisoformat("2026-08-01T00:00:00+08:00")
    with pytest.raises(ValidationError, match="UTC"):
        Partition(
            uri="s3://doku-train/input/day=2026-08-01/part-000.parquet",
            event_date=date(2026, 8, 1),
            min_event_time=local_offset,
            max_event_time=local_offset + timedelta(hours=1),
            row_count=10,
            size_bytes=100,
            sha256="a" * 64,
        )

    start = datetime(2026, 8, 1, tzinfo=UTC)
    partition = Partition(
        uri="s3://doku-train/input/day=2026-08-01/part-000.parquet",
        event_date=date(2026, 8, 1),
        min_event_time=start,
        max_event_time=start + timedelta(hours=23),
        row_count=10,
        size_bytes=100,
        sha256="a" * 64,
    )
    payload = DatasetManifest.unsigned_payload(partitions=(partition,), row_count=10)
    payload["as_of_ms"] = int(datetime(2026, 8, 2, tzinfo=UTC).timestamp() * 1000)
    payload["content_sha256"] = DatasetManifest.digest_payload(payload)
    with pytest.raises(ValidationError, match="LABEL_NOT_MATURE"):
        DatasetManifest.model_validate(payload)
```

```python
# recommend/train/tests/unit/test_s3_storage.py
from __future__ import annotations

import pytest

from recommend.train.comm.datasvr.s3_storage import S3Storage


class FakeClient:
    def put_object(self, **_: object) -> None:
        raise AssertionError("write must not occur for an invalid location")


def test_s3_storage_rejects_bucket_and_prefix_escape() -> None:
    store = S3Storage(client=FakeClient(), allowed_bucket="doku-train", allowed_prefix="models/")
    with pytest.raises(ValueError, match="allowed bucket"):
        store.put_bytes_if_absent("s3://other/models/a", b"x")
    with pytest.raises(ValueError, match="canonical"):
        store.put_bytes_if_absent("s3://doku-train/models/../secret", b"x")


def test_s3_storage_uses_create_only_write() -> None:
    class RecordingClient:
        request: dict[str, object] | None = None

        def put_object(self, **request: object) -> None:
            self.request = request

    client = RecordingClient()
    store = S3Storage(client=client, allowed_bucket="doku-train", allowed_prefix="models/")
    store.put_bytes_if_absent("s3://doku-train/models/rerank/model.onnx", b"model")
    assert client.request is not None
    assert client.request["IfNoneMatch"] == "*"
```

- [ ] **Step 2: Run both tests and verify missing-module failure**

Run: `uv run pytest recommend/train/tests/contract/test_dataset_manifest.py recommend/train/tests/unit/test_s3_storage.py -v`
Expected: FAIL because the Manifest and storage modules do not exist.

- [ ] **Step 3: Implement the strict Manifest and storage interfaces**

```python
# recommend/train/comm/datasvr/dataset_manifest.py
from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from typing import Literal

from pydantic import Field, model_validator

from recommend.train.comm.datasvr.storage import parse_storage_uri
from recommend.train.comm.model_params import StrictModel


class Partition(StrictModel):
    uri: str
    event_date: date
    min_event_time: datetime
    max_event_time: datetime
    row_count: int = Field(ge=1)
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_time_range(self) -> "Partition":
        if self.min_event_time.tzinfo is None or self.max_event_time.tzinfo is None:
            raise ValueError("partition times must be timezone-aware")
        if self.min_event_time.utcoffset() != timedelta(0) or self.max_event_time.utcoffset() != timedelta(0):
            raise ValueError("partition times must use UTC")
        parsed = parse_storage_uri(self.uri)
        if parsed.scheme not in {"s3", "file"}:
            raise ValueError("partition URI must use an allowed storage scheme")
        if self.min_event_time.astimezone(UTC).date() != self.event_date:
            raise ValueError("min_event_time must match event_date")
        if self.max_event_time.astimezone(UTC).date() != self.event_date:
            raise ValueError("max_event_time must match event_date")
        if self.max_event_time < self.min_event_time:
            raise ValueError("partition time range is invalid")
        return self


class DatasetManifest(StrictModel):
    manifest_version: Literal["1"]
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    feature_schema_version: str = Field(min_length=1)
    feature_schema_uri: str
    feature_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_definition_version: str = Field(min_length=1)
    label_definition_uri: str
    label_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    as_of_ms: int = Field(ge=0)
    label_maturity_hours: int = Field(ge=24)
    partitions: tuple[Partition, ...]
    row_count: int = Field(ge=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @staticmethod
    def unsigned_payload(*, partitions: tuple[Partition, ...], row_count: int) -> dict[str, object]:
        return {
            "manifest_version": "1",
            "dataset_id": "rerank-exposures",
            "dataset_version": "2026-08-29",
            "feature_schema_version": "1",
            "feature_schema_uri": "s3://doku-train/input/contracts/features-v1.json",
            "feature_schema_sha256": "b" * 64,
            "label_definition_version": "1",
            "label_definition_uri": "s3://doku-train/input/contracts/labels-v1.json",
            "label_definition_sha256": "c" * 64,
            "as_of_ms": 1_788_048_000_000,
            "label_maturity_hours": 24,
            "partitions": [item.model_dump(mode="json") for item in partitions],
            "row_count": row_count,
        }

    @staticmethod
    def digest_payload(payload: dict[str, object]) -> str:
        unsigned = {key: value for key, value in payload.items() if key != "content_sha256"}
        body = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(body).hexdigest()

    @model_validator(mode="after")
    def validate_integrity(self) -> "DatasetManifest":
        for uri in (self.feature_schema_uri, self.label_definition_uri):
            if parse_storage_uri(uri).scheme not in {"s3", "file"}:
                raise ValueError("contract URI must use an allowed storage scheme")
        if sum(item.row_count for item in self.partitions) != self.row_count:
            raise ValueError("DATA_INTEGRITY_MISMATCH: row_count does not match partitions")
        as_of = datetime.fromtimestamp(self.as_of_ms / 1000, tz=UTC)
        mature_before = as_of - timedelta(hours=self.label_maturity_hours)
        if any(item.max_event_time.astimezone(UTC) > mature_before for item in self.partitions):
            raise ValueError("LABEL_NOT_MATURE: partition exceeds maturity cutoff")
        payload = self.model_dump(mode="json")
        if self.digest_payload(payload) != self.content_sha256:
            raise ValueError("DATA_INTEGRITY_MISMATCH: content_sha256 mismatch")
        return self
```

```python
# recommend/train/comm/datasvr/storage.py
from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Protocol
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class ObjectHead:
    size_bytes: int
    sha256: str | None


@dataclass(frozen=True, slots=True)
class StorageUri:
    scheme: str
    bucket: str
    key: str


def parse_storage_uri(uri: str) -> StorageUri:
    parsed = urlparse(uri)
    if parsed.scheme not in {"s3", "file"} or parsed.query or parsed.fragment:
        raise ValueError("storage URI must be canonical s3:// or file://")
    key = parsed.path.lstrip("/")
    if not key or "\\" in key or any(part in {"", ".", ".."} for part in key.split("/")):
        raise ValueError("storage URI key must be canonical")
    return StorageUri(parsed.scheme, parsed.netloc, key)


class ObjectStore(Protocol):
    def get_bytes(self, uri: str, *, max_bytes: int) -> bytes: ...
    def download_to(self, uri: str, sink: BinaryIO) -> ObjectHead: ...
    def head(self, uri: str) -> ObjectHead: ...
    def put_bytes_if_absent(self, uri: str, body: bytes) -> ObjectHead: ...
```

```python
# recommend/train/comm/datasvr/s3_storage.py
from __future__ import annotations

import base64
import hashlib
from typing import Any, BinaryIO

from recommend.train.comm.datasvr.storage import ObjectHead, parse_storage_uri


class S3Storage:
    def __init__(self, *, client: Any, allowed_bucket: str, allowed_prefix: str) -> None:
        if not allowed_prefix.endswith("/"):
            raise ValueError("allowed prefix must end with /")
        self._client = client
        self._allowed_bucket = allowed_bucket
        self._allowed_prefix = allowed_prefix

    def _location(self, uri: str) -> tuple[str, str]:
        parsed = parse_storage_uri(uri)
        if parsed.scheme != "s3" or parsed.bucket != self._allowed_bucket:
            raise ValueError("URI is outside allowed bucket")
        if not parsed.key.startswith(self._allowed_prefix):
            raise ValueError("URI is outside canonical allowed prefix")
        return parsed.bucket, parsed.key

    def get_bytes(self, uri: str, *, max_bytes: int) -> bytes:
        bucket, key = self._location(uri)
        response = self._client.get_object(Bucket=bucket, Key=key)
        body = response["Body"].read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("object exceeds configured read limit")
        return body

    def download_to(self, uri: str, sink: BinaryIO) -> ObjectHead:
        bucket, key = self._location(uri)
        self._client.download_fileobj(bucket, key, sink)
        return self.head(uri)

    def head(self, uri: str) -> ObjectHead:
        bucket, key = self._location(uri)
        response = self._client.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
        checksum = response.get("ChecksumSHA256")
        decoded = base64.b64decode(checksum).hex() if isinstance(checksum, str) else None
        return ObjectHead(size_bytes=int(response["ContentLength"]), sha256=decoded)

    def put_bytes_if_absent(self, uri: str, body: bytes) -> ObjectHead:
        bucket, key = self._location(uri)
        digest = hashlib.sha256(body).digest()
        self._client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ChecksumSHA256=base64.b64encode(digest).decode(),
            IfNoneMatch="*",
        )
        return ObjectHead(size_bytes=len(body), sha256=digest.hex())
```

Implement `LocalStorage` with the same four methods, require `file://` URIs below an injected root, use `xb` for
immutable writes, and compute SHA-256 while streaming `download_to`.

```python
# recommend/train/comm/datasvr/local_storage.py
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import BinaryIO

from recommend.train.comm.datasvr.storage import ObjectHead, parse_storage_uri


class LocalStorage:
    def __init__(self, *, root: Path) -> None:
        self._root = root.resolve()

    def _path(self, uri: str) -> Path:
        parsed = parse_storage_uri(uri)
        if parsed.scheme != "file":
            raise ValueError("LocalStorage requires file:// URI")
        path = Path("/" + parsed.key).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("file URI is outside configured root")
        return path

    def get_bytes(self, uri: str, *, max_bytes: int) -> bytes:
        path = self._path(uri)
        with path.open("rb") as handle:
            body = handle.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("object exceeds configured read limit")
        return body

    def download_to(self, uri: str, sink: BinaryIO) -> ObjectHead:
        path = self._path(uri)
        with path.open("rb") as source:
            shutil.copyfileobj(source, sink, length=1024 * 1024)
        return self.head(uri)

    def head(self, uri: str) -> ObjectHead:
        path = self._path(uri)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        return ObjectHead(size_bytes=path.stat().st_size, sha256=digest)

    def put_bytes_if_absent(self, uri: str, body: bytes) -> ObjectHead:
        path = self._path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(body)
        return self.head(uri)
```

- [ ] **Step 4: Run Manifest and storage tests**

Run: `uv run pytest recommend/train/tests/contract/test_dataset_manifest.py recommend/train/tests/unit/test_s3_storage.py -v`
Expected: 4 tests pass; no AWS request is made.

- [ ] **Step 5: Commit contracts and storage boundaries**

```bash
git add recommend/train/comm/datasvr recommend/train/tests/contract recommend/train/tests/unit/test_s3_storage.py
git commit -m "feat: add immutable dataset manifest"
```

### Task 4: Stream Parquet and build deterministic temporal splits

**Requirements:** R004, NFR001, NFR002
**Depends on:** Task 3

**Files:**
- Create: `recommend/train/comm/datasvr/parquet_dataset.py`
- Create: `recommend/train/models/rerank/data_utils.py`
- Create: `recommend/train/tests/unit/test_parquet_dataset.py`
- Create: `recommend/train/tests/unit/test_data_utils.py`

**Completion conditions:** only one shard is materialized at a time; batches are bounded; checksum and row totals
are checked; 28 dates split exactly 27/1/refit-all; backtest is exactly 24/2/2; no sample is dropped.

- [ ] **Step 1: Write failing bounded-reader and split tests**

```python
# recommend/train/tests/unit/test_data_utils.py
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from recommend.train.comm.datasvr.dataset_manifest import Partition
from recommend.train.models.rerank.data_utils import build_backtest_split, build_daily_split


def partitions() -> tuple[Partition, ...]:
    first = date(2026, 8, 1)
    return tuple(
        Partition(
            uri=f"file://fixture/day={first + timedelta(days=index)}/part.parquet",
            event_date=first + timedelta(days=index),
            min_event_time=datetime.combine(first + timedelta(days=index), datetime.min.time(), UTC),
            max_event_time=datetime.combine(first + timedelta(days=index), datetime.max.time(), UTC),
            row_count=index + 1,
            size_bytes=100,
            sha256=f"{index:064x}",
        )
        for index in range(28)
    )


def test_daily_split_preserves_every_row() -> None:
    split = build_daily_split(partitions())
    assert len({item.event_date for item in split.train}) == 27
    assert len({item.event_date for item in split.validation}) == 1
    assert split.refit == partitions()
    assert sum(item.row_count for item in split.refit) == sum(range(1, 29))


def test_backtest_is_24_2_2() -> None:
    split = build_backtest_split(partitions())
    assert [len(part) for part in (split.train, split.validation, split.test)] == [24, 2, 2]


def test_non_contiguous_dates_are_rejected() -> None:
    values = list(partitions())
    skipped = values[-1].event_date + timedelta(days=2)
    values[-1] = values[-1].model_copy(update={
        "event_date": skipped,
        "min_event_time": datetime.combine(skipped, datetime.min.time(), UTC),
        "max_event_time": datetime.combine(skipped, datetime.max.time(), UTC),
    })
    try:
        build_daily_split(tuple(values))
    except ValueError as error:
        assert "SPLIT_INVALID" in str(error)
    else:
        raise AssertionError("non-contiguous dates were accepted")
```

```python
# recommend/train/tests/unit/test_parquet_dataset.py
from __future__ import annotations

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from recommend.train.comm.datasvr.local_storage import LocalStorage
from recommend.train.comm.datasvr.parquet_dataset import ParquetDataset


def test_reader_yields_bounded_batches_and_exact_rows(tmp_path) -> None:
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"x": list(range(10))}), path)
    partition = type("P", (), {
        "uri": path.as_uri(),
        "row_count": 10,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })()
    reader = ParquetDataset(LocalStorage(root=tmp_path), spool_limit_bytes=1024)
    batches = list(reader.iter_batches((partition,), columns=("x",), batch_size=4))
    assert [batch.num_rows for batch in batches] == [4, 4, 2]


def test_reader_rejects_partition_checksum_mismatch(tmp_path) -> None:
    path = tmp_path / "part.parquet"
    pq.write_table(pa.table({"x": [1]}), path)
    partition = type("P", (), {
        "uri": path.as_uri(),
        "row_count": 1,
        "sha256": "0" * 64,
    })()
    reader = ParquetDataset(LocalStorage(root=tmp_path), spool_limit_bytes=1024)
    with pytest.raises(ValueError, match="DATA_INTEGRITY_MISMATCH"):
        list(reader.iter_batches((partition,), columns=("x",), batch_size=1))
```

- [ ] **Step 2: Run focused tests and verify missing readers/splits**

Run: `uv run pytest recommend/train/tests/unit/test_parquet_dataset.py recommend/train/tests/unit/test_data_utils.py -v`
Expected: FAIL because `ParquetDataset`, `build_daily_split`, and `build_backtest_split` are absent.

- [ ] **Step 3: Implement bounded reading and exact date splits**

```python
# recommend/train/models/rerank/data_utils.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import torch

from recommend.train.comm.datasvr.dataset_manifest import Partition

TARGETS = ("effective_watch", "completion", "non_fast_swipe", "immersive_click")


@dataclass(frozen=True, slots=True)
class DailySplit:
    train: tuple[Partition, ...]
    validation: tuple[Partition, ...]
    refit: tuple[Partition, ...]


@dataclass(frozen=True, slots=True)
class BacktestSplit:
    train: tuple[Partition, ...]
    validation: tuple[Partition, ...]
    test: tuple[Partition, ...]


@dataclass(frozen=True, slots=True)
class RerankBatch:
    numeric: torch.Tensor
    categorical: torch.Tensor
    labels: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]


def _group_days(partitions: tuple[Partition, ...]) -> list[tuple[Partition, ...]]:
    ordered = sorted(partitions, key=lambda item: (item.event_date, item.uri))
    days = sorted({item.event_date for item in ordered})
    if len(days) != 28:
        raise ValueError("SPLIT_INVALID: exactly 28 mature event dates are required")
    if days != [days[0] + timedelta(days=index) for index in range(28)]:
        raise ValueError("SPLIT_INVALID: mature event dates must be contiguous")
    return [tuple(item for item in ordered if item.event_date == day) for day in days]


def build_daily_split(partitions: tuple[Partition, ...]) -> DailySplit:
    days = _group_days(partitions)
    return DailySplit(
        train=tuple(item for day in days[:27] for item in day),
        validation=days[27],
        refit=tuple(item for day in days for item in day),
    )


def build_backtest_split(partitions: tuple[Partition, ...]) -> BacktestSplit:
    days = _group_days(partitions)
    return BacktestSplit(
        train=tuple(item for day in days[:24] for item in day),
        validation=tuple(item for day in days[24:26] for item in day),
        test=tuple(item for day in days[26:] for item in day),
    )


def derive_completion(watch_duration_ms: torch.Tensor, content_duration_ms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = content_duration_ms > 0
    label = valid & (watch_duration_ms / content_duration_ms.clamp_min(1) >= 0.95)
    return label.to(torch.float32), valid
```

```python
# recommend/train/comm/datasvr/parquet_dataset.py
from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from recommend.train.comm.datasvr.dataset_manifest import Partition
from recommend.train.comm.datasvr.storage import ObjectStore


class ParquetDataset:
    def __init__(self, store: ObjectStore, *, spool_limit_bytes: int) -> None:
        self._store = store
        self._spool_limit_bytes = spool_limit_bytes

    def iter_batches(
        self,
        partitions: Sequence[Partition],
        *,
        columns: tuple[str, ...],
        batch_size: int,
    ) -> Iterator[pa.RecordBatch]:
        for partition in partitions:
            with tempfile.SpooledTemporaryFile(max_size=self._spool_limit_bytes) as handle:
                self._store.download_to(partition.uri, handle)
                handle.seek(0)
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
                if digest != partition.sha256:
                    raise ValueError("DATA_INTEGRITY_MISMATCH: partition checksum mismatch")
                handle.seek(0)
                rows = 0
                for batch in pq.ParquetFile(handle).iter_batches(
                    batch_size=batch_size, columns=list(columns)
                ):
                    rows += batch.num_rows
                    yield batch
                if rows != partition.row_count:
                    raise ValueError("DATA_INTEGRITY_MISMATCH: partition row_count mismatch")
```

- [ ] **Step 4: Run reader and split tests**

Run: `uv run pytest recommend/train/tests/unit/test_parquet_dataset.py recommend/train/tests/unit/test_data_utils.py -v`
Expected: 5 tests pass and the observed batch sizes are 4, 4, 2.

- [ ] **Step 5: Commit the bounded data path**

```bash
git add recommend/train/comm/datasvr/parquet_dataset.py recommend/train/models/rerank/data_utils.py recommend/train/tests/unit
git commit -m "feat: stream parquet training partitions"
```

### Task 5: Fit and freeze point-in-time feature state

**Requirements:** R004, R005, R006
**Depends on:** Tasks 2, 3, 4

**Files:**
- Create: `recommend/train/comm/feature_check.py`
- Create: `recommend/train/models/rerank/feature_utils.py`
- Modify: `recommend/train/models/rerank/data_utils.py`
- Create: `recommend/train/tests/unit/test_feature_check.py`
- Create: `recommend/train/tests/unit/test_feature_utils.py`
- Create: `recommend/train/tests/unit/test_labels.py`

**Completion conditions:** only training batches fit state; numeric missingness is distinct from zero; missing and OOV
categories use indices 0 and 1; seven-day aggregate metadata is mandatory; raw `user_id` is rejected; completion
is positive at exactly 95%; all four target masks survive batching.

- [ ] **Step 1: Write failing feature and label boundary tests**

```python
# recommend/train/tests/unit/test_feature_check.py
from __future__ import annotations

import pytest

from recommend.train.comm.feature_check import FeatureSchema, FeatureSpec, validate_feature_schema


def test_raw_user_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="user_id"):
        validate_feature_schema(
            FeatureSchema(version="1", features=(FeatureSpec(
                name="user_id", source_column="user_id", kind="categorical", window_days=None
            ),))
        )


def test_aggregate_requires_seven_day_event_time_left_closed_window() -> None:
    with pytest.raises(ValueError, match="exclude the current and future"):
        validate_feature_schema(
            FeatureSchema(version="1", features=(FeatureSpec(
                name="user_watch_count_7d",
                source_column="user_watch_count_7d",
                kind="numeric",
                window_days=7,
                as_of_column="event_time",
                window_closed=None,
            ),))
        )
    with pytest.raises(ValueError, match="7-day"):
        validate_feature_schema(
            FeatureSchema(version="1", features=(FeatureSpec(
                name="user_watch_count_3d",
                source_column="user_watch_count_3d",
                kind="numeric",
                window_days=3,
            ),))
        )
```

```python
# recommend/train/tests/unit/test_feature_utils.py
from __future__ import annotations

import pyarrow as pa

from recommend.train.comm.feature_check import FeatureSchema, FeatureSpec
from recommend.train.models.rerank.feature_utils import FeatureStateBuilder, transform_features


def test_missing_zero_and_oov_are_distinct() -> None:
    schema = FeatureSchema(version="1", features=(
        FeatureSpec(
            name="watch_count_7d",
            source_column="watch_count_7d",
            kind="numeric",
            window_days=7,
            as_of_column="event_time",
            window_closed="left",
        ),
        FeatureSpec(name="language", source_column="language", kind="categorical", window_days=None),
    ))
    builder = FeatureStateBuilder(schema=schema, maximum_vocabulary_size=8)
    builder.update(pa.record_batch([[0.0, None, 2.0], ["en", "zh", "en"]], names=["watch_count_7d", "language"]))
    state = builder.finalize()
    numeric, categorical = transform_features(
        pa.record_batch([[0.0, None], ["en", "ja"]], names=["watch_count_7d", "language"]),
        state,
    )
    assert numeric[:, 1].tolist() == [0.0, 1.0]
    assert categorical[:, 0].tolist() == [state.categorical["language"].vocabulary["en"], 1]
```

```python
# recommend/train/tests/unit/test_labels.py
from __future__ import annotations

import torch

from recommend.train.models.rerank.data_utils import build_labels, derive_completion


def test_completion_boundary_is_inclusive_and_invalid_duration_is_masked() -> None:
    label, valid = derive_completion(
        torch.tensor([949.0, 950.0, 951.0, 10.0]),
        torch.tensor([1000.0, 1000.0, 1000.0, 0.0]),
    )
    assert label.tolist() == [0.0, 1.0, 1.0, 0.0]
    assert valid.tolist() == [True, True, True, False]


def test_invalid_binary_label_is_counted_and_rejected() -> None:
    columns = {
        "watch_duration_ms": torch.tensor([100.0]),
        "content_duration_ms": torch.tensor([1000.0]),
        "effective_watch": torch.tensor([2]),
        "effective_watch_valid": torch.tensor([True]),
        "non_fast_swipe": torch.tensor([0]),
        "non_fast_swipe_valid": torch.tensor([True]),
        "immersive_click": torch.tensor([1]),
        "immersive_click_valid": torch.tensor([True]),
    }
    try:
        build_labels(columns)
    except ValueError as error:
        assert "bad_row_count=1" in str(error)
    else:
        raise AssertionError("invalid binary label was accepted")
```

- [ ] **Step 2: Run feature/label tests and verify missing-module failures**

Run: `uv run pytest recommend/train/tests/unit/test_feature_check.py recommend/train/tests/unit/test_feature_utils.py recommend/train/tests/unit/test_labels.py -v`
Expected: FAIL because feature contract/state code is absent.

- [ ] **Step 3: Implement strict feature contracts and incremental fitted state**

```python
# recommend/train/comm/feature_check.py
from __future__ import annotations

from typing import Literal

from recommend.train.comm.model_params import StrictModel


class FeatureSpec(StrictModel):
    name: str
    source_column: str
    kind: Literal["numeric", "categorical"]
    window_days: int | None
    as_of_column: str | None = None
    window_closed: Literal["left"] | None = None


class FeatureSchema(StrictModel):
    version: str
    features: tuple[FeatureSpec, ...]


class LabelDefinition(StrictModel):
    version: str
    targets: tuple[
        Literal["effective_watch", "completion", "non_fast_swipe", "immersive_click"], ...
    ]
    completion_expression: Literal[
        "content_duration_ms > 0 and watch_duration_ms / content_duration_ms >= 0.95"
    ]


def validate_feature_schema(schema: FeatureSchema) -> None:
    names = [item.name for item in schema.features]
    if len(names) != len(set(names)):
        raise ValueError("feature names must be unique")
    if "user_id" in names:
        raise ValueError("raw user_id is forbidden in the model signature")
    for feature in schema.features:
        if feature.window_days is not None and feature.window_days != 7:
            raise ValueError("aggregate features must use a point-in-time 7-day window")
        if feature.window_days is not None and (
            feature.as_of_column != "event_time" or feature.window_closed != "left"
        ):
            raise ValueError("aggregate window must exclude the current and future events")
```

```python
# recommend/train/models/rerank/feature_utils.py
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pyarrow as pa
import torch

from recommend.train.comm.feature_check import FeatureSchema, validate_feature_schema


@dataclass(frozen=True, slots=True)
class NumericState:
    mean: float
    standard_deviation: float


@dataclass(frozen=True, slots=True)
class CategoricalState:
    vocabulary: dict[str, int]
    missing_index: int = 0
    oov_index: int = 1


@dataclass(frozen=True, slots=True)
class FittedFeatureState:
    schema_version: str
    source_columns: dict[str, str]
    numeric: dict[str, NumericState]
    categorical: dict[str, CategoricalState]


class FeatureStateBuilder:
    def __init__(self, *, schema: FeatureSchema, maximum_vocabulary_size: int) -> None:
        validate_feature_schema(schema)
        self._schema = schema
        self._maximum_vocabulary_size = maximum_vocabulary_size
        self._numeric: dict[str, tuple[int, float, float]] = {}
        self._categorical: dict[str, Counter[str]] = {}

    def update(self, batch: pa.RecordBatch) -> None:
        for feature in self._schema.features:
            values = batch.column(batch.schema.get_field_index(feature.source_column)).to_pylist()
            if feature.kind == "numeric":
                count, total, square_total = self._numeric.get(feature.name, (0, 0.0, 0.0))
                valid = [float(value) for value in values if value is not None]
                self._numeric[feature.name] = (
                    count + len(valid),
                    total + sum(valid),
                    square_total + sum(value * value for value in valid),
                )
            else:
                counter = self._categorical.setdefault(feature.name, Counter())
                counter.update(str(value) for value in values if value is not None)

    def finalize(self) -> FittedFeatureState:
        numeric: dict[str, NumericState] = {}
        for name, (count, total, square_total) in self._numeric.items():
            if count == 0:
                raise ValueError(f"numeric feature {name} has no valid training values")
            mean = total / count
            variance = max(square_total / count - mean * mean, 0.0)
            numeric[name] = NumericState(mean, max(sqrt(variance), 1e-12))
        categorical = {
            name: CategoricalState({value: index + 2 for index, (value, _) in enumerate(
                counter.most_common(self._maximum_vocabulary_size)
            )})
            for name, counter in self._categorical.items()
        }
        sources = {feature.name: feature.source_column for feature in self._schema.features}
        return FittedFeatureState(self._schema.version, sources, numeric, categorical)


def transform_features(
    batch: pa.RecordBatch, state: FittedFeatureState
) -> tuple[torch.Tensor, torch.Tensor]:
    numeric_columns: list[np.ndarray] = []
    for name, fitted in state.numeric.items():
        values = batch.column(batch.schema.get_field_index(state.source_columns[name])).to_pylist()
        missing = np.asarray([value is None for value in values], dtype=np.float32)
        filled = np.asarray([fitted.mean if value is None else float(value) for value in values])
        numeric_columns.extend(((filled - fitted.mean) / fitted.standard_deviation, missing))
    categorical_columns: list[np.ndarray] = []
    for name, fitted in state.categorical.items():
        values = batch.column(batch.schema.get_field_index(state.source_columns[name])).to_pylist()
        categorical_columns.append(np.asarray([
            fitted.missing_index if value is None else fitted.vocabulary.get(str(value), fitted.oov_index)
            for value in values
        ], dtype=np.int64))
    numeric = torch.from_numpy(np.stack(numeric_columns, axis=1).astype(np.float32))
    categorical_array = (
        np.stack(categorical_columns, axis=1)
        if categorical_columns
        else np.empty((batch.num_rows, 0), dtype=np.int64)
    )
    categorical = torch.from_numpy(categorical_array)
    return numeric, categorical
```

- [ ] **Step 4: Extend `data_utils.py` to build all target labels and masks**

```python
# append to recommend/train/models/rerank/data_utils.py
def build_labels(columns: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    completion, completion_valid = derive_completion(
        columns["watch_duration_ms"], columns["content_duration_ms"]
    )
    labels = {
        "effective_watch": columns["effective_watch"].to(torch.float32),
        "completion": completion,
        "non_fast_swipe": columns["non_fast_swipe"].to(torch.float32),
        "immersive_click": columns["immersive_click"].to(torch.float32),
    }
    masks = {
        "effective_watch": columns["effective_watch_valid"].to(torch.bool),
        "completion": completion_valid,
        "non_fast_swipe": columns["non_fast_swipe_valid"].to(torch.bool),
        "immersive_click": columns["immersive_click_valid"].to(torch.bool),
    }
    bad_rows = torch.zeros_like(completion_valid)
    for target in ("effective_watch", "non_fast_swipe", "immersive_click"):
        bad_rows |= masks[target] & ~((labels[target] == 0.0) | (labels[target] == 1.0))
    bad_count = int(bad_rows.sum().item())
    if bad_count:
        raise ValueError(f"DATA_INTEGRITY_MISMATCH: bad_row_count={bad_count}")
    return labels, masks


def build_rerank_batch(batch: "pa.RecordBatch", state: "FittedFeatureState") -> RerankBatch:
    from recommend.train.models.rerank.feature_utils import transform_features

    numeric, categorical = transform_features(batch, state)
    required = (
        "watch_duration_ms",
        "content_duration_ms",
        "effective_watch",
        "effective_watch_valid",
        "non_fast_swipe",
        "non_fast_swipe_valid",
        "immersive_click",
        "immersive_click_valid",
    )
    columns = {
        name: torch.as_tensor(batch.column(batch.schema.get_field_index(name)).to_numpy())
        for name in required
    }
    labels, masks = build_labels(columns)
    return RerankBatch(numeric, categorical, labels, masks)
```

- [ ] **Step 5: Run feature and label tests**

Run: `uv run pytest recommend/train/tests/unit/test_feature_check.py recommend/train/tests/unit/test_feature_utils.py recommend/train/tests/unit/test_labels.py -v`
Expected: 5 tests pass, including the exact 0.95 boundary, OOV index 1, and fail-policy bad-row count.

- [ ] **Step 6: Commit fitted feature state**

```bash
git add recommend/train/comm/feature_check.py recommend/train/models/rerank recommend/train/tests/unit
git commit -m "feat: freeze rerank feature state"
```

### Task 6: Implement the shared-bottom four-target reranker

**Requirements:** R006
**Depends on:** Tasks 2, 5

**Files:**
- Create: `recommend/train/models/rerank/rerank_model.py`
- Modify: `recommend/train/models/rerank/__init__.py`
- Create: `recommend/train/tests/unit/test_rerank_model.py`

**Completion conditions:** output names and batch dimension are stable; category embeddings are configurable;
masked rows do not affect loss; empty masks fail; model code performs no file/network/process operations.

- [ ] **Step 1: Write failing forward and masked-loss tests**

```python
# recommend/train/tests/unit/test_rerank_model.py
from __future__ import annotations

import torch

from recommend.train.models.rerank.model_params import RerankModelParams, TARGETS
from recommend.train.models.rerank.rerank_model import RerankModel, masked_multitask_loss


def params() -> RerankModelParams:
    return RerankModelParams(
        numeric_input_dim=4,
        categorical_cardinalities=(4, 5),
        embedding_dim=3,
        shared_hidden_dims=(8, 4),
        tower_hidden_dim=3,
        activation="relu",
        dropout=0.0,
        loss_weights={target: 1.0 for target in TARGETS},
    )


def test_forward_returns_four_named_logits() -> None:
    model = RerankModel(params())
    outputs = model(torch.zeros(2, 4), torch.tensor([[0, 1], [2, 3]]))
    assert tuple(outputs) == TARGETS
    assert all(value.shape == (2,) for value in outputs.values())


def test_masked_rows_do_not_change_multitask_loss() -> None:
    logits = {target: torch.tensor([0.0, 100.0]) for target in TARGETS}
    labels = {target: torch.tensor([1.0, 0.0]) for target in TARGETS}
    masks = {target: torch.tensor([True, False]) for target in TARGETS}
    loss = masked_multitask_loss(logits, labels, masks, params().loss_weights)
    assert torch.isclose(loss, torch.tensor(0.6931472), atol=1e-6)


def test_batch_with_no_valid_target_labels_is_rejected() -> None:
    logits = {target: torch.tensor([0.0]) for target in TARGETS}
    labels = {target: torch.tensor([0.0]) for target in TARGETS}
    masks = {target: torch.tensor([False]) for target in TARGETS}
    try:
        masked_multitask_loss(logits, labels, masks, params().loss_weights)
    except ValueError as error:
        assert "no valid target labels" in str(error)
    else:
        raise AssertionError("empty masks were accepted")
```

- [ ] **Step 2: Run model tests and verify missing model failure**

Run: `uv run pytest recommend/train/tests/unit/test_rerank_model.py -v`
Expected: FAIL because `RerankModel` and `masked_multitask_loss` do not exist.

- [ ] **Step 3: Implement embeddings, shared layers, towers, and normalized masked loss**

```python
# recommend/train/models/rerank/rerank_model.py
from __future__ import annotations

from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F

from recommend.train.models.rerank.model_params import RerankModelParams, TARGETS


def _activation(name: str) -> nn.Module:
    return nn.ReLU() if name == "relu" else nn.SiLU()


def _mlp(dimensions: tuple[int, ...], dropout: float, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    for input_dim, output_dim in pairwise(dimensions):
        layers.extend((
            nn.Linear(input_dim, output_dim),
            _activation(activation),
            nn.Dropout(dropout),
        ))
    return nn.Sequential(*layers)


class RerankModel(nn.Module):
    def __init__(self, params: RerankModelParams) -> None:
        super().__init__()
        self.params = params
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, params.embedding_dim, padding_idx=0)
            for cardinality in params.categorical_cardinalities
        )
        combined = params.numeric_input_dim + len(self.embeddings) * params.embedding_dim
        self.shared = _mlp(
            (combined, *params.shared_hidden_dims), params.dropout, params.activation
        )
        shared_output = params.shared_hidden_dims[-1]
        self.towers = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(shared_output, params.tower_hidden_dim),
                _activation(params.activation),
                nn.Linear(params.tower_hidden_dim, 1),
            )
            for target in TARGETS
        })

    def forward(
        self, numeric: torch.Tensor, categorical: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        embedded = [layer(categorical[:, index]) for index, layer in enumerate(self.embeddings)]
        combined = torch.cat((numeric, *embedded), dim=1) if embedded else numeric
        shared = self.shared(combined)
        return {target: tower(shared).squeeze(1) for target, tower in self.towers.items()}


def masked_multitask_loss(
    logits: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    weights: dict[str, float],
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    active_weight = 0.0
    for target in TARGETS:
        mask = masks[target]
        if mask.any():
            terms.append(F.binary_cross_entropy_with_logits(
                logits[target][mask], labels[target][mask]
            ) * weights[target])
            active_weight += weights[target]
    if not terms:
        raise ValueError("NUMERICAL_FAILURE: batch has no valid target labels")
    return torch.stack(terms).sum() / active_weight
```

`recommend/train/models/rerank/__init__.py` must export only `RerankModel`, `RerankModelParams`,
`masked_multitask_loss`, and the adapter added in Task 9; it must not instantiate storage or read configuration.

- [ ] **Step 4: Run forward/loss tests**

Run: `uv run pytest recommend/train/tests/unit/test_rerank_model.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit the reranker**

```bash
git add recommend/train/models/rerank recommend/train/tests/unit/test_rerank_model.py
git commit -m "feat: add four-target rerank model"
```

### Task 7: Add strict atomic checkpoints and seed control

**Requirements:** R007, NFR002
**Depends on:** Tasks 2, 6

**Files:**
- Create: `recommend/train/comm/checkpoint_agent.py`
- Create: `recommend/train/tests/unit/test_checkpoint_agent.py`
- Create: `recommend/train/tests/unit/test_reproducibility.py`

**Completion conditions:** Python/NumPy/Torch seeds are set; save uses temp-file replacement; restore checks format,
config, dataset and model digests before loading; mismatch leaves the model untouched.

- [ ] **Step 1: Write failing checkpoint-lineage and seed tests**

```python
# recommend/train/tests/unit/test_checkpoint_agent.py
from __future__ import annotations

import pytest
import torch

from recommend.train.comm.checkpoint_agent import CheckpointAgent, CheckpointLineage


def test_restore_rejects_lineage_mismatch_before_loading(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    agent = CheckpointAgent(tmp_path)
    lineage = CheckpointLineage("config-a", "dataset-a", "model-a")
    path = agent.save(
        "run-1",
        model,
        optimizer,
        epoch=2,
        step=10,
        lineage=lineage,
        training_state={"best_loss": 0.4, "stale_epochs": 0},
    )
    assert torch.load(path, map_location="cpu", weights_only=False)["step"] == 10
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="lineage"):
        agent.restore("run-1", model, optimizer, CheckpointLineage("config-b", "dataset-a", "model-a"))
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
```

```python
# recommend/train/tests/unit/test_reproducibility.py
import random

import numpy as np
import torch

from recommend.train.comm.checkpoint_agent import seed_everything


def test_seed_everything_repeats_python_numpy_and_torch_values() -> None:
    seed_everything(11)
    first = (random.random(), np.random.rand(3), torch.rand(3))
    seed_everything(11)
    assert random.random() == first[0]
    assert np.array_equal(np.random.rand(3), first[1])
    assert torch.equal(torch.rand(3), first[2])
```

- [ ] **Step 2: Run checkpoint tests and verify missing module**

Run: `uv run pytest recommend/train/tests/unit/test_checkpoint_agent.py recommend/train/tests/unit/test_reproducibility.py -v`
Expected: FAIL because `checkpoint_agent.py` is absent.

- [ ] **Step 3: Implement seed control and strict atomic checkpointing**

```python
# recommend/train/comm/checkpoint_agent.py
from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class CheckpointLineage:
    config_digest: str
    dataset_manifest_digest: str
    model_structure_digest: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


class CheckpointAgent:
    FORMAT_VERSION = "1"

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, run_id: str) -> Path:
        return self._root / run_id / "checkpoints" / "checkpoint.pt"

    def save(
        self,
        run_id: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        step: int,
        lineage: CheckpointLineage,
        training_state: dict[str, float | int],
    ) -> Path:
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save({
            "format_version": self.FORMAT_VERSION,
            "epoch": epoch,
            "step": step,
            "lineage": asdict(lineage),
            "training_state": training_state,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }, temporary)
        os.replace(temporary, path)
        return path

    def restore(
        self,
        run_id: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        expected: CheckpointLineage,
    ) -> int:
        payload = torch.load(self._path(run_id), map_location="cpu", weights_only=False)
        if payload["format_version"] != self.FORMAT_VERSION:
            raise ValueError("CHECKPOINT_INCOMPATIBLE: format version")
        if payload["lineage"] != asdict(expected):
            raise ValueError("CHECKPOINT_INCOMPATIBLE: lineage")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
        return int(payload["epoch"])
```

- [ ] **Step 4: Run checkpoint/reproducibility tests**

Run: `uv run pytest recommend/train/tests/unit/test_checkpoint_agent.py recommend/train/tests/unit/test_reproducibility.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit checkpoints**

```bash
git add recommend/train/comm/checkpoint_agent.py recommend/train/tests/unit
git commit -m "feat: add strict training checkpoints"
```

### Task 8: Compute four-target metrics and classify gates

**Requirements:** R008
**Depends on:** Tasks 5, 6

**Files:**
- Create: `recommend/train/comm/metrics_utils.py`
- Create: `recommend/train/comm/eval/__init__.py`
- Create: `recommend/train/comm/eval/eval_model_rerank.py`
- Create: `recommend/train/comm/eval/gates.py`
- Create: `recommend/train/tests/unit/test_metrics_utils.py`
- Create: `recommend/train/tests/unit/test_gates.py`

**Completion conditions:** per-target counts/loss/AUC carry a version; a single class becomes `NOT_EVALUABLE`;
schema/numeric/export failures block; AUC floors and candidate deltas only warn and compare on one slice.

- [ ] **Step 1: Write failing metrics and gate-severity tests**

```python
# recommend/train/tests/unit/test_metrics_utils.py
from __future__ import annotations

import torch

from recommend.train.comm.eval.eval_model_rerank import evaluate_rerank
from recommend.train.comm.metrics_utils import MetricStatus, compute_binary_metrics

TARGETS = ("effective_watch", "completion", "non_fast_swipe", "immersive_click")


def test_single_class_target_is_not_evaluable() -> None:
    metric = compute_binary_metrics(
        torch.tensor([0.1, 0.2]), torch.tensor([1.0, 1.0]), torch.tensor([True, True]),
        minimum_valid_rows=2,
    )
    assert metric.status is MetricStatus.NOT_EVALUABLE
    assert metric.auc is None


def test_evaluator_reports_counts_loss_and_auc_for_all_targets() -> None:
    class Batch:
        labels = {target: torch.tensor([0.0, 1.0]) for target in TARGETS}
        masks = {target: torch.tensor([True, True]) for target in TARGETS}

    class Adapter:
        def logits(self, _batch: Batch) -> dict[str, torch.Tensor]:
            return {target: torch.tensor([-2.0, 2.0]) for target in TARGETS}

    result = evaluate_rerank(Adapter(), [Batch()], targets=TARGETS, minimum_valid_rows=2)
    assert set(result) == set(TARGETS)
    for metric in result.values():
        assert metric.status is MetricStatus.OK
        assert (metric.valid_count, metric.positive_count, metric.negative_count) == (2, 1, 1)
        assert metric.loss is not None
        assert metric.auc == 1.0
```

```python
# recommend/train/tests/unit/test_gates.py
import pytest

from recommend.train.comm.eval.gates import GateSeverity, ScopedAuc, hard_gate, quality_gate


def test_integrity_failure_blocks_and_auc_drop_warns() -> None:
    assert hard_gate("checksum", passed=False).severity is GateSeverity.BLOCK
    result = quality_gate(
        "auc",
        candidate=ScopedAuc(0.61, "current-slice"),
        previous=ScopedAuc(0.63, "current-slice"),
        warning_floor=0.60,
    )
    assert result.severity is GateSeverity.WARN
    assert result.passed is True


def test_auc_comparison_rejects_different_validation_slices() -> None:
    with pytest.raises(ValueError, match="same validation slice"):
        quality_gate(
            "auc",
            candidate=ScopedAuc(0.61, "current-slice"),
            previous=ScopedAuc(0.63, "historical-slice"),
            warning_floor=0.60,
        )
```

- [ ] **Step 2: Run metric/gate tests and verify missing modules**

Run: `uv run pytest recommend/train/tests/unit/test_metrics_utils.py recommend/train/tests/unit/test_gates.py -v`
Expected: FAIL because metrics and gates are absent.

- [ ] **Step 3: Implement versioned binary metrics and hard/soft gate values**

```python
# recommend/train/comm/metrics_utils.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F


class MetricStatus(StrEnum):
    OK = "OK"
    NOT_EVALUABLE = "NOT_EVALUABLE"


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    implementation_version: str
    status: MetricStatus
    valid_count: int
    positive_count: int
    negative_count: int
    loss: float | None
    auc: float | None


def compute_binary_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    minimum_valid_rows: int,
) -> BinaryMetrics:
    selected_logits = logits[mask].detach().cpu()
    selected_labels = labels[mask].detach().cpu()
    positives = int(selected_labels.sum().item())
    valid = int(selected_labels.numel())
    negatives = valid - positives
    if valid < minimum_valid_rows or positives == 0 or negatives == 0:
        return BinaryMetrics("roc_auc/v1", MetricStatus.NOT_EVALUABLE, valid, positives, negatives, None, None)
    loss = F.binary_cross_entropy_with_logits(selected_logits, selected_labels).item()
    auc = roc_auc_score(selected_labels.numpy(), selected_logits.numpy())
    return BinaryMetrics("roc_auc/v1", MetricStatus.OK, valid, positives, negatives, loss, float(auc))
```

```python
# recommend/train/comm/eval/gates.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GateSeverity(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass(frozen=True, slots=True)
class GateResult:
    name: str
    passed: bool
    severity: GateSeverity
    detail: str


@dataclass(frozen=True, slots=True)
class ScopedAuc:
    value: float
    validation_slice_digest: str


def hard_gate(name: str, *, passed: bool) -> GateResult:
    return GateResult(name, passed, GateSeverity.PASS if passed else GateSeverity.BLOCK, name)


def quality_gate(
    name: str, *, candidate: ScopedAuc, previous: ScopedAuc | None, warning_floor: float
) -> GateResult:
    if previous is not None and previous.validation_slice_digest != candidate.validation_slice_digest:
        raise ValueError("candidate and previous AUC must use the same validation slice")
    warned = candidate.value < warning_floor or (
        previous is not None and candidate.value < previous.value
    )
    return GateResult(name, True, GateSeverity.WARN if warned else GateSeverity.PASS, name)
```

```python
# recommend/train/comm/eval/eval_model_rerank.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar

import torch

from recommend.train.comm.metrics_utils import BinaryMetrics, compute_binary_metrics


class EvaluationBatch(Protocol):
    labels: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]


BatchT = TypeVar("BatchT", bound=EvaluationBatch)
BatchT_contra = TypeVar("BatchT_contra", bound=EvaluationBatch, contravariant=True)


class EvaluationAdapter(Protocol[BatchT_contra]):
    def logits(self, batch: BatchT_contra) -> dict[str, torch.Tensor]: ...


def evaluate_rerank(
    adapter: EvaluationAdapter[BatchT],
    batches: Iterable[BatchT],
    *,
    targets: tuple[str, ...],
    minimum_valid_rows: int,
) -> dict[str, BinaryMetrics]:
    logits = {target: [] for target in targets}
    labels = {target: [] for target in targets}
    masks = {target: [] for target in targets}
    with torch.inference_mode():
        for batch in batches:
            output = adapter.logits(batch)
            for target in targets:
                logits[target].append(output[target])
                labels[target].append(batch.labels[target])
                masks[target].append(batch.masks[target])
    return {
        target: compute_binary_metrics(
            torch.cat(logits[target]),
            torch.cat(labels[target]),
            torch.cat(masks[target]),
            minimum_valid_rows=minimum_valid_rows,
        )
        for target in targets
    }
```

- [ ] **Step 4: Run metric/gate tests**

Run: `uv run pytest recommend/train/tests/unit/test_metrics_utils.py recommend/train/tests/unit/test_gates.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit metrics and gates**

```bash
git add recommend/train/comm/metrics_utils.py recommend/train/comm/eval recommend/train/tests/unit
git commit -m "feat: add rerank evaluation gates"
```

### Task 9: Orchestrate evaluation training, gates, and production refit

**Requirements:** R006, R007, R008, NFR002
**Depends on:** Tasks 4, 6, 7, 8

**Files:**
- Create: `recommend/train/comm/model_protocol.py`
- Create: `recommend/train/comm/training_pipeline.py`
- Modify: `recommend/train/models/rerank/__init__.py`
- Create: `recommend/train/tests/unit/test_training_pipeline.py`

**Completion conditions:** `TrainingPipeline` imports no concrete rerank module; evaluation fitting uses only 27-day
batches; any blocking gate or exportability preflight failure prevents a second model; successful evaluation creates
a fresh model and trains it on all 28 days for exactly `selected_epoch`; NaN/Inf and budget failures stop the run.

- [ ] **Step 1: Write failing state-machine tests with a fake adapter**

```python
# recommend/train/tests/unit/test_training_pipeline.py
from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest
import torch

from recommend.train.comm.eval.gates import GateResult, GateSeverity
from recommend.train.comm.checkpoint_agent import CheckpointLineage
from recommend.train.comm.training_pipeline import BatchFactories, PipelineBlocked, TrainingPipeline


class Adapter:
    def __init__(self) -> None:
        self.module = torch.nn.Linear(1, 1)
        self.epochs: list[str] = []

    def loss(self, batch: torch.Tensor) -> torch.Tensor:
        return self.module(batch).square().mean()

    def logits(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"score": self.module(batch).squeeze(1)}


def batches(label: str) -> Callable[[], Iterable[torch.Tensor]]:
    def factory() -> Iterable[torch.Tensor]:
        yield torch.ones(2, 1)
    factory.__name__ = label
    return factory


def test_success_refits_fresh_model_for_selected_epoch() -> None:
    created: list[Adapter] = []

    def factory(_role: str) -> Adapter:
        adapter = Adapter()
        created.append(adapter)
        return adapter

    pipeline = TrainingPipeline(
        adapter_factory=factory,
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01),
        evaluator=lambda _adapter, _batches: {"auc": 0.6},
        gate_builder=lambda _metrics: (GateResult("metrics", True, GateSeverity.PASS, "ok"),),
        checkpoint_agent=None,
        run_id="run-1",
        lineage=CheckpointLineage("config", "dataset", "model"),
        seed=7,
        max_epochs=1,
        patience=1,
        memory_limit_bytes=1_000_000_000,
    )
    result = pipeline.run(BatchFactories(batches("train"), batches("validation"), batches("refit")))
    assert len(created) == 2
    assert created[0] is not created[1]
    assert result.selected_epoch == 1
    assert result.production_adapter is created[1]
    assert [trace.role for trace in result.traces] == ["evaluation", "production"]


def test_blocking_gate_prevents_refit() -> None:
    created: list[Adapter] = []
    pipeline = TrainingPipeline(
        adapter_factory=lambda _role: created.append(Adapter()) or created[-1],
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01),
        evaluator=lambda _adapter, _batches: {},
        gate_builder=lambda _metrics: (GateResult("labels", False, GateSeverity.BLOCK, "single class"),),
        checkpoint_agent=None,
        run_id="run-2",
        lineage=CheckpointLineage("config", "dataset", "model"),
        seed=7,
        max_epochs=1,
        patience=1,
        memory_limit_bytes=1_000_000_000,
    )
    with pytest.raises(PipelineBlocked, match="labels"):
        pipeline.run(BatchFactories(batches("train"), batches("validation"), batches("refit")))
    assert len(created) == 1


def test_non_finite_loss_stops_before_refit() -> None:
    class NonFiniteAdapter(Adapter):
        def loss(self, batch: torch.Tensor) -> torch.Tensor:
            return super().loss(batch) * torch.tensor(float("nan"))

    pipeline = TrainingPipeline(
        adapter_factory=lambda _role: NonFiniteAdapter(),
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01),
        evaluator=lambda _adapter, _batches: {},
        gate_builder=lambda _metrics: (),
        checkpoint_agent=None,
        run_id="run-nan",
        lineage=CheckpointLineage("config", "dataset", "model"),
        seed=7,
        max_epochs=1,
        patience=1,
        memory_limit_bytes=1_000_000_000,
    )
    with pytest.raises(PipelineBlocked, match="NUMERICAL_FAILURE"):
        pipeline.run(BatchFactories(batches("train"), batches("validation"), batches("refit")))


def test_memory_budget_exceeded_stops_before_refit() -> None:
    pipeline = TrainingPipeline(
        adapter_factory=lambda _role: Adapter(),
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01),
        evaluator=lambda _adapter, _batches: {},
        gate_builder=lambda _metrics: (),
        checkpoint_agent=None,
        run_id="run-memory",
        lineage=CheckpointLineage("config", "dataset", "model"),
        seed=7,
        max_epochs=1,
        patience=1,
        memory_limit_bytes=0,
    )
    with pytest.raises(PipelineBlocked, match="RESOURCE_BUDGET_EXCEEDED"):
        pipeline.run(BatchFactories(batches("train"), batches("validation"), batches("refit")))


def test_exportability_preflight_failure_prevents_refit() -> None:
    created: list[Adapter] = []

    def reject_export(_adapter: Adapter) -> None:
        raise ValueError("ONNX_PARITY_FAILED")

    pipeline = TrainingPipeline(
        adapter_factory=lambda _role: created.append(Adapter()) or created[-1],
        optimizer_factory=lambda parameters: torch.optim.SGD(parameters, lr=0.01),
        evaluator=lambda _adapter, _batches: {"auc": 0.6},
        gate_builder=lambda _metrics: (),
        checkpoint_agent=None,
        run_id="run-export",
        lineage=CheckpointLineage("config", "dataset", "model"),
        seed=7,
        max_epochs=1,
        patience=1,
        memory_limit_bytes=1_000_000_000,
        pre_refit_validator=reject_export,
    )
    with pytest.raises(ValueError, match="ONNX_PARITY_FAILED"):
        pipeline.run(BatchFactories(batches("train"), batches("validation"), batches("refit")))
    assert len(created) == 1
```

- [ ] **Step 2: Run pipeline tests and verify missing protocol/pipeline**

Run: `uv run pytest recommend/train/tests/unit/test_training_pipeline.py -v`
Expected: FAIL because `model_protocol.py` and `training_pipeline.py` are absent.

- [ ] **Step 3: Define the model-agnostic adapter protocol**

```python
# recommend/train/comm/model_protocol.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal, Protocol, TypeVar

import torch

BatchT = TypeVar("BatchT")
BatchT_contra = TypeVar("BatchT_contra", contravariant=True)
BatchT_factory = TypeVar("BatchT_factory", contravariant=True)
ModelRole = Literal["evaluation", "production"]


class ModelAdapter(Protocol[BatchT_contra]):
    module: torch.nn.Module

    def loss(self, batch: BatchT_contra) -> torch.Tensor: ...
    def logits(self, batch: BatchT_contra) -> dict[str, torch.Tensor]: ...


class AdapterFactory(Protocol[BatchT_factory]):
    def __call__(self, role: ModelRole) -> ModelAdapter[BatchT_factory]: ...


BatchFactory = Callable[[], Iterable[BatchT]]
```

- [ ] **Step 4: Implement the generic CPU state machine**

```python
# recommend/train/comm/training_pipeline.py
from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

import psutil
import torch

from recommend.train.comm.checkpoint_agent import (
    CheckpointAgent,
    CheckpointLineage,
    seed_everything,
)
from recommend.train.comm.eval.gates import GateResult, GateSeverity
from recommend.train.comm.model_protocol import AdapterFactory, ModelAdapter

BatchT = TypeVar("BatchT")
MetricT = TypeVar("MetricT")


class PipelineBlocked(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BatchFactories(Generic[BatchT]):
    train: Callable[[], Iterable[BatchT]]
    validation: Callable[[], Iterable[BatchT]]
    refit: Callable[[], Iterable[BatchT]]


@dataclass(frozen=True, slots=True)
class PipelineResult(Generic[BatchT, MetricT]):
    selected_epoch: int
    metrics: MetricT
    gates: tuple[GateResult, ...]
    production_adapter: ModelAdapter[BatchT]
    traces: tuple["EpochTrace", ...]


@dataclass(frozen=True, slots=True)
class EpochTrace:
    role: str
    epoch: int
    step_losses: tuple[float, ...]


class TrainingPipeline(Generic[BatchT, MetricT]):
    def __init__(
        self,
        *,
        adapter_factory: AdapterFactory[BatchT],
        optimizer_factory: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
        evaluator: Callable[[ModelAdapter[BatchT], Iterable[BatchT]], MetricT],
        gate_builder: Callable[[MetricT], tuple[GateResult, ...]],
        checkpoint_agent: CheckpointAgent | None,
        run_id: str,
        lineage: CheckpointLineage,
        seed: int,
        max_epochs: int,
        patience: int,
        memory_limit_bytes: int,
        pre_refit_validator: Callable[[ModelAdapter[BatchT]], None] | None = None,
    ) -> None:
        self._adapter_factory = adapter_factory
        self._optimizer_factory = optimizer_factory
        self._evaluator = evaluator
        self._gate_builder = gate_builder
        self._checkpoint_agent = checkpoint_agent
        self._run_id = run_id
        self._lineage = lineage
        self._seed = seed
        self._max_epochs = max_epochs
        self._patience = patience
        self._memory_limit_bytes = memory_limit_bytes
        self._pre_refit_validator = pre_refit_validator

    def _train_epoch(
        self,
        adapter: ModelAdapter[BatchT],
        optimizer: torch.optim.Optimizer,
        batches: Iterable[BatchT],
    ) -> tuple[float, tuple[float, ...]]:
        adapter.module.train()
        losses: list[float] = []
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = adapter.loss(batch)
            if not torch.isfinite(loss):
                raise PipelineBlocked("NUMERICAL_FAILURE")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in adapter.module.parameters()):
                raise PipelineBlocked("NUMERICAL_FAILURE")
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in adapter.module.parameters()):
                raise PipelineBlocked("NUMERICAL_FAILURE")
            losses.append(float(loss.detach()))
            if psutil.Process().memory_info().rss > self._memory_limit_bytes:
                raise PipelineBlocked("RESOURCE_BUDGET_EXCEEDED")
        if not losses:
            raise PipelineBlocked("DATA_EXHAUSTED")
        return sum(losses) / len(losses), tuple(losses)

    @staticmethod
    def _mean_loss(adapter: ModelAdapter[BatchT], batches: Iterable[BatchT]) -> float:
        adapter.module.eval()
        values: list[float] = []
        with torch.inference_mode():
            for batch in batches:
                loss = adapter.loss(batch)
                if not torch.isfinite(loss):
                    raise PipelineBlocked("NUMERICAL_FAILURE")
                values.append(float(loss))
        if not values:
            raise PipelineBlocked("DATA_EXHAUSTED")
        return sum(values) / len(values)

    def run(self, batches: BatchFactories[BatchT]) -> PipelineResult[BatchT, MetricT]:
        seed_everything(self._seed)
        evaluation_adapter = self._adapter_factory("evaluation")
        evaluation_optimizer = self._optimizer_factory(evaluation_adapter.module.parameters())
        selected_epoch = 1
        best_loss = math.inf
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        traces: list[EpochTrace] = []
        for epoch in range(1, self._max_epochs + 1):
            _, step_losses = self._train_epoch(
                evaluation_adapter, evaluation_optimizer, batches.train()
            )
            traces.append(EpochTrace("evaluation", epoch, step_losses))
            validation_loss = self._mean_loss(evaluation_adapter, batches.validation())
            if validation_loss < best_loss:
                best_loss, selected_epoch, stale_epochs = validation_loss, epoch, 0
                best_state = copy.deepcopy(evaluation_adapter.module.state_dict())
                if self._checkpoint_agent is not None:
                    self._checkpoint_agent.save(
                        self._run_id,
                        evaluation_adapter.module,
                        evaluation_optimizer,
                        epoch=epoch,
                        step=sum(len(trace.step_losses) for trace in traces),
                        lineage=self._lineage,
                        training_state={"best_loss": best_loss, "stale_epochs": stale_epochs},
                    )
            else:
                stale_epochs += 1
            if stale_epochs >= self._patience:
                break
        if best_state is None:
            raise PipelineBlocked("DATA_EXHAUSTED")
        if self._checkpoint_agent is None:
            evaluation_adapter.module.load_state_dict(best_state, strict=True)
        else:
            self._checkpoint_agent.restore(
                self._run_id, evaluation_adapter.module, evaluation_optimizer, self._lineage
            )
        metrics = self._evaluator(evaluation_adapter, batches.validation())
        gates = self._gate_builder(metrics)
        blocked = [gate.name for gate in gates if gate.severity is GateSeverity.BLOCK]
        if blocked:
            raise PipelineBlocked("EVALUATION_GATE_FAILED:" + ",".join(blocked))
        if self._pre_refit_validator is not None:
            self._pre_refit_validator(evaluation_adapter)
        seed_everything(self._seed + 1)
        production_adapter = self._adapter_factory("production")
        production_optimizer = self._optimizer_factory(production_adapter.module.parameters())
        for epoch in range(1, selected_epoch + 1):
            _, step_losses = self._train_epoch(
                production_adapter, production_optimizer, batches.refit()
            )
            traces.append(EpochTrace("production", epoch, step_losses))
        return PipelineResult(selected_epoch, metrics, gates, production_adapter, tuple(traces))
```

Add the concrete adapter without storage/config side effects:

```python
# recommend/train/models/rerank/__init__.py
from __future__ import annotations

import torch

from recommend.train.models.rerank.data_utils import RerankBatch
from recommend.train.models.rerank.model_params import RerankModelParams
from recommend.train.models.rerank.rerank_model import RerankModel, masked_multitask_loss


class RerankAdapter:
    def __init__(self, params: RerankModelParams) -> None:
        self.module = RerankModel(params)
        self._weights = params.loss_weights

    def loss(self, batch: RerankBatch) -> torch.Tensor:
        return masked_multitask_loss(self.logits(batch), batch.labels, batch.masks, self._weights)

    def logits(self, batch: RerankBatch) -> dict[str, torch.Tensor]:
        return self.module(batch.numeric, batch.categorical)


__all__ = ["RerankAdapter", "RerankModel", "RerankModelParams", "masked_multitask_loss"]
```

- [ ] **Step 5: Run state-machine tests and the import-boundary check**

Run:
```bash
uv run pytest recommend/train/tests/unit/test_training_pipeline.py -v
! rg -n 'recommend\.train\.models\.rerank' recommend/train/comm
```
Expected: 5 tests pass and `rg` returns no match from `comm` to the concrete model namespace.

- [ ] **Step 6: Commit the training state machine**

```bash
git add recommend/train/comm/model_protocol.py recommend/train/comm/training_pipeline.py recommend/train/models/rerank/__init__.py recommend/train/tests/unit/test_training_pipeline.py
git commit -m "feat: orchestrate cpu rerank training"
```

### Task 10: Export ONNX and commit an immutable artifact

**Requirements:** R009, R008
**Depends on:** Tasks 3, 6, 8, 9

**Files:**
- Create: `recommend/train/tools/onnx/__init__.py`
- Create: `recommend/train/tools/onnx/gen_onnx.py`
- Create: `recommend/train/comm/artifact_utils.py`
- Create: `recommend/train/tests/unit/test_gen_onnx.py`
- Create: `recommend/train/tests/contract/test_artifact_utils.py`

**Completion conditions:** ONNX has dynamic batch and four stable output names; ONNX checker and Runtime parity
pass; artifact version excludes timestamps; every object is read back; `manifest.json` is written last and failed
verification leaves no final manifest.

- [ ] **Step 1: Write failing ONNX parity and manifest-last tests**

```python
# recommend/train/tests/unit/test_gen_onnx.py
from __future__ import annotations

import onnx
import pytest
import torch

from recommend.train.models.rerank.model_params import RerankModelParams, TARGETS
from recommend.train.models.rerank.rerank_model import RerankModel
from recommend.train.tools.onnx.gen_onnx import export_and_verify


def test_export_has_four_outputs_and_dynamic_batch(tmp_path) -> None:
    params = RerankModelParams(
        numeric_input_dim=2,
        categorical_cardinalities=(4,),
        embedding_dim=2,
        shared_hidden_dims=(4,),
        tower_hidden_dim=2,
        activation="relu",
        dropout=0.0,
        loss_weights={target: 1.0 for target in TARGETS},
    )
    result = export_and_verify(
        RerankModel(params).eval(),
        torch.zeros(2, 2),
        torch.zeros(2, 1, dtype=torch.long),
        tmp_path / "model.onnx",
        absolute_tolerance=1e-5,
    )
    assert result.output_names == TARGETS
    graph = onnx.load(result.path).graph
    assert graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert graph.input[1].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert [output.name for output in graph.output] == list(TARGETS)
    assert all(
        output.type.tensor_type.shape.dim[0].dim_param == "batch" for output in graph.output
    )


def test_export_rejects_runtime_parity_mismatch(tmp_path, monkeypatch) -> None:
    def mismatch(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("forced parity mismatch")

    monkeypatch.setattr(
        "recommend.train.tools.onnx.gen_onnx.np.testing.assert_allclose", mismatch
    )
    params = RerankModelParams(
        numeric_input_dim=2,
        categorical_cardinalities=(4,),
        embedding_dim=2,
        shared_hidden_dims=(4,),
        tower_hidden_dim=2,
        activation="relu",
        dropout=0.0,
        loss_weights={target: 1.0 for target in TARGETS},
    )
    with pytest.raises(ValueError, match="ONNX_PARITY_FAILED"):
        export_and_verify(
            RerankModel(params).eval(),
            torch.zeros(2, 2),
            torch.zeros(2, 1, dtype=torch.long),
            tmp_path / "bad.onnx",
            absolute_tolerance=1e-5,
        )
```

```python
# recommend/train/tests/contract/test_artifact_utils.py
import hashlib
import json
from typing import BinaryIO

import pytest

from recommend.train.comm.artifact_utils import ArtifactBuilder
from recommend.train.comm.datasvr.storage import ObjectHead


def logical_inputs() -> dict[str, str]:
    return {
        "config": "a",
        "dataset": "b",
        "model": "c",
        "source": "d",
        "dependencies": "e",
        "onnx_opset": "18",
    }


class RecordingStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.writes: list[str] = []

    def put_bytes_if_absent(self, uri: str, body: bytes) -> ObjectHead:
        if uri in self.objects:
            raise FileExistsError(uri)
        self.objects[uri] = body
        self.writes.append(uri)
        return ObjectHead(len(body), hashlib.sha256(body).hexdigest())

    def get_bytes(self, uri: str, *, max_bytes: int) -> bytes:
        return self.objects[uri]

    def head(self, uri: str) -> ObjectHead:
        body = self.objects[uri]
        return ObjectHead(len(body), hashlib.sha256(body).hexdigest())

    def download_to(self, uri: str, sink: BinaryIO) -> ObjectHead:
        sink.write(self.objects[uri])
        return self.head(uri)


def test_manifest_is_the_last_artifact_write() -> None:
    store = RecordingStore()
    builder = ArtifactBuilder(store=store, output_prefix="s3://doku/models/")
    manifest_uri = builder.commit(
        model_name="rerank",
        logical_inputs=logical_inputs(),
        audit_metadata={"created_at": "2026-08-30T00:00:00Z"},
        files={
            "model.onnx": b"onnx",
            "signature.json": b"{}",
            "model-config.json": b"{}",
            "feature-schema.json": b"{}",
            "fitted-feature-state/state.json": b"{}",
            "metrics.json": b"{}",
            "lineage.json": b"{}",
        },
    )
    assert store.writes[-1] == manifest_uri
    assert manifest_uri.endswith("/manifest.json")
    manifest = json.loads(store.objects[manifest_uri])
    assert manifest["state"] == "COMMITTED"
    assert manifest["logical_inputs"]["onnx_opset"] == "18"


def test_readback_mismatch_never_writes_final_manifest() -> None:
    store = RecordingStore()
    original_get = store.get_bytes
    store.get_bytes = lambda uri, max_bytes: b"corrupt" if uri.endswith("model.onnx") else original_get(uri, max_bytes=max_bytes)  # type: ignore[method-assign]
    builder = ArtifactBuilder(store=store, output_prefix="s3://doku/models/")
    files = {
        "model.onnx": b"onnx",
        "signature.json": b"{}",
        "model-config.json": b"{}",
        "feature-schema.json": b"{}",
        "fitted-feature-state/state.json": b"{}",
        "metrics.json": b"{}",
        "lineage.json": b"{}",
    }
    try:
        builder.commit(
            model_name="rerank",
            logical_inputs=logical_inputs(),
            audit_metadata={"created_at": "2026-08-30T00:00:00Z"},
            files=files,
        )
    except ValueError as error:
        assert "ARTIFACT_VERIFY_FAILED" in str(error)
    else:
        raise AssertionError("corrupt readback was accepted")
    assert not any(uri.endswith("manifest.json") for uri in store.writes)


def test_precommit_validator_failure_never_writes_final_manifest() -> None:
    store = RecordingStore()
    files = {
        "model.onnx": b"onnx",
        "signature.json": b"{}",
        "model-config.json": b"{}",
        "feature-schema.json": b"{}",
        "fitted-feature-state/state.json": b"{}",
        "metrics.json": b"{}",
        "lineage.json": b"{}",
    }

    def reject_model(_body: bytes) -> None:
        raise ValueError("ONNX_EXPORT_FAILED")

    with pytest.raises(ValueError, match="ONNX_EXPORT_FAILED"):
        ArtifactBuilder(store=store, output_prefix="s3://doku/models/").commit(
            model_name="rerank",
            logical_inputs=logical_inputs(),
            audit_metadata={"created_at": "2026-08-30T00:00:00Z"},
            files=files,
            validators={"model.onnx": reject_model},
        )
    assert not any(uri.endswith("manifest.json") for uri in store.writes)


def test_audit_timestamp_does_not_change_logical_version() -> None:
    files = {
        "model.onnx": b"onnx",
        "signature.json": b"{}",
        "model-config.json": b"{}",
        "feature-schema.json": b"{}",
        "fitted-feature-state/state.json": b"{}",
        "metrics.json": b"{}",
        "lineage.json": b"{}",
    }
    uris = []
    for created_at in ("2026-08-30T00:00:00Z", "2026-08-30T01:00:00Z"):
        store = RecordingStore()
        uris.append(ArtifactBuilder(store=store, output_prefix="s3://doku/models/").commit(
            model_name="rerank",
            logical_inputs=logical_inputs(),
            audit_metadata={"created_at": created_at},
            files=files,
        ))
    assert uris[0] == uris[1]
```

- [ ] **Step 2: Run export/artifact tests and verify missing modules**

Run: `uv run pytest recommend/train/tests/unit/test_gen_onnx.py recommend/train/tests/contract/test_artifact_utils.py -v`
Expected: FAIL because ONNX export and artifact builder are absent.

- [ ] **Step 3: Implement stable ONNX export and Runtime parity**

```python
# recommend/train/tools/onnx/gen_onnx.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from recommend.train.models.rerank.model_params import TARGETS
from recommend.train.models.rerank.rerank_model import RerankModel


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: RerankModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.model(numeric, categorical)
        return tuple(output[target] for target in TARGETS)


@dataclass(frozen=True, slots=True)
class OnnxExportResult:
    path: Path
    output_names: tuple[str, ...]
    opset: int


def export_and_verify(
    model: RerankModel,
    numeric: torch.Tensor,
    categorical: torch.Tensor,
    path: Path,
    *,
    absolute_tolerance: float,
) -> OnnxExportResult:
    wrapper = ExportWrapper(model).eval()
    batch_dimension = torch.export.Dim("batch", min=1)
    torch.onnx.export(
        wrapper,
        (numeric, categorical),
        path,
        input_names=["numeric", "categorical"],
        output_names=list(TARGETS),
        dynamo=True,
        dynamic_shapes={
            "numeric": {0: batch_dimension},
            "categorical": {0: batch_dimension},
        },
        external_data=False,
        opset_version=18,
    )
    try:
        onnx.checker.check_model(onnx.load(path))
        session = ort.InferenceSession(path.read_bytes(), providers=["CPUExecutionProvider"])
    except Exception as error:
        raise ValueError("ONNX_EXPORT_FAILED") from error
    actual = session.run(list(TARGETS), {
        "numeric": numeric.numpy(),
        "categorical": categorical.numpy(),
    })
    with torch.inference_mode():
        expected = wrapper(numeric, categorical)
    try:
        for torch_value, onnx_value in zip(expected, actual, strict=True):
            np.testing.assert_allclose(
                torch_value.numpy(), onnx_value, rtol=0.0, atol=absolute_tolerance
            )
    except AssertionError as error:
        raise ValueError("ONNX_PARITY_FAILED") from error
    return OnnxExportResult(path, TARGETS, 18)
```

- [ ] **Step 4: Implement deterministic, read-back-verified artifact commit**

```python
# recommend/train/comm/artifact_utils.py
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

from recommend.train.comm.datasvr.storage import ObjectStore


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True, slots=True)
class ArtifactBuilder:
    store: ObjectStore
    output_prefix: str

    def commit(
        self,
        *,
        model_name: str,
        logical_inputs: dict[str, str],
        audit_metadata: dict[str, str],
        files: dict[str, bytes],
        validators: dict[str, Callable[[bytes], None]] | None = None,
    ) -> str:
        required = {
            "model.onnx",
            "signature.json",
            "model-config.json",
            "feature-schema.json",
            "fitted-feature-state/state.json",
            "metrics.json",
            "lineage.json",
        }
        if set(files) != required:
            raise ValueError("artifact file set is incomplete")
        required_logical_inputs = {
            "config", "dataset", "model", "source", "dependencies", "onnx_opset"
        }
        if set(logical_inputs) != required_logical_inputs:
            raise ValueError("artifact logical inputs are incomplete")
        logical_digest = hashlib.sha256(canonical_json(logical_inputs)).hexdigest()
        root = f"{self.output_prefix.rstrip('/')}/{model_name}/{logical_digest}"
        file_records: list[dict[str, object]] = []
        for name in sorted(files):
            body = files[name]
            uri = f"{root}/{name}"
            digest = hashlib.sha256(body).hexdigest()
            self.store.put_bytes_if_absent(uri, body)
            readback = self.store.get_bytes(uri, max_bytes=len(body) + 1)
            if hashlib.sha256(readback).hexdigest() != digest:
                raise ValueError("ARTIFACT_VERIFY_FAILED")
            if validators is not None and name in validators:
                validators[name](readback)
            file_records.append({"name": name, "size_bytes": len(body), "sha256": digest})
        manifest = canonical_json({
            "artifact_format_version": "1",
            "state": "COMMITTED",
            "model_name": model_name,
            "model_version": logical_digest,
            "logical_inputs": logical_inputs,
            "audit_metadata": audit_metadata,
            "files": file_records,
        })
        manifest_uri = f"{root}/manifest.json"
        self.store.put_bytes_if_absent(manifest_uri, manifest)
        return manifest_uri
```

- [ ] **Step 5: Run ONNX and artifact contract tests**

Run: `uv run pytest recommend/train/tests/unit/test_gen_onnx.py recommend/train/tests/contract/test_artifact_utils.py -v`
Expected: 6 tests pass; ONNX Runtime uses only `CPUExecutionProvider`; manifest is the final write, timestamps do
not change the logical version, and corruption
does not create one.

- [ ] **Step 6: Commit framework-neutral artifacts**

```bash
git add recommend/train/tools/onnx recommend/train/comm/artifact_utils.py recommend/train/tests
git commit -m "feat: export verified onnx artifacts"
```

### Task 11: Wire the Finder-style CLI and full CPU smoke test

**Requirements:** R008, R009, R010, NFR003, NFR004
**Depends on:** Tasks 2, 3, 9, 10

**Files:**
- Create: `recommend/train/comm/errors.py`
- Create: `recommend/train/comm/offline_evaluate.py`
- Create: `recommend/train/run.py`
- Create: `recommend/train/train_rerank.py`
- Create: `recommend/train/tests/fixtures.py`
- Create: `recommend/train/tests/unit/test_cli.py`
- Create: `recommend/train/tests/smoke/test_train_rerank.py`
- Modify: `pyproject.toml`

**Completion conditions:** `train` requires exact manifest/config/output and read-only commands require exact
manifest/config; `validate`, `train`, and `backtest` use one composition root; errors are JSON with run/stage/code
and redacted detail; local smoke executes the complete
27/1 → gate → 28-day refit → ONNX reload path without AWS/network access.

- [ ] **Step 1: Write failing CLI contract and smoke tests**

```python
# recommend/train/tests/unit/test_cli.py
import json

import pytest
from typer.testing import CliRunner

from recommend.train.run import app


def test_train_requires_explicit_manifest_config_and_output() -> None:
    result = CliRunner().invoke(app, ["rerank", "train"])
    payload = json.loads(result.output)
    assert result.exit_code == 1
    assert payload["code"] == "INPUT_CONTRACT_INVALID"
    assert all(option in payload["detail"] for option in ("--manifest", "--config", "--output"))


def test_train_failure_is_structured_and_redacted(tmp_path, monkeypatch) -> None:
    config = tmp_path / "train.yml"
    config.write_text("model_name: rerank")
    monkeypatch.setattr(
        "recommend.train.train_rerank.execute_train",
        lambda **_: (_ for _ in ()).throw(ValueError("token=secret-value")),
    )
    result = CliRunner().invoke(app, [
        "rerank", "train",
        "--manifest", "file:///tmp/input/manifest.json",
        "--config", str(config),
        "--output", "file:///tmp/output",
    ])
    payload = json.loads(result.output)
    assert result.exit_code == 1
    assert payload["code"] == "INPUT_CONTRACT_INVALID"
    assert "secret-value" not in payload["detail"]


@pytest.mark.parametrize("command", ["validate", "backtest"])
def test_read_only_command_failures_are_structured(command, tmp_path) -> None:
    config = tmp_path / "invalid.yml"
    config.write_text("{")
    result = CliRunner().invoke(app, [
        "rerank", command,
        "--manifest", "file:///tmp/input/manifest.json",
        "--config", str(config),
    ])
    payload = json.loads(result.output)
    assert result.exit_code == 1
    assert payload["stage"] == command
    assert payload["code"] == "INPUT_CONTRACT_INVALID"
```

```python
# recommend/train/tests/smoke/test_train_rerank.py
from __future__ import annotations

from recommend.train.tests.fixtures import build_local_28_day_fixture
from recommend.train.train_rerank import execute_backtest, execute_train


def test_local_cpu_training_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "recommend.train.train_rerank.boto3.client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("AWS access is forbidden")),
    )
    fixture = build_local_28_day_fixture(tmp_path)
    result = execute_train(
        manifest_uri=fixture.manifest_uri,
        config_path=fixture.config_path,
        output_uri=(tmp_path / "artifacts").as_uri(),
        allowed_input_root=tmp_path,
    )
    assert result.manifest_uri.endswith("manifest.json")
    assert result.selected_epoch >= 1
    assert result.output_names == (
        "effective_watch", "completion", "non_fast_swipe", "immersive_click"
    )
    assert result.artifact_reloaded is True


def test_local_backtest_reports_validation_and_test(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "recommend.train.train_rerank.boto3.client",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("AWS access is forbidden")),
    )
    fixture = build_local_28_day_fixture(tmp_path)
    result = execute_backtest(
        manifest_uri=fixture.manifest_uri,
        config_path=fixture.config_path,
        allowed_input_root=tmp_path,
    )
    assert set(result) == {"validation", "test"}
    assert set(result["validation"]) == {
        "effective_watch", "completion", "non_fast_swipe", "immersive_click"
    }
```

- [ ] **Step 2: Run CLI/smoke tests and verify missing entrypoints**

Run: `uv run pytest recommend/train/tests/unit/test_cli.py recommend/train/tests/smoke/test_train_rerank.py -v`
Expected: FAIL because CLI, fixture builder, and `execute_train` are absent.

- [ ] **Step 3: Add structured errors and Typer entrypoints**

```python
# recommend/train/comm/errors.py
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RunError:
    run_id: str
    stage: str
    code: str
    detail: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def classify_error(error: Exception) -> str:
    text = str(error)
    for code in (
        "DATA_INTEGRITY_MISMATCH",
        "LABEL_NOT_MATURE",
        "SPLIT_INVALID",
        "RESOURCE_BUDGET_EXCEEDED",
        "NUMERICAL_FAILURE",
        "CHECKPOINT_INCOMPATIBLE",
        "EVALUATION_GATE_FAILED",
        "ONNX_EXPORT_FAILED",
        "ONNX_PARITY_FAILED",
        "ARTIFACT_VERIFY_FAILED",
    ):
        if code in text:
            return code
    return "INPUT_CONTRACT_INVALID"


def redact_detail(detail: str) -> str:
    detail = re.sub(r"(?i)(token|secret|password|access[_-]?key)=([^&\s]+)", r"\1=<redacted>", detail)
    return detail[:500]
```

```python
# recommend/train/run.py
from __future__ import annotations

import typer

from recommend.train.train_rerank import app as rerank_app

app = typer.Typer(no_args_is_help=True)
app.add_typer(rerank_app, name="rerank")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
```

```python
# recommend/train/train_rerank.py
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn, cast

import boto3
import onnxruntime as ort
import torch
import typer
import yaml

from recommend.train.comm.artifact_utils import ArtifactBuilder, canonical_json
from recommend.train.comm.checkpoint_agent import CheckpointAgent, CheckpointLineage
from recommend.train.comm.datasvr.dataset_manifest import DatasetManifest, Partition
from recommend.train.comm.datasvr.local_storage import LocalStorage
from recommend.train.comm.datasvr.parquet_dataset import ParquetDataset
from recommend.train.comm.datasvr.s3_storage import S3Storage
from recommend.train.comm.datasvr.storage import ObjectStore, parse_storage_uri
from recommend.train.comm.eval.gates import GateResult, ScopedAuc, hard_gate, quality_gate
from recommend.train.comm.eval.eval_model_rerank import evaluate_rerank
from recommend.train.comm.errors import RunError, classify_error, redact_detail
from recommend.train.comm.feature_check import FeatureSchema, LabelDefinition
from recommend.train.comm.metrics_utils import BinaryMetrics
from recommend.train.comm.model_params_validator import (
    ValidatedTrainConfig,
    validate_config_payload,
)
from recommend.train.comm.model_protocol import ModelAdapter, ModelRole
from recommend.train.comm.offline_evaluate import run_backtest
from recommend.train.comm.training_pipeline import BatchFactories, TrainingPipeline
from recommend.train.models.rerank import RerankAdapter
from recommend.train.models.rerank.data_utils import (
    RerankBatch,
    build_backtest_split,
    build_daily_split,
    build_rerank_batch,
)
from recommend.train.models.rerank.feature_utils import (
    FeatureStateBuilder,
    FittedFeatureState,
)
from recommend.train.models.rerank.model_params import RerankModelParams, TARGETS
from recommend.train.models.rerank.rerank_model import RerankModel
from recommend.train.tools.onnx.gen_onnx import export_and_verify

app = typer.Typer(no_args_is_help=True)


@dataclass(frozen=True, slots=True)
class TrainExecutionResult:
    manifest_uri: str
    selected_epoch: int
    output_names: tuple[str, ...]
    artifact_reloaded: bool


@dataclass(frozen=True, slots=True)
class LoadedInputs:
    store: ObjectStore
    manifest: DatasetManifest
    feature_schema: FeatureSchema
    feature_schema_body: bytes


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _source_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _store_for_uri(uri: str, *, local_root: Path | None) -> ObjectStore:
    parsed = parse_storage_uri(uri)
    if parsed.scheme == "file":
        root = local_root or Path("/" + parsed.key).parent
        return LocalStorage(root=root)
    top_level, separator, _remainder = parsed.key.partition("/")
    if not separator:
        raise ValueError("S3 URI must use a non-root input prefix")
    allowed_prefix = f"{top_level}/"
    return S3Storage(
        client=boto3.client("s3"),
        allowed_bucket=parsed.bucket,
        allowed_prefix=allowed_prefix,
    )


def _store_for_output_prefix(uri: str, *, local_root: Path | None) -> ObjectStore:
    parsed = parse_storage_uri(uri)
    if parsed.scheme == "file":
        root = local_root or Path("/" + parsed.key).parent
        return LocalStorage(root=root)
    return S3Storage(
        client=boto3.client("s3"),
        allowed_bucket=parsed.bucket,
        allowed_prefix=f"{parsed.key.rstrip('/')}/",
    )


def _load_inputs(manifest_uri: str, *, local_root: Path | None) -> LoadedInputs:
    store = _store_for_uri(manifest_uri, local_root=local_root)
    manifest = DatasetManifest.model_validate_json(store.get_bytes(manifest_uri, max_bytes=16 * 1024 * 1024))
    feature_body = store.get_bytes(manifest.feature_schema_uri, max_bytes=16 * 1024 * 1024)
    label_body = store.get_bytes(manifest.label_definition_uri, max_bytes=16 * 1024 * 1024)
    if _sha256(feature_body) != manifest.feature_schema_sha256:
        raise ValueError("DATA_INTEGRITY_MISMATCH: feature schema checksum mismatch")
    if _sha256(label_body) != manifest.label_definition_sha256:
        raise ValueError("DATA_INTEGRITY_MISMATCH: label definition checksum mismatch")
    schema = FeatureSchema.model_validate_json(feature_body)
    labels = LabelDefinition.model_validate_json(label_body)
    if schema.version != manifest.feature_schema_version:
        raise ValueError("feature schema version mismatch")
    if labels.version != manifest.label_definition_version:
        raise ValueError("label definition version mismatch")
    if tuple(labels.targets) != TARGETS:
        raise ValueError("label target order mismatch")
    for partition in manifest.partitions:
        head = store.head(partition.uri)
        if head.size_bytes != partition.size_bytes or head.sha256 != partition.sha256:
            raise ValueError("DATA_INTEGRITY_MISMATCH: partition preflight mismatch")
    return LoadedInputs(store, manifest, schema, feature_body)


def _load_config(config_path: Path) -> ValidatedTrainConfig[RerankModelParams]:
    payload = yaml.safe_load(config_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("training config must be an object")
    return validate_config_payload(payload, RerankModelParams)


def _columns(schema: FeatureSchema) -> tuple[str, ...]:
    label_columns = (
        "watch_duration_ms", "content_duration_ms",
        "effective_watch", "effective_watch_valid",
        "non_fast_swipe", "non_fast_swipe_valid",
        "immersive_click", "immersive_click_valid",
    )
    return tuple(dict.fromkeys([*(item.source_column for item in schema.features), *label_columns]))


def _fit_state(
    reader: ParquetDataset,
    partitions: tuple[Partition, ...],
    schema: FeatureSchema,
    *,
    batch_size: int,
    maximum_vocabulary_size: int,
) -> FittedFeatureState:
    builder = FeatureStateBuilder(
        schema=schema,
        maximum_vocabulary_size=maximum_vocabulary_size,
    )
    for batch in reader.iter_batches(partitions, columns=_columns(schema), batch_size=batch_size):
        builder.update(batch)
    return builder.finalize()


def _validate_state_shape(state: FittedFeatureState, params: RerankModelParams) -> None:
    if params.numeric_input_dim != 2 * len(state.numeric):
        raise ValueError("numeric_input_dim does not match fitted feature state")
    if len(params.categorical_cardinalities) != len(state.categorical):
        raise ValueError("categorical feature count does not match model config")
    for cardinality, fitted in zip(
        params.categorical_cardinalities, state.categorical.values(), strict=True
    ):
        if len(fitted.vocabulary) + 2 > cardinality:
            raise ValueError("categorical vocabulary exceeds configured cardinality")


def _batch_factory(
    reader: ParquetDataset,
    partitions: tuple[Partition, ...],
    schema: FeatureSchema,
    state: FittedFeatureState,
    *,
    batch_size: int,
) -> Callable[[], Iterator[RerankBatch]]:
    def batches() -> Iterator[RerankBatch]:
        for batch in reader.iter_batches(partitions, columns=_columns(schema), batch_size=batch_size):
            yield build_rerank_batch(batch, state)
    return batches


_CPU_THREADS: tuple[int, int] | None = None


def _configure_cpu_threads(intra_op: int, inter_op: int) -> None:
    global _CPU_THREADS
    requested = (intra_op, inter_op)
    if _CPU_THREADS is None:
        torch.set_num_threads(intra_op)
        torch.set_num_interop_threads(inter_op)
        _CPU_THREADS = requested
    elif _CPU_THREADS != requested:
        raise ValueError("CPU thread policy cannot change inside one process")


def execute_train(
    *,
    manifest_uri: str,
    config_path: Path,
    output_uri: str,
    allowed_input_root: Path | None = None,
    run_id: str | None = None,
) -> TrainExecutionResult:
    config = _load_config(config_path)
    _configure_cpu_threads(
        config.common.execution.intra_op_threads,
        config.common.execution.inter_op_threads,
    )
    loaded = _load_inputs(manifest_uri, local_root=allowed_input_root)
    split = build_daily_split(loaded.manifest.partitions)
    validation_slice_digest = _sha256(canonical_json([
        {
            "event_date": item.event_date.isoformat(),
            "uri": item.uri,
            "sha256": item.sha256,
            "row_count": item.row_count,
        }
        for item in split.validation
    ]))
    reader = ParquetDataset(loaded.store, spool_limit_bytes=64 * 1024 * 1024)
    max_vocab = min(config.model.categorical_cardinalities, default=2) - 2
    evaluation_state = _fit_state(
        reader,
        split.train,
        loaded.feature_schema,
        batch_size=config.common.batch_size,
        maximum_vocabulary_size=max_vocab,
    )
    _validate_state_shape(evaluation_state, config.model)
    states = {"evaluation": evaluation_state}

    def adapter_factory(_role: ModelRole) -> RerankAdapter:
        return RerankAdapter(config.model)

    train_batches = _batch_factory(
        reader, split.train, loaded.feature_schema, evaluation_state,
        batch_size=config.common.batch_size,
    )
    validation_batches = _batch_factory(
        reader, split.validation, loaded.feature_schema, evaluation_state,
        batch_size=config.common.batch_size,
    )

    def refit_batches() -> Iterator[RerankBatch]:
        if "production" not in states:
            states["production"] = _fit_state(
                reader,
                split.refit,
                loaded.feature_schema,
                batch_size=config.common.batch_size,
                maximum_vocabulary_size=max_vocab,
            )
            _validate_state_shape(states["production"], config.model)
        yield from _batch_factory(
            reader,
            split.refit,
            loaded.feature_schema,
            states["production"],
            batch_size=config.common.batch_size,
        )()

    def evaluator(
        adapter: ModelAdapter[RerankBatch], batches: Iterable[RerankBatch]
    ) -> dict[str, BinaryMetrics]:
        return evaluate_rerank(
            adapter,
            batches,
            targets=TARGETS,
            minimum_valid_rows=config.common.metrics.minimum_valid_rows,
        )

    def gates(metrics: dict[str, BinaryMetrics]) -> tuple[GateResult, ...]:
        results: list[GateResult] = []
        for target in TARGETS:
            metric = metrics[target]
            results.append(hard_gate(f"{target}.evaluable", passed=metric.auc is not None))
            if metric.auc is not None:
                results.append(quality_gate(
                    f"{target}.auc",
                    candidate=ScopedAuc(metric.auc, validation_slice_digest),
                    previous=None,
                    warning_floor=config.common.metrics.auc_warning_floor,
                ))
        return tuple(results)

    def pre_refit_validator(adapter: ModelAdapter[RerankBatch]) -> None:
        example = next(validation_batches())
        with tempfile.TemporaryDirectory() as temporary:
            export_and_verify(
                cast(RerankModel, adapter.module),
                example.numeric,
                example.categorical,
                Path(temporary) / "evaluation-preflight.onnx",
                absolute_tolerance=config.common.export.absolute_tolerance,
            )

    resolved_run_id = run_id or str(uuid.uuid4())
    config_digest = _sha256(canonical_json(
        config.common.model_dump(mode="json", exclude={"checkpoint_dir"})
    ))
    model_digest = _sha256(canonical_json(config.model.model_dump(mode="json")))
    source_revision = _source_revision()
    dependency_lock_digest = _sha256(Path("uv.lock").read_bytes())
    lineage = CheckpointLineage(config_digest, loaded.manifest.content_sha256, model_digest)
    pipeline = TrainingPipeline(
        adapter_factory=adapter_factory,
        optimizer_factory=lambda parameters: torch.optim.AdamW(
            parameters,
            lr=config.common.optimizer.learning_rate,
            weight_decay=config.common.optimizer.weight_decay,
        ),
        evaluator=evaluator,
        gate_builder=gates,
        checkpoint_agent=CheckpointAgent(config.common.checkpoint_dir),
        run_id=resolved_run_id,
        lineage=lineage,
        seed=config.common.seed,
        max_epochs=config.common.max_epochs,
        patience=config.common.early_stopping_patience,
        memory_limit_bytes=config.common.execution.memory_limit_bytes,
        pre_refit_validator=pre_refit_validator,
    )
    result = pipeline.run(BatchFactories(train_batches, validation_batches, refit_batches))
    example = next(refit_batches())
    with tempfile.TemporaryDirectory() as temporary:
        onnx_path = Path(temporary) / "model.onnx"
        exported = export_and_verify(
            cast(RerankModel, result.production_adapter.module),
            example.numeric,
            example.categorical,
            onnx_path,
            absolute_tolerance=config.common.export.absolute_tolerance,
        )
        files = {
            "model.onnx": onnx_path.read_bytes(),
            "signature.json": canonical_json({
                "inputs": {
                    "numeric": {"dtype": "float32", "shape": ["batch", config.model.numeric_input_dim]},
                    "categorical": {
                        "dtype": "int64",
                        "shape": ["batch", len(config.model.categorical_cardinalities)],
                    },
                },
                "outputs": list(exported.output_names),
                "opset": exported.opset,
            }),
            "model-config.json": canonical_json(config.model.model_dump(mode="json")),
            "feature-schema.json": loaded.feature_schema_body,
            "fitted-feature-state/state.json": canonical_json(asdict(states["production"])),
            "metrics.json": canonical_json({
                "targets": {
                    target: asdict(metric) for target, metric in result.metrics.items()
                },
                "epochs": [asdict(trace) for trace in result.traces],
            }),
            "lineage.json": canonical_json({
                **asdict(lineage),
                "source_revision": source_revision,
                "dependency_lock_digest": dependency_lock_digest,
                "seed": config.common.seed,
            }),
        }
    output_store = _store_for_output_prefix(output_uri, local_root=allowed_input_root)
    artifact_reloaded = False

    def validate_onnx_artifact(body: bytes) -> None:
        nonlocal artifact_reloaded
        try:
            reloaded = ort.InferenceSession(body, providers=["CPUExecutionProvider"])
        except Exception as error:
            raise ValueError("ONNX_EXPORT_FAILED: artifact reload failed") from error
        if tuple(item.name for item in reloaded.get_outputs()) != exported.output_names:
            raise ValueError("ONNX_EXPORT_FAILED: artifact output signature changed")
        artifact_reloaded = True

    manifest_uri_out = ArtifactBuilder(output_store, output_uri).commit(
        model_name="rerank",
        logical_inputs={
            "config": config_digest,
            "dataset": loaded.manifest.content_sha256,
            "model": model_digest,
            "source": source_revision,
            "dependencies": dependency_lock_digest,
            "onnx_opset": str(exported.opset),
        },
        audit_metadata={
            "run_id": resolved_run_id,
            "created_at": datetime.now(UTC).isoformat(),
        },
        files=files,
        validators={"model.onnx": validate_onnx_artifact},
    )
    return TrainExecutionResult(
        manifest_uri_out, result.selected_epoch, exported.output_names, artifact_reloaded
    )


def execute_backtest(
    *, manifest_uri: str, config_path: Path, allowed_input_root: Path | None = None
) -> dict[str, dict[str, object]]:
    config = _load_config(config_path)
    _configure_cpu_threads(
        config.common.execution.intra_op_threads,
        config.common.execution.inter_op_threads,
    )
    loaded = _load_inputs(manifest_uri, local_root=allowed_input_root)
    split = build_backtest_split(loaded.manifest.partitions)
    reader = ParquetDataset(loaded.store, spool_limit_bytes=64 * 1024 * 1024)
    state = _fit_state(
        reader,
        split.train,
        loaded.feature_schema,
        batch_size=config.common.batch_size,
        maximum_vocabulary_size=min(config.model.categorical_cardinalities, default=2) - 2,
    )
    _validate_state_shape(state, config.model)
    adapter = RerankAdapter(config.model)
    optimizer = torch.optim.AdamW(
        adapter.module.parameters(),
        lr=config.common.optimizer.learning_rate,
        weight_decay=config.common.optimizer.weight_decay,
    )

    def evaluate(
        adapter_to_score: ModelAdapter[RerankBatch], batches: Iterable[RerankBatch]
    ) -> dict[str, BinaryMetrics]:
        return evaluate_rerank(
            adapter_to_score,
            batches,
            targets=TARGETS,
            minimum_valid_rows=config.common.metrics.minimum_valid_rows,
        )

    result = run_backtest(
        adapter=adapter,
        optimizer=optimizer,
        train_batches=_batch_factory(
            reader, split.train, loaded.feature_schema, state,
            batch_size=config.common.batch_size,
        ),
        validation_batches=_batch_factory(
            reader, split.validation, loaded.feature_schema, state,
            batch_size=config.common.batch_size,
        ),
        test_batches=_batch_factory(
            reader, split.test, loaded.feature_schema, state,
            batch_size=config.common.batch_size,
        ),
        epochs=config.common.max_epochs,
        evaluator=evaluate,
    )
    return {
        "validation": {target: asdict(metric) for target, metric in result.validation.items()},
        "test": {target: asdict(metric) for target, metric in result.test.items()},
    }


def _exit_with_error(run_id: str, stage: str, error: Exception) -> NoReturn:
    typer.echo(RunError(
        run_id=run_id,
        stage=stage,
        code=classify_error(error),
        detail=redact_detail(str(error)),
    ).to_json(), err=True)
    raise typer.Exit(1) from None


@app.command()
def train(
    manifest: str | None = typer.Option(None, "--manifest"),
    config: Path | None = typer.Option(None, "--config", dir_okay=False),
    output: str | None = typer.Option(None, "--output"),
) -> None:
    run_id = str(uuid.uuid4())
    try:
        if manifest is None or config is None or output is None:
            raise ValueError("missing required options: --manifest, --config, --output")
        result = execute_train(
            manifest_uri=manifest,
            config_path=config,
            output_uri=output,
            run_id=run_id,
        )
    except Exception as error:
        _exit_with_error(run_id, "train", error)
    typer.echo(result.manifest_uri)


@app.command()
def validate(
    manifest: str | None = typer.Option(None, "--manifest"),
    config: Path | None = typer.Option(None, "--config", dir_okay=False),
) -> None:
    run_id = str(uuid.uuid4())
    try:
        if manifest is None or config is None:
            raise ValueError("missing required options: --manifest, --config")
        _load_config(config)
        loaded = _load_inputs(manifest, local_root=None)
    except Exception as error:
        _exit_with_error(run_id, "validate", error)
    typer.echo(loaded.manifest.content_sha256)


@app.command()
def backtest(
    manifest: str | None = typer.Option(None, "--manifest"),
    config: Path | None = typer.Option(None, "--config", dir_okay=False),
) -> None:
    run_id = str(uuid.uuid4())
    try:
        if manifest is None or config is None:
            raise ValueError("missing required options: --manifest, --config")
        result = execute_backtest(manifest_uri=manifest, config_path=config)
    except Exception as error:
        _exit_with_error(run_id, "backtest", error)
    typer.echo(json.dumps(result, sort_keys=True))
```

Keep the backtest training loop model-agnostic through the exact helper below:

```python
# recommend/train/comm/offline_evaluate.py
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from recommend.train.comm.model_protocol import ModelAdapter

BatchT = TypeVar("BatchT")
MetricT = TypeVar("MetricT")


@dataclass(frozen=True, slots=True)
class BacktestResult(Generic[MetricT]):
    validation: MetricT
    test: MetricT
def run_backtest(
    *,
    adapter: ModelAdapter[BatchT],
    optimizer: torch.optim.Optimizer,
    train_batches: Callable[[], Iterable[BatchT]],
    validation_batches: Callable[[], Iterable[BatchT]],
    test_batches: Callable[[], Iterable[BatchT]],
    epochs: int,
    evaluator: Callable[[ModelAdapter[BatchT], Iterable[BatchT]], MetricT],
) -> BacktestResult[MetricT]:
    for _ in range(epochs):
        adapter.module.train()
        observed = False
        for batch in train_batches():
            observed = True
            optimizer.zero_grad(set_to_none=True)
            loss = adapter.loss(batch)
            if not torch.isfinite(loss):
                raise ValueError("NUMERICAL_FAILURE")
            loss.backward()
            optimizer.step()
        if not observed:
            raise ValueError("DATA_EXHAUSTED")
    return BacktestResult(
        validation=evaluator(adapter, validation_batches()),
        test=evaluator(adapter, test_batches()),
    )
```

- [ ] **Step 4: Build a deterministic 28-day local fixture and finish direct composition**

```python
# recommend/train/tests/fixtures.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from recommend.train.comm.datasvr.dataset_manifest import DatasetManifest, Partition


@dataclass(frozen=True, slots=True)
class LocalTrainingFixture:
    manifest_uri: str
    config_path: Path


def one_row(day_index: int, row_index: int) -> dict[str, object]:
    positive = (day_index + row_index) % 2
    return {
        "event_time": f"2026-08-{day_index + 1:02d}T12:00:00Z",
        "watch_count_7d": float(day_index + row_index),
        "language": "en" if positive else "zh",
        "watch_duration_ms": 950 if positive else 100,
        "content_duration_ms": 1000,
        "effective_watch": positive,
        "effective_watch_valid": True,
        "non_fast_swipe": positive,
        "non_fast_swipe_valid": True,
        "immersive_click": positive,
        "immersive_click_valid": True,
    }


def build_local_28_day_fixture(root: Path) -> LocalTrainingFixture:
    input_root = root / "input"
    contracts_root = input_root / "contracts"
    contracts_root.mkdir(parents=True)
    feature_body = json.dumps({
        "version": "1",
        "features": [
            {
                "name": "watch_count_7d",
                "source_column": "watch_count_7d",
                "kind": "numeric",
                "window_days": 7,
                "as_of_column": "event_time",
                "window_closed": "left",
            },
            {
                "name": "language",
                "source_column": "language",
                "kind": "categorical",
                "window_days": None,
            },
        ],
    }, sort_keys=True).encode()
    label_body = json.dumps({
        "version": "1",
        "targets": ["effective_watch", "completion", "non_fast_swipe", "immersive_click"],
        "completion_expression": "content_duration_ms > 0 and watch_duration_ms / content_duration_ms >= 0.95",
    }, sort_keys=True).encode()
    feature_path = contracts_root / "features-v1.json"
    label_path = contracts_root / "labels-v1.json"
    feature_path.write_bytes(feature_body)
    label_path.write_bytes(label_body)

    first = date(2026, 8, 1)
    partitions: list[Partition] = []
    for day_index in range(28):
        event_date = first + timedelta(days=day_index)
        path = input_root / f"day={event_date.isoformat()}" / "part-000.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([one_row(day_index, row) for row in range(4)]), path)
        partitions.append(Partition(
            uri=path.as_uri(),
            event_date=event_date,
            min_event_time=datetime.combine(event_date, time.min, UTC),
            max_event_time=datetime.combine(event_date, time.max, UTC),
            row_count=4,
            size_bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ))
    payload: dict[str, object] = {
        "manifest_version": "1",
        "dataset_id": "rerank-smoke",
        "dataset_version": "2026-08-28",
        "feature_schema_version": "1",
        "feature_schema_uri": feature_path.as_uri(),
        "feature_schema_sha256": hashlib.sha256(feature_body).hexdigest(),
        "label_definition_version": "1",
        "label_definition_uri": label_path.as_uri(),
        "label_definition_sha256": hashlib.sha256(label_body).hexdigest(),
        "as_of_ms": 1_788_048_000_000,
        "label_maturity_hours": 24,
        "partitions": [item.model_dump(mode="json") for item in partitions],
        "row_count": 112,
    }
    payload["content_sha256"] = DatasetManifest.digest_payload(payload)
    manifest = DatasetManifest.model_validate(payload)
    manifest_path = input_root / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json())

    config = {
        "model_name": "rerank",
        "checkpoint_dir": str(root / "runs"),
        "seed": 7,
        "batch_size": 4,
        "max_epochs": 1,
        "early_stopping_patience": 1,
        "bad_row_policy": "fail",
        "execution": {
            "strategy": "single_process_cpu",
            "intra_op_threads": 2,
            "inter_op_threads": 1,
            "reader_workers": 0,
            "prefetch_batches": 1,
            "memory_limit_bytes": 1_073_741_824,
        },
        "split": {"train_days": 27, "validation_days": 1, "refit_days": 28},
        "optimizer": {"name": "adamw", "learning_rate": 0.001, "weight_decay": 0.0},
        "metrics": {"minimum_valid_rows": 2, "auc_warning_floor": 0.5},
        "export": {"onnx_opset": 18, "absolute_tolerance": 1e-5},
        "model_params": {
            "numeric_input_dim": 2,
            "categorical_cardinalities": [8],
            "embedding_dim": 2,
            "shared_hidden_dims": [4],
            "tower_hidden_dim": 2,
            "activation": "relu",
            "dropout": 0.0,
            "loss_weights": {
                "effective_watch": 1.0,
                "completion": 1.0,
                "non_fast_swipe": 1.0,
                "immersive_click": 1.0,
            },
        },
    }
    config_path = root / "train.yml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    return LocalTrainingFixture(manifest_path.as_uri(), config_path)
```

- [ ] **Step 5: Register the console alias and run CLI/smoke tests**

Add to `pyproject.toml`:

```toml
[project.scripts]
doku-train = "recommend.train.run:main"
```

Run: `uv lock && uv run pytest recommend/train/tests/unit/test_cli.py recommend/train/tests/smoke/test_train_rerank.py -v`
Expected: 6 tests pass; train smoke creates a reloadable local `model.onnx`, backtest reports validation/test, and
no S3 client is constructed.

- [ ] **Step 6: Commit CLI and smoke path**

```bash
git add pyproject.toml uv.lock recommend/train/run.py recommend/train/train_rerank.py recommend/train/comm/errors.py recommend/train/comm/offline_evaluate.py recommend/train/tests
git commit -m "feat: add rerank training cli"
```

### Task 12: Enforce quality policy and capture capacity evidence

**Requirements:** R001, NFR001, NFR002, NFR003, NFR004; acceptance criteria 1–7
**Depends on:** Tasks 1–11

**Files:**
- Create: `recommend/train/comm/resource_report.py`
- Create: `recommend/train/tests/unit/test_dependency_policy.py`
- Create: `recommend/train/tests/unit/test_security_contract.py`
- Create: `recommend/train/tests/performance/test_capacity_report.py`
- Modify: `Makefile`
- Modify: `recommend/train/train_rerank.py`
- Modify: `recommend/train/comm/artifact_utils.py`
- Modify: `AGENTS.md`
- Modify: `docs/architecture/system-overview.md`
- Modify: `docs/architecture/components.md`
- Modify: `docs/architecture/data-model.md`
- Modify: `docs/architecture/interfaces.md`
- Modify: `specs/001-cpu-training-scaffold/verification.md`

**Completion conditions:** `make check` covers format/lint/type/unit/contract/smoke/schema checks; lockfile has no
GPU runtime packages; logs/signatures contain no raw `user_id` or credentials; capacity report records required
fields without asserting a production SLA; verification evidence contains fresh command outputs.

- [ ] **Step 1: Write failing dependency, security, and resource-report tests**

```python
# recommend/train/tests/unit/test_dependency_policy.py
from pathlib import Path


def test_lock_has_no_gpu_runtime_packages() -> None:
    lock = Path("uv.lock").read_text().lower()
    forbidden = ("nvidia-cuda", "nvidia-cudnn", "nvidia-nccl", "torch-cuda")
    assert not any(package in lock for package in forbidden)
```

```python
# recommend/train/tests/unit/test_security_contract.py
from recommend.train.comm.feature_check import FeatureSchema, FeatureSpec, validate_feature_schema


def test_raw_user_id_cannot_reach_signature() -> None:
    schema = FeatureSchema(version="1", features=(
        FeatureSpec(name="user_id", source_column="user_id", kind="categorical", window_days=None),
    ))
    try:
        validate_feature_schema(schema)
    except ValueError as error:
        assert "user_id" in str(error)
    else:
        raise AssertionError("raw user_id was accepted")
```

```python
# recommend/train/tests/performance/test_capacity_report.py
import pytest

from recommend.train.comm.resource_report import CapacityRecorder


@pytest.mark.performance
def test_capacity_report_contains_auditable_fields() -> None:
    report = CapacityRecorder.start(
        row_count=28_000_000, shard_count=280, total_bytes=10_000_000_000
    ).finish(processed_rows=10, processed_bytes=1_000)
    assert report.row_count == 28_000_000
    assert report.cpu_count >= 1
    assert report.cpu_model
    assert report.available_memory_bytes > 0
    assert report.peak_rss_bytes > 0
    assert report.elapsed_seconds >= 0.0
    assert report.samples_per_second >= 0.0
    assert report.bytes_per_second >= 0.0
    assert not hasattr(report, "sla_passed")
```

- [ ] **Step 2: Run policy tests and verify missing capacity recorder**

Run: `uv run pytest recommend/train/tests/unit/test_dependency_policy.py recommend/train/tests/unit/test_security_contract.py recommend/train/tests/performance/test_capacity_report.py -v`
Expected: capacity test fails because `resource_report.py` is absent; dependency and security tests pass.

- [ ] **Step 3: Implement an auditable resource report without an SLA gate**

```python
# recommend/train/comm/resource_report.py
from __future__ import annotations

import os
import platform
import resource
import sys
import time
from dataclasses import dataclass

import psutil


@dataclass(frozen=True, slots=True)
class CapacityReport:
    row_count: int
    processed_rows: int
    shard_count: int
    total_bytes: int
    processed_bytes: int
    cpu_count: int
    cpu_model: str
    available_memory_bytes: int
    peak_rss_bytes: int
    elapsed_seconds: float
    samples_per_second: float
    bytes_per_second: float


@dataclass(frozen=True, slots=True)
class CapacityRecorder:
    row_count: int
    shard_count: int
    total_bytes: int
    started_at: float

    @classmethod
    def start(cls, *, row_count: int, shard_count: int, total_bytes: int) -> "CapacityRecorder":
        return cls(row_count, shard_count, total_bytes, time.monotonic())

    def finish(self, *, processed_rows: int, processed_bytes: int) -> CapacityReport:
        elapsed = time.monotonic() - self.started_at
        maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_bytes = maximum_rss if sys.platform == "darwin" else maximum_rss * 1024
        return CapacityReport(
            row_count=self.row_count,
            processed_rows=processed_rows,
            shard_count=self.shard_count,
            total_bytes=self.total_bytes,
            processed_bytes=processed_bytes,
            cpu_count=os.cpu_count() or 1,
            cpu_model=platform.processor() or platform.machine(),
            available_memory_bytes=psutil.virtual_memory().available,
            peak_rss_bytes=peak_rss_bytes,
            elapsed_seconds=elapsed,
            samples_per_second=processed_rows / elapsed if elapsed > 0.0 else 0.0,
            bytes_per_second=processed_bytes / elapsed if elapsed > 0.0 else 0.0,
        )
```

Integrate the recorder into `execute_train` immediately after Manifest validation and add the result to the
artifact before commit:

```python
# additions in recommend/train/train_rerank.py
from recommend.train.comm.resource_report import CapacityRecorder

capacity = CapacityRecorder.start(
    row_count=loaded.manifest.row_count,
    shard_count=len(loaded.manifest.partitions),
    total_bytes=sum(item.size_bytes for item in loaded.manifest.partitions),
)

# after train/refit/export and before ArtifactBuilder.commit
resource_report = capacity.finish(
    processed_rows=loaded.manifest.row_count,
    processed_bytes=sum(item.size_bytes for item in loaded.manifest.partitions),
)
files["resource-report.json"] = canonical_json(asdict(resource_report))
```

Apply the same exact key to `ArtifactBuilder` and its contract fixture:

```python
required = {
    "model.onnx",
    "signature.json",
    "model-config.json",
    "feature-schema.json",
    "fitted-feature-state/state.json",
    "metrics.json",
    "lineage.json",
    "resource-report.json",
}
```

- [ ] **Step 4: Make all automated gates explicit**

Replace the `Makefile` with:

```makefile
.PHONY: format lint typecheck schema-check unit contract smoke test check capacity
format:
	uv run ruff format --check .
lint:
	uv run ruff check .
typecheck:
	uv run mypy recommend
schema-check:
	uv run python -c 'from pathlib import Path; from recommend.train.comm.model_params import TrainConfig; from recommend.train.comm.model_params_validator import schema_matches; from recommend.train.models.rerank.model_params import RerankModelParams; assert schema_matches(TrainConfig, Path("recommend/train/comm/config.schema.yml")); assert schema_matches(RerankModelParams, Path("recommend/train/models/rerank/config.schema.yml"))'
unit:
	uv run pytest recommend/train/tests/unit -q
contract:
	uv run pytest recommend/train/tests/contract -q
smoke:
	uv run pytest recommend/train/tests/smoke -q
test: unit contract smoke
check: format lint typecheck schema-check test
capacity:
	uv run pytest recommend/train/tests/performance -q -m performance
```

- [ ] **Step 5: Run the full verification suite and opt-in capacity probe**

Run:
```bash
uv run ruff format .
uv run ruff check --fix .
make check
make capacity
git diff --check
```
Expected: every command exits 0; format/import fixes leave no remaining lint diff; `make check` runs unit,
contract, smoke and ONNX reload; `make capacity` emits a
`CapacityReport` but no SLA verdict; `git diff --check` is silent.

- [ ] **Step 6: Record fresh evidence and synchronize docs**

In `verification.md`, change each implemented matrix row from `PLANNED` to `PASS`, record the exact command,
test count, UTC execution time, source revision, machine CPU/memory, and artifact Manifest digest. Update architecture
docs only when implementation differs from the approved decision table; never rewrite `docs/source/`.

- [ ] **Step 7: Commit verified Spec 001 implementation**

```bash
git add Makefile AGENTS.md docs/architecture specs/001-cpu-training-scaffold/verification.md recommend/train
git commit -m "test: verify cpu training scaffold"
```

## Execution handoff

After this plan is approved, choose one implementation mode:

1. **Subagent-Driven (recommended):** use `superpowers:subagent-driven-development`, dispatch one fresh worker per
   task, and run requirements/compliance review between tasks.
2. **Inline Execution:** use `superpowers:executing-plans`, execute in dependency-ordered batches, and stop at each
   review checkpoint.

Neither mode may implement code until the user explicitly chooses execution.
