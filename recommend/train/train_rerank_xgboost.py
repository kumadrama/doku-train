from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

# XGBoost must initialize OpenMP before PyTorch on macOS.
# isort: off
import xgboost as _xgboost

# isort: on
import pyarrow as pa
import torch

from recommend.train.comm.datasvr.dataset_manifest import DatasetManifest, ShardManifest
from recommend.train.comm.datasvr.parquet_dataset import ParquetDataset
from recommend.train.comm.datasvr.storage import ObjectStore
from recommend.train.comm.errors import PipelineBlocked, error_stage
from recommend.train.comm.eval.eval_model_rerank import evaluate_rerank
from recommend.train.comm.eval.gates import GateResult, GateSeverity, build_metric_gates
from recommend.train.comm.feature_check import validate_feature_names, validate_point_in_time
from recommend.train.comm.offline_evaluate import BacktestWindow, build_backtest_windows
from recommend.train.comm.resource_report import ResourceTracker
from recommend.train.comm.run_support import (
    build_output_store,
    digest_json,
    lock_digest,
    source_revision,
    validate_dataset,
)
from recommend.train.comm.training_output import (
    ModelTrainingOutputUris,
    ModelTrainingOutputWriter,
)
from recommend.train.models.rerank.data_utils import RerankBatch, TemporalSplit, build_daily_split
from recommend.train.models.rerank.feature_utils import (
    FittedFeatureState,
    fit_feature_state,
    transform_batch,
)
from recommend.train.models.rerank_train import (
    RerankTrainModel,
    RerankTrainTrainer,
    ValidatedRerankTrainConfig,
    load_rerank_train_config,
)

XGBOOST_ARTIFACT_NAME = "checkpoint.xgb.zip"


@dataclass(frozen=True, slots=True)
class ValidatedXGBoostInput:
    manifest: DatasetManifest
    config: ValidatedRerankTrainConfig
    store: ObjectStore
    split: TemporalSplit


@dataclass(frozen=True, slots=True)
class XGBoostTrainExecutionResult:
    outputs: ModelTrainingOutputUris
    selected_rounds: dict[str, int]
    gates: tuple[GateResult, ...]
    checkpoint_reloaded: bool


def validate_xgboost_input(*, manifest_uri: str, config_path: Path) -> ValidatedXGBoostInput:
    dataset = validate_dataset(manifest_uri)
    config = load_rerank_train_config(config_path)
    validate_feature_names((*config.model.numeric_features, *config.model.categorical_features))
    return ValidatedXGBoostInput(
        manifest=dataset.manifest,
        config=config,
        store=dataset.store,
        split=build_daily_split(dataset.manifest),
    )


def execute_xgboost_backtest(*, manifest_uri: str, config_path: Path) -> tuple[BacktestWindow, ...]:
    validated = validate_xgboost_input(manifest_uri=manifest_uri, config_path=config_path)
    dates = tuple(shard.event_date for shard in validated.manifest.shards)
    return build_backtest_windows(dates, train_days=24, validation_days=2, test_days=2)


def _write_local_checkpoint(root: Path, run_id: str, body: bytes) -> Path:
    if not run_id or "/" in run_id or ".." in run_id:
        raise ValueError("invalid run_id")
    path = root / run_id / "checkpoints" / "production" / XGBOOST_ARTIFACT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix="checkpoint-",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(body)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def execute_xgboost_train(
    *,
    manifest_uri: str,
    config_path: Path,
    output_prefix: str,
    run_id: str,
) -> XGBoostTrainExecutionResult:
    resources = ResourceTracker()
    with error_stage("validate"), resources.stage("validate_input"):
        validated = validate_xgboost_input(manifest_uri=manifest_uri, config_path=config_path)
    common = validated.config.common
    params = validated.config.model
    torch.set_num_threads(common.execution.intra_op_threads)
    if torch.get_num_interop_threads() != common.execution.inter_op_threads:
        torch.set_num_interop_threads(common.execution.inter_op_threads)
    dataset = ParquetDataset(
        validated.store,
        validated.manifest,
        batch_size=common.batch_size,
    )
    statistical_features = tuple(name for name in params.numeric_features if name.endswith("_7d"))

    def raw_batches(shards: Iterable[ShardManifest]) -> Iterator[pa.RecordBatch]:
        for batch in dataset.iter_batches(shards):
            validate_point_in_time(batch, statistical_features)
            yield batch

    with error_stage("evaluate"), resources.stage("feature_fit_evaluation"):
        evaluation_state = fit_feature_state(
            raw_batches(validated.split.train),
            numeric_features=params.numeric_features,
            categorical_features=params.categorical_features,
            feature_schema_version=validated.manifest.feature_schema_version,
        )
    with error_stage("refit"), resources.stage("feature_fit_production"):
        production_state = fit_feature_state(
            raw_batches(validated.split.refit),
            numeric_features=params.numeric_features,
            categorical_features=params.categorical_features,
            feature_schema_version=validated.manifest.feature_schema_version,
        )

    def transformed(
        shards: Iterable[ShardManifest], state: FittedFeatureState
    ) -> Iterator[RerankBatch]:
        return (transform_batch(batch, state) for batch in raw_batches(shards))

    def train_batches() -> Iterator[RerankBatch]:
        return transformed(validated.split.train, evaluation_state)

    def validation_batches() -> Iterator[RerankBatch]:
        return transformed(validated.split.validation, evaluation_state)

    def refit_batches() -> Iterator[RerankBatch]:
        return transformed(validated.split.refit, production_state)

    trainer = RerankTrainTrainer(
        params,
        seed=common.seed,
        nthread=common.execution.intra_op_threads,
        memory_limit_bytes=common.execution.memory_limit_bytes,
    )
    with error_stage("evaluate"), resources.stage("train_evaluation"):
        evaluation = trainer.fit_evaluation(train_batches, validation_batches)
        validation_slice_digest = digest_json(
            [shard.model_dump(mode="json") for shard in validated.split.validation]
        )
        metrics = evaluate_rerank(
            evaluation.model,
            validation_batches(),
            minimum_valid_rows=common.metrics.minimum_valid_rows,
            validation_slice_digest=validation_slice_digest,
        )
        gates = build_metric_gates(metrics, auc_warning_floor=common.metrics.auc_warning_floor)
        blocked = [gate.name for gate in gates if gate.severity is GateSeverity.BLOCK]
        if blocked:
            raise PipelineBlocked(
                "EVALUATION_GATE_FAILED:" + ",".join(blocked),
                stage="evaluate",
            )

    with error_stage("refit"), resources.stage("train_production_refit"):
        production_model = trainer.fit_production(refit_batches, evaluation.selected_rounds)
        checkpoint_body = production_model.to_bytes()
        checkpoint_path = _write_local_checkpoint(common.checkpoint_dir, run_id, checkpoint_body)
        checkpoint_body = checkpoint_path.read_bytes()
        verified_model = RerankTrainModel.from_bytes(checkpoint_body, params)
        verification_batch = next(refit_batches(), None)
        if verification_batch is None:
            raise PipelineBlocked("DATA_EXHAUSTED", stage="refit")
        verified_model.logits(verification_batch)

    batch_count = sum(
        (shard.row_count + common.batch_size - 1) // common.batch_size
        for shard in validated.manifest.shards
    )
    resource_report = resources.report(
        row_count=validated.manifest.row_count,
        batch_count=batch_count,
        byte_count=sum(shard.size_bytes for shard in validated.manifest.shards),
    )
    config_digest = digest_json(
        {
            "common": common.model_dump(mode="json"),
            "model": params.model_dump(mode="json"),
        }
    )
    metrics_body = json.dumps(
        {
            "metric_set_version": "1",
            "validation_slice_digest": metrics.validation_slice_digest,
            "targets": {
                metric.target: metric.model_dump(mode="json") for metric in metrics.targets
            },
            "gates": [asdict(gate) for gate in gates],
            "selected_rounds": evaluation.selected_rounds,
            "traces": [asdict(trace) for trace in evaluation.traces],
            "resource_report": asdict(resource_report),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    lineage_body = json.dumps(
        {
            "run_id": run_id,
            "model_name": "rerank_train",
            "xgboost_version": _xgboost.__version__,
            "source_revision": source_revision(),
            "dependency_lock_digest": lock_digest(),
            "config_digest": config_digest,
            "model_structure_digest": digest_json(params.model_dump(mode="json")),
            "dataset_manifest_digest": validated.manifest.content_sha256,
            "feature_schema_version": validated.manifest.feature_schema_version,
            "label_definition_version": validated.manifest.label_definition_version,
            "seed": common.seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with error_stage("write-output"):
        output_store = build_output_store(output_prefix)
        outputs = ModelTrainingOutputWriter(
            output_store,
            output_prefix,
            XGBOOST_ARTIFACT_NAME,
        ).write(
            run_id,
            model_artifact=checkpoint_body,
            metrics=metrics_body,
            fitted_feature_state=production_state.to_json_bytes(),
            lineage=lineage_body,
        )
        readback = output_store.get_bytes(
            outputs.model_artifact,
            max_bytes=len(checkpoint_body),
        )
        reloaded = RerankTrainModel.from_bytes(readback, params)
        reloaded.logits(verification_batch)
    return XGBoostTrainExecutionResult(
        outputs=outputs,
        selected_rounds=evaluation.selected_rounds,
        gates=gates,
        checkpoint_reloaded=True,
    )
