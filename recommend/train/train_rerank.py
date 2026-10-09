from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa
import torch

from recommend.train.comm.checkpoint_agent import (
    CheckpointAgent,
    CheckpointLineage,
    CheckpointRole,
)
from recommend.train.comm.datasvr.dataset_manifest import DatasetManifest, ShardManifest
from recommend.train.comm.datasvr.parquet_dataset import ParquetDataset
from recommend.train.comm.datasvr.storage import ObjectStore
from recommend.train.comm.errors import error_stage
from recommend.train.comm.eval.eval_model_rerank import evaluate_rerank
from recommend.train.comm.eval.gates import GateResult, build_metric_gates
from recommend.train.comm.feature_check import validate_feature_names, validate_point_in_time
from recommend.train.comm.metrics_utils import MetricSet
from recommend.train.comm.model_params_validator import ValidatedTrainConfig, load_config
from recommend.train.comm.model_protocol import AdapterFactory, ModelAdapter, ModelRole
from recommend.train.comm.offline_evaluate import BacktestWindow, build_backtest_windows
from recommend.train.comm.resource_report import ResourceTracker
from recommend.train.comm.run_support import (
    build_output_store,
    digest_json,
    lock_digest,
    source_revision,
    validate_dataset,
)
from recommend.train.comm.training_output import TrainingOutputUris, TrainingOutputWriter
from recommend.train.comm.training_pipeline import BatchFactories, PipelineResult, TrainingPipeline
from recommend.train.models.rerank import RerankAdapter
from recommend.train.models.rerank.data_utils import RerankBatch, TemporalSplit, build_daily_split
from recommend.train.models.rerank.feature_utils import (
    FittedFeatureState,
    fit_feature_state,
    transform_batch,
)
from recommend.train.models.rerank.model_params import RerankModelParams


@dataclass(frozen=True, slots=True)
class ValidatedInput:
    manifest: DatasetManifest
    config: ValidatedTrainConfig[RerankModelParams]
    store: ObjectStore
    split: TemporalSplit


@dataclass(frozen=True, slots=True)
class TrainExecutionResult:
    outputs: TrainingOutputUris
    selected_epoch: int
    gates: tuple[GateResult, ...]
    checkpoint_reloaded: bool


def validate_input(*, manifest_uri: str, config_path: Path) -> ValidatedInput:
    dataset = validate_dataset(manifest_uri)
    store = dataset.store
    manifest = dataset.manifest
    config = load_config(config_path, RerankModelParams)
    if config.common.model_name != "rerank":
        raise ValueError("INPUT_CONTRACT_INVALID: model_name must be rerank")
    validate_feature_names((*config.model.numeric_features, *config.model.categorical_features))
    return ValidatedInput(
        manifest=manifest,
        config=config,
        store=store,
        split=build_daily_split(manifest),
    )


def execute_backtest(*, manifest_uri: str, config_path: Path) -> tuple[BacktestWindow, ...]:
    validated = validate_input(manifest_uri=manifest_uri, config_path=config_path)
    dates = tuple(shard.event_date for shard in validated.manifest.shards)
    return build_backtest_windows(dates, train_days=24, validation_days=2, test_days=2)


def _params_for_state(params: RerankModelParams, state: FittedFeatureState) -> RerankModelParams:
    return params.model_copy(
        update={"categorical_cardinalities": tuple(item.cardinality for item in state.categorical)}
    )


def _configure_threads(intra_op_threads: int, inter_op_threads: int) -> None:
    torch.set_num_threads(intra_op_threads)
    if torch.get_num_interop_threads() != inter_op_threads:
        torch.set_num_interop_threads(inter_op_threads)


def execute_train(
    *,
    manifest_uri: str,
    config_path: Path,
    output_prefix: str,
    run_id: str,
) -> TrainExecutionResult:
    resources = ResourceTracker()
    with error_stage("validate"), resources.stage("validate_input"):
        validated = validate_input(manifest_uri=manifest_uri, config_path=config_path)
    config = validated.config
    common = config.common
    _configure_threads(
        common.execution.intra_op_threads,
        common.execution.inter_op_threads,
    )
    dataset = ParquetDataset(
        validated.store,
        validated.manifest,
        batch_size=common.batch_size,
    )
    statistical_features = tuple(
        name for name in config.model.numeric_features if name.endswith("_7d")
    )

    def raw_batches(shards: Iterable[ShardManifest]) -> Iterator[pa.RecordBatch]:
        for batch in dataset.iter_batches(shards):
            validate_point_in_time(batch, statistical_features)
            yield batch

    with error_stage("evaluate"), resources.stage("feature_fit_evaluation"):
        evaluation_state = fit_feature_state(
            raw_batches(validated.split.train),
            numeric_features=config.model.numeric_features,
            categorical_features=config.model.categorical_features,
            feature_schema_version=validated.manifest.feature_schema_version,
        )
    with error_stage("refit"), resources.stage("feature_fit_production"):
        production_state = fit_feature_state(
            raw_batches(validated.split.refit),
            numeric_features=config.model.numeric_features,
            categorical_features=config.model.categorical_features,
            feature_schema_version=validated.manifest.feature_schema_version,
        )
    evaluation_params = _params_for_state(config.model, evaluation_state)
    production_params = _params_for_state(config.model, production_state)

    def transformed(
        shards: Iterable[ShardManifest], state: FittedFeatureState
    ) -> Iterator[RerankBatch]:
        return (transform_batch(batch, state) for batch in raw_batches(shards))

    factories = BatchFactories[RerankBatch](
        train=lambda: transformed(validated.split.train, evaluation_state),
        validation=lambda: transformed(validated.split.validation, evaluation_state),
        refit=lambda: transformed(validated.split.refit, production_state),
    )

    def adapter_factory(role: ModelRole) -> RerankAdapter:
        return RerankAdapter(evaluation_params if role == "evaluation" else production_params)

    config_digest = digest_json(
        {
            "common": common.model_dump(mode="json"),
            "model": config.model.model_dump(mode="json"),
        }
    )
    lineage = CheckpointLineage(
        config_digest=config_digest,
        dataset_manifest_digest=validated.manifest.content_sha256,
        model_structure_digest=digest_json(config.model.model_dump(mode="json")),
    )
    checkpoint_agent = CheckpointAgent(common.checkpoint_dir)
    validation_slice_digest = digest_json(
        [shard.model_dump(mode="json") for shard in validated.split.validation]
    )

    def evaluator(adapter: ModelAdapter[RerankBatch], batches: Iterable[RerankBatch]) -> MetricSet:
        return evaluate_rerank(
            adapter,
            batches,
            minimum_valid_rows=common.metrics.minimum_valid_rows,
            validation_slice_digest=validation_slice_digest,
        )

    pipeline = TrainingPipeline[RerankBatch, MetricSet](
        adapter_factory=cast(AdapterFactory[RerankBatch], adapter_factory),
        optimizer_factory=lambda parameters: torch.optim.AdamW(
            parameters,
            lr=common.optimizer.learning_rate,
            weight_decay=common.optimizer.weight_decay,
        ),
        evaluator=evaluator,
        gate_builder=lambda metrics: build_metric_gates(
            metrics, auc_warning_floor=common.metrics.auc_warning_floor
        ),
        checkpoint_agent=checkpoint_agent,
        run_id=run_id,
        lineage=lineage,
        seed=common.seed,
        max_epochs=common.max_epochs,
        patience=common.early_stopping_patience,
        memory_limit_bytes=common.execution.memory_limit_bytes,
    )
    with resources.stage("train_evaluate_refit"):
        result: PipelineResult[RerankBatch, MetricSet] = pipeline.run(factories)
    checkpoint_body = result.production_checkpoint.read_bytes()
    with error_stage("refit"), resources.stage("checkpoint_verify"):
        verification_adapter = RerankAdapter(production_params)
        verification_optimizer = torch.optim.AdamW(verification_adapter.module.parameters())
        checkpoint_agent.load_bytes(
            checkpoint_body,
            CheckpointRole.PRODUCTION,
            verification_adapter.module,
            verification_optimizer,
            lineage,
        )

    batch_count = sum(
        (shard.row_count + common.batch_size - 1) // common.batch_size
        for shard in validated.manifest.shards
    )
    resource_report = resources.report(
        row_count=validated.manifest.row_count,
        batch_count=batch_count,
        byte_count=sum(shard.size_bytes for shard in validated.manifest.shards),
    )

    metrics_body = json.dumps(
        {
            "metric_set_version": "1",
            "validation_slice_digest": result.metrics.validation_slice_digest,
            "targets": {
                metric.target: metric.model_dump(mode="json") for metric in result.metrics.targets
            },
            "gates": [asdict(gate) for gate in result.gates],
            "selected_epoch": result.selected_epoch,
            "traces": [asdict(trace) for trace in result.traces],
            "resource_report": asdict(resource_report),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    lineage_body = json.dumps(
        {
            "run_id": run_id,
            "source_revision": source_revision(),
            "dependency_lock_digest": lock_digest(),
            "config_digest": config_digest,
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
        outputs = TrainingOutputWriter(output_store, output_prefix).write(
            run_id,
            {
                "checkpoint.pt": checkpoint_body,
                "metrics.json": metrics_body,
                "fitted-feature-state/state.json": production_state.to_json_bytes(),
                "lineage.json": lineage_body,
            },
        )
        readback = output_store.get_bytes(outputs.checkpoint, max_bytes=len(checkpoint_body))
        reloaded_adapter = RerankAdapter(production_params)
        reloaded_optimizer = torch.optim.AdamW(reloaded_adapter.module.parameters())
        checkpoint_agent.load_bytes(
            readback,
            CheckpointRole.PRODUCTION,
            reloaded_adapter.module,
            reloaded_optimizer,
            lineage,
        )
    return TrainExecutionResult(
        outputs=outputs,
        selected_epoch=result.selected_epoch,
        gates=result.gates,
        checkpoint_reloaded=True,
    )
