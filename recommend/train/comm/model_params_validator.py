from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from recommend.train.comm.model_params import TrainConfig


@dataclass(frozen=True, slots=True)
class ValidatedTrainConfig[ModelParamsT: BaseModel]:
    common: TrainConfig
    model: ModelParamsT


def validate_config_payload[ModelParamsT: BaseModel](
    payload: dict[str, object], model_type: type[ModelParamsT]
) -> ValidatedTrainConfig[ModelParamsT]:
    common = TrainConfig.model_validate(payload)
    model = model_type.model_validate(common.model_params)
    return ValidatedTrainConfig(common=common, model=model)


def load_config[ModelParamsT: BaseModel](
    path: Path, model_type: type[ModelParamsT]
) -> ValidatedTrainConfig[ModelParamsT]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("training config must be a mapping")
    return validate_config_payload(payload, model_type)


def write_schema(model_type: type[BaseModel], path: Path) -> None:
    path.write_text(yaml.safe_dump(model_type.model_json_schema(), sort_keys=True))


def schema_matches(model_type: type[BaseModel], path: Path) -> bool:
    loaded: object = yaml.safe_load(path.read_text())
    return loaded == model_type.model_json_schema()
