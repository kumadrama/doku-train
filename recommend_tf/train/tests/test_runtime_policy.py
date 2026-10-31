import io
import json
from unittest.mock import Mock

import pytest

from recommend_tf.platforms.runtime_policy import PolicyUnavailable, RuntimePolicy, validate


def fixture():
    return dict(
        schema_version="doku-tf-runtime/v1",
        training_enabled=True,
        publication_enabled=False,
        recovery_enabled=False,
        recovery_max_items=1,
    )


def test_validation_fails_closed():
    for change in (
        {"training_enabled": 1},
        {"recovery_max_items": True},
        {"recovery_max_items": 0},
        {"extra": "value"},
    ):
        with pytest.raises(ValueError):
            validate(dict(fixture(), **change))


def test_token_reuse_empty_response_and_expiry():
    now = [0]
    client = Mock()
    client.start_configuration_session.return_value = {"InitialConfigurationToken": "one"}
    client.get_latest_configuration.side_effect = [
        dict(
            NextPollConfigurationToken="two",
            NextPollIntervalInSeconds=15,
            Configuration=io.BytesIO(json.dumps(fixture()).encode()),
            VersionLabel="v1",
        ),
        dict(
            NextPollConfigurationToken="three",
            NextPollIntervalInSeconds=120,
            Configuration=io.BytesIO(b""),
        ),
    ]
    policy = RuntimePolicy(
        client, dict(application_id="a", environment_id="e", profile_id="p"), clock=lambda: now[0]
    )
    assert policy.allowed("training")
    assert not policy.allowed("publication")
    assert client.get_latest_configuration.call_count == 1
    now[0] = 16
    assert policy.allowed("training")
    client.get_latest_configuration.assert_called_with(ConfigurationToken="two")
    now[0] = 77
    with pytest.raises(PolicyUnavailable):
        policy.read()


def test_bad_refresh_never_uses_previous_allow():
    client = Mock()
    client.start_configuration_session.return_value = {"InitialConfigurationToken": "one"}
    client.get_latest_configuration.return_value = dict(
        NextPollConfigurationToken="two",
        NextPollIntervalInSeconds=15,
        Configuration=io.BytesIO(b"{}"),
    )
    policy = RuntimePolicy(client, dict(application_id="a", environment_id="e", profile_id="p"))
    with pytest.raises(PolicyUnavailable):
        policy.allowed("training")
