import hashlib
import json
from unittest.mock import Mock

import pytest

from recommend_tf.platforms.sagemaker.publish_model import ReleaseState
from recommend_tf.platforms.sagemaker.ready_event import with_release
from recommend_tf.platforms.sagemaker.recover_daily import enqueue_ready, recover
from recommend_tf.train.tests.test_daily_publication import S3


class Store(S3):
    def list_objects_v2(self, Bucket, Prefix, MaxKeys, ContinuationToken=None):
        keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = {"Contents": [{"Key": k} for k in keys[start : start + MaxKeys]]}
        if start + MaxKeys < len(keys):
            page["NextContinuationToken"] = str(start + MaxKeys)
        return page


def fixture():
    s3, events, policy = Store(), Mock(), Mock()
    policy.allowed.return_value = True
    policy.read.return_value = {"recovery_max_items": 1}
    events.put_events.return_value = {"FailedEntryCount": 0}
    settings = dict(
        deferred_uri="s3://bucket/deferred",
        state_uri="s3://bucket/state",
        kms_key="key",
        cluster_arn="cluster",
        event_source="source",
        event_bus="bus",
        detail_type="ready",
        release=dict(
            config_uri="s3://bucket/config",
            config_sha256="c" * 64,
            binding_uri="s3://bucket/binding",
            binding_sha256="b" * 64,
        ),
    )
    return s3, events, policy, settings


def pending(s3, settings, cutoff="2026-10-31"):
    return enqueue_ready(
        s3,
        settings,
        {"cutoff": cutoff},
        dict(
            version="doku-tf-ready-event/v1",
            status="published",
            ready_uri="s3://bucket/ready",
            ready_sha256="a" * 64,
        ),
    )


def test_durable_producer_request_is_idempotent_and_replays():
    s3, events, policy, settings = fixture()
    first = pending(s3, settings)
    assert pending(s3, settings) == first
    assert recover(s3, events, settings, policy)["count"] == 1
    detail = json.loads(events.put_events.call_args.kwargs["Entries"][0]["Detail"])
    assert detail["ready_uri"] == first["ready_uri"]
    assert recover(s3, events, settings, policy)["count"] == 1
    # Delivery accepted is not a terminal ACK: the journal closes the intent.
    journal = ReleaseState(s3, settings["state_uri"], "key")
    journal.value["high_water"] = "2026-10-31"
    journal.save()
    assert recover(s3, events, settings, policy)["count"] == 0


def test_failed_delivery_does_not_advance_cursor():
    s3, events, policy, settings = fixture()
    pending(s3, settings)
    events.put_events.return_value = {"FailedEntryCount": 1}
    with pytest.raises(RuntimeError, match="delivery failed"):
        recover(s3, events, settings, policy)
    assert ("bucket", "deferred/cursor.json") not in s3.objects


def test_paused_recovery_and_active_executor_do_not_start_work():
    s3, events, policy, settings = fixture()
    pending(s3, settings)
    policy.allowed.return_value = False
    assert recover(s3, events, settings, policy)["status"] == "paused"
    policy.allowed.return_value = True
    journal = ReleaseState(s3, settings["state_uri"], "key")
    journal.claim("active-operation", "2026-10-31")
    assert recover(s3, events, settings, policy)["count"] == 0
    events.put_events.assert_not_called()


def test_packaged_recipe_owns_window_and_digest():
    raw = json.dumps({"online_config": {"training_date": 28, "validation_date": 1}})
    packaged = dict(
        schema_version="doku-tf-ready-release/v1",
        config_json=raw,
        release={"config_sha256": hashlib.sha256(raw.encode()).hexdigest()},
    )
    settings = with_release(
        {"deferred_uri": "s3://bucket/deferred", "training_days": 999}, packaged
    )
    assert (settings["training_days"], settings["validation_days"]) == (28, 1)
    packaged["config_json"] += " "
    with pytest.raises(ValueError, match="digest"):
        with_release(settings, packaged)


def test_manual_rollback_is_not_superseded_by_high_water():
    s3, events, policy, settings = fixture()
    record = dict(
        version="doku-tf-deferred/v1",
        operation="rollback-op",
        cutoff="2026-10-30",
        arguments={"rollback_version": "retained-model"},
    )
    raw = json.dumps(record).encode()
    sha = hashlib.sha256(raw).hexdigest()
    s3.put_object(Bucket="bucket", Key="deferred/requests/one/" + sha + ".json", Body=raw)
    journal = ReleaseState(s3, settings["state_uri"], "key")
    journal.value["high_water"] = "2026-10-31"
    journal.save()
    assert recover(s3, events, settings, policy)["count"] == 1
