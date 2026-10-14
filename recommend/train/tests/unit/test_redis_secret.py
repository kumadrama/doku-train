import json
from urllib.parse import unquote, urlsplit

import boto3
import pytest

from recommend.train.publish_model import distribution_from_environment, redis_url_from_secret


def test_platform_secret_is_encoded_and_tls_required(monkeypatch):
    value = dict(
        host="master.example.com",
        port=6379,
        db=0,
        tls=True,
        username="model-user",
        password="p@ss:/?#word",
    )

    class Secrets:
        def get_secret_value(self, *, SecretId):
            assert SecretId == "test/model/readwrite"
            return {"SecretString": json.dumps(value)}

    monkeypatch.setattr(boto3, "client", lambda name: Secrets())
    url = urlsplit(redis_url_from_secret("test/model/readwrite"))
    assert url.scheme == "rediss" and url.hostname == "master.example.com"
    assert unquote(url.password) == value["password"]
    value["tls"] = False
    with pytest.raises(ValueError) as error:
        redis_url_from_secret("test/model/readwrite")
    assert value["password"] not in str(error.value)


def test_ambiguous_secret_url_rejected(monkeypatch):
    monkeypatch.setenv("DOKU_TRAIN_REDIS_URL", "redis://127.0.0.1")
    monkeypatch.setenv("DOKU_TRAIN_REDIS_SECRET_ID", "test/model/readwrite")
    with pytest.raises(ValueError, match="only one"):
        distribution_from_environment("file:///tmp/test", "test")
