"""Publish run-scoped training artifacts; exclude raw prepared input and caches."""

from urllib.parse import urlsplit


class ArtifactUpload:
    def __init__(self, client, uri, kms_key):
        location = urlsplit(uri)
        if location.scheme != "s3" or not location.netloc or not location.path.strip("/"):
            raise ValueError("Explicit run-scoped S3 artifact URI required")
        self.client = client
        self.bucket = location.netloc
        self.prefix = location.path.strip("/") + "/"
        self.kms_key = kms_key
        self.sent = {}

    def __call__(self, root):
        files = []
        for name in ["input", "training", "test"]:
            files.extend(p for p in (root / name).rglob("*") if p.is_file())
        files.extend(p for p in root.glob("*.json") if p.is_file())
        # Publish the terminal status last; no remote success marker before artifacts.
        files.sort(
            key=lambda p: (
                2
                if p.name == "COMMITTED.json"
                else 1
                if p.name in {"experiment-status.json", "daily-status.json"}
                else 0,
                str(p),
            )
        )
        for path in files:
            if path.suffix in (".sqlite", ".tmp"):
                continue
            rel = str(path.relative_to(root))
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if self.sent.get(rel) == signature:
                continue
            self.client.upload_file(
                str(path),
                self.bucket,
                self.prefix + rel,
                ExtraArgs={"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": self.kms_key},
            )
            self.sent[rel] = signature
