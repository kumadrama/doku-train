"""Bounded SageMaker HTTP adapter supervising the TensorFlow Serving process."""

import argparse
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from .artifact_loader import load
from .request_adapter import prepare
from .response_adapter import format_response


class Backend:
    def __init__(self, timeout):
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))

    def call(self, path, payload=None):
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        request = Request(
            "http://127.0.0.1:8501/v1/models/rerank" + path,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def ready(self):
        record = self.call("/versions/1")
        return any(
            item.get("version") == "1"
            and item.get("state") == "AVAILABLE"
            and item.get("status", {}).get("error_code") == "OK"
            for item in record.get("model_version_status", [])
        )


def handler(artifact, backend, limits):
    slots = threading.BoundedSemaphore(limits["concurrency"])

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(limits["timeout"])

        def log_message(self, *args):
            pass  # Do not log raw paths, requests, features, or backend error bodies.

        def send(self, status, body):
            payload = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True

        def do_GET(self):
            if self.path != "/ping":
                return self.send(404, {"error": "not_found"})
            try:
                ready = backend.ready()
            except Exception:
                ready = False
            return self.send(200 if ready else 503, {"ready": ready})

        def do_POST(self):
            if self.path == "/ping":
                return self.do_GET()
            if self.path != "/invocations":
                return self.send(404, {"error": "not_found"})
            if self.headers.get_content_type() != "application/json":
                return self.send(415, {"error": "content_type"})
            if self.headers.get("Transfer-Encoding"):
                return self.send(400, {"error": "content_length_required"})
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if not 0 < length <= limits["request_bytes"]:
                return self.send(413, {"error": "request_size"})
            if not slots.acquire(blocking=False):
                return self.send(503, {"error": "busy"})
            try:
                try:
                    body = self.rfile.read(length)
                    if len(body) != length:
                        raise ValueError("truncated body")
                    payload, size = prepare(
                        json.loads(body),
                        artifact,
                        max_batch=limits["batch"],
                        max_history=limits["history"],
                    )
                except (ValueError, TypeError, KeyError, OverflowError):
                    return self.send(400, {"error": "invalid_request"})
                try:
                    result = backend.call("/versions/1:predict", payload)
                    result = format_response(result, artifact["version"], size)
                except Exception:
                    return self.send(503, {"error": "inference_failed"})
                return self.send(200, result)
            finally:
                slots.release()

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve"], nargs="?", default="serve")
    args = parser.parse_args()
    del args
    limits = {
        name: int(os.environ[env])
        for name, env in {
            "batch": "DOKU_MAX_BATCH",
            "history": "DOKU_MAX_HISTORY",
            "request_bytes": "DOKU_MAX_REQUEST_BYTES",
            "concurrency": "DOKU_CONCURRENCY",
            "timeout": "DOKU_INFERENCE_TIMEOUT_SECONDS",
        }.items()
    }
    threads = int(os.environ["DOKU_TF_THREADS"])
    startup = int(os.environ["DOKU_STARTUP_TIMEOUT_SECONDS"])
    if min(*limits.values(), threads, startup) <= 0 or limits["timeout"] > 50:
        raise ValueError("positive explicit budgets required; inference timeout <= 50 seconds")
    artifact = load(os.environ.get("DOKU_MODEL_ROOT", "/opt/ml/model"))
    if os.environ.get("DOKU_MODEL_VERSION", artifact["version"]) != artifact["version"]:
        raise ValueError("deployment version differs from loaded artifact")
    backend = Backend(limits["timeout"])
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    with tempfile.TemporaryDirectory(prefix="tf-model-") as directory:
        (Path(directory) / "1").symlink_to(
            artifact["root"] / "saved_model", target_is_directory=True
        )
        process = subprocess.Popen(
            [
                "tensorflow_model_server",
                "--port=8500",
                "--rest_api_port=8501",
                "--model_name=rerank",
                "--model_base_path=" + directory,
                "--tensorflow_intra_op_parallelism=" + str(threads),
                "--tensorflow_inter_op_parallelism=1",
                "--rest_api_num_threads=" + str(limits["concurrency"]),
            ]
        )
        server = None
        try:
            deadline = time.monotonic() + startup
            while not stop.is_set():
                if process.poll() is not None:
                    raise RuntimeError("TensorFlow Serving exited before readiness")
                try:
                    if backend.ready():
                        break
                except Exception:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("model startup timeout")
                stop.wait(0.2)
            if stop.is_set():
                return
            server = ThreadingHTTPServer(("0.0.0.0", 8080), handler(artifact, backend, limits))
            server.timeout = 0.2
            while not stop.is_set():
                if process.poll() is not None:
                    raise RuntimeError("TensorFlow Serving exited")
                server.handle_request()
        finally:
            if server:
                server.server_close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
