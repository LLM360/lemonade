#!/usr/bin/env python3
"""Unit tests for llama.cpp validation model selection."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from test.utils import llamacpp_capability_validation as capabilities
from test.utils import validation_model_selection as selection

MODEL_REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "cpp"
    / "resources"
    / "server_models.json"
)
VALIDATION_MODEL_CATALOG = (
    Path(__file__).resolve().parent / "fixtures" / "llamacpp_validation_models.json"
)
VALIDATION_ARTIFACTS = (
    Path(__file__).resolve().parent / "fixtures" / "llamacpp_validation_artifacts.json"
)
LLAMACPP_GUIDE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "guide"
    / "configuration"
    / "llamacpp.md"
)


def load_validation_module():
    requests_module = types.ModuleType("requests")
    requests_module.RequestException = RuntimeError

    class StubSession:
        trust_env = True

        @staticmethod
        def request(*args, **kwargs):
            return requests_module.request(*args, **kwargs)

        @staticmethod
        def close():
            return None

    requests_module.Session = StubSession

    server_base_module = types.ModuleType("utils.server_base")
    server_base_module._auth_headers = lambda: {}
    server_base_module.pull_model_with_retry = lambda *_args, **_kwargs: None
    server_base_module.unload_all_models = lambda **_kwargs: None
    server_base_module.wait_for_server = lambda **_kwargs: None

    test_models_module = types.ModuleType("utils.test_models")
    test_models_module.PORT = 13305
    test_models_module.TIMEOUT_DEFAULT = 60

    utils_package = types.ModuleType("utils")
    utils_package.__path__ = []
    module_path = Path(__file__).resolve().parent / "validate_llamacpp.py"
    spec = importlib.util.spec_from_file_location(
        "validate_llamacpp_under_test", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")

    module = importlib.util.module_from_spec(spec)
    stubs = {
        "requests": requests_module,
        "utils": utils_package,
        "utils.llamacpp_capability_validation": capabilities,
        "utils.server_base": server_base_module,
        "utils.test_models": test_models_module,
        "utils.validation_model_selection": selection,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


VALIDATION = load_validation_module()


@contextlib.contextmanager
def stalled_http_response(prefix: bytes, suffix: bytes, stall_seconds=10):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(30)
    address = listener.getsockname()
    release = threading.Event()
    prefix_sent = threading.Event()
    stall_elapsed = threading.Event()

    def serve():
        try:
            connection, _address = listener.accept()
            with connection:
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = connection.recv(4096)
                    if not chunk:
                        return
                    request += chunk
                connection.sendall(prefix)
                prefix_sent.set()
                if not release.wait(stall_seconds):
                    stall_elapsed.set()
                connection.sendall(suffix)
        except OSError:
            pass

    server = threading.Thread(target=serve)
    server.start()
    try:
        yield (
            f"http://127.0.0.1:{address[1]}/stream",
            prefix_sent,
            stall_elapsed,
        )
    finally:
        release.set()
        try:
            socket.create_connection(address, timeout=0.1).close()
        except OSError:
            pass
        listener.close()
        server.join(timeout=5)
        if server.is_alive():
            raise AssertionError("local test server did not stop")


@contextlib.contextmanager
def trickling_http_response():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(30)
    address = listener.getsockname()
    release = threading.Event()
    body_started = threading.Event()
    trickle_elapsed = threading.Event()

    def serve():
        try:
            connection, _address = listener.accept()
            with connection:
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = connection.recv(4096)
                    if not chunk:
                        return
                    request += chunk
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Connection: close\r\n\r\n"
                )
                body_started.set()
                automatic_deadline = time.monotonic() + 30
                while not release.wait(0.01):
                    if time.monotonic() >= automatic_deadline:
                        trickle_elapsed.set()
                        return
                    connection.sendall(b"x")
        except OSError:
            pass

    server = threading.Thread(target=serve)
    server.start()
    try:
        yield (
            f"http://127.0.0.1:{address[1]}/stream",
            body_started,
            trickle_elapsed,
        )
    finally:
        release.set()
        try:
            socket.create_connection(address, timeout=0.1).close()
        except OSError:
            pass
        listener.close()
        server.join(timeout=5)
        if server.is_alive():
            raise AssertionError("local trickle server did not stop")


@contextlib.contextmanager
def trickling_http_headers():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(30)
    address = listener.getsockname()
    release = threading.Event()
    headers_started = threading.Event()
    trickle_elapsed = threading.Event()

    def serve():
        try:
            connection, _address = listener.accept()
            with connection:
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = connection.recv(4096)
                    if not chunk:
                        return
                    request += chunk
                connection.sendall(b"HTTP/1.1 200 OK\r\nX-Padding: ")
                headers_started.set()
                automatic_deadline = time.monotonic() + 30
                while not release.wait(0.01):
                    if time.monotonic() >= automatic_deadline:
                        trickle_elapsed.set()
                        return
                    connection.sendall(b"x")
        except OSError:
            pass

    server = threading.Thread(target=serve)
    server.start()
    try:
        yield (
            f"http://127.0.0.1:{address[1]}/json",
            headers_started,
            trickle_elapsed,
        )
    finally:
        release.set()
        try:
            socket.create_connection(address, timeout=0.1).close()
        except OSError:
            pass
        listener.close()
        server.join(timeout=5)
        if server.is_alive():
            raise AssertionError("local header trickle server did not stop")


@contextlib.contextmanager
def fixed_http_response(body: bytes):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(30)
    address = listener.getsockname()
    response_sent = threading.Event()

    def serve():
        try:
            connection, _address = listener.accept()
            with connection:
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = connection.recv(4096)
                    if not chunk:
                        return
                    request += chunk
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/plain; charset=utf-8\r\n"
                    + f"Content-Length: {len(body)}\r\n".encode("ascii")
                    + b"X-Test: bounded\r\nConnection: close\r\n\r\n"
                    + body
                )
                response_sent.set()
        except OSError:
            pass

    server = threading.Thread(target=serve)
    server.start()
    try:
        yield f"http://127.0.0.1:{address[1]}/json", response_sent
    finally:
        try:
            socket.create_connection(address, timeout=0.1).close()
        except OSError:
            pass
        listener.close()
        server.join(timeout=5)
        if server.is_alive():
            raise AssertionError("local fixed-response server did not stop")


@contextlib.contextmanager
def held_open_pull_response(body: bytes):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(30)
    address = listener.getsockname()
    terminal_sent = threading.Event()
    release = threading.Event()

    def serve():
        try:
            connection, _address = listener.accept()
            with connection:
                request = b""
                while b"\r\n\r\n" not in request:
                    chunk = connection.recv(4096)
                    if not chunk:
                        return
                    request += chunk
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/event-stream\r\n"
                    b"Connection: close\r\n\r\n" + body
                )
                terminal_sent.set()
                release.wait(30)
        except OSError:
            pass

    server = threading.Thread(target=serve)
    server.start()
    try:
        yield (
            f"http://127.0.0.1:{address[1]}/api/v1/pull",
            terminal_sent,
            release,
        )
    finally:
        release.set()
        try:
            socket.create_connection(address, timeout=0.1).close()
        except OSError:
            pass
        listener.close()
        server.join(timeout=5)
        if server.is_alive():
            raise AssertionError("held-open pull response server did not stop")


class ChunkedResponse:
    def __init__(self, chunks, status_code=200, content_type="text/event-stream"):
        self.chunks = chunks
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.encoding = "utf-8"
        self.closed = False

    def iter_content(self, **_kwargs):
        yield from self.chunks

    def close(self):
        self.closed = True


class LlamaCppValidationSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = [
            {
                "id": "Zulu-Hot",
                "recipe": "llamacpp",
                "labels": ["chat", "hot"],
                "checkpoint": "example/zulu:Zulu.gguf",
                "size": 4.0,
            },
            {
                "id": "Alpha-Hot",
                "recipe": "llamacpp",
                "labels": ["hot", "chat"],
                "checkpoint": "example/alpha:Alpha.gguf",
                "size": 1.0,
            },
            {
                "id": "Cold-Llama",
                "recipe": "llamacpp",
                "labels": ["chat"],
                "checkpoint": "example/cold:Cold.gguf",
                "size": 2.0,
            },
            {
                "id": "Hot-Image",
                "recipe": "sdcpp",
                "labels": ["hot", "image"],
                "checkpoint": "example/image:Image.gguf",
                "size": 0.5,
            },
        ]

    def select(self, requested_model_ids=None, lite=False):
        def resolve_builtin_model(canonical_model_id):
            model_id = canonical_model_id.removeprefix("builtin.")
            return next(
                (model for model in self.catalog if model["id"] == model_id), None
            )

        return selection.select_llamacpp_models(
            self.catalog,
            requested_model_ids=requested_model_ids,
            lite=lite,
            builtin_model_resolver=resolve_builtin_model,
        )

    def test_default_selection_filters_and_sorts_hot_llamacpp_models(self) -> None:
        selected = self.select()

        self.assertEqual([model["id"] for model in selected], ["Alpha-Hot", "Zulu-Hot"])

    def test_lite_selection_uses_smallest_hot_llamacpp_model(self) -> None:
        selected = self.select(lite=True)

        self.assertEqual([model["id"] for model in selected], ["Alpha-Hot"])

    def test_explicit_selection_preserves_order_and_accepts_non_hot_model(self) -> None:
        selected = self.select(["Cold-Llama", "Alpha-Hot"])

        self.assertEqual(
            [model["id"] for model in selected], ["Cold-Llama", "Alpha-Hot"]
        )

    def test_explicit_selection_preserves_duplicate_ids(self) -> None:
        selected = self.select(["Alpha-Hot", "Alpha-Hot"])

        self.assertEqual(
            [model["id"] for model in selected], ["Alpha-Hot", "Alpha-Hot"]
        )

    def test_explicit_selection_rejects_lite_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.select(["Alpha-Hot"], lite=True)

    def test_explicit_selection_resolves_and_loads_canonical_builtin(self) -> None:
        resolved_ids = []
        shadow = {
            "id": "K2-Horizon-0.9B-GGUF",
            "recipe": "llamacpp",
            "labels": ["chat"],
            "checkpoint": "user/model:shadow.gguf",
        }
        built_in = {
            "id": "K2-Horizon-0.9B-GGUF",
            "recipe": "llamacpp",
            "labels": ["chat"],
            "checkpoint": "IFM/K2-Horizon-0.9B-GGUF:model.gguf",
        }

        def resolve_builtin_model(canonical_model_id):
            resolved_ids.append(canonical_model_id)
            return built_in

        selected = selection.select_llamacpp_models(
            [shadow],
            requested_model_ids=["K2-Horizon-0.9B-GGUF"],
            builtin_model_resolver=resolve_builtin_model,
        )

        self.assertEqual(resolved_ids, ["builtin.K2-Horizon-0.9B-GGUF"])
        self.assertEqual(selected[0]["id"], "K2-Horizon-0.9B-GGUF")
        self.assertEqual(selected[0]["load_id"], "builtin.K2-Horizon-0.9B-GGUF")
        self.assertEqual(
            selected[0]["checkpoint"],
            "IFM/K2-Horizon-0.9B-GGUF:model.gguf",
        )

    def test_explicit_selection_rejects_resolver_response_for_another_model(
        self,
    ) -> None:
        wrong_model = {
            "id": "Different-Llama",
            "recipe": "llamacpp",
            "labels": ["chat"],
            "checkpoint": "example/different:model.gguf",
        }

        with self.assertRaisesRegex(
            selection.ModelSelectionError,
            "requested.*K2-Horizon-0.9B-GGUF.*Different-Llama",
        ):
            selection.select_llamacpp_models(
                [],
                requested_model_ids=["K2-Horizon-0.9B-GGUF"],
                builtin_model_resolver=lambda _model_id: wrong_model,
            )

    def test_shadowed_builtin_keeps_canonical_id_for_validation_evidence(self) -> None:
        built_in = {
            "id": "builtin.Shadowed-Llama",
            "recipe": "llamacpp",
            "labels": ["chat"],
            "checkpoint": "example/builtin:Shadowed-Llama.gguf",
        }

        selected = selection.select_llamacpp_models(
            [],
            requested_model_ids=["Shadowed-Llama"],
            builtin_model_resolver=lambda _model_id: built_in,
        )

        self.assertEqual(selected[0]["id"], "builtin.Shadowed-Llama")
        self.assertEqual(selected[0]["load_id"], "builtin.Shadowed-Llama")

    def test_explicit_selection_rejects_unverified_bare_catalog_entry(self) -> None:
        registered_model = [
            {
                "id": "K2-Horizon-0.9B-GGUF",
                "recipe": "llamacpp",
                "labels": ["chat"],
                "checkpoint": "user/model:shadow.gguf",
            }
        ]

        with self.assertRaisesRegex(ValueError, "built-in"):
            selection.select_llamacpp_models(
                registered_model,
                requested_model_ids=["K2-Horizon-0.9B-GGUF"],
            )

    def test_unknown_explicit_model_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing-Model"):
            self.select(["Missing-Model"])

    def test_non_llamacpp_explicit_model_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Hot-Image.*sdcpp"):
            self.select(["Hot-Image"])

    def test_non_chat_llamacpp_explicit_models_have_clear_errors(self) -> None:
        for model_id, label in (
            ("Embed-Llama", "embeddings"),
            ("Rerank-Llama", "reranking"),
        ):
            with self.subTest(label=label):
                self.catalog.append(
                    {
                        "id": model_id,
                        "recipe": "llamacpp",
                        "labels": [label],
                        "checkpoint": f"example/{label}:model.gguf",
                    }
                )

                with self.assertRaisesRegex(
                    ValueError, f"{model_id}.*does not support chat"
                ):
                    self.select([model_id])

    def test_invalid_checkpoint_marks_explicit_model_unpullable(self) -> None:
        records = [
            {"id": "Broken-Llama", "recipe": "llamacpp", "labels": ["chat"]},
            {
                "id": "Broken-Llama",
                "recipe": "llamacpp",
                "labels": ["chat"],
                "checkpoint": "",
            },
            {
                "id": "Broken-Llama",
                "recipe": "llamacpp",
                "labels": ["chat"],
                "checkpoint": 123,
            },
            {
                "id": "Broken-Llama",
                "recipe": "llamacpp",
                "labels": ["chat"],
                "checkpoints": {"main": ""},
            },
        ]
        for record in records:
            with self.subTest(record=record):
                with self.assertRaisesRegex(ValueError, "Broken-Llama.*checkpoint"):
                    selection.select_llamacpp_models(
                        [],
                        requested_model_ids=["Broken-Llama"],
                        builtin_model_resolver=lambda _model_id, value=record: value,
                    )

    def test_explicit_selection_accepts_checkpoints_main(self) -> None:
        model = {
            "id": "Multi-Checkpoint-Llama",
            "recipe": "llamacpp",
            "labels": ["chat"],
            "checkpoints": {
                "main": "example/model:main.gguf",
                "draft": "example/model:draft.gguf",
            },
        }

        selected = selection.select_llamacpp_models(
            [],
            requested_model_ids=["Multi-Checkpoint-Llama"],
            builtin_model_resolver=lambda _model_id: model,
        )

        self.assertEqual(selected[0]["id"], "Multi-Checkpoint-Llama")

    def test_lite_selection_does_not_validate_unselected_models(self) -> None:
        self.catalog.append(
            {
                "id": "Broken-Large-Hot",
                "recipe": "llamacpp",
                "labels": ["chat", "hot"],
                "size": 100.0,
            }
        )

        selected = self.select(lite=True)

        self.assertEqual([model["id"] for model in selected], ["Alpha-Hot"])

    def test_default_selection_keeps_hot_models_without_catalog_checkpoints(
        self,
    ) -> None:
        self.catalog.append(
            {
                "id": "Local-Hot",
                "recipe": "llamacpp",
                "labels": ["chat", "hot"],
                "size": 3.0,
            }
        )

        selected = self.select()

        self.assertEqual(
            [model["id"] for model in selected],
            ["Alpha-Hot", "Local-Hot", "Zulu-Hot"],
        )

    def test_empty_explicit_list_uses_default_selection(self) -> None:
        selected = self.select([])

        self.assertEqual([model["id"] for model in selected], ["Alpha-Hot", "Zulu-Hot"])

    def test_model_argument_is_repeatable(self) -> None:
        parser = argparse.ArgumentParser()
        selection.add_model_selection_arguments(parser)

        args = parser.parse_args(["--model", "Cold-Llama", "--model", "Alpha-Hot"])

        self.assertEqual(args.model, ["Cold-Llama", "Alpha-Hot"])
        self.assertFalse(args.lite)

    def test_model_and_lite_arguments_are_mutually_exclusive(self) -> None:
        parser = argparse.ArgumentParser()
        selection.add_model_selection_arguments(parser)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                parser.parse_args(["--model", "Alpha-Hot", "--lite"])

        self.assertEqual(raised.exception.code, 2)


class K2HorizonCatalogTests(unittest.TestCase):
    @staticmethod
    def expected_models() -> dict:
        return {
            "K2-Horizon-0.9B-GGUF": {
                "checkpoint": "IFM/K2-Horizon-0.9B-GGUF:K2-Horizon-1B-BF16.gguf",
                "recipe": "llamacpp",
                "suggested": True,
                "labels": ["chat", "reasoning", "tool-calling", "hot"],
                "size": 2.16,
            },
            "K2-Horizon-3.7B-GGUF": {
                "checkpoint": "IFM/K2-Horizon-3.7B-GGUF:K2-Horizon-4B-BF16.gguf",
                "recipe": "llamacpp",
                "suggested": True,
                "labels": ["chat", "reasoning", "tool-calling"],
                "size": 10.13,
            },
            "K2-Horizon-7B-GGUF": {
                "checkpoint": "IFM/K2-Horizon-7B-GGUF:K2-Horizon-7B-BF16.gguf",
                "recipe": "llamacpp",
                "suggested": True,
                "labels": ["chat", "reasoning", "tool-calling"],
                "size": 18.01,
            },
        }

    def test_product_catalog_does_not_advertise_k2_horizon(self) -> None:
        product_models = json.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))

        for model_id in self.expected_models():
            with self.subTest(model_id=model_id):
                self.assertNotIn(model_id, product_models)

    def test_public_guide_does_not_advertise_validation_candidates(self) -> None:
        guide = LLAMACPP_GUIDE.read_text(encoding="utf-8")

        self.assertNotIn("## K2-Horizon", guide)
        for model_id in self.expected_models():
            with self.subTest(model_id=model_id):
                self.assertNotIn(model_id, guide)

    def test_k2_horizon_validation_models_have_exact_catalog_contract(self) -> None:
        models = json.loads(VALIDATION_MODEL_CATALOG.read_text(encoding="utf-8"))
        self.assertEqual(models, self.expected_models())

    def test_k2_horizon_validation_artifacts_are_immutably_bound(self) -> None:
        artifacts = json.loads(VALIDATION_ARTIFACTS.read_text(encoding="utf-8"))
        expected = {
            "K2-Horizon-0.9B-GGUF": {
                "revision": "8496259ac62d33192d0fffe708303ed2ca29a384",
                "filename": "K2-Horizon-1B-BF16.gguf",
                "size_bytes": 2159424896,
                "sha256": "371010db1807bb07b62e738422ee0de26c1e15a347f31108ed2c6e219095a8b8",
                "gguf_architecture": "k2-horizon",
                "max_context_window": 131072,
            },
            "K2-Horizon-3.7B-GGUF": {
                "revision": "67b824b8079635b77e19a2a4cd2682a9716478c1",
                "filename": "K2-Horizon-4B-BF16.gguf",
                "size_bytes": 10128343424,
                "sha256": "372ecf9977eb22c92739fec478f2c887daef659d7c55a609ac97b905d1a243b1",
                "gguf_architecture": "k2-horizon",
                "max_context_window": 524288,
            },
            "K2-Horizon-7B-GGUF": {
                "revision": "bcb8c25b76112ce96a962f5b8ab624435d1ee0c9",
                "filename": "K2-Horizon-7B-BF16.gguf",
                "size_bytes": 18010413440,
                "sha256": "088c5d0814ef955d137fd1073ee3a68f6a53411fca44b75dc58a320640000444",
                "gguf_architecture": "k2-horizon",
                "max_context_window": 524288,
            },
        }

        self.assertEqual(artifacts, expected)

    def test_model_catalogs_have_no_duplicate_ids(self) -> None:
        for path in (MODEL_REGISTRY, VALIDATION_MODEL_CATALOG):
            duplicate_ids = []

            def unique_object(pairs):
                parsed = {}
                for key, value in pairs:
                    if key in parsed:
                        duplicate_ids.append(key)
                    parsed[key] = value
                return parsed

            json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=unique_object,
            )

            with self.subTest(path=path):
                self.assertEqual(duplicate_ids, [])


class LlamaCppValidationRuntimeTests(unittest.TestCase):
    @staticmethod
    def response(status_code=200):
        return types.SimpleNamespace(status_code=status_code)

    def test_request_json_rejects_duplicate_response_members(self) -> None:
        response_text = '{"model":"wrong","model":"right"}'
        with fixed_http_response(response_text.encode("utf-8")) as (url, _sent):
            _response, body = VALIDATION.request_json(
                "GET",
                url,
                timeout=5,
            )

        self.assertEqual(body, {"raw_text": response_text})

    def test_json_transport_deadline_interrupts_unterminated_trickle(self) -> None:
        workers = []
        real_popen = subprocess.Popen

        def record_worker(*args, **kwargs):
            worker = real_popen(*args, **kwargs)
            workers.append(worker)
            return worker

        with trickling_http_response() as (url, body_started, trickle_elapsed):
            with (
                mock.patch.object(
                    VALIDATION.subprocess,
                    "Popen",
                    side_effect=record_worker,
                ),
                self.assertRaisesRegex(
                    capabilities.CapabilityValidationError,
                    "wall-clock deadline",
                ),
            ):
                VALIDATION.request_json("GET", url, timeout=10)

            self.assertTrue(body_started.is_set())
            self.assertFalse(trickle_elapsed.is_set())

        self.assertEqual(len(workers), 1)
        self.assertIsNotNone(workers[0].poll())

    def test_json_transport_deadline_covers_response_header_trickle(self) -> None:
        workers = []
        real_popen = subprocess.Popen

        def record_worker(*args, **kwargs):
            worker = real_popen(*args, **kwargs)
            workers.append(worker)
            return worker

        with trickling_http_headers() as (url, headers_started, trickle_elapsed):
            with (
                mock.patch.object(
                    VALIDATION.subprocess,
                    "Popen",
                    side_effect=record_worker,
                ),
                self.assertRaisesRegex(
                    capabilities.CapabilityValidationError,
                    "wall-clock deadline",
                ),
            ):
                VALIDATION.request_json("GET", url, timeout=5)

            self.assertTrue(headers_started.is_set())
            self.assertFalse(trickle_elapsed.is_set())

        self.assertEqual(len(workers), 1)
        self.assertIsNotNone(workers[0].poll())

    def test_json_transport_accepts_exact_response_byte_bound(self) -> None:
        body = b"x" * VALIDATION.MAX_STREAM_SSE_BYTES
        workers = []
        real_popen = subprocess.Popen

        def record_worker(*args, **kwargs):
            worker = real_popen(*args, **kwargs)
            workers.append(worker)
            return worker

        with fixed_http_response(body) as (url, response_sent):
            with mock.patch.object(
                VALIDATION.subprocess,
                "Popen",
                side_effect=record_worker,
            ):
                response, parsed = VALIDATION.request_json("GET", url, timeout=30)

        self.assertTrue(response_sent.is_set())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, body)
        self.assertEqual(response.text, body.decode("utf-8"))
        self.assertEqual(response.headers["X-Test"], "bounded")
        self.assertEqual(parsed, {"raw_text": body.decode("utf-8")})
        self.assertEqual(len(workers), 1)
        self.assertIsNotNone(workers[0].poll())

    def test_json_transport_rejects_response_above_byte_bound(self) -> None:
        body = b"x" * (VALIDATION.MAX_STREAM_SSE_BYTES + 1)
        workers = []
        real_popen = subprocess.Popen

        def record_worker(*args, **kwargs):
            worker = real_popen(*args, **kwargs)
            workers.append(worker)
            return worker

        with fixed_http_response(body) as (url, _response_sent):
            with (
                mock.patch.object(
                    VALIDATION.subprocess,
                    "Popen",
                    side_effect=record_worker,
                ),
                self.assertRaisesRegex(
                    capabilities.CapabilityValidationError,
                    "response byte count exceeds bounds",
                ),
            ):
                VALIDATION.request_json("GET", url, timeout=30)

        self.assertEqual(len(workers), 1)
        self.assertIsNotNone(workers[0].poll())

    def test_json_transport_rejects_malformed_worker_framing_after_reap(self) -> None:
        class CompletedProcess:
            returncode = 0

            def __init__(self):
                self.communicate_calls = 0

            def communicate(self, input=None, timeout=None):
                del input, timeout
                self.communicate_calls += 1
                return b"invalid framing", b""

            def poll(self):
                return self.returncode

        process = CompletedProcess()
        with (
            mock.patch.object(
                VALIDATION.subprocess,
                "Popen",
                return_value=process,
            ),
            self.assertRaisesRegex(
                capabilities.CapabilityValidationError,
                "invalid framing",
            ),
        ):
            VALIDATION.request_json(
                "GET",
                "http://127.0.0.1:13305/json",
                timeout=10,
            )

        self.assertEqual(process.communicate_calls, 1)
        self.assertIsNotNone(process.poll())

    def test_http_worker_is_reaped_when_post_spawn_clock_read_is_interrupted(
        self,
    ) -> None:
        class StartedProcess:
            def __init__(self):
                self.returncode = None
                self.killed = False
                self.communicate_calls = []

            def communicate(self, input=None, timeout=None):
                self.communicate_calls.append((input, timeout))
                self.returncode = -9
                return b"", b""

            def kill(self):
                self.killed = True

            def poll(self):
                return self.returncode

        interruption = KeyboardInterrupt()
        process = StartedProcess()
        with (
            mock.patch.object(
                VALIDATION.subprocess,
                "Popen",
                return_value=process,
            ),
            mock.patch.object(
                VALIDATION.time,
                "monotonic",
                side_effect=(0.0, interruption),
            ),
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            VALIDATION._execute_http_worker(
                "GET",
                "http://127.0.0.1:13305/json",
                60,
                {},
                None,
                "json",
                60,
                "HTTP response",
            )

        self.assertIs(raised.exception, interruption)
        self.assertTrue(process.killed)
        self.assertEqual(process.communicate_calls, [(None, None)])
        self.assertIsNotNone(process.poll())

    def test_http_worker_preserves_control_exception_during_spawn(self) -> None:
        interruption = KeyboardInterrupt()
        with (
            mock.patch.object(
                VALIDATION.subprocess,
                "Popen",
                side_effect=interruption,
            ),
            mock.patch.object(
                VALIDATION,
                "_kill_and_reap_stream_worker",
            ) as cleanup,
            mock.patch.object(VALIDATION.time, "monotonic", return_value=0.0),
            self.assertRaises(KeyboardInterrupt) as raised,
        ):
            VALIDATION._execute_http_worker(
                "GET",
                "http://127.0.0.1:13305/json",
                60,
                {},
                None,
                "json",
                60,
                "HTTP response",
            )

        self.assertIs(raised.exception, interruption)
        cleanup.assert_not_called()

    def test_stream_transport_deadline_covers_slow_response_headers(self) -> None:
        prefix = b"HTTP/1.1 "
        suffix = (
            b"200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 0\r\n\r\n"
        )
        workers = []
        real_popen = subprocess.Popen

        def record_worker(*args, **kwargs):
            worker = real_popen(*args, **kwargs)
            workers.append(worker)
            return worker

        with stalled_http_response(prefix, suffix) as (
            url,
            prefix_sent,
            stall_elapsed,
        ):
            with (
                mock.patch.object(
                    VALIDATION.subprocess,
                    "Popen",
                    side_effect=record_worker,
                ),
                self.assertRaisesRegex(
                    capabilities.CapabilityValidationError,
                    "wall-clock deadline",
                ),
            ):
                _response, lines = VALIDATION.request_stream(
                    "POST",
                    url,
                    timeout=5,
                    wall_clock_deadline=time.monotonic() + 5,
                    json={"stream": True},
                )
                list(lines)

            self.assertTrue(prefix_sent.is_set())
            self.assertFalse(stall_elapsed.is_set())

        self.assertEqual(len(workers), 1)
        self.assertIsNotNone(workers[0].poll())

    def test_stream_transport_deadline_covers_blocked_body_cleanup(self) -> None:
        prefix = (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Connection: close\r\n\r\n"
        )
        suffix = b"data: [DONE]\n"
        with stalled_http_response(prefix, suffix) as (
            url,
            prefix_sent,
            stall_elapsed,
        ):
            with self.assertRaisesRegex(
                capabilities.CapabilityValidationError,
                "wall-clock deadline",
            ):
                _response, lines = VALIDATION.request_stream(
                    "POST",
                    url,
                    timeout=5,
                    wall_clock_deadline=time.monotonic() + 5,
                    json={"stream": True},
                )
                list(lines)

            self.assertTrue(prefix_sent.is_set())
            self.assertFalse(stall_elapsed.is_set())

    def test_stream_transport_timeout_kills_and_reaps_worker(self) -> None:
        class TimedOutProcess:
            def __init__(self):
                self.returncode = None
                self.killed = False
                self.communicate_calls = []

            def communicate(self, input=None, timeout=None):
                self.communicate_calls.append((input, timeout))
                if timeout is not None:
                    raise subprocess.TimeoutExpired(["stream-worker"], timeout)
                self.returncode = -9
                return b"", b""

            def kill(self):
                self.killed = True

            def poll(self):
                return self.returncode

        process = TimedOutProcess()
        with (
            mock.patch.object(
                VALIDATION.subprocess,
                "Popen",
                return_value=process,
            ),
            self.assertRaisesRegex(
                capabilities.CapabilityValidationError,
                "wall-clock deadline",
            ),
        ):
            _response, lines = VALIDATION.request_stream(
                "POST",
                "http://127.0.0.1:13305/stream",
                timeout=60,
                wall_clock_deadline=time.monotonic() + 60,
                json={"stream": True},
            )
            list(lines)

        self.assertTrue(process.killed)
        self.assertEqual(len(process.communicate_calls), 2)
        self.assertIsNotNone(process.poll())

    def test_stream_transport_deadline_interrupts_unterminated_trickle(self) -> None:
        with trickling_http_response() as (url, body_started, trickle_elapsed):
            with self.assertRaisesRegex(
                capabilities.CapabilityValidationError,
                "wall-clock deadline",
            ):
                _response, lines = VALIDATION.request_stream(
                    "POST",
                    url,
                    timeout=2,
                    wall_clock_deadline=time.monotonic() + 5,
                    json={"stream": True},
                )
                list(lines)

            self.assertTrue(body_started.is_set())
            self.assertFalse(trickle_elapsed.is_set())

    def test_stream_transport_reassembles_lines_and_bounds_worker_input(self) -> None:
        class CompletedProcess:
            def __init__(self, output):
                self.output = output
                self.returncode = 0
                self.communicate_calls = []

            def communicate(self, input=None, timeout=None):
                self.communicate_calls.append((input, timeout))
                return self.output, b""

            def poll(self):
                return self.returncode

        output = VALIDATION._encode_stream_worker_output(
            200,
            b"data: one\r\ndata: two\n\n",
        )
        process = CompletedProcess(output)
        authorization = "Bearer test-secret"
        with (
            mock.patch.object(
                VALIDATION.subprocess,
                "Popen",
                return_value=process,
            ) as popen,
            mock.patch.object(
                VALIDATION,
                "_auth_headers",
                return_value={"Authorization": authorization},
            ),
        ):
            returned, lines = VALIDATION.request_stream(
                "POST",
                "http://localhost:13305/api/v1/chat/completions",
                timeout=60,
                json={"stream": True},
            )

            self.assertEqual(
                list(lines),
                [b"data: one\r", b"data: two"],
            )

        self.assertEqual(returned.status_code, 200)
        request_spec = json.loads(process.communicate_calls[0][0])
        self.assertNotIn("wall_clock_deadline", request_spec)
        self.assertEqual(
            request_spec["max_response_bytes"],
            VALIDATION.MAX_STREAM_SSE_BYTES,
        )
        self.assertEqual(request_spec["headers"]["Authorization"], authorization)
        self.assertNotIn(authorization, popen.call_args.args[0])

    def test_stream_transport_worker_success_is_reaped(self) -> None:
        prefix = (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Connection: close\r\n\r\n"
        )
        suffix = b"data: one\ndata: [DONE]\n"
        workers = []
        real_popen = subprocess.Popen

        def record_worker(*args, **kwargs):
            worker = real_popen(*args, **kwargs)
            workers.append(worker)
            return worker

        with stalled_http_response(prefix, suffix, stall_seconds=0) as (
            url,
            prefix_sent,
            _stall_elapsed,
        ):
            with mock.patch.object(
                VALIDATION.subprocess,
                "Popen",
                side_effect=record_worker,
            ):
                response, lines = VALIDATION.request_stream(
                    "POST",
                    url,
                    timeout=5,
                    wall_clock_deadline=time.monotonic() + 10,
                    json={"stream": True},
                )

            self.assertTrue(prefix_sent.is_set())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(list(lines), [b"data: one", b"data: [DONE]"])
        self.assertEqual(len(workers), 1)
        self.assertIsNotNone(workers[0].poll())

    def test_stream_transport_rejects_worker_event_count_above_bound(self) -> None:
        output = VALIDATION._encode_stream_worker_output(
            200,
            b"data: {}\n\ndata: {}\n\ndata: {}\n\n",
        )

        with (
            mock.patch.object(VALIDATION, "MAX_STREAM_EVENTS", 2),
            self.assertRaisesRegex(
                capabilities.CapabilityValidationError,
                "event count",
            ),
        ):
            VALIDATION._decode_stream_worker_output(output)

    def test_stream_transport_rejects_a_line_above_its_independent_bound(
        self,
    ) -> None:
        output = VALIDATION._encode_stream_worker_output(200, b"data: 123456\n\n")

        with (
            mock.patch.object(VALIDATION, "MAX_STREAM_SSE_LINE_BYTES", 8),
            self.assertRaisesRegex(
                capabilities.CapabilityValidationError,
                "SSE line exceeds bounds",
            ),
        ):
            VALIDATION._decode_stream_worker_output(output)

    def test_stream_transport_counts_semantic_frames_not_blank_lines(self) -> None:
        chunk_count = 2500
        chunk = json.dumps(
            {
                "model": "backend-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "x"},
                        "finish_reason": None,
                    }
                ],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        terminal = json.dumps(
            {
                "model": "backend-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            },
            separators=(",", ":"),
        ).encode("utf-8")
        body = (b"data: " + chunk + b"\n\n") * chunk_count
        body += b"data: " + terminal + b"\n\ndata: [DONE]\n\n"

        lines = VALIDATION._split_stream_lines(body)
        message, finish_reason, saw_done, model = capabilities._decode_openai_stream(
            lines
        )

        self.assertEqual(len(lines), chunk_count + 2)
        self.assertEqual(message["content"], "x" * chunk_count)
        self.assertEqual(finish_reason, "stop")
        self.assertTrue(saw_done)
        self.assertEqual(model, "backend-model")
        self.assertGreaterEqual(capabilities.MAX_STREAM_EVENTS, 3072 + 2)

    def test_stream_transport_rejects_unterminated_body_above_byte_bound(self) -> None:
        class OversizedResponse:
            def __init__(self):
                self.closed = False

            def iter_content(self, **_kwargs):
                yield b"abc"
                yield b"def"

            def close(self):
                self.closed = True

        response = OversizedResponse()
        with (
            mock.patch.object(
                VALIDATION.requests,
                "request",
                return_value=response,
                create=True,
            ),
            mock.patch.object(
                VALIDATION,
                "_load_stream_worker_request",
                return_value=(
                    "stream",
                    "POST",
                    "http://127.0.0.1:13305/stream",
                    60,
                    {},
                    {"stream": True},
                    5,
                ),
            ),
            self.assertRaisesRegex(
                capabilities.CapabilityValidationError,
                "SSE byte count",
            ),
        ):
            VALIDATION._run_stream_worker()

        self.assertTrue(response.closed)

    def test_json_worker_closes_response_above_byte_bound(self) -> None:
        class OversizedResponse:
            def __init__(self):
                self.closed = False

            def iter_content(self, **_kwargs):
                yield b"abc"
                yield b"def"

            def close(self):
                self.closed = True

        response = OversizedResponse()
        with (
            mock.patch.object(
                VALIDATION.requests,
                "request",
                return_value=response,
                create=True,
            ),
            mock.patch.object(
                VALIDATION,
                "_load_stream_worker_request",
                return_value=(
                    "json",
                    "GET",
                    "http://127.0.0.1:13305/json",
                    60,
                    {},
                    None,
                    5,
                ),
            ),
            self.assertRaisesRegex(
                capabilities.CapabilityValidationError,
                "HTTP response byte count",
            ),
        ):
            VALIDATION._run_stream_worker()

        self.assertTrue(response.closed)

    def test_json_worker_accepts_exact_custom_response_bound(self) -> None:
        class BoundedResponse:
            status_code = 200
            headers = {"Content-Type": "application/json"}
            encoding = "utf-8"

            def __init__(self):
                self.closed = False

            def iter_content(self, **_kwargs):
                yield b"abc"
                yield b"de"

            def close(self):
                self.closed = True

        response = BoundedResponse()
        captured_bodies = []
        stdout = types.SimpleNamespace(buffer=io.BytesIO())

        def encode(_status, body, _headers, _encoding):
            captured_bodies.append(body)
            return b"encoded"

        with (
            mock.patch.object(
                VALIDATION.requests,
                "request",
                return_value=response,
                create=True,
            ),
            mock.patch.object(
                VALIDATION,
                "_load_stream_worker_request",
                return_value=(
                    "json",
                    "GET",
                    "http://127.0.0.1:13305/json",
                    60,
                    {},
                    None,
                    5,
                ),
            ),
            mock.patch.object(
                VALIDATION,
                "_encode_stream_worker_output",
                side_effect=encode,
            ),
            mock.patch.object(VALIDATION.sys, "stdout", stdout),
        ):
            VALIDATION._run_stream_worker()

        self.assertEqual(captured_bodies, [b"abcde"])
        self.assertEqual(stdout.buffer.getvalue(), b"encoded")
        self.assertTrue(response.closed)

    def request_json_for_chat(
        self,
        chat_message,
        operations=None,
        load_payloads=None,
        response_model="builtin.Test-Llama",
        unload_status_code=200,
    ):
        loaded = False

        def request_json(method, url, timeout, **kwargs):
            nonlocal loaded
            del method, timeout
            operation = url.rsplit("/", maxsplit=1)[-1]
            if operations is not None:
                operations.append(operation)
            if operation == "load":
                loaded = True
                if load_payloads is not None:
                    load_payloads.append(kwargs["json"])
                return self.response(), {}
            if operation == "unload":
                loaded = False
                return self.response(unload_status_code), {}
            if operation == "health":
                return self.response(), {
                    "all_models_loaded": (
                        [
                            {
                                "model_name": "builtin.Test-Llama",
                                "recipe_options": {"llamacpp_backend": "vulkan"},
                            }
                        ]
                        if loaded
                        else []
                    )
                }
            if operation == "completions":
                message = (
                    chat_message(kwargs["json"])
                    if callable(chat_message)
                    else chat_message
                )
                return self.response(), {
                    "model": response_model,
                    "choices": [{"message": message}],
                }
            if operation == "stats":
                return self.response(), {"output_tokens": 4}
            return self.response(), {}

        return request_json

    def test_validation_uses_ipv4_loopback_for_server_readiness(self) -> None:
        with (
            mock.patch.object(VALIDATION, "wait_for_server") as wait,
            mock.patch.object(
                VALIDATION,
                "request_json",
                return_value=(self.response(), {"status": "ok"}),
            ) as request,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            VALIDATION.require_running_server("http://127.0.0.1:13305/api/v1", 13305)

        wait.assert_called_once_with(
            port=13305,
            timeout=VALIDATION.TIMEOUT_HEALTH,
            host="127.0.0.1",
        )
        self.assertTrue(request.call_args.args[1].startswith("http://127.0.0.1:"))

    def test_server_readiness_bounds_each_connect_attempt(self) -> None:
        clock = types.SimpleNamespace(now=0.0)
        connect_timeouts = []

        def connect(_address, timeout):
            connect_timeouts.append(timeout)
            clock.now += timeout
            raise OSError("not ready")

        def sleep(delay):
            clock.now += delay

        with (
            mock.patch.object(VALIDATION.time, "monotonic", lambda: clock.now),
            mock.patch.object(VALIDATION.time, "sleep", side_effect=sleep),
            mock.patch.object(
                VALIDATION.socket,
                "create_connection",
                side_effect=connect,
            ),
            self.assertRaisesRegex(TimeoutError, "2.5 seconds"),
        ):
            VALIDATION.wait_for_server(port=13305, timeout=2.5, host="127.0.0.1")

        self.assertGreater(len(connect_timeouts), 1)
        self.assertTrue(
            all(
                0 < timeout <= VALIDATION.SERVER_CONNECT_ATTEMPT_TIMEOUT_SECONDS
                for timeout in connect_timeouts
            )
        )

    def test_unload_all_models_uses_bounded_worker_request_and_retries(self) -> None:
        failed = self.response(500)
        succeeded = self.response(404)
        with (
            mock.patch.object(
                VALIDATION,
                "request_json",
                side_effect=[(failed, {}), (succeeded, {})],
            ) as request,
            mock.patch.object(VALIDATION.time, "sleep") as sleep,
        ):
            response = VALIDATION.unload_all_models(
                port=13305,
                attempts=3,
                host="127.0.0.1",
            )

        self.assertIs(response, succeeded)
        self.assertEqual(request.call_count, 2)
        request.assert_called_with(
            "POST",
            "http://127.0.0.1:13305/api/v1/unload",
            timeout=VALIDATION.TIMEOUT_DEFAULT,
            json={},
        )
        sleep.assert_called_once_with(1)

    def test_http_worker_ignores_ambient_proxy_configuration(self) -> None:
        response = ChunkedResponse(
            (b"ok",),
            status_code=503,
            content_type="text/plain",
        )
        session = mock.Mock()
        session.request.return_value = response
        stdout = types.SimpleNamespace(buffer=io.BytesIO())
        with (
            mock.patch.object(
                VALIDATION.requests,
                "Session",
                return_value=session,
            ),
            mock.patch.object(
                VALIDATION,
                "_load_stream_worker_request",
                return_value=(
                    "pull_sse",
                    "POST",
                    "http://127.0.0.1:13305/api/v1/pull",
                    60,
                    {},
                    {"model_name": "builtin.Test-Llama", "stream": True},
                    VALIDATION.MAX_PULL_SSE_WIRE_BYTES,
                ),
            ),
            mock.patch.object(VALIDATION.sys, "stdout", stdout),
        ):
            VALIDATION._run_stream_worker()

        self.assertFalse(session.trust_env)
        session.request.assert_called_once()
        session.close.assert_called_once()
        self.assertTrue(response.closed)

    def test_pull_rejects_non_loopback_host_before_spawning_worker(self) -> None:
        with (
            mock.patch.object(VALIDATION, "_execute_http_worker") as execute,
            self.assertRaisesRegex(
                capabilities.CapabilityValidationError,
                "host must be loopback",
            ),
        ):
            VALIDATION._pull_model_streaming(
                "builtin.Test-Llama",
                13305,
                host="example.com",
            )

        execute.assert_not_called()

    def test_pull_uses_legacy_sse_inside_absolute_deadline_worker(self) -> None:
        deadline = time.monotonic() + 60
        response = VALIDATION._BufferedResponse(
            200,
            {"Content-Type": "text/event-stream"},
            "utf-8",
            b'{"terminal":"complete"}',
        )
        with (
            mock.patch.object(
                VALIDATION,
                "_execute_http_worker",
                return_value=response,
            ) as execute,
            mock.patch.object(
                VALIDATION,
                "_auth_headers",
                return_value={"Authorization": "Bearer secret"},
            ),
        ):
            result = VALIDATION._pull_model_streaming(
                "builtin.Test-Llama",
                13305,
                host="127.0.0.1",
                wall_clock_deadline=deadline,
            )

        self.assertEqual(result, (200, ""))
        execute.assert_called_once_with(
            "POST",
            "http://127.0.0.1:13305/api/v1/pull",
            VALIDATION.TIMEOUT_INFERENCE,
            {"Authorization": "Bearer secret"},
            {
                "model_name": "builtin.Test-Llama",
                "stream": True,
                "subscribe": True,
            },
            "pull_sse",
            deadline,
            "model pull",
            VALIDATION.MAX_PULL_SSE_WIRE_BYTES,
        )

    def test_pull_maps_only_bounded_worker_terminal_summaries(self) -> None:
        cases = (
            ({"terminal": "complete"}, (200, "")),
            (
                {"terminal": "error", "status_code": 500, "error": "disk full"},
                (500, "disk full"),
            ),
            (
                {
                    "terminal": "error",
                    "status_code": 400,
                    "error": "unknown_model",
                },
                (400, "unknown_model"),
            ),
        )
        for summary, expected in cases:
            with self.subTest(summary=summary):
                response = VALIDATION._BufferedResponse(
                    200,
                    {},
                    "utf-8",
                    json.dumps(summary).encode("utf-8"),
                )
                with mock.patch.object(
                    VALIDATION,
                    "_execute_http_worker",
                    return_value=response,
                ):
                    result = VALIDATION._pull_model_streaming(
                        "builtin.Test-Llama",
                        13305,
                        wall_clock_deadline=time.monotonic() + 60,
                    )
                self.assertEqual(result, expected)

        malformed = VALIDATION._BufferedResponse(200, {}, "utf-8", b"{}")
        with (
            mock.patch.object(
                VALIDATION,
                "_execute_http_worker",
                return_value=malformed,
            ),
            self.assertRaisesRegex(
                capabilities.CapabilityValidationError,
                "terminal summary",
            ),
        ):
            VALIDATION._pull_model_streaming(
                "builtin.Test-Llama",
                13305,
                wall_clock_deadline=time.monotonic() + 60,
            )

    def test_pull_sse_reducer_handles_crlf_and_chunk_boundaries(self) -> None:
        body = (
            b'event: progress\r\ndata: {"percent":50}\r\n\r\n'
            b'event: complete\r\ndata: {"status":"ok"}\r\n\r\n'
        )
        response = ChunkedResponse(
            (body[:1], body[1:17], body[17:41], body[41:]),
            content_type="text/event-stream; charset=utf-8",
        )

        summary = VALIDATION._reduce_pull_sse_response(response, len(body))

        self.assertEqual(json.loads(summary), {"terminal": "complete"})
        self.assertLessEqual(
            len(summary),
            VALIDATION.MAX_PULL_TERMINAL_SUMMARY_BYTES,
        )

    def test_pull_sse_reducer_discards_large_progress_history(self) -> None:
        progress = b'event: progress\ndata: {"percent":50}\n\n'
        complete = b'event: complete\ndata: {"status":"ok"}\n\n'
        event_total = 20_000

        def chunks():
            for _index in range(event_total):
                yield progress
            yield complete

        response = ChunkedResponse(chunks())
        summary = VALIDATION._reduce_pull_sse_response(
            response,
            len(progress) * event_total + len(complete),
        )

        self.assertEqual(summary, b'{"terminal":"complete"}')
        self.assertLess(len(summary), 64)

    def test_pull_sse_reducer_accepts_exact_wire_bound(self) -> None:
        body = b'event: complete\ndata: {"status":"ok"}\n\n'

        summary = VALIDATION._reduce_pull_sse_response(
            ChunkedResponse((body,)),
            len(body),
        )

        self.assertEqual(summary, b'{"terminal":"complete"}')

    def test_pull_sse_reducer_enforces_wire_line_data_and_event_bounds(self) -> None:
        complete = b'event: complete\ndata: {"status":"ok"}\n\n'
        cases = (
            (
                ChunkedResponse((complete,)),
                len(complete) - 1,
                contextlib.nullcontext(),
                "wire byte count",
            ),
            (
                ChunkedResponse((b"x" * 9 + b"\n",)),
                100,
                mock.patch.object(VALIDATION, "MAX_PULL_SSE_LINE_BYTES", 8),
                "line exceeds bounds",
            ),
            (
                ChunkedResponse((b'event: progress\ndata: {"x":1}\n\n',)),
                100,
                mock.patch.object(VALIDATION, "MAX_PULL_SSE_EVENT_DATA_BYTES", 2),
                "event data exceeds bounds",
            ),
            (
                ChunkedResponse(
                    (b'event: progress\ndata: {"percent":1}\n\n' + complete,)
                ),
                1000,
                mock.patch.object(VALIDATION, "MAX_PULL_SSE_EVENTS", 1),
                "event count exceeds bounds",
            ),
        )
        for response, wire_bound, patched_bound, expected_error in cases:
            with (
                self.subTest(expected_error=expected_error),
                patched_bound,
                self.assertRaisesRegex(
                    capabilities.CapabilityValidationError,
                    expected_error,
                ),
            ):
                VALIDATION._reduce_pull_sse_response(response, wire_bound)

    def test_pull_sse_reducer_requires_one_terminal_and_clean_eof(self) -> None:
        progress = b'event: progress\ndata: {"percent":1}\n\n'
        complete = b'event: complete\ndata: {"status":"ok"}\n\n'
        cases = (
            (progress, "without a terminal"),
            (complete + complete, "after its terminal"),
            (complete + progress, "after its terminal"),
            (complete[:-1], "incomplete event"),
            (complete.rstrip(b"\n"), "incomplete line"),
            (complete + b"data: {}\n\n", "after its terminal"),
        )
        for body, expected_error in cases:
            with (
                self.subTest(expected_error=expected_error),
                self.assertRaisesRegex(
                    capabilities.CapabilityValidationError,
                    expected_error,
                ),
            ):
                VALIDATION._reduce_pull_sse_response(
                    ChunkedResponse((body,)),
                    len(body),
                )

    def test_pull_sse_reducer_rejects_invalid_content_and_json(self) -> None:
        cases = (
            (
                ChunkedResponse(
                    (b'event: complete\ndata: {"status":"ok"}\n\n',),
                    content_type="application/json",
                ),
                "content type",
            ),
            (
                ChunkedResponse((b'event: progress\ndata: {"x":1,"x":2}\n\n',)),
                "event data is invalid",
            ),
            (
                ChunkedResponse((b'event: progress\ndata: {"x":NaN}\n\n',)),
                "event data is invalid",
            ),
            (
                ChunkedResponse((b"event: complete\ndata: []\n\n",)),
                "event data is invalid",
            ),
        )
        for response, expected_error in cases:
            body_size = sum(len(chunk) for chunk in response.chunks)
            with (
                self.subTest(expected_error=expected_error),
                self.assertRaisesRegex(
                    capabilities.CapabilityValidationError,
                    expected_error,
                ),
            ):
                VALIDATION._reduce_pull_sse_response(response, body_size)

    def test_pull_worker_always_closes_response_on_parser_failure(self) -> None:
        response = ChunkedResponse((b'event: progress\ndata: {"percent":1}\n\n',))
        with (
            mock.patch.object(
                VALIDATION.requests,
                "request",
                return_value=response,
                create=True,
            ),
            mock.patch.object(
                VALIDATION,
                "_load_stream_worker_request",
                return_value=(
                    "pull_sse",
                    "POST",
                    "http://127.0.0.1:13305/api/v1/pull",
                    60,
                    {},
                    {"model_name": "builtin.Test-Llama", "stream": True},
                    VALIDATION.MAX_PULL_SSE_WIRE_BYTES,
                ),
            ),
            self.assertRaisesRegex(
                capabilities.CapabilityValidationError,
                "without a terminal event",
            ),
        ):
            VALIDATION._run_stream_worker()

        self.assertTrue(response.closed)

    def test_pull_worker_bounds_and_fully_consumes_non_200_body(self) -> None:
        response = ChunkedResponse(
            (b"temporarily ", b"unavailable"),
            status_code=503,
            content_type="application/json",
        )
        stdout = types.SimpleNamespace(buffer=io.BytesIO())
        with (
            mock.patch.object(
                VALIDATION.requests,
                "request",
                return_value=response,
                create=True,
            ),
            mock.patch.object(
                VALIDATION,
                "_load_stream_worker_request",
                return_value=(
                    "pull_sse",
                    "POST",
                    "http://127.0.0.1:13305/api/v1/pull",
                    60,
                    {},
                    {"model_name": "builtin.Test-Llama", "stream": True},
                    VALIDATION.MAX_PULL_SSE_WIRE_BYTES,
                ),
            ),
            mock.patch.object(VALIDATION.sys, "stdout", stdout),
        ):
            VALIDATION._run_stream_worker()

        returned = VALIDATION._decode_http_worker_output(stdout.buffer.getvalue())
        self.assertEqual(returned.status_code, 503)
        self.assertEqual(returned.text, "temporarily unavailable")
        self.assertTrue(response.closed)

        oversized = ChunkedResponse(
            (b"x" * VALIDATION.MAX_PULL_RESPONSE_BYTES, b"x"),
            status_code=503,
            content_type="application/json",
        )
        with (
            mock.patch.object(
                VALIDATION.requests,
                "request",
                return_value=oversized,
                create=True,
            ),
            mock.patch.object(
                VALIDATION,
                "_load_stream_worker_request",
                return_value=(
                    "pull_sse",
                    "POST",
                    "http://127.0.0.1:13305/api/v1/pull",
                    60,
                    {},
                    {"model_name": "builtin.Test-Llama", "stream": True},
                    VALIDATION.MAX_PULL_SSE_WIRE_BYTES,
                ),
            ),
            self.assertRaisesRegex(
                capabilities.CapabilityValidationError,
                "HTTP response byte count exceeds bounds",
            ),
        ):
            VALIDATION._run_stream_worker()
        self.assertTrue(oversized.closed)

    def test_pull_deadline_kills_and_reaps_trickling_worker(self) -> None:
        workers = []
        real_popen = subprocess.Popen

        def record_worker(*args, **kwargs):
            worker = real_popen(*args, **kwargs)
            workers.append(worker)
            return worker

        with trickling_http_response() as (url, body_started, trickle_elapsed):
            host, port = url.split("//", maxsplit=1)[1].split(":")
            port = int(port.split("/", maxsplit=1)[0])
            with (
                mock.patch.object(
                    VALIDATION.subprocess,
                    "Popen",
                    side_effect=record_worker,
                ),
                self.assertRaisesRegex(
                    capabilities.CapabilityValidationError,
                    "model pull exceeded its wall-clock deadline",
                ),
            ):
                VALIDATION._pull_model_streaming(
                    "builtin.Test-Llama",
                    port,
                    host=host,
                    wall_clock_deadline=time.monotonic() + 5,
                )

            self.assertTrue(body_started.is_set())
            self.assertFalse(trickle_elapsed.is_set())

        self.assertEqual(len(workers), 1)
        self.assertIsNotNone(workers[0].poll())

    def test_pull_success_worker_is_reaped_after_clean_terminal_eof(self) -> None:
        prefix = (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Connection: close\r\n\r\n"
        )
        body = b'event: complete\ndata: {"status":"ok"}\n\n'
        workers = []
        real_popen = subprocess.Popen

        def record_worker(*args, **kwargs):
            worker = real_popen(*args, **kwargs)
            workers.append(worker)
            return worker

        with stalled_http_response(prefix, body, stall_seconds=0) as (
            url,
            response_started,
            _stall_elapsed,
        ):
            host, port = url.split("//", maxsplit=1)[1].split(":")
            port = int(port.split("/", maxsplit=1)[0])
            with mock.patch.object(
                VALIDATION.subprocess,
                "Popen",
                side_effect=record_worker,
            ):
                result = VALIDATION._pull_model_streaming(
                    "builtin.Test-Llama",
                    port,
                    host=host,
                    wall_clock_deadline=time.monotonic() + 10,
                )
            self.assertTrue(response_started.is_set())

        self.assertEqual(result, (200, ""))
        self.assertEqual(len(workers), 1)
        self.assertIsNotNone(workers[0].poll())

    def test_pull_waits_for_clean_eof_after_terminal_event(self) -> None:
        body = b'event: complete\ndata: {"status":"ok"}\n\n'
        workers = []
        result = []
        errors = []
        finished = threading.Event()
        real_popen = subprocess.Popen

        def record_worker(*args, **kwargs):
            worker = real_popen(*args, **kwargs)
            workers.append(worker)
            return worker

        def pull(host, port):
            try:
                result.append(
                    VALIDATION._pull_model_streaming(
                        "builtin.Test-Llama",
                        port,
                        host=host,
                        wall_clock_deadline=time.monotonic() + 20,
                    )
                )
            except BaseException as exc:  # pylint: disable=broad-exception-caught
                errors.append(exc)
            finally:
                finished.set()

        with held_open_pull_response(body) as (url, terminal_sent, release):
            host, port = url.split("//", maxsplit=1)[1].split(":")
            port = int(port.split("/", maxsplit=1)[0])
            with mock.patch.object(
                VALIDATION.subprocess,
                "Popen",
                side_effect=record_worker,
            ):
                caller = threading.Thread(target=pull, args=(host, port))
                caller.start()
                self.assertTrue(terminal_sent.wait(10))
                self.assertFalse(finished.wait(0.25))
                self.assertEqual(len(workers), 1)
                self.assertIsNone(workers[0].poll())
                release.set()
                caller.join(timeout=10)

        self.assertFalse(caller.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result, [(200, "")])
        self.assertIsNotNone(workers[0].poll())

    def test_pull_ambiguous_failures_and_control_exceptions_are_not_retried(
        self,
    ) -> None:
        failures = (
            capabilities.CapabilityValidationError("malformed SSE"),
            KeyboardInterrupt(),
        )
        for failure in failures:
            with self.subTest(failure=failure.__class__.__name__):
                with (
                    mock.patch.object(
                        VALIDATION,
                        "_pull_model_streaming",
                        side_effect=failure,
                    ) as pull,
                    mock.patch.object(VALIDATION.time, "sleep") as sleep,
                    self.assertRaises(failure.__class__) as raised,
                ):
                    VALIDATION.pull_model_with_retry(
                        "builtin.Test-Llama",
                        port=13305,
                        host="127.0.0.1",
                    )

                self.assertIs(raised.exception, failure)
                pull.assert_called_once()
                sleep.assert_not_called()

    def test_pull_retries_only_returned_transient_terminal_status(self) -> None:
        with (
            mock.patch.object(
                VALIDATION,
                "_pull_model_streaming",
                side_effect=[(500, "download failed"), (200, "")],
            ) as pull,
            mock.patch.object(VALIDATION.time, "sleep") as sleep,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            VALIDATION.pull_model_with_retry(
                "builtin.Test-Llama",
                port=13305,
                host="127.0.0.1",
            )

        self.assertEqual(pull.call_count, 2)
        self.assertEqual(
            pull.call_args_list[0].kwargs["wall_clock_deadline"],
            pull.call_args_list[1].kwargs["wall_clock_deadline"],
        )
        sleep.assert_called_once_with(2)

    def test_model_rejects_raw_ifm_markers_in_response_fields(self) -> None:
        cases = {
            "content": (
                {"content": "answer <ifm|think>hidden"},
                "content",
            ),
            "nested_content": (
                {
                    "content": [
                        {"type": "text", "text": "answer </ifm|think>"},
                    ]
                },
                "content",
            ),
            "reasoning_content": (
                {
                    "content": "answer",
                    "reasoning_content": "<ifm|think>hidden",
                },
                "reasoning_content",
            ),
            "reasoning": (
                {
                    "content": "answer",
                    "reasoning": {"text": "hidden <ifm|think>"},
                },
                "reasoning",
            ),
            "tool_calls": (
                {
                    "content": "answer",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": '{"query":"<ifm|tool_calls>"}',
                            }
                        }
                    ],
                },
                "tool_calls",
            ),
            "unexpected_field": (
                {
                    "content": "answer",
                    "metadata": {"debug": "<IFM|unexpected"},
                },
                "metadata",
            ),
        }

        for case_name, (message, response_field) in cases.items():
            with self.subTest(case_name=case_name):
                with (
                    mock.patch.object(
                        VALIDATION,
                        "request_json",
                        side_effect=self.request_json_for_chat(message),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    success, error, _stats = VALIDATION.test_model(
                        "http://localhost:13305/api/v1",
                        "builtin.Test-Llama",
                        "vulkan",
                    )

                self.assertFalse(success)
                self.assertRegex(
                    error,
                    rf"(?i)raw IFM control marker.*{response_field}",
                )

    def test_model_rejects_ifm_chat_boundary_tokens(self) -> None:
        for token in ("<|ifm|im_start|>", "<|ifm|im_end|>"):
            with self.subTest(token=token):
                message = {"content": f"answer {token} leaked"}
                with (
                    mock.patch.object(
                        VALIDATION,
                        "request_json",
                        side_effect=self.request_json_for_chat(message),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    success, error, _stats = VALIDATION.test_model(
                        "http://localhost:13305/api/v1",
                        "builtin.Test-Llama",
                        "vulkan",
                    )

                self.assertFalse(success)
                self.assertRegex(error, "(?i)raw IFM control marker.*content")

    def test_model_accepts_clean_nested_tool_call_fields(self) -> None:
        message = {
            "content": "I will use the lookup tool.",
            "reasoning_content": "The tool can answer this request.",
            "tool_calls": [
                {
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"weather"}',
                    }
                }
            ],
        }

        with (
            mock.patch.object(
                VALIDATION,
                "request_json",
                side_effect=self.request_json_for_chat(message),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            success, response_text, _stats = VALIDATION.test_model(
                "http://localhost:13305/api/v1",
                "builtin.Test-Llama",
                "vulkan",
            )

        self.assertTrue(success)
        self.assertEqual(response_text, message["content"])

    def test_plain_chat_requires_visible_final_content(self) -> None:
        message = {
            "content": "",
            "reasoning_content": "The answer should be four.",
        }

        with (
            mock.patch.object(
                VALIDATION,
                "request_json",
                side_effect=self.request_json_for_chat(message),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            success, error, _stats = VALIDATION.test_model(
                "http://localhost:13305/api/v1",
                "builtin.Test-Llama",
                "vulkan",
            )

        self.assertFalse(success)
        self.assertIn("visible final content", error)

    def test_plain_smoke_disables_reasoning_to_get_visible_final_content(self) -> None:
        def reasoning_model_response(payload):
            self.assertEqual(payload["max_completion_tokens"], 50)
            if payload.get("chat_template_kwargs") == {"enable_thinking": False}:
                return {"content": "The answer is 4."}
            return {
                "content": "",
                "reasoning_content": "The answer should be four.",
            }

        with (
            mock.patch.object(
                VALIDATION,
                "request_json",
                side_effect=self.request_json_for_chat(reasoning_model_response),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            success, response, _stats = VALIDATION.test_model(
                "http://localhost:13305/api/v1",
                "builtin.Test-Llama",
                "vulkan",
            )

        self.assertTrue(success, response)
        self.assertEqual(response, "The answer is 4.")

    def test_model_runs_requested_capability_profile_and_records_evidence(self) -> None:
        matrix = {
            "profile": capabilities.K2_HORIZON_PROFILE,
            "model": "builtin.Test-Llama",
            "pass": True,
            "cases": [],
        }
        evidence = {}

        with (
            mock.patch.object(
                VALIDATION,
                "request_json",
                side_effect=self.request_json_for_chat({"content": "The answer is 4."}),
            ) as request,
            mock.patch.object(
                VALIDATION,
                "validate_capabilities",
                return_value=(
                    matrix,
                    True,
                    f"{len(capabilities.K2_HORIZON_CASE_IDS)}/"
                    f"{len(capabilities.K2_HORIZON_CASE_IDS)} capability cases passed",
                ),
            ) as validate,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            success, _response, _stats = VALIDATION.test_model(
                "http://localhost:13305/api/v1",
                "builtin.Test-Llama",
                "vulkan",
                capability_profile=capabilities.K2_HORIZON_PROFILE,
                capability_evidence=evidence,
            )

        self.assertTrue(success)
        self.assertEqual(evidence, matrix)
        validate.assert_called_once_with(
            capabilities.K2_HORIZON_PROFILE,
            base_url="http://localhost:13305/api/v1",
            model="builtin.Test-Llama",
            request_json=request,
            request_stream=VALIDATION.request_stream,
            timeout=VALIDATION.TIMEOUT_INFERENCE,
        )

    def test_capability_failure_fails_model_and_preserves_matrix(self) -> None:
        matrix = {
            "profile": capabilities.K2_HORIZON_PROFILE,
            "model": "builtin.Test-Llama",
            "pass": False,
            "cases": [{"id": "openai_reasoning_low", "pass": False}],
        }
        evidence = {}

        with (
            mock.patch.object(
                VALIDATION,
                "request_json",
                side_effect=self.request_json_for_chat({"content": "The answer is 4."}),
            ),
            mock.patch.object(
                VALIDATION,
                "validate_capabilities",
                return_value=(matrix, False, "reasoning low failed"),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            success, error, _stats = VALIDATION.test_model(
                "http://localhost:13305/api/v1",
                "builtin.Test-Llama",
                "vulkan",
                capability_profile=capabilities.K2_HORIZON_PROFILE,
                capability_evidence=evidence,
            )

        self.assertFalse(success)
        self.assertIn("reasoning low failed", error)
        self.assertEqual(evidence, matrix)

    def test_model_requires_health_to_confirm_the_requested_backend(self) -> None:
        cases = (
            (503, {}, "health inventory returned HTTP 503"),
            (200, {}, "missing all_models_loaded"),
            (200, {"all_models_loaded": "invalid"}, "invalid model inventory"),
            (200, {"all_models_loaded": []}, "is absent from the loaded inventory"),
            (
                200,
                {"all_models_loaded": [{"model_name": "builtin.Test-Llama"}]},
                "missing recipe_options",
            ),
            (
                200,
                {
                    "all_models_loaded": [
                        {
                            "model_name": "builtin.Test-Llama",
                            "recipe_options": "invalid",
                        }
                    ]
                },
                "invalid recipe_options",
            ),
            (
                200,
                {
                    "all_models_loaded": [
                        {
                            "model_name": "builtin.Test-Llama",
                            "recipe_options": {},
                        }
                    ]
                },
                "missing llamacpp_backend",
            ),
            (
                200,
                {
                    "all_models_loaded": [
                        {
                            "model_name": "builtin.Test-Llama",
                            "recipe_options": {"llamacpp_backend": "cpu"},
                        }
                    ]
                },
                "backend 'cpu' instead of 'vulkan'",
            ),
            (
                200,
                {
                    "all_models_loaded": [
                        {
                            "model_name": "builtin.Test-Llama",
                            "recipe_options": {"llamacpp_backend": "vulkan"},
                        },
                        {
                            "model_name": "builtin.Stale-Llama",
                            "recipe_options": {"llamacpp_backend": "vulkan"},
                        },
                    ]
                },
                "exactly the requested model",
            ),
        )

        for status_code, health, expected_error in cases:
            with self.subTest(health=health, status_code=status_code):

                def request_json(method, url, timeout, **_kwargs):
                    del method, timeout
                    operation = url.rsplit("/", maxsplit=1)[-1]
                    if operation == "load":
                        return self.response(), {}
                    if operation == "health":
                        return self.response(status_code), health
                    if operation == "unload":
                        return self.response(), {}
                    self.fail(f"Unexpected request after invalid health: {operation}")

                with (
                    mock.patch.object(
                        VALIDATION,
                        "request_json",
                        side_effect=request_json,
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    success, error, _stats = VALIDATION.test_model(
                        "http://localhost:13305/api/v1",
                        "builtin.Test-Llama",
                        "vulkan",
                    )

                self.assertFalse(success)
                self.assertIn(expected_error, error)

    def test_model_binds_top_level_response_model_to_the_request(self) -> None:
        for response_model in (None, "builtin.Other-Llama"):
            with self.subTest(response_model=response_model):
                with (
                    mock.patch.object(
                        VALIDATION,
                        "request_json",
                        side_effect=self.request_json_for_chat(
                            {"content": "The answer is 4."},
                            response_model=response_model,
                        ),
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    success, error, _stats = VALIDATION.test_model(
                        "http://localhost:13305/api/v1",
                        "builtin.Test-Llama",
                        "vulkan",
                    )

                self.assertFalse(success)
                self.assertIn("response model", error)

    def test_model_accepts_canonically_equivalent_bare_response_model(self) -> None:
        with (
            mock.patch.object(
                VALIDATION,
                "request_json",
                side_effect=self.request_json_for_chat(
                    {"content": "The answer is 4."},
                    response_model="Test-Llama",
                ),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            success, response, _stats = VALIDATION.test_model(
                "http://localhost:13305/api/v1",
                "builtin.Test-Llama",
                "vulkan",
            )

        self.assertTrue(success, response)

    def test_model_treats_post_inference_unload_failure_as_fatal(self) -> None:
        with (
            mock.patch.object(
                VALIDATION,
                "request_json",
                side_effect=self.request_json_for_chat(
                    {"content": "The answer is 4."},
                    unload_status_code=500,
                ),
            ),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "unload.*HTTP 500"),
        ):
            VALIDATION.test_model(
                "http://localhost:13305/api/v1",
                "builtin.Test-Llama",
                "vulkan",
            )

    def run_validation(self, explicit):
        operations = []
        model = {
            "id": "Test-Llama",
            "recipe": "llamacpp",
            "labels": ["chat", "hot"],
            "checkpoint": "example/Test-Llama:model.gguf",
        }
        argv = [
            "validate_llamacpp.py",
            "--backend",
            "vulkan",
            "--skip-install",
        ]
        if explicit:
            argv.extend(["--model", "Test-Llama"])

        with tempfile.TemporaryDirectory() as temp_dir:
            argv.extend(["--output", str(Path(temp_dir) / "results.json")])
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(VALIDATION, "require_running_server"),
                mock.patch.object(
                    VALIDATION, "get_model_catalog", return_value=[model]
                ),
                mock.patch.object(VALIDATION, "get_builtin_model", return_value=model),
                mock.patch.object(
                    VALIDATION,
                    "unload_all_models",
                    return_value=self.response(),
                ),
                mock.patch.object(
                    VALIDATION,
                    "request_json",
                    side_effect=self.request_json_for_chat(
                        {"content": "The answer is 4."}, operations
                    ),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                VALIDATION.main()

        return operations

    def test_explicit_models_reload_after_unload_but_hot_models_do_not(self) -> None:
        expected_loads = {"explicit": 2, "hot": 1}

        for selection_mode, expected_load_count in expected_loads.items():
            with self.subTest(selection_mode=selection_mode):
                operations = self.run_validation(selection_mode == "explicit")

                self.assertEqual(operations.count("load"), expected_load_count)
                self.assertEqual(operations.count("unload"), expected_load_count)
                if selection_mode == "explicit":
                    first_unload = operations.index("unload")
                    second_load = operations.index("load", first_unload + 1)
                    self.assertLess(first_unload, second_load)

    def test_reload_fails_when_the_first_model_remains_loaded(self) -> None:
        health = {
            "all_models_loaded": [{"model_name": "builtin.Test-Llama"}],
        }
        with (
            mock.patch.object(
                VALIDATION,
                "test_model",
                return_value=(True, "answer", {}),
            ) as test_model,
            mock.patch.object(
                VALIDATION,
                "request_json",
                return_value=(self.response(), health),
            ),
        ):
            success, error, _stats = VALIDATION.validate_model_lifecycle(
                "http://localhost:13305/api/v1",
                "builtin.Test-Llama",
                "vulkan",
                reload_after_unload=True,
            )

        self.assertFalse(success)
        self.assertRegex(error, "still loaded")
        test_model.assert_called_once()

    def test_reload_rejects_any_model_remaining_after_unload(self) -> None:
        health = {
            "all_models_loaded": [{"model_name": "builtin.Stale-Llama"}],
        }
        with (
            mock.patch.object(
                VALIDATION,
                "test_model",
                return_value=(True, "answer", {}),
            ) as test_model,
            mock.patch.object(
                VALIDATION,
                "request_json",
                return_value=(self.response(), health),
            ),
        ):
            success, error, _stats = VALIDATION.validate_model_lifecycle(
                "http://localhost:13305/api/v1",
                "builtin.Test-Llama",
                "vulkan",
                reload_after_unload=True,
            )

        self.assertFalse(success)
        self.assertIn("still loaded", error)
        test_model.assert_called_once()

    def test_reload_requires_a_valid_loaded_model_inventory(self) -> None:
        for health in ({}, {"all_models_loaded": "invalid"}):
            with self.subTest(health=health):
                with (
                    mock.patch.object(
                        VALIDATION,
                        "test_model",
                        return_value=(True, "answer", {}),
                    ) as test_model,
                    mock.patch.object(
                        VALIDATION,
                        "request_json",
                        return_value=(self.response(), health),
                    ),
                ):
                    success, error, _stats = VALIDATION.validate_model_lifecycle(
                        "http://localhost:13305/api/v1",
                        "builtin.Test-Llama",
                        "vulkan",
                        reload_after_unload=True,
                    )

                self.assertFalse(success)
                self.assertRegex(error, "Could not verify unload")
                test_model.assert_called_once()

    def test_reload_smoke_does_not_repeat_the_capability_profile(self) -> None:
        evidence = {}
        health = {"all_models_loaded": []}

        with (
            mock.patch.object(
                VALIDATION,
                "test_model",
                side_effect=[(True, "first", {}), (True, "second", {})],
            ) as test_model,
            mock.patch.object(
                VALIDATION,
                "request_json",
                return_value=(self.response(), health),
            ),
        ):
            result = VALIDATION.validate_model_lifecycle(
                "http://localhost:13305/api/v1",
                "builtin.Test-Llama",
                "vulkan",
                reload_after_unload=True,
                capability_profile=capabilities.K2_HORIZON_PROFILE,
                capability_evidence=evidence,
            )

        self.assertTrue(result[0])
        self.assertEqual(test_model.call_count, 2)
        first_call, second_call = test_model.call_args_list
        self.assertEqual(
            first_call.kwargs,
            {
                "capability_profile": capabilities.K2_HORIZON_PROFILE,
                "capability_evidence": evidence,
            },
        )
        self.assertEqual(second_call.kwargs, {})


class LlamaCppValidationSideEffectTests(unittest.TestCase):
    @staticmethod
    def response(status_code=200):
        return types.SimpleNamespace(status_code=status_code)

    def test_prepare_only_installs_and_pulls_without_running_inference(self) -> None:
        model = {
            "id": "Test-Llama",
            "load_id": "builtin.Test-Llama",
            "recipe": "llamacpp",
            "labels": ["chat"],
            "checkpoint": "example/Test-Llama:model.gguf",
        }
        argv = [
            "validate_llamacpp.py",
            "--backend",
            "vulkan",
            "--prepare-only",
            "--model",
            "Test-Llama",
        ]

        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(VALIDATION, "require_running_server"),
            mock.patch.object(VALIDATION, "get_builtin_model", return_value=model),
            mock.patch.object(
                VALIDATION,
                "unload_all_models",
                return_value=self.response(),
            ),
            mock.patch.object(VALIDATION, "install_backend") as install,
            mock.patch.object(VALIDATION, "pull_model_with_retry") as pull,
            mock.patch.object(VALIDATION, "validate_model_lifecycle") as validate,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                VALIDATION.main()
            except SystemExit as exc:
                self.fail(f"prepare-only exited unexpectedly: {exc}")

        install.assert_called_once()
        pull.assert_called_once_with("builtin.Test-Llama", port=13305, host="127.0.0.1")
        validate.assert_not_called()

    def test_result_file_creation_is_exclusive_and_rejects_non_finite_values(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.json"
            output_path.write_text("preserved", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                VALIDATION.write_results_file(output_path, [])
            self.assertEqual(output_path.read_text(encoding="utf-8"), "preserved")

            non_finite_path = Path(directory) / "non-finite.json"
            with self.assertRaises(ValueError):
                VALIDATION.write_results_file(
                    non_finite_path,
                    [{"tokens_per_second": float("inf")}],
                )

    def test_prepare_only_stops_when_a_model_pull_fails(self) -> None:
        model = {
            "id": "Test-Llama",
            "load_id": "builtin.Test-Llama",
            "recipe": "llamacpp",
            "labels": ["chat"],
            "checkpoint": "example/Test-Llama:model.gguf",
        }
        argv = [
            "validate_llamacpp.py",
            "--backend",
            "vulkan",
            "--prepare-only",
            "--model",
            "Test-Llama",
        ]

        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(VALIDATION, "require_running_server"),
            mock.patch.object(VALIDATION, "get_builtin_model", return_value=model),
            mock.patch.object(
                VALIDATION,
                "unload_all_models",
                return_value=self.response(),
            ),
            mock.patch.object(VALIDATION, "install_backend"),
            mock.patch.object(
                VALIDATION,
                "pull_model_with_retry",
                side_effect=AssertionError("pull failed"),
            ),
            mock.patch.object(VALIDATION, "validate_model_lifecycle") as validate,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            try:
                with self.assertRaisesRegex(AssertionError, "pull failed"):
                    VALIDATION.main()
            except SystemExit as exc:
                self.fail(f"prepare-only exited before the pull: {exc}")

        validate.assert_not_called()

    def test_prepare_only_rejects_invalid_selection_before_server_mutation(
        self,
    ) -> None:
        argv = [
            "validate_llamacpp.py",
            "--backend",
            "vulkan",
            "--prepare-only",
            "--model",
            "Missing-Llama",
        ]

        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(VALIDATION, "require_running_server"),
            mock.patch.object(VALIDATION, "get_builtin_model", return_value=None),
            mock.patch.object(VALIDATION, "install_backend") as install,
            mock.patch.object(VALIDATION, "pull_model_with_retry") as pull,
            mock.patch.object(VALIDATION, "unload_all_models") as unload_models,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            VALIDATION.main()

        self.assertEqual(raised.exception.code, 1)
        install.assert_not_called()
        pull.assert_not_called()
        unload_models.assert_not_called()

    def test_invalid_explicit_model_is_rejected_before_server_mutation(self) -> None:
        argv = [
            "validate_llamacpp.py",
            "--backend",
            "rocm",
            "--channel",
            "nightly",
            "--model",
            "Missing-Llama",
        ]

        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(VALIDATION, "require_running_server"),
            mock.patch.object(VALIDATION, "get_builtin_model", return_value=None),
            mock.patch.object(VALIDATION, "set_rocm_channel") as set_channel,
            mock.patch.object(VALIDATION, "unload_all_models") as unload_models,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            VALIDATION.main()

        self.assertEqual(raised.exception.code, 1)
        set_channel.assert_not_called()
        unload_models.assert_not_called()

    def test_wrong_builtin_resolver_id_is_rejected_before_server_mutation(self) -> None:
        wrong_model = {
            "id": "Different-Llama",
            "recipe": "llamacpp",
            "labels": ["chat"],
            "checkpoint": "example/different:model.gguf",
        }
        argv = [
            "validate_llamacpp.py",
            "--backend",
            "vulkan",
            "--model",
            "Test-Llama",
        ]

        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(VALIDATION, "require_running_server"),
            mock.patch.object(
                VALIDATION,
                "get_builtin_model",
                return_value=wrong_model,
            ),
            mock.patch.object(VALIDATION, "install_backend") as install,
            mock.patch.object(VALIDATION, "unload_all_models") as unload_models,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            VALIDATION.main()

        self.assertEqual(raised.exception.code, 1)
        install.assert_not_called()
        unload_models.assert_not_called()

    def test_initial_clean_state_unload_http_failure_is_fatal(self) -> None:
        model = {
            "id": "Test-Llama",
            "load_id": "builtin.Test-Llama",
            "recipe": "llamacpp",
            "labels": ["chat"],
            "checkpoint": "example/Test-Llama:model.gguf",
        }
        argv = [
            "validate_llamacpp.py",
            "--backend",
            "vulkan",
            "--skip-install",
            "--model",
            "Test-Llama",
        ]

        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(VALIDATION, "require_running_server"),
            mock.patch.object(
                VALIDATION, "select_llamacpp_models", return_value=[model]
            ),
            mock.patch.object(
                VALIDATION,
                "unload_all_models",
                return_value=self.response(500),
            ),
            mock.patch.object(
                VALIDATION,
                "validate_model_lifecycle",
                return_value=(True, "Four.", {}),
            ) as lifecycle,
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "clean state.*HTTP 500"),
        ):
            VALIDATION.main()

        lifecycle.assert_not_called()

    def test_final_cleanup_unload_request_failure_is_fatal(self) -> None:
        model = {
            "id": "Test-Llama",
            "load_id": "builtin.Test-Llama",
            "recipe": "llamacpp",
            "labels": ["chat"],
            "checkpoint": "example/Test-Llama:model.gguf",
        }
        argv = [
            "validate_llamacpp.py",
            "--backend",
            "vulkan",
            "--skip-install",
            "--model",
            "Test-Llama",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            argv.extend(["--output", str(Path(temp_dir) / "results.json")])
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(VALIDATION, "require_running_server"),
                mock.patch.object(
                    VALIDATION, "select_llamacpp_models", return_value=[model]
                ),
                mock.patch.object(
                    VALIDATION,
                    "unload_all_models",
                    side_effect=[
                        self.response(),
                        VALIDATION.requests.RequestException("cleanup failed"),
                    ],
                ),
                mock.patch.object(
                    VALIDATION,
                    "validate_model_lifecycle",
                    return_value=(True, "Four.", {}),
                ),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "cleanup failed"),
            ):
                VALIDATION.main()

    def test_capability_execution_uses_the_requested_canonical_load_id(self) -> None:
        model = {
            "id": "Untrusted-Display-Id",
            "load_id": "builtin.Test-Llama",
            "recipe": "llamacpp",
            "labels": ["chat"],
            "checkpoint": "example/Test-Llama:model.gguf",
        }
        argv = [
            "validate_llamacpp.py",
            "--backend",
            "vulkan",
            "--skip-install",
            "--model",
            "Test-Llama",
            "--capability-profile",
            capabilities.K2_HORIZON_PROFILE,
            "--capability-model",
            "Test-Llama",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            argv.extend(["--output", str(Path(temp_dir) / "results.json")])
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(VALIDATION, "require_running_server"),
                mock.patch.object(
                    VALIDATION, "select_llamacpp_models", return_value=[model]
                ),
                mock.patch.object(
                    VALIDATION,
                    "unload_all_models",
                    return_value=self.response(),
                ),
                mock.patch.object(
                    VALIDATION,
                    "validate_model_lifecycle",
                    return_value=(True, "Four.", {}),
                ) as lifecycle,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                VALIDATION.main()

        self.assertEqual(
            lifecycle.call_args.kwargs["capability_profile"],
            capabilities.K2_HORIZON_PROFILE,
        )

    def test_capability_options_must_be_paired(self) -> None:
        cases = (
            ["--capability-profile", capabilities.K2_HORIZON_PROFILE],
            ["--capability-model", "Test-Llama"],
        )

        for extra_args in cases:
            with self.subTest(extra_args=extra_args):
                argv = [
                    "validate_llamacpp.py",
                    "--backend",
                    "rocm",
                    "--channel",
                    "nightly",
                    *extra_args,
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(VALIDATION, "require_running_server"),
                    mock.patch.object(VALIDATION, "set_rocm_channel") as set_channel,
                    mock.patch.object(VALIDATION, "unload_all_models") as unload_models,
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    VALIDATION.main()

                set_channel.assert_not_called()
                unload_models.assert_not_called()

    def test_capability_profile_can_target_a_selected_hot_model(self) -> None:
        model = {
            "id": "K2-Horizon-0.9B-GGUF",
            "recipe": "llamacpp",
            "labels": ["chat", "hot"],
            "checkpoint": "IFM/example:model.gguf",
        }
        matrix = {
            "profile": capabilities.K2_HORIZON_PROFILE,
            "model": "K2-Horizon-0.9B-GGUF",
            "pass": True,
            "cases": [],
        }

        def validate_lifecycle(
            _base_url,
            _model_name,
            _backend,
            reload_after_unload=False,
            *,
            capability_profile="",
            capability_evidence=None,
        ):
            self.assertFalse(reload_after_unload)
            self.assertEqual(capability_profile, capabilities.K2_HORIZON_PROFILE)
            capability_evidence.update(matrix)
            return True, "Four.", {}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.json"
            argv = [
                "validate_llamacpp.py",
                "--backend",
                "vulkan",
                "--skip-install",
                "--capability-profile",
                capabilities.K2_HORIZON_PROFILE,
                "--capability-model",
                model["id"],
                "--output",
                str(output_path),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(VALIDATION, "require_running_server"),
                mock.patch.object(
                    VALIDATION, "get_model_catalog", return_value=[model]
                ),
                mock.patch.object(
                    VALIDATION,
                    "unload_all_models",
                    return_value=self.response(),
                ),
                mock.patch.object(
                    VALIDATION,
                    "validate_model_lifecycle",
                    side_effect=validate_lifecycle,
                ) as lifecycle,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                VALIDATION.main()

            records = json.loads(output_path.read_text(encoding="utf-8"))

        lifecycle.assert_called_once()
        self.assertEqual(records[0]["capability_matrix"], matrix)

    def test_capability_model_must_be_selected_and_unique(self) -> None:
        cases = (
            ["--capability-model", "Other-Llama"],
            [
                "--capability-model",
                "Test-Llama",
                "--capability-model",
                "Test-Llama",
            ],
        )
        model = {
            "id": "Test-Llama",
            "recipe": "llamacpp",
            "labels": ["chat"],
            "checkpoint": "example/Test-Llama:model.gguf",
        }

        for capability_args in cases:
            with self.subTest(capability_args=capability_args):
                argv = [
                    "validate_llamacpp.py",
                    "--backend",
                    "vulkan",
                    "--model",
                    "Test-Llama",
                    "--capability-profile",
                    capabilities.K2_HORIZON_PROFILE,
                    *capability_args,
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(VALIDATION, "require_running_server"),
                    mock.patch.object(
                        VALIDATION,
                        "get_builtin_model",
                        return_value=model,
                    ),
                    mock.patch.object(VALIDATION, "install_backend") as install,
                    mock.patch.object(VALIDATION, "unload_all_models") as unload_models,
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    VALIDATION.main()

                install.assert_not_called()
                unload_models.assert_not_called()


if __name__ == "__main__":
    unittest.main()
