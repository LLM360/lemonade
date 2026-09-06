#!/usr/bin/env python3
"""Unit tests for llama.cpp validation model selection."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from test.utils import validation_model_selection as selection

MODEL_REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "cpp"
    / "resources"
    / "server_models.json"
)


def load_validation_module():
    requests_module = types.ModuleType("requests")
    requests_module.RequestException = RuntimeError

    server_base_module = types.ModuleType("utils.server_base")
    server_base_module._auth_headers = lambda: {}
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
        "utils.server_base": server_base_module,
        "utils.test_models": test_models_module,
        "utils.validation_model_selection": selection,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


VALIDATION = load_validation_module()


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
    def test_k2_horizon_bf16_models_have_exact_catalog_contract(self) -> None:
        models = json.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))
        expected = {
            "K2-Horizon-0.9B-GGUF": {
                "checkpoint": ("IFM/K2-Horizon-0.9B-GGUF:K2-Horizon-1B-BF16.gguf"),
                "recipe": "llamacpp",
                "suggested": True,
                "labels": ["chat", "reasoning", "tool-calling", "hot"],
                "size": 2.16,
            },
            "K2-Horizon-3.7B-GGUF": {
                "checkpoint": ("IFM/K2-Horizon-3.7B-GGUF:K2-Horizon-4B-BF16.gguf"),
                "recipe": "llamacpp",
                "suggested": True,
                "labels": ["chat", "reasoning", "tool-calling"],
                "size": 10.13,
            },
            "K2-Horizon-7B-GGUF": {
                "checkpoint": ("IFM/K2-Horizon-7B-GGUF:K2-Horizon-7B-BF16.gguf"),
                "recipe": "llamacpp",
                "suggested": True,
                "labels": ["chat", "reasoning", "tool-calling"],
                "size": 18.01,
            },
        }

        for model_id, expected_record in expected.items():
            with self.subTest(model_id=model_id):
                self.assertEqual(models.get(model_id), expected_record)

    def test_model_registry_has_no_duplicate_ids(self) -> None:
        duplicate_ids = []

        def unique_object(pairs):
            parsed = {}
            for key, value in pairs:
                if key in parsed:
                    duplicate_ids.append(key)
                parsed[key] = value
            return parsed

        json.loads(
            MODEL_REGISTRY.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )

        self.assertEqual(duplicate_ids, [])


class LlamaCppValidationRuntimeTests(unittest.TestCase):
    @staticmethod
    def response(status_code=200):
        return types.SimpleNamespace(status_code=status_code)

    def request_json_for_chat(self, chat_message, operations=None):
        loaded = False

        def request_json(method, url, timeout, **_kwargs):
            nonlocal loaded
            del method, timeout
            operation = url.rsplit("/", maxsplit=1)[-1]
            if operations is not None:
                operations.append(operation)
            if operation == "load":
                loaded = True
                return self.response(), {}
            if operation == "unload":
                loaded = False
                return self.response(), {}
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
                return self.response(), {"choices": [{"message": chat_message}]}
            if operation == "stats":
                return self.response(), {"output_tokens": 4}
            return self.response(), {}

        return request_json

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
                mock.patch.object(VALIDATION, "unload_all_models"),
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


class LlamaCppValidationSideEffectTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
