#!/usr/bin/env python3
"""Unit tests for llama.cpp validation model selection."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import unittest
from pathlib import Path

from test.utils import validation_model_selection as selection

MODEL_REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "cpp"
    / "resources"
    / "server_models.json"
)


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


if __name__ == "__main__":
    unittest.main()
