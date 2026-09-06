#!/usr/bin/env python3
"""Tests for scheduled llama.cpp validation evidence checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test.utils import llamacpp_validation_evidence as evidence

K2_SMALL = "K2-Horizon-0.9B-GGUF"


def expected_matrix(models: list[str] | None = None) -> dict:
    planned_models = models or [K2_SMALL]
    return {
        "include": [
            {
                "backend": "vulkan",
                "channel": "",
                "models": [],
                "expected_models": planned_models,
            },
            {
                "backend": "rocm",
                "channel": "stable",
                "models": [],
                "expected_models": planned_models,
            },
            {
                "backend": "rocm",
                "channel": "nightly",
                "models": [],
                "expected_models": planned_models,
            },
        ]
    }


def passing_result(model: str = "K2-Horizon-0.9B-GGUF") -> dict:
    return {
        "model": model,
        "pass": True,
        "response": "Four.",
        "input_tokens": 10,
        "output_tokens": 2,
        "time_to_first_token": 0.5,
        "tokens_per_second": 4.0,
    }


class LlamaCppValidationEvidenceTests(unittest.TestCase):
    def write_expected_results(self, root: Path) -> None:
        for filename in evidence.EXPECTED_RESULT_FILES:
            (root / filename).write_text(
                json.dumps([passing_result()]),
                encoding="utf-8",
            )

    def test_all_expected_nonempty_passing_results_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root)

            results = evidence.load_and_validate_results(root, expected_matrix())

        self.assertEqual(set(results), set(evidence.EXPECTED_RESULT_FILES))
        self.assertTrue(all(len(records) == 1 for records in results.values()))

    def test_missing_empty_and_invalid_json_files_fail_closed(self) -> None:
        cases = {
            "missing": None,
            "empty": "",
            "invalid": "not json",
            "empty list": "[]",
        }
        for name, replacement in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_expected_results(root)
                target = root / evidence.EXPECTED_RESULT_FILES[0]
                if replacement is None:
                    target.unlink()
                else:
                    target.write_text(replacement, encoding="utf-8")

                with self.assertRaises(evidence.ValidationEvidenceError):
                    evidence.load_and_validate_results(root, expected_matrix())

    def test_failed_or_malformed_records_fail_closed(self) -> None:
        cases = {
            "failed": [{**passing_result(), "pass": False}],
            "non_boolean pass": [{**passing_result(), "pass": 1}],
            "invalid metric": [{**passing_result(), "tokens_per_second": "unexpected"}],
            "negative token count": [{**passing_result(), "input_tokens": -1}],
            "missing field": [
                {
                    key: value
                    for key, value in passing_result().items()
                    if key != "response"
                }
            ],
            "non_object": ["PASS"],
            "duplicate model": [passing_result(), passing_result()],
        }
        for name, replacement in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_expected_results(root)
                (root / evidence.EXPECTED_RESULT_FILES[0]).write_text(
                    json.dumps(replacement),
                    encoding="utf-8",
                )

                with self.assertRaises(evidence.ValidationEvidenceError):
                    evidence.load_and_validate_results(root, expected_matrix())

    def test_runtime_models_must_exactly_match_each_planned_lane(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root)
            target = root / "llamacpp_validation_rocm-stable.json"
            target.write_text(
                json.dumps([passing_result("some-other-model")]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                evidence.ValidationEvidenceError,
                "do not match planned models",
            ):
                evidence.load_and_validate_results(root, expected_matrix())

    def test_every_planned_lane_must_include_k2_small(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in evidence.EXPECTED_RESULT_FILES:
                (root / filename).write_text(
                    json.dumps([passing_result("some-other-model")]),
                    encoding="utf-8",
                )

            with self.assertRaisesRegex(
                evidence.ValidationEvidenceError,
                K2_SMALL,
            ):
                evidence.load_and_validate_results(
                    root,
                    expected_matrix(["some-other-model"]),
                )

    def test_inconsistent_planned_lane_sets_fail_closed(self) -> None:
        matrix = expected_matrix()
        matrix["include"][2]["expected_models"] = [K2_SMALL, "another-model"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root)
            (root / "llamacpp_validation_rocm-nightly.json").write_text(
                json.dumps([passing_result(K2_SMALL), passing_result("another-model")]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                evidence.ValidationEvidenceError,
                "same planned model set",
            ):
                evidence.load_and_validate_results(root, matrix)


if __name__ == "__main__":
    unittest.main()
