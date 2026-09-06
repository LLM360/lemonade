#!/usr/bin/env python3
"""Tests for scheduled llama.cpp validation evidence checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test.utils import llamacpp_capability_validation as capabilities
from test.utils import llamacpp_validation_evidence as evidence

K2_SMALL = "K2-Horizon-0.9B-GGUF"


def expected_matrix(models: list[str] | None = None) -> dict:
    planned_models = models or [K2_SMALL]
    return {
        "include": [
            {
                "target": "windows-vulkan",
                "backend": "vulkan",
                "channel": "",
                "models": [],
                "expected_models": planned_models,
                "capability_profile": capabilities.K2_HORIZON_PROFILE,
                "capability_models": [K2_SMALL],
            },
            {
                "target": "rocm-stable",
                "backend": "rocm",
                "channel": "stable",
                "models": [],
                "expected_models": planned_models,
                "capability_profile": capabilities.K2_HORIZON_PROFILE,
                "capability_models": [K2_SMALL],
            },
            {
                "target": "rocm-nightly",
                "backend": "rocm",
                "channel": "nightly",
                "models": [],
                "expected_models": planned_models,
                "capability_profile": capabilities.K2_HORIZON_PROFILE,
                "capability_models": [K2_SMALL],
            },
        ]
    }


def passing_capability_matrix(model: str = K2_SMALL) -> dict:
    cases = []
    for case_id in capabilities.K2_HORIZON_CASE_IDS:
        protocol = "openai"
        if case_id == "ollama_thinking":
            protocol = "ollama"
        elif case_id == "anthropic_translation":
            protocol = "anthropic"
        stream = case_id in {
            "openai_plain_off_stream",
            "openai_reasoning_high_stream",
            "openai_tool_stream_xml",
        }
        tool_case = case_id in {
            "openai_tool_json",
            "openai_tool_xml",
            "openai_tool_xml_typed",
            "openai_tool_default_xml",
            "openai_tool_stream_xml",
        }
        reasoning_case = case_id.startswith("openai_reasoning_") or case_id == (
            "ollama_thinking"
        )
        content_case = case_id in {
            "openai_plain_off_nonstream",
            "openai_plain_off_stream",
            "openai_tool_result_followup",
            "anthropic_translation",
            "ollama_thinking",
        } or case_id.startswith("openai_reasoning_")
        finish_reason = "stop"
        if tool_case:
            finish_reason = "tool_calls"
        elif case_id == "anthropic_translation":
            finish_reason = "end_turn"
        cases.append(
            {
                "id": case_id,
                "protocol": protocol,
                "stream": stream,
                "pass": True,
                "status_code": 200,
                "finish_reason": finish_reason,
                "content_chars": 2 if content_case else 0,
                "reasoning_chars": 8 if reasoning_case else 0,
                "tool_call_count": 1 if tool_case else 0,
                "terminal_frame": True,
                "tool_name": capabilities.TOOL_NAME if tool_case else None,
                "tool_arguments": (
                    capabilities.EXPECTED_TOOL_ARGUMENTS if tool_case else None
                ),
                "parity_with": (
                    "openai_tool_xml"
                    if case_id in {"openai_tool_default_xml", "openai_tool_stream_xml"}
                    else None
                ),
                "raw_ifm_marker": None,
                "error": None,
            }
        )
    return {
        "profile": capabilities.K2_HORIZON_PROFILE,
        "model": f"builtin.{model.removeprefix('builtin.')}",
        "pass": True,
        "cases": cases,
    }


def passing_result(model: str = K2_SMALL, *, capability: bool | None = None) -> dict:
    result = {
        "model": model,
        "pass": True,
        "response": "Four.",
        "input_tokens": 10,
        "output_tokens": 2,
        "time_to_first_token": 0.5,
        "tokens_per_second": 4.0,
    }
    if capability if capability is not None else model == K2_SMALL:
        result["capability_matrix"] = passing_capability_matrix(model)
    return result


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
            "blank response": [{**passing_result(), "response": " \t"}],
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

    def test_capability_evidence_is_required_and_validated(self) -> None:
        mutations = {}
        missing = passing_result()
        missing.pop("capability_matrix")
        mutations["missing"] = missing
        failed = passing_result()
        failed["capability_matrix"]["pass"] = False
        mutations["failed"] = failed
        marker = passing_result()
        marker["capability_matrix"]["cases"][0]["raw_ifm_marker"] = "<ifm|think>"
        mutations["marker"] = marker
        wrong_model = passing_result()
        wrong_model["capability_matrix"]["model"] = "builtin.Other-Llama"
        mutations["wrong model"] = wrong_model

        for name, result in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_expected_results(root)
                (root / evidence.EXPECTED_RESULT_FILES[0]).write_text(
                    json.dumps([result]),
                    encoding="utf-8",
                )

                with self.assertRaises(evidence.ValidationEvidenceError):
                    evidence.load_and_validate_results(root, expected_matrix())

    def test_duplicate_json_keys_in_evidence_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root)
            target = root / evidence.EXPECTED_RESULT_FILES[0]
            raw = json.dumps([passing_result()])
            raw = raw.replace(
                '"city": "Paris"',
                '"city": "London", "city": "Paris"',
                1,
            )
            target.write_text(raw, encoding="utf-8")

            with self.assertRaisesRegex(
                evidence.ValidationEvidenceError,
                "duplicate JSON key",
            ):
                evidence.load_and_validate_results(root, expected_matrix())

    def test_non_designated_model_cannot_attach_capability_evidence(self) -> None:
        other_model = "Other-Llama"
        matrix = expected_matrix([K2_SMALL, other_model])
        records = [
            passing_result(),
            passing_result(other_model, capability=True),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in evidence.EXPECTED_RESULT_FILES:
                (root / filename).write_text(json.dumps(records), encoding="utf-8")

            with self.assertRaisesRegex(
                evidence.ValidationEvidenceError,
                "unexpected capability matrix",
            ):
                evidence.load_and_validate_results(root, matrix)

    def test_capability_plan_fields_are_paired_and_model_bound(self) -> None:
        mutations = {}
        missing_profile = expected_matrix()
        missing_profile["include"][0].pop("capability_profile")
        mutations["missing profile"] = missing_profile
        missing_models = expected_matrix()
        missing_models["include"][0].pop("capability_models")
        mutations["missing models"] = missing_models
        unselected = expected_matrix()
        unselected["include"][0]["capability_models"] = ["Other-Llama"]
        mutations["unselected model"] = unselected
        unknown_profile = expected_matrix()
        unknown_profile["include"][0]["capability_profile"] = "unknown"
        mutations["unknown profile"] = unknown_profile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root)
            for name, matrix in mutations.items():
                with self.subTest(name=name):
                    with self.assertRaises(evidence.ValidationEvidenceError):
                        evidence.load_and_validate_results(root, matrix)


if __name__ == "__main__":
    unittest.main()
