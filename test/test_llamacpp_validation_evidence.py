#!/usr/bin/env python3
"""Tests for scheduled llama.cpp validation evidence checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.utils import llamacpp_capability_validation as capabilities
from test.utils import llamacpp_validation_evidence as evidence
from test.utils import llamacpp_validation_plan as planning

K2_SMALL = "K2-Horizon-0.9B-GGUF"


def expected_matrix(
    models_by_target: dict[str, list[str]] | None = None,
) -> dict:
    matrix = planning.create_validation_plan("schedule")
    for row in matrix["include"]:
        if models_by_target is None or row["target"] not in models_by_target:
            continue
        planned_models = models_by_target[row["target"]]
        row["models"] = list(planned_models)
        row["expected_models"] = list(planned_models)
    return matrix


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
        reasoning_case = (
            case_id.startswith("openai_reasoning_")
            and case_id != "openai_reasoning_low"
        ) or case_id == "ollama_thinking"
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
    def write_expected_results(self, root: Path, matrix: dict | None = None) -> None:
        selected_matrix = matrix or expected_matrix()
        for row in selected_matrix["include"]:
            filename = f"llamacpp_validation_{row['target']}.json"
            records = [
                passing_result(model, capability=model == K2_SMALL)
                for model in row["expected_models"]
            ]
            (root / filename).write_text(
                json.dumps(records),
                encoding="utf-8",
            )
            restart_filename = f"llamacpp_restart_validation_{row['target']}.json"
            restart_model = row["capability_models"][0]
            (root / restart_filename).write_text(
                json.dumps([passing_result(restart_model, capability=False)]),
                encoding="utf-8",
            )

    def test_all_expected_nonempty_passing_results_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root)

            results = evidence.load_and_validate_results(root, expected_matrix())

        self.assertEqual(
            set(results),
            set(evidence.EXPECTED_RESULT_FILES)
            | set(evidence.EXPECTED_RESTART_RESULT_FILES),
        )
        self.assertGreater(
            len(results["llamacpp_validation_windows-vulkan.json"]),
            1,
        )
        self.assertTrue(
            all(
                len(records) == 1
                for filename, records in results.items()
                if filename != "llamacpp_validation_windows-vulkan.json"
            )
        )

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

    def test_restart_evidence_must_be_present_exact_and_passing(self) -> None:
        mutations = {
            "missing": None,
            "empty": "",
            "invalid": "not json",
            "empty list": "[]",
            "failed": json.dumps([{**passing_result(capability=False), "pass": False}]),
            "wrong model": json.dumps(
                [passing_result("Other-Llama", capability=False)]
            ),
            "extra model": json.dumps(
                [
                    passing_result(capability=False),
                    passing_result("Other-Llama", capability=False),
                ]
            ),
            "unexpected capabilities": json.dumps([passing_result()]),
        }
        target_name = "llamacpp_restart_validation_linux-vulkan.json"
        for name, replacement in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_expected_results(root)
                target = root / target_name
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

    def test_exact_producer_record_schema_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            expected = passing_result()
            path.write_text(json.dumps([expected]), encoding="utf-8")

            records = evidence.load_and_validate_result_file(
                path,
                [K2_SMALL],
                capability_profile=capabilities.K2_HORIZON_PROFILE,
                capability_models=[K2_SMALL],
            )

        self.assertEqual(records, [expected])

    def test_result_file_above_byte_limit_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(
                json.dumps([passing_result(capability=False)]),
                encoding="utf-8",
            )

            with (
                mock.patch.object(
                    evidence,
                    "MAX_VALIDATION_EVIDENCE_BYTES",
                    path.stat().st_size - 1,
                ),
                self.assertRaisesRegex(
                    evidence.ValidationEvidenceError,
                    "byte limit",
                ),
            ):
                evidence.load_and_validate_result_file(path, [K2_SMALL])

    def test_result_file_at_byte_limit_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            expected = passing_result(capability=False)
            path.write_text(json.dumps([expected]), encoding="utf-8")

            with mock.patch.object(
                evidence,
                "MAX_VALIDATION_EVIDENCE_BYTES",
                path.stat().st_size,
            ):
                records = evidence.load_and_validate_result_file(path, [K2_SMALL])

        self.assertEqual(records, [expected])

    def test_result_file_symlink_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(
                json.dumps([passing_result(capability=False)]),
                encoding="utf-8",
            )
            path = root / "result.json"
            try:
                path.symlink_to(source.name)
            except OSError as exc:
                self.skipTest(f"symlinks unavailable: {exc}")

            with self.assertRaisesRegex(
                evidence.ValidationEvidenceError,
                "regular file",
            ):
                evidence.load_and_validate_result_file(path, [K2_SMALL])

    def test_result_and_capability_schemas_reject_unknown_fields(self) -> None:
        def add_result_field(records: list[dict]) -> None:
            records[0]["schema_version"] = 999

        def add_matrix_field(records: list[dict]) -> None:
            capability_record = next(
                record for record in records if "capability_matrix" in record
            )
            capability_record["capability_matrix"]["verification"] = "failed"

        def add_case_field(records: list[dict]) -> None:
            capability_record = next(
                record for record in records if "capability_matrix" in record
            )
            capability_record["capability_matrix"]["cases"][0][
                "verification"
            ] = "failed"

        mutations = {
            "result": add_result_field,
            "capability matrix": add_matrix_field,
            "capability case": add_case_field,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_expected_results(root)
                target = root / evidence.EXPECTED_RESULT_FILES[0]
                records = json.loads(target.read_text(encoding="utf-8"))
                mutate(records)
                target.write_text(json.dumps(records), encoding="utf-8")

                with self.assertRaisesRegex(
                    evidence.ValidationEvidenceError,
                    "unexpected.*fields",
                ):
                    evidence.load_and_validate_results(root, expected_matrix())

    def test_runtime_models_must_exactly_match_each_planned_lane(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root)
            target = root / "llamacpp_validation_windows-rocm-stable.json"
            target.write_text(
                json.dumps([passing_result("some-other-model")]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                evidence.ValidationEvidenceError,
                "do not match planned models",
            ):
                evidence.load_and_validate_results(root, expected_matrix())

    def test_selected_models_must_exactly_match_expected_models(self) -> None:
        matrix = expected_matrix()
        matrix["include"][0]["models"] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root, matrix)

            with self.assertRaisesRegex(
                evidence.ValidationEvidenceError,
                "selected models",
            ):
                evidence.load_and_validate_results(root, matrix)

    def test_every_planned_lane_must_include_k2_small(self) -> None:
        matrix = expected_matrix()
        for row in matrix["include"]:
            row["models"] = ["some-other-model"]
            row["expected_models"] = ["some-other-model"]
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
                    matrix,
                )

    def test_exact_per_lane_planned_model_sets_are_accepted(self) -> None:
        matrix = expected_matrix()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root, matrix)

            results = evidence.load_and_validate_results(root, matrix)

        self.assertEqual(
            len(results["llamacpp_validation_windows-vulkan.json"]),
            len(matrix["include"][0]["expected_models"]),
        )

    def test_exact_lane_mapping_is_required(self) -> None:
        mutations = {}
        missing = expected_matrix()
        missing["include"].pop()
        mutations["missing"] = missing
        wrong_backend = expected_matrix()
        wrong_backend["include"][2]["backend"] = "cuda"
        mutations["backend"] = wrong_backend
        wrong_pin = expected_matrix()
        wrong_pin["include"][3]["managed_pin"] = "cpu"
        mutations["pin"] = wrong_pin
        wrong_platform = expected_matrix()
        wrong_platform["include"][1]["build_platform"] = "windows"
        mutations["platform"] = wrong_platform
        wrong_runner = expected_matrix()
        wrong_runner["include"][0]["runner"] = ["windows-latest"]
        mutations["runner"] = wrong_runner

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root)
            for name, matrix in mutations.items():
                with self.subTest(name=name):
                    with self.assertRaisesRegex(
                        evidence.ValidationEvidenceError,
                        "scheduled validation lanes|mapping",
                    ):
                        evidence.load_and_validate_results(root, matrix)

    def test_large_k2_models_require_the_128gb_vulkan_lane(self) -> None:
        for model in (
            planning.K2_MEDIUM,
            planning.K2_LARGE,
            f"builtin.{planning.K2_MEDIUM}",
            f"builtin.{planning.K2_LARGE}",
        ):
            matrix = expected_matrix({"linux-vulkan": [K2_SMALL, model]})
            with self.subTest(model=model), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_expected_results(root, matrix)
                with self.assertRaisesRegex(
                    evidence.ValidationEvidenceError,
                    "128gb",
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

    def test_nonstandard_json_numeric_constants_fail_closed(self) -> None:
        for constant in ("NaN", "Infinity", "-Infinity"):
            with (
                self.subTest(constant=constant),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                self.write_expected_results(root)
                target = root / evidence.EXPECTED_RESULT_FILES[0]
                raw = target.read_text(encoding="utf-8").replace(
                    '"tokens_per_second": 4.0',
                    f'"tokens_per_second": {constant}',
                    1,
                )
                target.write_text(raw, encoding="utf-8")

                with self.assertRaisesRegex(
                    evidence.ValidationEvidenceError,
                    "nonstandard JSON numeric constant",
                ):
                    evidence.load_and_validate_results(root, expected_matrix())

    def test_overflowed_numeric_metrics_fail_closed(self) -> None:
        for field, value in (
            ("time_to_first_token", "1e400"),
            ("tokens_per_second", "1e400"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_expected_results(root)
                target = root / evidence.EXPECTED_RESULT_FILES[0]
                raw = target.read_text(encoding="utf-8")
                raw = raw.replace(
                    f'"{field}": {passing_result()[field]}',
                    f'"{field}": {value}',
                    1,
                )
                target.write_text(raw, encoding="utf-8")

                with self.assertRaisesRegex(
                    evidence.ValidationEvidenceError,
                    "non-finite JSON number",
                ):
                    evidence.load_and_validate_results(root, expected_matrix())

    def test_overflowed_numeric_extra_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root)
            target = root / evidence.EXPECTED_RESULT_FILES[0]
            raw = target.read_text(encoding="utf-8").replace(
                "{", '{"untrusted_extra": 1e400,', 1
            )
            target.write_text(raw, encoding="utf-8")

            with self.assertRaisesRegex(
                evidence.ValidationEvidenceError,
                "non-finite JSON number",
            ):
                evidence.load_and_validate_results(root, expected_matrix())

    def test_overflowed_numeric_extra_matrix_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root)
            raw_matrix = json.dumps(expected_matrix()).replace(
                "{", '{"untrusted_extra": 1e400,', 1
            )

            with (
                mock.patch(
                    "sys.argv",
                    [
                        "llamacpp_validation_evidence.py",
                        "--directory",
                        str(root),
                        "--expected-matrix-json",
                        raw_matrix,
                    ],
                ),
                self.assertRaisesRegex(SystemExit, "1"),
            ):
                evidence.main()

    def test_non_designated_model_cannot_attach_capability_evidence(self) -> None:
        matrix = expected_matrix()
        planned_models = matrix["include"][0]["expected_models"]
        other_model = next(model for model in planned_models if model != K2_SMALL)
        records = [
            passing_result(
                model,
                capability=model in {K2_SMALL, other_model},
            )
            for model in planned_models
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_expected_results(root, matrix)
            (root / "llamacpp_validation_windows-vulkan.json").write_text(
                json.dumps(records), encoding="utf-8"
            )

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
