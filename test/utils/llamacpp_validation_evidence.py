#!/usr/bin/env python3
"""Validate complete scheduled llama.cpp result evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_RESULT_FILES = (
    "llamacpp_validation_vulkan.json",
    "llamacpp_validation_rocm-stable.json",
    "llamacpp_validation_rocm-nightly.json",
)
K2_SMALL = "K2-Horizon-0.9B-GGUF"
REQUIRED_RESULT_FIELDS = {
    "model",
    "pass",
    "response",
    "input_tokens",
    "output_tokens",
    "time_to_first_token",
    "tokens_per_second",
}


class ValidationEvidenceError(ValueError):
    """Raised when scheduled validation evidence is incomplete or failed."""


def _validate_record(filename: str, index: int, record: object) -> dict:
    if not isinstance(record, dict):
        raise ValidationEvidenceError(f"{filename}: result {index} must be an object")
    missing_fields = REQUIRED_RESULT_FIELDS - set(record)
    if missing_fields:
        raise ValidationEvidenceError(
            f"{filename}: result {index} missing fields: "
            + ", ".join(sorted(missing_fields))
        )
    model = record["model"]
    if not isinstance(model, str) or not model:
        raise ValidationEvidenceError(
            f"{filename}: result {index} model must be nonempty"
        )
    passed = record["pass"]
    if not isinstance(passed, bool):
        raise ValidationEvidenceError(
            f"{filename}: result {index} pass must be boolean"
        )
    if not passed:
        raise ValidationEvidenceError(f"{filename}: {model} did not pass validation")
    response = record["response"]
    if not isinstance(response, str) or not response:
        raise ValidationEvidenceError(
            f"{filename}: result {index} response must be nonempty"
        )
    for field in ("input_tokens", "output_tokens"):
        value = record[field]
        if not (
            value == "N/A"
            or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)
        ):
            raise ValidationEvidenceError(
                f"{filename}: result {index} {field} must be nonnegative or N/A"
            )
    for field in ("time_to_first_token", "tokens_per_second"):
        value = record[field]
        if not (
            value == "N/A"
            or (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value >= 0
            )
        ):
            raise ValidationEvidenceError(
                f"{filename}: result {index} {field} must be nonnegative or N/A"
            )
    return record


def _expected_models_by_file(expected_matrix: object) -> dict[str, list[str]]:
    if not isinstance(expected_matrix, dict):
        raise ValidationEvidenceError("validation matrix must be an object")
    rows = expected_matrix.get("include")
    if not isinstance(rows, list):
        raise ValidationEvidenceError("validation matrix include must be an array")

    expected = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValidationEvidenceError(
                f"validation matrix row {index} must be an object"
            )
        backend = row.get("backend")
        channel = row.get("channel")
        if not isinstance(backend, str) or not isinstance(channel, str):
            raise ValidationEvidenceError(
                f"validation matrix row {index} backend and channel must be strings"
            )
        label = f"{backend}-{channel}" if channel else backend
        filename = f"llamacpp_validation_{label}.json"
        if filename in expected:
            raise ValidationEvidenceError(f"duplicate validation lane: {label}")
        if row.get("models") != []:
            raise ValidationEvidenceError(
                f"scheduled validation lane {label} must keep runtime hot selection"
            )
        models = row.get("expected_models")
        if (
            not isinstance(models, list)
            or not models
            or not all(isinstance(model, str) and model for model in models)
            or len(models) != len(set(models))
        ):
            raise ValidationEvidenceError(
                f"validation matrix lane {label} needs unique expected models"
            )
        if K2_SMALL not in models:
            raise ValidationEvidenceError(
                f"validation matrix lane {label} must include {K2_SMALL}"
            )
        expected[filename] = models

    if set(expected) != set(EXPECTED_RESULT_FILES):
        raise ValidationEvidenceError(
            "validation matrix must contain exactly the scheduled validation lanes"
        )
    planned_model_lists = list(expected.values())
    if any(models != planned_model_lists[0] for models in planned_model_lists[1:]):
        raise ValidationEvidenceError(
            "scheduled validation lanes must use the same planned model set"
        )
    return expected


def load_and_validate_results(
    root: Path,
    expected_matrix: object,
) -> dict[str, list[dict]]:
    expected_models = _expected_models_by_file(expected_matrix)
    validated = {}
    for filename in EXPECTED_RESULT_FILES:
        path = root / filename
        if not path.is_file():
            raise ValidationEvidenceError(f"missing validation result: {filename}")
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValidationEvidenceError(f"could not read {filename}: {exc}") from exc
        if not raw.strip():
            raise ValidationEvidenceError(f"validation result is empty: {filename}")
        try:
            records = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationEvidenceError(
                f"validation result is invalid JSON: {filename}: {exc}"
            ) from exc
        if not isinstance(records, list) or not records:
            raise ValidationEvidenceError(
                f"validation result must be a nonempty array: {filename}"
            )
        checked = [
            _validate_record(filename, index, record)
            for index, record in enumerate(records)
        ]
        models = [record["model"] for record in checked]
        if len(models) != len(set(models)):
            raise ValidationEvidenceError(
                f"validation result contains duplicate models: {filename}"
            )
        if models != expected_models[filename]:
            raise ValidationEvidenceError(
                f"{filename}: result models do not match planned models"
            )
        validated[filename] = checked
    return validated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    parser.add_argument("--expected-matrix-json", required=True)
    args = parser.parse_args()
    try:
        expected_matrix = json.loads(args.expected_matrix_json)
        results = load_and_validate_results(args.directory, expected_matrix)
    except (json.JSONDecodeError, ValidationEvidenceError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    for filename, records in results.items():
        print(f"{filename}: {len(records)} passing result(s)")


if __name__ == "__main__":
    main()
