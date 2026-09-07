#!/usr/bin/env python3
"""Validate complete scheduled llama.cpp result evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import stat
from pathlib import Path

if __package__:
    from .llamacpp_capability_validation import (
        K2_HORIZON_PROFILE,
        CapabilityValidationError,
        validate_capability_matrix_evidence,
    )
    from .llamacpp_validation_plan import (
        K2_LARGE,
        K2_MEDIUM,
        K2_SMALL,
        VALIDATION_LANES,
        require_complete_promotion_coverage,
    )
else:
    from llamacpp_capability_validation import (  # type: ignore[import-not-found]
        K2_HORIZON_PROFILE,
        CapabilityValidationError,
        validate_capability_matrix_evidence,
    )
    from llamacpp_validation_plan import (  # type: ignore[import-not-found]
        K2_LARGE,
        K2_MEDIUM,
        K2_SMALL,
        VALIDATION_LANES,
        require_complete_promotion_coverage,
    )

EXPECTED_RESULT_FILES = tuple(
    f"llamacpp_validation_{lane['target']}.json" for lane in VALIDATION_LANES
)
EXPECTED_RESTART_RESULT_FILES = tuple(
    f"llamacpp_restart_validation_{lane['target']}.json" for lane in VALIDATION_LANES
)
REQUIRED_RESULT_FIELDS = {
    "model",
    "pass",
    "response",
    "input_tokens",
    "output_tokens",
    "time_to_first_token",
    "tokens_per_second",
}
MAX_VALIDATION_EVIDENCE_BYTES = 4 * 1024 * 1024
_EVIDENCE_READ_CHUNK_BYTES = 64 * 1024


class ValidationEvidenceError(ValueError):
    """Raised when scheduled validation evidence is incomplete or failed."""


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationEvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValidationEvidenceError(f"nonstandard JSON numeric constant: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValidationEvidenceError(f"non-finite JSON number: {value}")
    return parsed


def _validate_record(filename: str, index: int, record: object) -> dict:
    if not isinstance(record, dict):
        raise ValidationEvidenceError(f"{filename}: result {index} must be an object")
    unexpected_fields = set(record) - REQUIRED_RESULT_FIELDS - {"capability_matrix"}
    if unexpected_fields:
        raise ValidationEvidenceError(
            f"{filename}: result {index} has unexpected fields: "
            + ", ".join(sorted(unexpected_fields))
        )
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
    if not isinstance(response, str) or not response.strip():
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
                and (isinstance(value, int) or math.isfinite(value))
            )
        ):
            raise ValidationEvidenceError(
                f"{filename}: result {index} {field} must be nonnegative or N/A"
            )
    return record


def _canonical_model_id(model_id: str) -> str:
    return model_id.removeprefix("builtin.")


def _stat_snapshot(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _read_bounded_result_file(path: Path) -> str:
    filename = path.name
    try:
        path_status = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValidationEvidenceError(f"missing validation result: {filename}") from exc
    except OSError as exc:
        raise ValidationEvidenceError(f"could not read {filename}: {exc}") from exc
    if not stat.S_ISREG(path_status.st_mode):
        raise ValidationEvidenceError(
            f"validation result must be a regular file: {filename}"
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValidationEvidenceError(f"could not read {filename}: {exc}") from exc

    try:
        opened_status = os.fstat(descriptor)
        if not stat.S_ISREG(opened_status.st_mode):
            raise ValidationEvidenceError(
                f"validation result must be a regular file: {filename}"
            )
        if (opened_status.st_dev, opened_status.st_ino) != (
            path_status.st_dev,
            path_status.st_ino,
        ):
            raise ValidationEvidenceError(
                f"validation result changed before reading: {filename}"
            )
        if opened_status.st_size > MAX_VALIDATION_EVIDENCE_BYTES:
            raise ValidationEvidenceError(
                f"validation result exceeds byte limit: {filename}"
            )

        chunks = []
        total_bytes = 0
        while total_bytes <= MAX_VALIDATION_EVIDENCE_BYTES:
            chunk = os.read(
                descriptor,
                min(
                    _EVIDENCE_READ_CHUNK_BYTES,
                    MAX_VALIDATION_EVIDENCE_BYTES + 1 - total_bytes,
                ),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total_bytes += len(chunk)
        final_status = os.fstat(descriptor)
    except ValidationEvidenceError:
        raise
    except OSError as exc:
        raise ValidationEvidenceError(f"could not read {filename}: {exc}") from exc
    finally:
        os.close(descriptor)

    if total_bytes > MAX_VALIDATION_EVIDENCE_BYTES:
        raise ValidationEvidenceError(
            f"validation result exceeds byte limit: {filename}"
        )
    if _stat_snapshot(opened_status) != _stat_snapshot(final_status):
        raise ValidationEvidenceError(
            f"validation result changed while reading: {filename}"
        )
    if total_bytes != final_status.st_size:
        raise ValidationEvidenceError(
            f"validation result size changed while reading: {filename}"
        )
    try:
        final_path_status = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValidationEvidenceError(
            f"validation result changed while reading: {filename}"
        ) from exc
    if not stat.S_ISREG(final_path_status.st_mode) or (
        final_path_status.st_dev,
        final_path_status.st_ino,
    ) != (final_status.st_dev, final_status.st_ino):
        raise ValidationEvidenceError(
            f"validation result changed while reading: {filename}"
        )
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeError as exc:
        raise ValidationEvidenceError(f"could not read {filename}: {exc}") from exc


def load_and_validate_result_file(
    path: Path,
    expected_models: list[str],
    *,
    capability_profile: str = "",
    capability_models: list[str] | None = None,
) -> list[dict]:
    filename = path.name
    if (
        not expected_models
        or not all(isinstance(model, str) and model for model in expected_models)
        or len({_canonical_model_id(model) for model in expected_models})
        != len(expected_models)
    ):
        raise ValidationEvidenceError(
            f"{filename}: expected models must be unique nonempty model identifiers"
        )
    expected_capability_models = capability_models or []
    if bool(capability_profile) != bool(expected_capability_models):
        raise ValidationEvidenceError(
            f"{filename}: capability profile and models must be paired"
        )
    canonical_expected_models = [
        _canonical_model_id(model) for model in expected_models
    ]
    canonical_capability_models = {
        _canonical_model_id(model) for model in expected_capability_models
    }
    if len(canonical_capability_models) != len(expected_capability_models):
        raise ValidationEvidenceError(f"{filename}: capability models must be unique")
    if not canonical_capability_models.issubset(canonical_expected_models):
        raise ValidationEvidenceError(f"{filename}: capability models must be selected")
    raw = _read_bounded_result_file(path)
    if not raw.strip():
        raise ValidationEvidenceError(f"validation result is empty: {filename}")
    try:
        records = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
        )
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
    canonical_models = [_canonical_model_id(model) for model in models]
    if len(canonical_models) != len(set(canonical_models)):
        raise ValidationEvidenceError(
            f"validation result contains duplicate models: {filename}"
        )
    if canonical_models != canonical_expected_models:
        raise ValidationEvidenceError(
            f"{filename}: result models do not match planned models"
        )
    for record, model_id in zip(checked, canonical_models):
        if model_id not in canonical_capability_models:
            if "capability_matrix" in record:
                raise ValidationEvidenceError(
                    f"{filename}: {record['model']} has an unexpected capability matrix"
                )
            continue
        if "capability_matrix" not in record:
            raise ValidationEvidenceError(
                f"{filename}: {record['model']} is missing capability evidence"
            )
        try:
            validate_capability_matrix_evidence(
                record["capability_matrix"],
                capability_profile,
                record["model"],
            )
        except CapabilityValidationError as exc:
            raise ValidationEvidenceError(
                f"{filename}: {record['model']} capability evidence is invalid: {exc}"
            ) from exc
    return checked


def _expected_models_by_file(expected_matrix: object) -> dict[str, dict]:
    if not isinstance(expected_matrix, dict):
        raise ValidationEvidenceError("validation matrix must be an object")
    try:
        require_complete_promotion_coverage(expected_matrix)
    except ValueError as exc:
        raise ValidationEvidenceError(
            f"invalid scheduled validation lanes: {exc}"
        ) from exc
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
        target = row.get("target")
        if not isinstance(backend, str) or not isinstance(channel, str):
            raise ValidationEvidenceError(
                f"validation matrix row {index} backend and channel must be strings"
            )
        if not isinstance(target, str) or not re.fullmatch(r"[a-z0-9-]+", target):
            raise ValidationEvidenceError(
                f"validation matrix row {index} target must be a safe nonempty label"
            )
        filename = f"llamacpp_validation_{target}.json"
        if filename in expected:
            raise ValidationEvidenceError(f"duplicate validation lane: {target}")
        selected_models = row.get("models")
        if (
            not isinstance(selected_models, list)
            or not all(isinstance(model, str) and model for model in selected_models)
            or len(selected_models) != len(set(selected_models))
        ):
            raise ValidationEvidenceError(
                f"validation matrix lane {target} needs unique selected models"
            )
        models = row.get("expected_models")
        if (
            not isinstance(models, list)
            or not models
            or not all(isinstance(model, str) and model for model in models)
            or len(models) != len(set(models))
        ):
            raise ValidationEvidenceError(
                f"validation matrix lane {target} needs unique expected models"
            )
        if selected_models != models:
            raise ValidationEvidenceError(
                f"validation matrix lane {target} selected and expected models differ"
            )
        if not any(_canonical_model_id(model) == K2_SMALL for model in models):
            raise ValidationEvidenceError(
                f"validation matrix lane {target} must include {K2_SMALL}"
            )
        has_capability_profile = "capability_profile" in row
        has_capability_models = "capability_models" in row
        if has_capability_profile != has_capability_models:
            raise ValidationEvidenceError(
                f"validation matrix lane {target} must pair capability profile and models"
            )
        if not has_capability_profile:
            raise ValidationEvidenceError(
                f"scheduled validation lane {target} needs capability evidence"
            )
        capability_profile = row["capability_profile"]
        capability_models = row["capability_models"]
        if capability_profile != K2_HORIZON_PROFILE:
            raise ValidationEvidenceError(
                f"validation matrix lane {target} has unsupported capability profile"
            )
        if (
            not isinstance(capability_models, list)
            or not capability_models
            or not all(isinstance(model, str) and model for model in capability_models)
            or len(capability_models) != len(set(capability_models))
        ):
            raise ValidationEvidenceError(
                f"validation matrix lane {target} needs unique capability models"
            )
        canonical_models = {_canonical_model_id(model) for model in models}
        if any(
            _canonical_model_id(model) not in canonical_models
            for model in capability_models
        ):
            raise ValidationEvidenceError(
                f"validation matrix lane {target} has an unselected capability model"
            )
        if K2_SMALL not in capability_models:
            raise ValidationEvidenceError(
                f"validation matrix lane {target} must validate {K2_SMALL} capabilities"
            )
        if any(
            _canonical_model_id(model) in {K2_MEDIUM, K2_LARGE} for model in models
        ) and (target != "windows-vulkan" or "128gb" not in row["runner"]):
            raise ValidationEvidenceError(
                f"validation matrix lane {target} needs the 128gb Vulkan runner "
                "for K2-Horizon-3.7B/7B"
            )
        expected[filename] = {
            "models": models,
            "capability_profile": capability_profile,
            "capability_models": capability_models,
            "restart_model": capability_models[0],
        }

    if set(expected) != set(EXPECTED_RESULT_FILES):
        raise ValidationEvidenceError(
            "validation matrix must contain exactly the scheduled validation lanes"
        )
    return expected


def load_and_validate_results(
    root: Path,
    expected_matrix: object,
) -> dict[str, list[dict]]:
    expected_lanes = _expected_models_by_file(expected_matrix)
    validated = {}
    for filename in EXPECTED_RESULT_FILES:
        expected_lane = expected_lanes[filename]
        validated[filename] = load_and_validate_result_file(
            root / filename,
            expected_lane["models"],
            capability_profile=expected_lane["capability_profile"],
            capability_models=expected_lane["capability_models"],
        )
        restart_filename = filename.replace(
            "llamacpp_validation_", "llamacpp_restart_validation_", 1
        )
        validated[restart_filename] = load_and_validate_result_file(
            root / restart_filename,
            [expected_lane["restart_model"]],
        )
    return validated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path.cwd())
    parser.add_argument("--expected-matrix-json", required=True)
    args = parser.parse_args()
    try:
        expected_matrix = json.loads(
            args.expected_matrix_json,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_float,
        )
        results = load_and_validate_results(args.directory, expected_matrix)
    except (json.JSONDecodeError, ValidationEvidenceError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    for filename, records in results.items():
        print(f"{filename}: {len(records)} passing result(s)")


if __name__ == "__main__":
    main()
