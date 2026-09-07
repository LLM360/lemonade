#!/usr/bin/env python3
"""Merge validation-only model records into a disposable Lemonade catalog."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


class ModelCatalogOverlayError(ValueError):
    """Raised when a validation catalog overlay cannot be merged safely."""


def _reject_duplicate_members(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ModelCatalogOverlayError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _reject_non_finite_number(value: str):
    raise ModelCatalogOverlayError(f"non-finite JSON number: {value}")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_non_finite_number(value)
    return parsed


def _json_values_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_values_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    return left == right


def load_model_catalog(path: Path | str) -> dict:
    path = Path(path)
    try:
        catalog = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_members,
            parse_constant=_reject_non_finite_number,
            parse_float=_parse_finite_float,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelCatalogOverlayError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(catalog, dict):
        raise ModelCatalogOverlayError(f"{path}: catalog must be a JSON object")
    for model_id, model in catalog.items():
        if not isinstance(model, dict):
            raise ModelCatalogOverlayError(
                f"{path}: model {model_id} must be a JSON object"
            )
    return catalog


def merge_model_catalogs(
    base_path: Path | str,
    overlay_path: Path | str,
) -> dict:
    base_path = Path(base_path)
    overlay_path = Path(overlay_path)
    merged = load_model_catalog(base_path)
    overlay = load_model_catalog(overlay_path)

    for model_id, candidate in overlay.items():
        if model_id in merged and not _json_values_equal(merged[model_id], candidate):
            raise ModelCatalogOverlayError(f"conflicting model definition: {model_id}")
        merged[model_id] = candidate
    return merged


def merge_model_catalog_files(
    base_path: Path | str,
    overlay_path: Path | str,
    output_path: Path | str,
) -> None:
    output_path = Path(output_path)
    merged = merge_model_catalogs(base_path, overlay_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=output_path.parent,
            encoding="utf-8",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(merged, temporary, indent=4, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a validation-only model overlay into a catalog"
    )
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    merge_model_catalog_files(args.base, args.overlay, args.output)


if __name__ == "__main__":
    main()
