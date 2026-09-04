"""Model selection shared by llama.cpp validation entry points."""

from __future__ import annotations

import argparse
import math
from typing import Any, Callable, Iterable

BuiltinModelResolver = Callable[[str], dict[str, Any] | None]


class ModelSelectionError(ValueError):
    """Raised when requested validation models cannot be selected safely."""


def add_model_selection_arguments(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--model",
        action="append",
        default=None,
        help="Built-in llama.cpp model to validate. May be repeated.",
    )
    selection.add_argument(
        "--lite",
        action="store_true",
        help="Lite mode: only test the smallest hot model",
    )


def _model_size(model: dict[str, Any]) -> float:
    size = model.get("size")
    if isinstance(size, (int, float)) and not isinstance(size, bool):
        return float(size)
    return math.inf


def _require_pullable_checkpoint(model: dict[str, Any]) -> None:
    checkpoint = model.get("checkpoint")
    if isinstance(checkpoint, str) and checkpoint.strip():
        return

    checkpoints = model.get("checkpoints")
    if isinstance(checkpoints, dict):
        main_checkpoint = checkpoints.get("main")
        if isinstance(main_checkpoint, str) and main_checkpoint.strip():
            return

    model_id = model.get("id", "<unknown>")
    raise ModelSelectionError(
        f"Model '{model_id}' cannot be pulled because its catalog checkpoint "
        "is missing or invalid"
    )


def select_llamacpp_models(
    catalog: Iterable[dict[str, Any]],
    requested_model_ids: list[str] | None = None,
    lite: bool = False,
    builtin_model_resolver: BuiltinModelResolver | None = None,
) -> list[dict[str, Any]]:
    models = list(catalog)

    if requested_model_ids:
        if builtin_model_resolver is None:
            raise ModelSelectionError(
                "Explicit model selection requires a built-in model resolver"
            )
        selected = []
        for requested_model_id in requested_model_ids:
            canonical_model_id = (
                requested_model_id
                if requested_model_id.startswith("builtin.")
                else f"builtin.{requested_model_id}"
            )
            model = builtin_model_resolver(canonical_model_id)
            if model is None:
                raise ModelSelectionError(
                    f"Model '{requested_model_id}' was not found in the built-in "
                    "model catalog"
                )
            model = dict(model)
            model["id"] = requested_model_id
            model["load_id"] = canonical_model_id
            recipe = model.get("recipe")
            if recipe != "llamacpp":
                raise ModelSelectionError(
                    f"Model '{requested_model_id}' uses recipe '{recipe}', "
                    "expected 'llamacpp'"
                )
            _require_pullable_checkpoint(model)
            selected.append(model)
        return selected

    selected = [
        model
        for model in models
        if model.get("recipe") == "llamacpp" and "hot" in model.get("labels", [])
    ]
    selected.sort(key=lambda model: model["id"])
    if not selected:
        raise ModelSelectionError("No hot llama.cpp models found in the model catalog")

    if lite:
        selected = [min(selected, key=_model_size)]

    for model in selected:
        _require_pullable_checkpoint(model)

    return selected
