"""Build the llama.cpp validation matrix for GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if __package__:
    from .validation_model_catalog import (
        load_model_catalog,
        merge_model_catalogs,
    )
else:
    from validation_model_catalog import (  # type: ignore[import-not-found]
        load_model_catalog,
        merge_model_catalogs,
    )

K2_SMALL = "K2-Horizon-0.9B-GGUF"
K2_MEDIUM = "K2-Horizon-3.7B-GGUF"
K2_LARGE = "K2-Horizon-7B-GGUF"
SUPPORTED_EVENTS = {"merge_group", "pull_request", "schedule", "workflow_dispatch"}
K2_CAPABILITY_PROFILE = "k2-horizon-v1"
MANAGED_PINS = frozenset(
    {"cpu", "cuda", "metal", "rocm-nightly", "rocm-stable", "vulkan"}
)
MODEL_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "cpp"
    / "resources"
    / "server_models.json"
)
VALIDATION_MODEL_OVERLAY = (
    Path(__file__).resolve().parents[1] / "fixtures" / "llamacpp_validation_models.json"
)

LARGE_VULKAN_RUNNER = [
    "self-hosted",
    "Windows",
    "128gb",
    "vulkan",
    "lemon-prod",
]
LINUX_VULKAN_RUNNER = [
    "self-hosted",
    "Linux",
    "vulkan",
    "lemon-prod",
]
LINUX_CPU_RUNNER = ["self-hosted", "Linux", "lemon-prod"]
MACOS_METAL_RUNNER = ["macos-latest"]
WINDOWS_CUDA_RUNNER = [
    "self-hosted",
    "Windows",
    "cuda",
    "lemon-prod",
]
WINDOWS_ROCM_RUNNER = [
    "self-hosted",
    "Windows",
    "stx-halo",
    "rocm",
    "lemon-prod",
]

VALIDATION_LANES = (
    {
        "target": "windows-vulkan",
        "backend": "vulkan",
        "channel": "",
        "managed_pin": "vulkan",
        "build_platform": "windows",
        "runner": LARGE_VULKAN_RUNNER,
    },
    {
        "target": "linux-vulkan",
        "backend": "vulkan",
        "channel": "",
        "managed_pin": "vulkan",
        "build_platform": "linux",
        "runner": LINUX_VULKAN_RUNNER,
    },
    {
        "target": "linux-cpu",
        "backend": "cpu",
        "channel": "",
        "managed_pin": "cpu",
        "build_platform": "linux",
        "runner": LINUX_CPU_RUNNER,
    },
    {
        "target": "macos-metal",
        "backend": "metal",
        "channel": "",
        "managed_pin": "metal",
        "build_platform": "macos",
        "runner": MACOS_METAL_RUNNER,
    },
    {
        "target": "windows-cuda",
        "backend": "cuda",
        "channel": "",
        "managed_pin": "cuda",
        "build_platform": "windows",
        "runner": WINDOWS_CUDA_RUNNER,
    },
    {
        "target": "windows-rocm-stable",
        "backend": "rocm",
        "channel": "stable",
        "managed_pin": "rocm-stable",
        "build_platform": "windows",
        "runner": WINDOWS_ROCM_RUNNER,
    },
    {
        "target": "windows-rocm-nightly",
        "backend": "rocm",
        "channel": "nightly",
        "managed_pin": "rocm-nightly",
        "build_platform": "windows",
        "runner": WINDOWS_ROCM_RUNNER,
    },
)


def _parse_models(models_csv: str) -> list[str]:
    return [model.strip() for model in models_csv.split(",") if model.strip()]


def _canonical_model_id(model_id: str) -> str:
    return model_id.removeprefix("builtin.")


def _hot_llamacpp_model_ids(registry: dict, source: Path) -> list[str]:
    models = sorted(
        model_id
        for model_id, model in registry.items()
        if isinstance(model, dict)
        and model.get("recipe") == "llamacpp"
        and isinstance(model.get("labels"), list)
        and "hot" in model["labels"]
    )
    if not models:
        raise ValueError(f"Model registry {source} has no hot llama.cpp models")
    return models


def load_hot_llamacpp_model_ids(path: Path = MODEL_REGISTRY) -> list[str]:
    return _hot_llamacpp_model_ids(load_model_catalog(path), path)


def load_validation_candidate_model_ids(
    path: Path = MODEL_REGISTRY,
    overlay_path: Path = VALIDATION_MODEL_OVERLAY,
) -> list[str]:
    product_hot_models = _hot_llamacpp_model_ids(load_model_catalog(path), path)
    overlay = load_model_catalog(overlay_path)
    merged = merge_model_catalogs(path, overlay_path)

    for model_id in overlay:
        model = merged[model_id]
        if model.get("recipe") != "llamacpp":
            raise ValueError(
                f"Validation model {model_id} must use the llamacpp recipe"
            )
        labels = model.get("labels")
        if not isinstance(labels, list) or "chat" not in labels:
            raise ValueError(f"Validation model {model_id} must support chat")

    return sorted(set(product_hot_models) | set(overlay))


def _row(
    lane: dict[str, object],
    models: list[str],
    expected_models: list[str] | None = None,
) -> dict[str, object]:
    row = {
        "target": lane["target"],
        "backend": lane["backend"],
        "channel": lane["channel"],
        "managed_pin": lane["managed_pin"],
        "build_platform": lane["build_platform"],
        "runner": list(lane["runner"]),
        "models": list(models),
        "lite": False,
        "capability_profile": K2_CAPABILITY_PROFILE,
        "capability_models": [K2_SMALL],
    }
    if expected_models is not None:
        row["expected_models"] = list(expected_models)
    return row


def _with_required_k2_small(models: list[str]) -> list[str]:
    if any(_canonical_model_id(model) == K2_SMALL for model in models):
        return list(models)
    return [K2_SMALL, *models]


def _small_lane_models(models: list[str]) -> list[str]:
    return [
        model
        for model in _with_required_k2_small(models)
        if _canonical_model_id(model) not in {K2_MEDIUM, K2_LARGE}
    ]


def require_complete_promotion_coverage(
    plan: dict[str, list[dict[str, object]]],
) -> None:
    rows = plan.get("include") if isinstance(plan, dict) else None
    if not isinstance(rows, list) or len(rows) != len(VALIDATION_LANES):
        raise ValueError("Promotion requires exactly the seven validation lanes")

    candidate_models = load_validation_candidate_model_ids()
    actual_targets = []
    for index, (row, lane) in enumerate(zip(rows, VALIDATION_LANES)):
        if not isinstance(row, dict):
            raise ValueError(f"Validation lane {index} must be an object")
        actual_targets.append(row.get("target"))
        for field in (
            "target",
            "backend",
            "channel",
            "managed_pin",
            "build_platform",
        ):
            if row.get(field) != lane[field]:
                raise ValueError(
                    f"Validation lane {index} has an invalid {field} mapping"
                )
        if row.get("runner") != lane["runner"]:
            raise ValueError(f"Validation lane {index} has an invalid runner mapping")
        if row.get("capability_profile") != K2_CAPABILITY_PROFILE:
            raise ValueError(f"Validation lane {index} lacks the K2 capability profile")
        if row.get("capability_models") != [K2_SMALL]:
            raise ValueError(
                f"Validation lane {index} lacks the K2 small capability model"
            )

        if row.get("lite") is not False:
            raise ValueError(f"Validation lane {index} cannot use lite mode")

        selected_models = row.get("models")
        expected_models = row.get("expected_models")
        for field_name, model_ids in (
            ("selected", selected_models),
            ("expected", expected_models),
        ):
            if (
                not isinstance(model_ids, list)
                or not model_ids
                or not all(
                    isinstance(model_id, str) and model_id for model_id in model_ids
                )
                or len(model_ids) != len(set(model_ids))
            ):
                raise ValueError(
                    f"Validation lane {index} {field_name} models must be "
                    "unique nonempty strings"
                )
        if selected_models != expected_models:
            raise ValueError(
                f"Validation lane {index} selected models do not match expected models"
            )

        planned_models = expected_models
        if not any(
            _canonical_model_id(model_id) == K2_SMALL for model_id in planned_models
        ):
            raise ValueError(f"Validation lane {index} does not include {K2_SMALL}")
        if any(
            _canonical_model_id(model_id) in {K2_MEDIUM, K2_LARGE}
            for model_id in planned_models
        ):
            if row["target"] != "windows-vulkan" or "128gb" not in row["runner"]:
                raise ValueError(
                    "K2-Horizon-3.7B/7B validation requires the 128gb Vulkan lane"
                )

        if row["target"] == "windows-vulkan":
            if planned_models != candidate_models:
                raise ValueError(
                    "The 128gb Vulkan lane must use the exact candidate model set"
                )
        elif planned_models != [K2_SMALL]:
            raise ValueError(
                f"Validation lane {index} must use the exact K2 small model set"
            )

    if actual_targets != [lane["target"] for lane in VALIDATION_LANES]:
        raise ValueError("Promotion validation lanes must use the exact target order")
    if {row["managed_pin"] for row in rows} != MANAGED_PINS:
        raise ValueError("Promotion validation does not cover every managed pin")


def create_validation_plan(
    event_name: str,
    models_csv: str = "",
    lite: bool = False,
) -> dict[str, list[dict[str, object]]]:
    if event_name not in SUPPORTED_EVENTS:
        raise ValueError(f"Unsupported workflow event: {event_name}")

    models = _parse_models(models_csv)
    if event_name != "workflow_dispatch" and (models or lite):
        raise ValueError("Manual model inputs are valid only for workflow_dispatch")
    if models and lite:
        raise ValueError("Explicit models and lite mode are mutually exclusive")

    if event_name in {"pull_request", "merge_group"}:
        return {
            "include": [
                _row(
                    lane,
                    (
                        [K2_SMALL, K2_MEDIUM, K2_LARGE]
                        if lane["target"] == "windows-vulkan"
                        else [K2_SMALL]
                    ),
                )
                for lane in VALIDATION_LANES
            ]
        }

    if event_name == "schedule":
        candidate_models = load_validation_candidate_model_ids()
        plan = {
            "include": [
                _row(
                    lane,
                    (
                        candidate_models
                        if lane["target"] == "windows-vulkan"
                        else [K2_SMALL]
                    ),
                    (
                        candidate_models
                        if lane["target"] == "windows-vulkan"
                        else [K2_SMALL]
                    ),
                )
                for lane in VALIDATION_LANES
            ]
        }
        require_complete_promotion_coverage(plan)
        return plan

    if lite:
        large_lane_models = [K2_SMALL]
        small_lane_models = [K2_SMALL]
    elif models:
        large_lane_models = _with_required_k2_small(models)
        small_lane_models = _small_lane_models(models)
    else:
        large_lane_models = _with_required_k2_small(load_hot_llamacpp_model_ids())
        small_lane_models = [K2_SMALL]
    return {
        "include": [
            _row(
                lane,
                (
                    large_lane_models
                    if lane["target"] == "windows-vulkan"
                    else small_lane_models
                ),
            )
            for lane in VALIDATION_LANES
        ]
    }


def is_promotion_eligible(
    event_name: str,
    models_csv: str = "",
    lite: bool = False,
) -> bool:
    plan = create_validation_plan(event_name, models_csv, lite)
    if event_name != "schedule":
        return False
    require_complete_promotion_coverage(plan)
    return True


def write_github_output(
    output_path: Path,
    plan: dict[str, list[dict[str, object]]],
    *,
    promotion_eligible: bool,
) -> None:
    matrix = json.dumps(plan, separators=(",", ":"))
    with output_path.open("a", encoding="utf-8") as output:
        output.write(f"matrix={matrix}\n")
        output.write(f"promotion_eligible={str(promotion_eligible).lower()}\n")


def _bool_argument(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized in {"", "false"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True)
    parser.add_argument("--models", default="")
    parser.add_argument("--lite", default=False, type=_bool_argument)
    args = parser.parse_args()

    output_value = os.environ.get("GITHUB_OUTPUT")
    if not output_value:
        parser.error("GITHUB_OUTPUT is required")

    plan = create_validation_plan(args.event, args.models, args.lite)
    promotion_eligible = is_promotion_eligible(args.event, args.models, args.lite)
    write_github_output(
        Path(output_value),
        plan,
        promotion_eligible=promotion_eligible,
    )
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
