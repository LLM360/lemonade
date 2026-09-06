"""Build the llama.cpp validation matrix for GitHub Actions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

K2_SMALL = "K2-Horizon-0.9B-GGUF"
K2_MEDIUM = "K2-Horizon-3.7B-GGUF"
K2_LARGE = "K2-Horizon-7B-GGUF"
SUPPORTED_EVENTS = {"merge_group", "pull_request", "schedule", "workflow_dispatch"}
MODEL_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "cpp"
    / "resources"
    / "server_models.json"
)

LARGE_VULKAN_RUNNER = [
    "self-hosted",
    "Windows",
    "128gb",
    "vulkan",
    "lemon-prod",
]
LARGE_ROCM_RUNNER = [
    "self-hosted",
    "Windows",
    "128gb",
    "rocm",
    "lemon-prod",
]
SMALL_VULKAN_RUNNER = [
    "self-hosted",
    "Windows",
    "stx-halo",
    "vulkan",
    "lemon-prod",
]
SMALL_ROCM_RUNNER = [
    "self-hosted",
    "Windows",
    "stx-halo",
    "rocm",
    "lemon-prod",
]


def _parse_models(models_csv: str) -> list[str]:
    return [model.strip() for model in models_csv.split(",") if model.strip()]


def load_hot_llamacpp_model_ids(path: Path = MODEL_REGISTRY) -> list[str]:
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model registry {path}: {exc}") from exc
    if not isinstance(registry, dict):
        raise ValueError(f"Model registry {path} must be an object")
    models = sorted(
        model_id
        for model_id, model in registry.items()
        if isinstance(model, dict)
        and model.get("recipe") == "llamacpp"
        and isinstance(model.get("labels"), list)
        and "hot" in model["labels"]
    )
    if not models:
        raise ValueError(f"Model registry {path} has no hot llama.cpp models")
    return models


def _row(
    backend: str,
    channel: str,
    runner: list[str],
    models: list[str],
    lite: bool,
    expected_models: list[str] | None = None,
) -> dict[str, object]:
    row = {
        "backend": backend,
        "channel": channel,
        "runner": list(runner),
        "models": list(models),
        "lite": lite,
    }
    if expected_models is not None:
        row["expected_models"] = list(expected_models)
    return row


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
                    "vulkan",
                    "",
                    LARGE_VULKAN_RUNNER,
                    [K2_SMALL, K2_MEDIUM, K2_LARGE],
                    False,
                ),
                _row("rocm", "stable", SMALL_ROCM_RUNNER, [K2_SMALL], False),
                _row("rocm", "nightly", SMALL_ROCM_RUNNER, [K2_SMALL], False),
            ]
        }

    runner_vulkan = SMALL_VULKAN_RUNNER if lite else LARGE_VULKAN_RUNNER
    runner_rocm = SMALL_ROCM_RUNNER if lite else LARGE_ROCM_RUNNER
    expected_models = (
        load_hot_llamacpp_model_ids() if event_name == "schedule" else None
    )
    return {
        "include": [
            _row("vulkan", "", runner_vulkan, models, lite, expected_models),
            _row(
                "rocm",
                "stable",
                runner_rocm,
                models,
                lite,
                expected_models,
            ),
            _row(
                "rocm",
                "nightly",
                runner_rocm,
                models,
                lite,
                expected_models,
            ),
        ]
    }


def is_promotion_eligible(
    event_name: str,
    models_csv: str = "",
    lite: bool = False,
) -> bool:
    create_validation_plan(event_name, models_csv, lite)
    return event_name == "schedule"


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
