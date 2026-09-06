"""Managed llama.cpp release-asset requirements used by CI."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping

_RELEASE_RE = re.compile(r"^b[0-9]+$")
_CUDA_COMPUTE_CAPABILITIES = (
    "sm_75",
    "sm_80",
    "sm_86",
    "sm_89",
    "sm_90",
    "sm_100",
    "sm_120",
    "sm_121",
)
_ROCM_NIGHTLY_TARGETS_BY_PLATFORM = {
    "windows": (
        "gfx1151",
        "gfx1150",
        "gfx120X",
        "gfx110X",
        "gfx103X",
        "gfx90a",
        "gfx908",
        "gfx1152",
    ),
    "ubuntu": (
        "gfx1151",
        "gfx1150",
        "gfx120X",
        "gfx110X",
        "gfx103X",
        "gfx90a",
        "gfx908",
        "gfx1152",
        "gfx942",
    ),
}


def _validated_release(value: str, name: str) -> str:
    release = value.strip()
    if not _RELEASE_RE.fullmatch(release):
        raise ValueError(f"Invalid {name}: {release!r}")
    return release


def _therock_major_minor(backend_versions: Mapping[str, object]) -> str:
    therock = backend_versions.get("therock", {})
    if not isinstance(therock, Mapping):
        raise ValueError("backend_versions.json therock must be an object")
    version = str(therock.get("version", "")).strip().removeprefix("v")
    major_minor = ".".join(version.split(".")[:2])
    if not major_minor:
        raise ValueError("backend_versions.json is missing therock.version")
    return major_minor


def _mapped_rocm_targets(
    targets: Iterable[str], asset_families: Mapping[str, object]
) -> list[str]:
    mapped = []
    for target in targets:
        family = asset_families.get(target, target)
        if not isinstance(family, str) or not family:
            raise ValueError(f"Invalid ROCm asset family for {target}: {family!r}")
        if family not in mapped:
            mapped.append(family)
    return mapped


def build_asset_requirements(
    *,
    ggml_release: str,
    rocm_release: str,
    lemonade_release: str,
    backend_versions: Mapping[str, object],
) -> dict[str, dict[str, list[str]]]:
    """Return required asset names grouped by publisher and Lemonade backend."""
    ggml_release = _validated_release(ggml_release, "GGML_RELEASE")
    rocm_release = _validated_release(rocm_release, "ROCM_RELEASE")
    lemonade_release = _validated_release(lemonade_release, "LEMONADE_RELEASE")
    therock = _therock_major_minor(backend_versions)

    asset_families = backend_versions.get("rocm_asset_families", {})
    if not isinstance(asset_families, Mapping):
        raise ValueError("backend_versions.json rocm_asset_families must be an object")

    rocm_nightly = []
    for platform, targets in _ROCM_NIGHTLY_TARGETS_BY_PLATFORM.items():
        for target in _mapped_rocm_targets(targets, asset_families):
            rocm_nightly.append(
                f"llama-{rocm_release}-{platform}-rocm-{target}-x64.zip"
            )

    cuda = []
    for compute_capability in _CUDA_COMPUTE_CAPABILITIES:
        cuda.extend(
            (
                f"llama-{lemonade_release}-windows-cuda-{compute_capability}-x64.7z",
                f"llama-{lemonade_release}-ubuntu-cuda-{compute_capability}-x64.tar.xz",
                f"llama-{lemonade_release}-ubuntu-cuda-{compute_capability}-arm64.tar.xz",
            )
        )

    return {
        "ggml": {
            "vulkan": [
                f"llama-{ggml_release}-bin-win-vulkan-x64.zip",
                f"llama-{ggml_release}-bin-ubuntu-vulkan-x64.tar.gz",
                f"llama-{ggml_release}-bin-ubuntu-vulkan-arm64.tar.gz",
            ],
            "cpu": [
                f"llama-{ggml_release}-bin-win-cpu-x64.zip",
                f"llama-{ggml_release}-bin-ubuntu-x64.tar.gz",
                f"llama-{ggml_release}-bin-ubuntu-arm64.tar.gz",
            ],
            "metal": [
                f"llama-{ggml_release}-bin-macos-arm64.tar.gz",
            ],
        },
        "rocm": {"rocm-nightly": rocm_nightly},
        "lemonade": {
            "rocm-stable": [
                f"llama-{lemonade_release}-bin-win-rocm-{therock}-x64.zip",
                f"llama-{lemonade_release}-bin-ubuntu-rocm-{therock}-x64.tar.gz",
            ],
            "cuda": cuda,
        },
    }


def evaluate_asset_group(
    available_assets: Collection[str],
    backend_requirements: Mapping[str, Iterable[str]],
) -> tuple[list[str], dict[str, list[str]]]:
    """Return eligible backends and their missing required assets."""
    eligible = []
    missing_by_backend = {}
    for backend, required_assets in backend_requirements.items():
        missing = [asset for asset in required_assets if asset not in available_assets]
        missing_by_backend[backend] = missing
        if not missing:
            eligible.append(backend)
    return eligible, missing_by_backend
