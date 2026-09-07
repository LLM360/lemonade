"""Managed llama.cpp release-asset requirements used by CI."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping

_RELEASE_RE = re.compile(r"^b[0-9]+$")
_ROCM_CONCRETE_ISA_RE = re.compile(r"^gfx[0-9a-f]{3,4}$")
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
_ROCM_RDNA_NIGHTLY_ISAS = (
    "gfx1030",
    "gfx1031",
    "gfx1032",
    "gfx1033",
    "gfx1034",
    "gfx1035",
    "gfx1036",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1103",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1200",
    "gfx1201",
)
_ROCM_NIGHTLY_ISAS_BY_PLATFORM = {
    "windows": _ROCM_RDNA_NIGHTLY_ISAS,
    "ubuntu": (*_ROCM_RDNA_NIGHTLY_ISAS, "gfx908", "gfx90a", "gfx942"),
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


def _validate_rocm_asset_families(asset_families: Mapping[str, object]) -> None:
    for isa, family in asset_families.items():
        if isa == "comment":
            continue
        if not isinstance(isa, str) or _ROCM_CONCRETE_ISA_RE.fullmatch(isa) is None:
            raise ValueError(
                f"ROCm asset family key must be a concrete ROCm ISA: {isa!r}"
            )
        if not isinstance(family, str) or not family:
            raise ValueError(f"Invalid ROCm asset family for {isa}: {family!r}")


def _rocm_asset_families(
    backend_versions: Mapping[str, object],
) -> Mapping[str, object]:
    asset_families = backend_versions.get("rocm_asset_families", {})
    if not isinstance(asset_families, Mapping):
        raise ValueError("backend_versions.json rocm_asset_families must be an object")
    _validate_rocm_asset_families(asset_families)
    return asset_families


def build_rocm_asset_target_requirements(
    *,
    rocm_release: str,
    backend_versions: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    """Bind every required ROCm nightly asset to its concrete build targets."""
    release = _validated_release(rocm_release, "ROCM_RELEASE")
    asset_families = _rocm_asset_families(backend_versions)
    grouped_targets: dict[str, list[str]] = {}
    for platform, concrete_isas in _ROCM_NIGHTLY_ISAS_BY_PLATFORM.items():
        for concrete_isa in concrete_isas:
            family = asset_families.get(concrete_isa, concrete_isa)
            if not isinstance(family, str) or not family:
                raise ValueError(
                    f"Invalid ROCm asset family for {concrete_isa}: {family!r}"
                )
            asset_name = f"llama-{release}-{platform}-rocm-{family}-x64.zip"
            grouped_targets.setdefault(asset_name, []).append(concrete_isa)
    return {
        asset_name: tuple(concrete_isas)
        for asset_name, concrete_isas in grouped_targets.items()
    }


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

    rocm_nightly = list(
        build_rocm_asset_target_requirements(
            rocm_release=rocm_release,
            backend_versions=backend_versions,
        )
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
    available_assets: Collection[str] | Mapping[str, object],
    backend_requirements: Mapping[str, Iterable[str]],
    *,
    required_asset_targets: Mapping[str, Iterable[str]] | None = None,
    attested_asset_targets: Mapping[str, Iterable[str]] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """Return eligible backends and their missing required assets."""

    def has_required_targets(asset: str) -> bool:
        if required_asset_targets is None:
            return True
        required = required_asset_targets.get(asset)
        if required is None or isinstance(required, str):
            return False
        if attested_asset_targets is None:
            return False
        attested = attested_asset_targets.get(asset)
        if attested is None or isinstance(attested, str):
            return False
        return set(required).issubset(attested)

    def is_available(asset: str) -> bool:
        if not isinstance(available_assets, Mapping):
            has_asset = asset in available_assets
        else:
            size = available_assets.get(asset)
            has_asset = (
                isinstance(size, int) and not isinstance(size, bool) and size > 0
            )
        return has_asset and has_required_targets(asset)

    eligible = []
    missing_by_backend = {}
    for backend, required_assets in backend_requirements.items():
        missing = [asset for asset in required_assets if not is_available(asset)]
        missing_by_backend[backend] = missing
        if not missing:
            eligible.append(backend)
    return eligible, missing_by_backend
