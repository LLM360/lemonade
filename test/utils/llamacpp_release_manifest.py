#!/usr/bin/env python3
"""Capture and compare canonical GitHub release-asset manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
LOWER_SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^[0-9a-fA-F]{40}$")
RELEASE_TAG_RE = re.compile(r"^b[0-9]+$")
SOURCE_MANIFEST_ASSET_NAME = ".llamacpp-source.json"
SOURCE_MANIFEST_MAX_BYTES = 65_536
TRUSTED_SOURCE_REPOSITORY = "ggml-org/llama.cpp"
PUBLISHER_CLAIM_TYPES = frozenset({"immutable-source-manifest", "source-release-tag"})
FORK_RELEASE_REPOSITORIES = frozenset(
    {"lemonade-sdk/llama.cpp", "lemonade-sdk/llamacpp-rocm"}
)
MANAGED_PIN_REPOSITORIES = {
    "cpu": "ggml-org/llama.cpp",
    "cuda": "lemonade-sdk/llama.cpp",
    "metal": "ggml-org/llama.cpp",
    "rocm-nightly": "lemonade-sdk/llamacpp-rocm",
    "rocm-stable": "lemonade-sdk/llama.cpp",
    "vulkan": "ggml-org/llama.cpp",
}


class ReleaseManifestError(ValueError):
    """Raised when release metadata is incomplete, malformed, or changed."""


def _is_git_oid(value: object) -> bool:
    return isinstance(value, str) and GIT_OID_RE.fullmatch(value) is not None


def _reject_duplicate_json_members(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseManifestError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _strict_json_loads(value: str, label: str) -> object:
    try:
        return json.loads(value, object_pairs_hook=_reject_duplicate_json_members)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError(f"{label} is not valid JSON: {exc}") from exc


def require_upstream_ancestry(
    comparison: object,
    source_repository: str,
    publisher_claimed_source_commit: str,
    upstream_reference_head: str,
) -> None:
    expected_path = (
        f"/repos/{source_repository}/compare/"
        f"{publisher_claimed_source_commit}...{upstream_reference_head}"
    )
    if not isinstance(comparison, dict):
        raise ReleaseManifestError("publisher-claimed source commit is not reachable")
    base_commit = comparison.get("base_commit")
    merge_base_commit = comparison.get("merge_base_commit")
    comparison_url = comparison.get("url")
    if (
        not isinstance(source_repository, str)
        or source_repository.count("/") != 1
        or not _is_git_oid(publisher_claimed_source_commit)
        or not _is_git_oid(upstream_reference_head)
        or not isinstance(base_commit, dict)
        or not isinstance(merge_base_commit, dict)
        or not isinstance(comparison_url, str)
        or urlparse(comparison_url).path != expected_path
        or base_commit.get("sha", "").lower() != publisher_claimed_source_commit.lower()
        or merge_base_commit.get("sha", "").lower()
        != publisher_claimed_source_commit.lower()
        or comparison.get("status") not in {"ahead", "identical"}
    ):
        raise ReleaseManifestError("publisher-claimed source commit is not reachable")


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReleaseManifestError(f"{label} must be a positive integer")
    return value


def locate_immutable_source_manifest_asset(
    release_payload: object,
    release_repository: str,
) -> dict:
    """Return the single bounded source-manifest asset for a fork release."""
    if release_repository not in FORK_RELEASE_REPOSITORIES:
        raise ReleaseManifestError(
            f"{release_repository}: immutable source manifests are only valid for "
            "managed fork releases"
        )
    if not isinstance(release_payload, dict):
        raise ReleaseManifestError(
            f"{release_repository}: release payload must be an object"
        )
    if release_payload.get("immutable") is not True:
        raise ReleaseManifestError(
            f"{release_repository}: release immutable must be true"
        )
    if release_payload.get("draft") is not False:
        raise ReleaseManifestError(f"{release_repository}: release draft must be false")
    raw_assets = release_payload.get("assets")
    if not isinstance(raw_assets, list):
        raise ReleaseManifestError(
            f"{release_repository}: release assets must be an array"
        )
    matches = [
        asset
        for asset in raw_assets
        if isinstance(asset, dict) and asset.get("name") == SOURCE_MANIFEST_ASSET_NAME
    ]
    if len(matches) != 1:
        raise ReleaseManifestError(
            f"{release_repository}: release must contain exactly one "
            f"{SOURCE_MANIFEST_ASSET_NAME} asset"
        )
    asset = matches[0]
    _positive_integer(asset.get("id"), f"{release_repository}: source manifest id")
    size = _positive_integer(
        asset.get("size"), f"{release_repository}: source manifest size"
    )
    if size > SOURCE_MANIFEST_MAX_BYTES:
        raise ReleaseManifestError(
            f"{release_repository}: source manifest size exceeds "
            f"{SOURCE_MANIFEST_MAX_BYTES} bytes"
        )
    if asset.get("state") != "uploaded":
        raise ReleaseManifestError(
            f"{release_repository}: source manifest must be uploaded"
        )
    digest = asset.get("digest")
    if not isinstance(digest, str) or LOWER_SHA256_DIGEST_RE.fullmatch(digest) is None:
        raise ReleaseManifestError(
            f"{release_repository}: source manifest must have a lowercase sha256 digest"
        )
    _source_manifest_asset_bindings(release_payload, release_repository)
    return asset


def _source_manifest_asset_bindings(
    release_payload: object,
    release_repository: str,
) -> list[dict]:
    if not isinstance(release_payload, dict) or not isinstance(
        release_payload.get("assets"), list
    ):
        raise ReleaseManifestError(
            f"{release_repository}: release assets must be an array"
        )
    bindings = []
    names: set[str] = set()
    for index, asset in enumerate(release_payload["assets"]):
        prefix = f"{release_repository}: release asset {index}"
        if not isinstance(asset, dict):
            raise ReleaseManifestError(f"{prefix} must be an object")
        name = asset.get("name")
        if not isinstance(name, str) or not name:
            raise ReleaseManifestError(f"{prefix} name must be nonempty")
        if name in names:
            raise ReleaseManifestError(
                f"{release_repository}: duplicate release asset name {name!r}"
            )
        names.add(name)
        if asset.get("state") != "uploaded":
            raise ReleaseManifestError(f"{prefix} must be uploaded")
        size = asset.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ReleaseManifestError(f"{prefix} size must be nonnegative")
        digest = asset.get("digest")
        if (
            not isinstance(digest, str)
            or LOWER_SHA256_DIGEST_RE.fullmatch(digest) is None
        ):
            raise ReleaseManifestError(f"{prefix} must have a lowercase sha256 digest")
        if name != SOURCE_MANIFEST_ASSET_NAME:
            bindings.append({"digest": digest, "name": name, "size": size})
    if not bindings:
        raise ReleaseManifestError(
            f"{release_repository}: source manifest must bind at least one release asset"
        )
    return sorted(bindings, key=lambda asset: asset["name"])


def validate_immutable_source_manifest(
    evidence: bytes,
    release_payload: object,
    release_repository: str,
    release_tag: str,
    release_tag_commit: str,
) -> str:
    """Validate one immutable publisher claim and return its upstream source OID."""
    source_asset = locate_immutable_source_manifest_asset(
        release_payload,
        release_repository,
    )
    if not isinstance(evidence, bytes):
        raise ReleaseManifestError(
            f"{release_repository}: source manifest must be raw bytes"
        )
    if len(evidence) != source_asset["size"]:
        raise ReleaseManifestError(
            f"{release_repository}: source manifest size does not match release metadata"
        )
    measured_digest = "sha256:" + hashlib.sha256(evidence).hexdigest()
    if measured_digest != source_asset["digest"]:
        raise ReleaseManifestError(
            f"{release_repository}: source manifest sha256 digest does not match "
            "release metadata"
        )
    try:
        evidence_text = evidence.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseManifestError(
            f"{release_repository}: source manifest must be UTF-8 JSON"
        ) from exc
    document = _strict_json_loads(
        evidence_text,
        f"{release_repository}: source manifest",
    )
    if not isinstance(document, dict):
        raise ReleaseManifestError(
            f"{release_repository}: source manifest must be an object"
        )
    expected_keys = {
        "assets",
        "release_repository",
        "release_tag",
        "release_tag_commit",
        "schema_version",
        "source_commit",
        "source_repository",
    }
    if set(document) != expected_keys:
        raise ReleaseManifestError(
            f"{release_repository}: source manifest has unexpected fields"
        )
    if (
        isinstance(document["schema_version"], bool)
        or not isinstance(document["schema_version"], int)
        or document["schema_version"] != 1
    ):
        raise ReleaseManifestError(
            f"{release_repository}: source manifest schema_version must be 1"
        )
    if document["release_repository"] != release_repository:
        raise ReleaseManifestError(
            f"{release_repository}: source manifest release repository mismatch"
        )
    if (
        not isinstance(release_tag, str)
        or RELEASE_TAG_RE.fullmatch(release_tag) is None
        or document["release_tag"] != release_tag
    ):
        raise ReleaseManifestError(
            f"{release_repository}: source manifest release tag mismatch"
        )
    if not _is_git_oid(release_tag_commit) or (
        not isinstance(document["release_tag_commit"], str)
        or document["release_tag_commit"].lower() != release_tag_commit.lower()
    ):
        raise ReleaseManifestError(
            f"{release_repository}: source manifest release tag commit mismatch"
        )
    if document["source_repository"] != TRUSTED_SOURCE_REPOSITORY:
        raise ReleaseManifestError(
            f"{release_repository}: source manifest source repository must be "
            f"{TRUSTED_SOURCE_REPOSITORY}"
        )
    source_commit = document["source_commit"]
    if not _is_git_oid(source_commit):
        raise ReleaseManifestError(
            f"{release_repository}: source manifest source commit must be a full "
            "40-character Git OID"
        )
    raw_bindings = document["assets"]
    if not isinstance(raw_bindings, list):
        raise ReleaseManifestError(
            f"{release_repository}: source manifest assets must be an array"
        )
    bindings = []
    names: set[str] = set()
    for index, binding in enumerate(raw_bindings):
        prefix = f"{release_repository}: source manifest asset {index}"
        if not isinstance(binding, dict) or set(binding) != {"digest", "name", "size"}:
            raise ReleaseManifestError(f"{prefix} has unexpected fields")
        name = binding["name"]
        if not isinstance(name, str) or not name:
            raise ReleaseManifestError(f"{prefix} name must be nonempty")
        if name == SOURCE_MANIFEST_ASSET_NAME:
            raise ReleaseManifestError(
                f"{release_repository}: source manifest must exclude itself"
            )
        if name in names:
            raise ReleaseManifestError(
                f"{release_repository}: source manifest has duplicate asset {name!r}"
            )
        names.add(name)
        size = binding["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ReleaseManifestError(f"{prefix} size must be nonnegative")
        digest = binding["digest"]
        if (
            not isinstance(digest, str)
            or LOWER_SHA256_DIGEST_RE.fullmatch(digest) is None
        ):
            raise ReleaseManifestError(f"{prefix} must have a lowercase sha256 digest")
        bindings.append({"digest": digest, "name": name, "size": size})
    expected_bindings = _source_manifest_asset_bindings(
        release_payload,
        release_repository,
    )
    if sorted(bindings, key=lambda asset: asset["name"]) != expected_bindings:
        raise ReleaseManifestError(
            f"{release_repository}: source manifest asset metadata does not match "
            "the immutable release"
        )
    return source_commit.lower()


def _canonical_release(
    repository: str,
    expected_tag: str,
    payload: object,
) -> dict:
    if not isinstance(repository, str) or repository.count("/") != 1:
        raise ReleaseManifestError(f"invalid repository name: {repository!r}")
    if not isinstance(payload, dict):
        raise ReleaseManifestError(f"{repository}: release payload must be an object")

    actual_tag = payload.get("tag_name")
    if actual_tag != expected_tag:
        raise ReleaseManifestError(
            f"{repository}: release tag mismatch: expected {expected_tag!r}, "
            f"got {actual_tag!r}"
        )
    if (
        not isinstance(expected_tag, str)
        or RELEASE_TAG_RE.fullmatch(expected_tag) is None
    ):
        raise ReleaseManifestError(f"{repository}: release tag must match bNNNN")
    release_id = _positive_integer(payload.get("id"), f"{repository}: release id")
    release_immutable = payload.get("immutable")
    if release_immutable is not True:
        raise ReleaseManifestError(f"{repository}: release immutable must be true")
    release_draft = payload.get("draft")
    if release_draft is not False:
        raise ReleaseManifestError(f"{repository}: release draft must be false")
    release_tag_commit = payload.get("release_tag_commit")
    if not _is_git_oid(release_tag_commit):
        raise ReleaseManifestError(
            f"{repository}: release tag commit must be a full 40-character Git OID"
        )
    source_repository = payload.get("source_repository")
    if source_repository != TRUSTED_SOURCE_REPOSITORY:
        raise ReleaseManifestError(
            f"{repository}: source repository must be {TRUSTED_SOURCE_REPOSITORY}"
        )
    publisher_claim_type = payload.get("publisher_claim_type")
    if publisher_claim_type not in PUBLISHER_CLAIM_TYPES:
        raise ReleaseManifestError(
            f"{repository}: publisher claim type must identify its source"
        )
    publisher_claimed_source_commit = payload.get("publisher_claimed_source_commit")
    if not _is_git_oid(publisher_claimed_source_commit):
        raise ReleaseManifestError(
            f"{repository}: publisher-claimed source commit must be a full "
            "40-character Git OID"
        )
    upstream_reference = payload.get("upstream_reference")
    if not isinstance(upstream_reference, str) or not upstream_reference.startswith(
        "refs/heads/"
    ):
        raise ReleaseManifestError(
            f"{repository}: upstream reference must be a branch reference"
        )
    upstream_reference_head = payload.get("upstream_reference_head")
    if not _is_git_oid(upstream_reference_head):
        raise ReleaseManifestError(
            f"{repository}: upstream reference head must be a full 40-character Git OID"
        )
    if (
        repository == source_repository
        and release_tag_commit.lower() != publisher_claimed_source_commit.lower()
    ):
        raise ReleaseManifestError(
            f"{repository}: release tag commit must equal publisher-claimed source commit "
            "for a source release"
        )
    expected_claim_type = (
        "source-release-tag"
        if repository == source_repository
        else "immutable-source-manifest"
    )
    if publisher_claim_type != expected_claim_type:
        raise ReleaseManifestError(
            f"{repository}: publisher claim type must be {expected_claim_type}"
        )
    raw_assets = payload.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ReleaseManifestError(f"{repository}: release assets must be nonempty")

    asset_ids: set[int] = set()
    asset_names: set[str] = set()
    assets = []
    for index, raw_asset in enumerate(raw_assets):
        prefix = f"{repository}: asset {index}"
        if not isinstance(raw_asset, dict):
            raise ReleaseManifestError(f"{prefix} must be an object")

        name = raw_asset.get("name")
        if not isinstance(name, str) or not name:
            raise ReleaseManifestError(f"{prefix}: asset name must be nonempty")
        asset_id = _positive_integer(raw_asset.get("id"), f"{prefix}: asset id")
        size = raw_asset.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ReleaseManifestError(
                f"{prefix}: asset size must be a nonnegative integer"
            )
        digest = raw_asset.get("digest")
        if not isinstance(digest, str) or not SHA256_DIGEST_RE.fullmatch(digest):
            raise ReleaseManifestError(
                f"{prefix}: asset must have a complete sha256 digest"
            )
        updated_at = raw_asset.get("updated_at")
        if not isinstance(updated_at, str) or not updated_at:
            raise ReleaseManifestError(f"{prefix}: asset updated_at must be nonempty")
        state = raw_asset.get("state")
        if state != "uploaded":
            raise ReleaseManifestError(f"{prefix}: asset must be uploaded")

        if asset_id in asset_ids:
            raise ReleaseManifestError(f"{repository}: duplicate asset id {asset_id}")
        if name in asset_names:
            raise ReleaseManifestError(f"{repository}: duplicate asset name {name!r}")
        asset_ids.add(asset_id)
        asset_names.add(name)
        assets.append(
            {
                "digest": digest.lower(),
                "id": asset_id,
                "name": name,
                "size": size,
                "state": state,
                "updated_at": updated_at,
            }
        )

    assets.sort(key=lambda asset: (asset["name"], asset["id"]))
    if publisher_claim_type == "immutable-source-manifest":
        source_assets = [
            asset for asset in assets if asset["name"] == SOURCE_MANIFEST_ASSET_NAME
        ]
        if len(source_assets) != 1:
            raise ReleaseManifestError(
                f"{repository}: immutable source claim requires exactly one "
                f"{SOURCE_MANIFEST_ASSET_NAME} asset"
            )
        if (
            source_assets[0]["size"] <= 0
            or source_assets[0]["size"] > SOURCE_MANIFEST_MAX_BYTES
            or LOWER_SHA256_DIGEST_RE.fullmatch(source_assets[0]["digest"]) is None
        ):
            raise ReleaseManifestError(
                f"{repository}: immutable source manifest metadata is invalid"
            )
    return {
        "assets": assets,
        "publisher_claim_type": publisher_claim_type,
        "publisher_claimed_source_commit": publisher_claimed_source_commit.lower(),
        "release_draft": release_draft,
        "release_id": release_id,
        "release_immutable": release_immutable,
        "repository": repository,
        "release_tag_commit": release_tag_commit.lower(),
        "source_repository": source_repository,
        "tag_name": expected_tag,
        "upstream_reference": upstream_reference,
        "upstream_reference_head": upstream_reference_head.lower(),
    }


def build_release_asset_manifest(
    releases: Iterable[tuple[str, str, object]],
) -> str:
    canonical_releases = [
        _canonical_release(repository, expected_tag, payload)
        for repository, expected_tag, payload in releases
    ]
    if not canonical_releases:
        raise ReleaseManifestError("release manifest must contain at least one release")
    repositories = [release["repository"] for release in canonical_releases]
    if len(repositories) != len(set(repositories)):
        raise ReleaseManifestError("release manifest contains duplicate repositories")
    upstream_bindings = {
        (
            release["source_repository"],
            release["upstream_reference"],
            release["upstream_reference_head"],
        )
        for release in canonical_releases
    }
    if len(upstream_bindings) != 1:
        raise ReleaseManifestError(
            "release manifest must use one immutable upstream reference head"
        )
    canonical_releases.sort(key=lambda release: release["repository"])
    return json.dumps(
        {"releases": canonical_releases, "schema_version": 4},
        separators=(",", ":"),
        sort_keys=True,
    )


def canonicalize_release_asset_manifest(manifest_json: str) -> str:
    document = _strict_json_loads(manifest_json, "release manifest")
    if (
        not isinstance(document, dict)
        or isinstance(document.get("schema_version"), bool)
        or not isinstance(document.get("schema_version"), int)
        or document.get("schema_version") != 4
    ):
        raise ReleaseManifestError("release manifest schema_version must be 4")
    releases = document.get("releases")
    if not isinstance(releases, list):
        raise ReleaseManifestError("release manifest releases must be an array")

    inputs = []
    for release in releases:
        if not isinstance(release, dict):
            raise ReleaseManifestError("release manifest entry must be an object")
        expected_keys = {
            "assets",
            "publisher_claim_type",
            "publisher_claimed_source_commit",
            "release_draft",
            "release_id",
            "release_immutable",
            "repository",
            "release_tag_commit",
            "source_repository",
            "tag_name",
            "upstream_reference",
            "upstream_reference_head",
        }
        if set(release) != expected_keys:
            raise ReleaseManifestError("release manifest entry has unexpected fields")
        inputs.append(
            (
                release["repository"],
                release["tag_name"],
                {
                    "assets": release["assets"],
                    "draft": release["release_draft"],
                    "id": release["release_id"],
                    "immutable": release["release_immutable"],
                    "publisher_claim_type": release["publisher_claim_type"],
                    "publisher_claimed_source_commit": release[
                        "publisher_claimed_source_commit"
                    ],
                    "release_tag_commit": release["release_tag_commit"],
                    "source_repository": release["source_repository"],
                    "tag_name": release["tag_name"],
                    "upstream_reference": release["upstream_reference"],
                    "upstream_reference_head": release["upstream_reference_head"],
                },
            )
        )

    canonical = build_release_asset_manifest(inputs)
    if json.loads(canonical) != document:
        raise ReleaseManifestError("release manifest has unexpected fields")
    return canonical


def require_exact_manifest_match(expected_json: str, actual_json: str) -> None:
    expected = canonicalize_release_asset_manifest(expected_json)
    actual = canonicalize_release_asset_manifest(actual_json)
    if expected != actual:
        raise ReleaseManifestError(
            "release manifest changed since validation; refusing publication"
        )


def managed_llamacpp_pin_changes(
    base_versions: object,
    candidate_versions: object,
) -> dict[str, tuple[str, str]]:
    sections = []
    for label, document in (
        ("base", base_versions),
        ("candidate", candidate_versions),
    ):
        if not isinstance(document, dict) or not isinstance(
            document.get("llamacpp"), dict
        ):
            raise ReleaseManifestError(
                f"{label} backend versions must contain a llamacpp object"
            )
        section = document["llamacpp"]
        for backend in MANAGED_PIN_REPOSITORIES:
            value = section.get(backend)
            if not isinstance(value, str) or RELEASE_TAG_RE.fullmatch(value) is None:
                raise ReleaseManifestError(
                    f"{label} llamacpp.{backend} must be a bNNNN release tag"
                )
        sections.append(section)

    base_section, candidate_section = sections
    return {
        backend: (base_section[backend], candidate_section[backend])
        for backend in MANAGED_PIN_REPOSITORIES
        if base_section[backend] != candidate_section[backend]
    }


def require_release_manifest_for_managed_pin_changes(
    base_versions: object,
    candidate_versions: object,
    manifest_json: str | None,
) -> bool:
    changes = managed_llamacpp_pin_changes(base_versions, candidate_versions)
    if not changes:
        return False
    if manifest_json is None:
        raise ReleaseManifestError(
            "changed managed llama.cpp pins require a committed release manifest"
        )

    canonical = canonicalize_release_asset_manifest(manifest_json)
    releases = {
        release["repository"]: release for release in json.loads(canonical)["releases"]
    }
    expected_repositories = set(MANAGED_PIN_REPOSITORIES.values())
    if set(releases) != expected_repositories:
        raise ReleaseManifestError(
            "changed managed llama.cpp pins require all managed release repositories"
        )
    for backend, (_, candidate_tag) in changes.items():
        repository = MANAGED_PIN_REPOSITORIES[backend]
        if releases[repository]["tag_name"] != candidate_tag:
            raise ReleaseManifestError(
                f"release manifest does not match changed managed llama.cpp pin "
                f"llamacpp.{backend}={candidate_tag}"
            )
    return True


def materialize_release_asset_manifest(
    manifest_json: str,
    output_path: Path,
) -> tuple[str, str]:
    canonical = canonicalize_release_asset_manifest(manifest_json)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    document = json.loads(canonical)
    summary_lines = [
        "## Release manifest evidence",
        "",
        (
            "Only immutable GitHub releases are accepted. Fork source commits are "
            "bound to the release tag and binary assets by an immutable publisher "
            "manifest; this is not cryptographic build provenance."
        ),
        "",
        f"Canonical manifest SHA-256: `{digest}`",
        "",
        "| Publisher release | Immutable | Release tag commit | Publisher source claim | Upstream boundary |",
        "|---|---|---|---|---|",
    ]
    for release in document["releases"]:
        summary_lines.append(
            f"| `{release['repository']}@{release['tag_name']}` "
            f"| `{str(release['release_immutable']).lower()}` "
            f"| `{release['release_tag_commit']}` "
            f"| `{release['publisher_claimed_source_commit']}` "
            f"(`{release['publisher_claim_type']}`) "
            f"| `{release['source_repository']}@{release['upstream_reference_head']}` "
            f"(`{release['upstream_reference']}`) |"
        )
    try:
        output_path.write_text(canonical + "\n", encoding="utf-8")
    except OSError as exc:
        raise ReleaseManifestError(f"could not write {output_path}: {exc}") from exc
    return digest, "\n".join(summary_lines)


def _load_json_file(path: Path) -> object:
    try:
        return _strict_json_loads(path.read_text(encoding="utf-8"), str(path))
    except OSError as exc:
        raise ReleaseManifestError(f"could not read {path}: {exc}") from exc


def _load_release_specs(specs: list[list[str]]) -> list[tuple[str, str, object]]:
    releases = []
    for repository, expected_tag, path_text in specs:
        path = Path(path_text)
        releases.append((repository, expected_tag, _load_json_file(path)))
    return releases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        action="append",
        nargs=3,
        metavar=("REPOSITORY", "TAG", "JSON_PATH"),
    )
    parser.add_argument(
        "--source-manifest-asset-id",
        nargs=2,
        metavar=("RELEASE_JSON_PATH", "RELEASE_REPOSITORY"),
    )
    parser.add_argument(
        "--validate-source-manifest",
        nargs=5,
        metavar=(
            "SOURCE_MANIFEST_PATH",
            "RELEASE_JSON_PATH",
            "RELEASE_REPOSITORY",
            "RELEASE_TAG",
            "RELEASE_TAG_COMMIT",
        ),
    )
    parser.add_argument(
        "--require-upstream-ancestry",
        nargs=4,
        metavar=("JSON_PATH", "REPOSITORY", "CLAIMED_COMMIT", "UPSTREAM_HEAD"),
    )
    parser.add_argument("--expected-json")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--manifest-json")
    parser.add_argument("--materialize-manifest", type=Path)
    parser.add_argument("--append-summary-to", type=Path)
    parser.add_argument(
        "--managed-pin-changes",
        nargs=2,
        type=Path,
        metavar=("BASE_VERSIONS", "CANDIDATE_VERSIONS"),
    )
    parser.add_argument(
        "--require-managed-pin-manifest",
        nargs=3,
        type=Path,
        metavar=("BASE_VERSIONS", "CANDIDATE_VERSIONS", "MANIFEST"),
    )
    args = parser.parse_args()

    source_modes = sum(
        mode is not None
        for mode in (args.source_manifest_asset_id, args.validate_source_manifest)
    )
    if source_modes:
        if source_modes != 1 or any(
            (
                args.release,
                args.require_upstream_ancestry,
                args.expected_json is not None,
                args.github_output,
                args.manifest_json is not None,
                args.materialize_manifest,
                args.append_summary_to,
                args.managed_pin_changes,
                args.require_managed_pin_manifest,
            )
        ):
            parser.error("source-manifest modes cannot be combined with other modes")
        try:
            if args.source_manifest_asset_id is not None:
                release_path, release_repository = args.source_manifest_asset_id
                asset = locate_immutable_source_manifest_asset(
                    _load_json_file(Path(release_path)),
                    release_repository,
                )
                print(asset["id"])
            else:
                (
                    source_path,
                    release_path,
                    release_repository,
                    release_tag,
                    release_tag_commit,
                ) = args.validate_source_manifest
                print(
                    validate_immutable_source_manifest(
                        Path(source_path).read_bytes(),
                        _load_json_file(Path(release_path)),
                        release_repository,
                        release_tag,
                        release_tag_commit,
                    )
                )
        except (OSError, ReleaseManifestError) as exc:
            parser.exit(1, f"ERROR: {exc}\n")
        return

    if args.managed_pin_changes:
        if any(
            (
                args.release,
                args.require_upstream_ancestry,
                args.expected_json is not None,
                args.github_output,
                args.manifest_json is not None,
                args.materialize_manifest,
                args.append_summary_to,
                args.require_managed_pin_manifest,
            )
        ):
            parser.error("--managed-pin-changes cannot be combined with other modes")
        try:
            base_path, candidate_path = args.managed_pin_changes
            changes = managed_llamacpp_pin_changes(
                _load_json_file(base_path),
                _load_json_file(candidate_path),
            )
        except ReleaseManifestError as exc:
            parser.exit(1, f"ERROR: {exc}\n")
        print(
            json.dumps(
                {
                    backend: {"from": old, "to": new}
                    for backend, (old, new) in changes.items()
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return
    if args.require_managed_pin_manifest:
        if any(
            (
                args.release,
                args.require_upstream_ancestry,
                args.expected_json is not None,
                args.github_output,
                args.manifest_json is not None,
                args.materialize_manifest,
                args.append_summary_to,
            )
        ):
            parser.error(
                "--require-managed-pin-manifest cannot be combined with other modes"
            )
        base_path, candidate_path, manifest_path = args.require_managed_pin_manifest
        try:
            require_release_manifest_for_managed_pin_changes(
                _load_json_file(base_path),
                _load_json_file(candidate_path),
                manifest_path.read_text(encoding="utf-8"),
            )
        except (OSError, ReleaseManifestError) as exc:
            parser.exit(1, f"ERROR: {exc}\n")
        print("changed")
        return

    if args.require_upstream_ancestry:
        if (
            args.release
            or args.expected_json is not None
            or args.github_output
            or args.manifest_json is not None
            or args.materialize_manifest
            or args.append_summary_to
            or args.managed_pin_changes
            or args.require_managed_pin_manifest
        ):
            parser.error(
                "--require-upstream-ancestry cannot be combined with manifest options"
            )
        path_text, repository, claimed_commit, upstream_head = (
            args.require_upstream_ancestry
        )
        try:
            require_upstream_ancestry(
                _load_json_file(Path(path_text)),
                repository,
                claimed_commit,
                upstream_head,
            )
        except ReleaseManifestError as exc:
            parser.exit(1, f"ERROR: {exc}\n")
        return
    if args.materialize_manifest is not None:
        if (
            args.release
            or args.expected_json is not None
            or args.github_output
            or args.manifest_json is None
        ):
            parser.error(
                "--materialize-manifest requires only --manifest-json and optional "
                "--append-summary-to"
            )
        try:
            digest, summary = materialize_release_asset_manifest(
                args.manifest_json,
                args.materialize_manifest,
            )
            if args.append_summary_to is not None:
                with args.append_summary_to.open("a", encoding="utf-8") as output_file:
                    output_file.write(f"\n\n{summary}\n")
        except (OSError, ReleaseManifestError) as exc:
            parser.exit(1, f"ERROR: {exc}\n")
        print(digest)
        return
    if args.manifest_json is not None or args.append_summary_to is not None:
        parser.error("--manifest-json requires --materialize-manifest")
    if not args.release:
        parser.error("at least one --release is required")

    try:
        actual = build_release_asset_manifest(_load_release_specs(args.release))
        if args.expected_json is not None:
            require_exact_manifest_match(args.expected_json, actual)
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as output_file:
                output_file.write(f"release_asset_manifest={actual}\n")
    except ReleaseManifestError as exc:
        parser.exit(1, f"ERROR: {exc}\n")

    print(actual)


if __name__ == "__main__":
    main()
