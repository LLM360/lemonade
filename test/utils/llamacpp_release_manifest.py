#!/usr/bin/env python3
"""Capture and compare canonical GitHub release-asset manifests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


class ReleaseManifestError(ValueError):
    """Raised when release metadata is incomplete, malformed, or changed."""


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReleaseManifestError(f"{label} must be a positive integer")
    return value


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
    release_id = _positive_integer(payload.get("id"), f"{repository}: release id")
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
                "updated_at": updated_at,
            }
        )

    assets.sort(key=lambda asset: (asset["name"], asset["id"]))
    return {
        "assets": assets,
        "release_id": release_id,
        "repository": repository,
        "tag_name": expected_tag,
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
    canonical_releases.sort(key=lambda release: release["repository"])
    return json.dumps(
        {"releases": canonical_releases, "schema_version": 1},
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonicalize_manifest(manifest_json: str) -> str:
    try:
        document = json.loads(manifest_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError(
            f"release manifest is not valid JSON: {exc}"
        ) from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ReleaseManifestError("release manifest schema_version must be 1")
    releases = document.get("releases")
    if not isinstance(releases, list):
        raise ReleaseManifestError("release manifest releases must be an array")

    inputs = []
    for release in releases:
        if not isinstance(release, dict):
            raise ReleaseManifestError("release manifest entry must be an object")
        expected_keys = {"assets", "release_id", "repository", "tag_name"}
        if set(release) != expected_keys:
            raise ReleaseManifestError("release manifest entry has unexpected fields")
        inputs.append(
            (
                release["repository"],
                release["tag_name"],
                {
                    "assets": release["assets"],
                    "id": release["release_id"],
                    "tag_name": release["tag_name"],
                },
            )
        )

    canonical = build_release_asset_manifest(inputs)
    if json.loads(canonical) != document:
        raise ReleaseManifestError("release manifest has unexpected fields")
    return canonical


def require_exact_manifest_match(expected_json: str, actual_json: str) -> None:
    expected = _canonicalize_manifest(expected_json)
    actual = _canonicalize_manifest(actual_json)
    if expected != actual:
        raise ReleaseManifestError(
            "release assets changed since validation; refusing publication"
        )


def _load_release_specs(specs: list[list[str]]) -> list[tuple[str, str, object]]:
    releases = []
    for repository, expected_tag, path_text in specs:
        path = Path(path_text)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReleaseManifestError(f"could not read {path}: {exc}") from exc
        releases.append((repository, expected_tag, payload))
    return releases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        action="append",
        nargs=3,
        required=True,
        metavar=("REPOSITORY", "TAG", "JSON_PATH"),
    )
    parser.add_argument("--expected-json")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

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
