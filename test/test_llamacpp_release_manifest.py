#!/usr/bin/env python3
"""Tests for immutable llama.cpp release-asset manifests."""

from __future__ import annotations

import copy
import json
import unittest

from test.utils import llamacpp_release_manifest as manifest


def release_payload(tag: str, release_id: int, asset_seed: int) -> dict:
    return {
        "id": release_id,
        "tag_name": tag,
        "source_commit": f"{release_id:040x}",
        "assets": [
            {
                "name": "z-last.zip",
                "id": asset_seed + 1,
                "size": 200,
                "digest": "sha256:" + "B" * 64,
                "updated_at": "2026-09-05T12:34:56Z",
            },
            {
                "name": "a-first.zip",
                "id": asset_seed,
                "size": 100,
                "digest": "sha256:" + "a" * 64,
                "updated_at": "2026-09-05T12:30:00Z",
            },
        ],
    }


class LlamaCppReleaseManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ggml = release_payload("b1234", 10, 100)
        self.rocm = release_payload("b1235", 20, 200)

    def test_manifest_is_canonical_across_release_and_asset_order(self) -> None:
        first = manifest.build_release_asset_manifest(
            [
                ("lemonade-sdk/llamacpp-rocm", "b1235", self.rocm),
                ("ggml-org/llama.cpp", "b1234", self.ggml),
            ]
        )
        reversed_assets = copy.deepcopy(self.ggml)
        reversed_assets["assets"].reverse()
        second = manifest.build_release_asset_manifest(
            [
                ("ggml-org/llama.cpp", "b1234", reversed_assets),
                ("lemonade-sdk/llamacpp-rocm", "b1235", self.rocm),
            ]
        )

        self.assertEqual(first, second)
        parsed = json.loads(first)
        self.assertEqual(parsed["schema_version"], 2)
        self.assertEqual(
            [release["repository"] for release in parsed["releases"]],
            ["ggml-org/llama.cpp", "lemonade-sdk/llamacpp-rocm"],
        )
        self.assertEqual(
            [asset["name"] for asset in parsed["releases"][0]["assets"]],
            ["a-first.zip", "z-last.zip"],
        )
        self.assertEqual(
            parsed["releases"][0]["assets"][1]["digest"],
            "sha256:" + "b" * 64,
        )

    def test_every_asset_requires_a_sha256_digest(self) -> None:
        for digest in (None, "", "md5:" + "a" * 32, "sha256:abc"):
            with self.subTest(digest=digest):
                payload = copy.deepcopy(self.ggml)
                payload["assets"][0]["digest"] = digest

                with self.assertRaisesRegex(
                    manifest.ReleaseManifestError,
                    "sha256 digest",
                ):
                    manifest.build_release_asset_manifest(
                        [("ggml-org/llama.cpp", "b1234", payload)]
                    )

    def test_release_and_asset_identity_fields_are_required(self) -> None:
        cases = (
            ("release id", lambda payload: payload.pop("id")),
            ("source commit", lambda payload: payload.pop("source_commit")),
            ("asset name", lambda payload: payload["assets"][0].pop("name")),
            ("asset id", lambda payload: payload["assets"][0].pop("id")),
            ("asset size", lambda payload: payload["assets"][0].pop("size")),
            (
                "asset updated_at",
                lambda payload: payload["assets"][0].pop("updated_at"),
            ),
        )
        for expected_error, mutate in cases:
            with self.subTest(expected_error=expected_error):
                payload = copy.deepcopy(self.ggml)
                mutate(payload)

                with self.assertRaisesRegex(
                    manifest.ReleaseManifestError,
                    expected_error,
                ):
                    manifest.build_release_asset_manifest(
                        [("ggml-org/llama.cpp", "b1234", payload)]
                    )

    def test_source_commit_must_be_a_full_git_oid(self) -> None:
        for source_commit in (None, "", "abc123", "g" * 40):
            with self.subTest(source_commit=source_commit):
                payload = copy.deepcopy(self.ggml)
                payload["source_commit"] = source_commit

                with self.assertRaisesRegex(
                    manifest.ReleaseManifestError,
                    "source commit",
                ):
                    manifest.build_release_asset_manifest(
                        [("ggml-org/llama.cpp", "b1234", payload)]
                    )

    def test_tag_mismatch_and_duplicate_assets_fail_closed(self) -> None:
        with self.assertRaisesRegex(manifest.ReleaseManifestError, "tag mismatch"):
            manifest.build_release_asset_manifest(
                [("ggml-org/llama.cpp", "b9999", self.ggml)]
            )

        payload = copy.deepcopy(self.ggml)
        payload["assets"][1]["id"] = payload["assets"][0]["id"]
        with self.assertRaisesRegex(
            manifest.ReleaseManifestError, "duplicate asset id"
        ):
            manifest.build_release_asset_manifest(
                [("ggml-org/llama.cpp", "b1234", payload)]
            )

    def test_comparison_rejects_any_asset_metadata_change(self) -> None:
        expected = manifest.build_release_asset_manifest(
            [("ggml-org/llama.cpp", "b1234", self.ggml)]
        )
        changed = copy.deepcopy(self.ggml)
        changed["assets"][0]["updated_at"] = "2026-09-05T12:35:00Z"
        actual = manifest.build_release_asset_manifest(
            [("ggml-org/llama.cpp", "b1234", changed)]
        )

        with self.assertRaisesRegex(
            manifest.ReleaseManifestError,
            "changed since validation",
        ):
            manifest.require_exact_manifest_match(expected, actual)

        manifest.require_exact_manifest_match(expected, expected)


if __name__ == "__main__":
    unittest.main()
