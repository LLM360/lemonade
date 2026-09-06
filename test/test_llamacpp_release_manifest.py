#!/usr/bin/env python3
"""Tests for canonical llama.cpp release-asset manifests."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from test.utils import llamacpp_release_manifest as manifest


def release_payload(
    tag: str,
    release_id: int,
    asset_seed: int,
    *,
    repository: str = "ggml-org/llama.cpp",
    publisher_claim_type: str = "source-release-tag",
    publisher_claimed_source_commit: str | None = None,
) -> dict:
    release_tag_commit = f"{release_id:040x}"
    claimed_commit = publisher_claimed_source_commit or release_tag_commit
    assets = [
        {
            "name": "z-last.zip",
            "id": asset_seed + 1,
            "size": 200,
            "digest": "sha256:" + "B" * 64,
            "state": "uploaded",
            "updated_at": "2026-09-05T12:34:56Z",
        },
        {
            "name": "a-first.zip",
            "id": asset_seed,
            "size": 100,
            "digest": "sha256:" + "a" * 64,
            "state": "uploaded",
            "updated_at": "2026-09-05T12:30:00Z",
        },
    ]
    if publisher_claim_type == "immutable-source-manifest":
        assets.append(
            {
                "name": ".llamacpp-source.json",
                "id": asset_seed + 2,
                "size": 512,
                "digest": "sha256:" + "c" * 64,
                "state": "uploaded",
                "updated_at": "2026-09-05T12:36:00Z",
            }
        )
    return {
        "body": "release notes",
        "draft": False,
        "id": release_id,
        "immutable": True,
        "publisher_claim_type": publisher_claim_type,
        "publisher_claimed_source_commit": claimed_commit,
        "release_tag_commit": release_tag_commit,
        "source_repository": repository,
        "tag_name": tag,
        "upstream_reference": "refs/heads/master",
        "upstream_reference_head": "d" * 40,
        "assets": assets,
    }


class LlamaCppReleaseManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ggml = release_payload("b1234", 10, 100)
        self.rocm = release_payload(
            "b1235",
            20,
            200,
            publisher_claim_type="immutable-source-manifest",
        )

    def _immutable_source_manifest_fixture(self) -> tuple[dict, bytes, str]:
        release = copy.deepcopy(self.ggml)
        release["draft"] = False
        release["tag_name"] = "b4321"
        release["release_tag_commit"] = "b" * 40
        for asset in release["assets"]:
            asset["state"] = "uploaded"
            asset["digest"] = asset["digest"].lower()
        source_commit = "c" * 40
        document = {
            "assets": sorted(
                (
                    {
                        "digest": asset["digest"].lower(),
                        "name": asset["name"],
                        "size": asset["size"],
                    }
                    for asset in release["assets"]
                ),
                key=lambda asset: asset["name"],
            ),
            "release_repository": "lemonade-sdk/llama.cpp",
            "release_tag": "b4321",
            "release_tag_commit": "b" * 40,
            "schema_version": 1,
            "source_commit": source_commit,
            "source_repository": "ggml-org/llama.cpp",
        }
        evidence = json.dumps(
            document,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        release["assets"].append(
            {
                "digest": "sha256:" + hashlib.sha256(evidence).hexdigest(),
                "id": 999,
                "name": ".llamacpp-source.json",
                "size": len(evidence),
                "state": "uploaded",
                "updated_at": "2026-09-05T12:40:00Z",
            }
        )
        return release, evidence, source_commit

    @staticmethod
    def _bind_source_evidence(release: dict, evidence: bytes) -> None:
        source_asset = next(
            asset
            for asset in release["assets"]
            if asset["name"] == manifest.SOURCE_MANIFEST_ASSET_NAME
        )
        source_asset["size"] = len(evidence)
        source_asset["digest"] = "sha256:" + hashlib.sha256(evidence).hexdigest()

    def test_immutable_source_manifest_binds_release_source_and_assets(self) -> None:
        release, evidence, source_commit = self._immutable_source_manifest_fixture()

        asset = manifest.locate_immutable_source_manifest_asset(
            release,
            "lemonade-sdk/llama.cpp",
        )
        self.assertEqual(asset["id"], 999)
        self.assertEqual(
            manifest.validate_immutable_source_manifest(
                evidence,
                release,
                "lemonade-sdk/llama.cpp",
                "b4321",
                "b" * 40,
            ),
            source_commit,
        )

    def test_immutable_source_manifest_rejects_unbound_or_ambiguous_input(self) -> None:
        release, evidence, _source_commit = self._immutable_source_manifest_fixture()
        document = json.loads(evidence)

        document_cases = (
            ("release repository", "release_repository", "attacker/llama.cpp"),
            ("release tag", "release_tag", "b9999"),
            ("release tag commit", "release_tag_commit", "d" * 40),
            ("source repository", "source_repository", "attacker/llama.cpp"),
            ("source commit", "source_commit", "abc123"),
            ("asset metadata", "assets", document["assets"][:-1]),
        )
        for expected_error, field, value in document_cases:
            with self.subTest(field=field):
                changed = copy.deepcopy(document)
                changed[field] = value
                changed_bytes = json.dumps(
                    changed,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                changed_release = copy.deepcopy(release)
                self._bind_source_evidence(changed_release, changed_bytes)

                with self.assertRaisesRegex(
                    manifest.ReleaseManifestError,
                    expected_error,
                ):
                    manifest.validate_immutable_source_manifest(
                        changed_bytes,
                        changed_release,
                        "lemonade-sdk/llama.cpp",
                        "b4321",
                        "b" * 40,
                    )

        duplicate_key_evidence = (
            evidence[:-1] + b',"source_commit":"' + b"d" * 40 + b'"}'
        )
        duplicate_release = copy.deepcopy(release)
        self._bind_source_evidence(duplicate_release, duplicate_key_evidence)
        with self.assertRaisesRegex(
            manifest.ReleaseManifestError,
            "duplicate JSON member",
        ):
            manifest.validate_immutable_source_manifest(
                duplicate_key_evidence,
                duplicate_release,
                "lemonade-sdk/llama.cpp",
                "b4321",
                "b" * 40,
            )

        for mutate, expected_error in (
            (lambda payload: payload["assets"].pop(), "exactly one"),
            (
                lambda payload: payload["assets"].append(
                    copy.deepcopy(payload["assets"][-1])
                ),
                "exactly one",
            ),
            (
                lambda payload: payload["assets"][-1].update(size=0),
                "size",
            ),
            (
                lambda payload: payload["assets"][-1].update(size=65_537),
                "size",
            ),
            (
                lambda payload: payload["assets"][-1].update(
                    digest="sha256:" + "A" * 64
                ),
                "sha256 digest",
            ),
            (
                lambda payload: payload["assets"][-1].update(state="new"),
                "uploaded",
            ),
        ):
            with self.subTest(expected_error=expected_error):
                changed_release = copy.deepcopy(release)
                mutate(changed_release)
                with self.assertRaisesRegex(
                    manifest.ReleaseManifestError,
                    expected_error,
                ):
                    manifest.locate_immutable_source_manifest_asset(
                        changed_release,
                        "lemonade-sdk/llama.cpp",
                    )

    def test_immutable_source_manifest_schema_is_strict(self) -> None:
        release, evidence, _source_commit = self._immutable_source_manifest_fixture()
        document = json.loads(evidence)
        cases = []

        extra_top_level = copy.deepcopy(document)
        extra_top_level["unexpected"] = True
        cases.append(("unexpected fields", extra_top_level))
        for schema_version in (True, 1.0, 2, "1"):
            changed = copy.deepcopy(document)
            changed["schema_version"] = schema_version
            cases.append(("schema_version", changed))
        extra_asset_field = copy.deepcopy(document)
        extra_asset_field["assets"][0]["unexpected"] = True
        cases.append(("unexpected fields", extra_asset_field))
        duplicate_asset = copy.deepcopy(document)
        duplicate_asset["assets"].append(copy.deepcopy(duplicate_asset["assets"][0]))
        cases.append(("duplicate asset", duplicate_asset))
        self_reference = copy.deepcopy(document)
        self_reference["assets"][0]["name"] = manifest.SOURCE_MANIFEST_ASSET_NAME
        cases.append(("exclude itself", self_reference))
        for expected_error, changed in cases:
            with self.subTest(expected_error=expected_error):
                changed_bytes = json.dumps(
                    changed,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                changed_release = copy.deepcopy(release)
                self._bind_source_evidence(changed_release, changed_bytes)

                with self.assertRaisesRegex(
                    manifest.ReleaseManifestError,
                    expected_error,
                ):
                    manifest.validate_immutable_source_manifest(
                        changed_bytes,
                        changed_release,
                        "lemonade-sdk/llama.cpp",
                        "b4321",
                        "b" * 40,
                    )

        reversed_document = copy.deepcopy(document)
        reversed_document["assets"].reverse()
        reversed_evidence = json.dumps(
            reversed_document,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        reversed_release = copy.deepcopy(release)
        self._bind_source_evidence(reversed_release, reversed_evidence)
        self.assertEqual(
            manifest.validate_immutable_source_manifest(
                reversed_evidence,
                reversed_release,
                "lemonade-sdk/llama.cpp",
                "b4321",
                "b" * 40,
            ),
            document["source_commit"],
        )

    def test_immutable_source_manifest_measures_raw_evidence_before_parsing(
        self,
    ) -> None:
        release, evidence, _source_commit = self._immutable_source_manifest_fixture()

        with self.assertRaisesRegex(
            manifest.ReleaseManifestError, "size does not match"
        ):
            manifest.validate_immutable_source_manifest(
                evidence + b" ",
                release,
                "lemonade-sdk/llama.cpp",
                "b4321",
                "b" * 40,
            )

        same_size_corruption = b"[" + evidence[1:]
        with self.assertRaisesRegex(
            manifest.ReleaseManifestError,
            "sha256 digest does not match",
        ):
            manifest.validate_immutable_source_manifest(
                same_size_corruption,
                release,
                "lemonade-sdk/llama.cpp",
                "b4321",
                "b" * 40,
            )

        invalid_utf8 = b"\xff"
        invalid_utf8_release = copy.deepcopy(release)
        self._bind_source_evidence(invalid_utf8_release, invalid_utf8)
        with self.assertRaisesRegex(manifest.ReleaseManifestError, "UTF-8 JSON"):
            manifest.validate_immutable_source_manifest(
                invalid_utf8,
                invalid_utf8_release,
                "lemonade-sdk/llama.cpp",
                "b4321",
                "b" * 40,
            )

    def test_immutable_source_manifest_release_metadata_is_strict(self) -> None:
        release, _evidence, _source_commit = self._immutable_source_manifest_fixture()
        source_index = next(
            index
            for index, asset in enumerate(release["assets"])
            if asset["name"] == manifest.SOURCE_MANIFEST_ASSET_NAME
        )
        cases = (
            ("release draft", lambda payload: payload.update(draft=True)),
            (
                "positive integer",
                lambda payload: payload["assets"][source_index].update(id=True),
            ),
            (
                "duplicate release asset name",
                lambda payload: payload["assets"][1].update(
                    name=payload["assets"][0]["name"]
                ),
            ),
            (
                "must be uploaded",
                lambda payload: payload["assets"][0].update(state="new"),
            ),
            (
                "lowercase sha256 digest",
                lambda payload: payload["assets"][0].update(
                    digest="sha256:" + "A" * 64
                ),
            ),
        )
        for expected_error, mutate in cases:
            with self.subTest(expected_error=expected_error):
                changed_release = copy.deepcopy(release)
                mutate(changed_release)

                with self.assertRaisesRegex(
                    manifest.ReleaseManifestError,
                    expected_error,
                ):
                    manifest.locate_immutable_source_manifest_asset(
                        changed_release,
                        "lemonade-sdk/llama.cpp",
                    )

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
        self.assertEqual(parsed["schema_version"], 4)
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
            ("release immutable", lambda payload: payload.pop("immutable")),
            ("release draft", lambda payload: payload.pop("draft")),
            (
                "release tag commit",
                lambda payload: payload.pop("release_tag_commit"),
            ),
            (
                "source repository",
                lambda payload: payload.pop("source_repository"),
            ),
            (
                "publisher claim type",
                lambda payload: payload.pop("publisher_claim_type"),
            ),
            (
                "publisher-claimed source commit",
                lambda payload: payload.pop("publisher_claimed_source_commit"),
            ),
            (
                "upstream reference",
                lambda payload: payload.pop("upstream_reference"),
            ),
            (
                "upstream reference head",
                lambda payload: payload.pop("upstream_reference_head"),
            ),
            ("asset name", lambda payload: payload["assets"][0].pop("name")),
            ("asset id", lambda payload: payload["assets"][0].pop("id")),
            ("asset size", lambda payload: payload["assets"][0].pop("size")),
            (
                "asset updated_at",
                lambda payload: payload["assets"][0].pop("updated_at"),
            ),
            (
                "asset must be uploaded",
                lambda payload: payload["assets"][0].pop("state"),
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

    def test_mutable_releases_fail_closed(self) -> None:
        for immutable in (False, None, 1, "true"):
            with self.subTest(immutable=immutable):
                payload = copy.deepcopy(self.ggml)
                payload["immutable"] = immutable

                with self.assertRaisesRegex(
                    manifest.ReleaseManifestError,
                    "release immutable must be true",
                ):
                    manifest.build_release_asset_manifest(
                        [("ggml-org/llama.cpp", "b1234", payload)]
                    )

    def test_draft_releases_fail_closed(self) -> None:
        for draft in (True, None, 0, "false"):
            with self.subTest(draft=draft):
                payload = copy.deepcopy(self.ggml)
                payload["draft"] = draft

                with self.assertRaisesRegex(
                    manifest.ReleaseManifestError,
                    "release draft must be false",
                ):
                    manifest.build_release_asset_manifest(
                        [("ggml-org/llama.cpp", "b1234", payload)]
                    )

    def test_publisher_claim_commits_must_be_full_git_oids(self) -> None:
        for field in ("release_tag_commit", "publisher_claimed_source_commit"):
            for commit in (None, "", "abc123", "g" * 40):
                with self.subTest(field=field, commit=commit):
                    payload = copy.deepcopy(self.ggml)
                    payload[field] = commit

                    with self.assertRaisesRegex(
                        manifest.ReleaseManifestError,
                        (
                            "publisher-claimed source commit"
                            if field == "publisher_claimed_source_commit"
                            else field.replace("_", " ")
                        ),
                    ):
                        manifest.build_release_asset_manifest(
                            [("ggml-org/llama.cpp", "b1234", payload)]
                        )

    def test_packaging_commit_is_distinct_from_binary_source_commit(self) -> None:
        payload = release_payload(
            "b4321",
            30,
            300,
            repository="ggml-org/llama.cpp",
            publisher_claim_type="immutable-source-manifest",
            publisher_claimed_source_commit="f" * 40,
        )

        parsed = json.loads(
            manifest.build_release_asset_manifest(
                [("lemonade-sdk/llama.cpp", "b4321", payload)]
            )
        )["releases"][0]

        self.assertEqual(parsed["release_tag_commit"], f"{30:040x}")
        self.assertEqual(parsed["source_repository"], "ggml-org/llama.cpp")
        self.assertEqual(parsed["publisher_claimed_source_commit"], "f" * 40)
        self.assertNotEqual(
            parsed["release_tag_commit"],
            parsed["publisher_claimed_source_commit"],
        )

    def test_upstream_release_tag_commit_must_equal_source_commit(self) -> None:
        payload = copy.deepcopy(self.ggml)
        payload["publisher_claimed_source_commit"] = "f" * 40

        with self.assertRaisesRegex(
            manifest.ReleaseManifestError,
            "release tag commit must equal publisher-claimed source commit",
        ):
            manifest.build_release_asset_manifest(
                [("ggml-org/llama.cpp", "b1234", payload)]
            )

    def test_release_note_source_claims_are_rejected(self) -> None:
        payload = release_payload(
            "b4321",
            30,
            300,
            repository="ggml-org/llama.cpp",
            publisher_claim_type="immutable-source-manifest",
            publisher_claimed_source_commit="f" * 40,
        )
        payload["publisher_claim_type"] = "release-note-marker"
        payload["body"] = f"**Llama.cpp Source Commit**: {'f' * 40}"

        with self.assertRaisesRegex(
            manifest.ReleaseManifestError,
            "publisher claim type",
        ):
            manifest.build_release_asset_manifest(
                [("lemonade-sdk/llama.cpp", "b4321", payload)]
            )

    def test_fork_network_commit_is_not_upstream_ancestry(self) -> None:
        claimed_commit = "3" * 40
        upstream_head = "7" * 40
        comparison = {
            "base_commit": {"sha": claimed_commit},
            "merge_base_commit": {"sha": "4" * 40},
            "status": "diverged",
            "url": (
                "https://api.github.com/repos/ggml-org/llama.cpp/compare/"
                f"{claimed_commit}...{upstream_head}"
            ),
        }

        self.assertTrue(
            hasattr(manifest, "require_upstream_ancestry"),
            "upstream ancestry validator is required",
        )
        with self.assertRaisesRegex(
            manifest.ReleaseManifestError,
            "not reachable",
        ):
            manifest.require_upstream_ancestry(
                comparison,
                "ggml-org/llama.cpp",
                claimed_commit,
                upstream_head,
            )

    def test_untrusted_manifest_anchor_must_reach_live_upstream_head(self) -> None:
        manifest_anchor = "3" * 40
        live_upstream_head = "7" * 40
        comparison = {
            "base_commit": {"sha": manifest_anchor},
            "merge_base_commit": {"sha": "4" * 40},
            "status": "diverged",
            "url": (
                "https://api.github.com/repos/ggml-org/llama.cpp/compare/"
                f"{manifest_anchor}...{live_upstream_head}"
            ),
        }

        with self.assertRaisesRegex(
            manifest.ReleaseManifestError,
            "not reachable",
        ):
            manifest.require_upstream_ancestry(
                comparison,
                "ggml-org/llama.cpp",
                manifest_anchor,
                live_upstream_head,
            )

    def test_unchanged_managed_pins_do_not_require_a_manifest(self) -> None:
        versions = {
            "llamacpp": {
                "cpu": "b1",
                "cuda": "b2",
                "metal": "b1",
                "rocm-nightly": "b3",
                "rocm-stable": "b2",
                "vulkan": "b1",
            }
        }

        self.assertFalse(
            manifest.require_release_manifest_for_managed_pin_changes(
                versions,
                copy.deepcopy(versions),
                None,
            )
        )

    def test_changed_managed_pin_requires_a_matching_release_manifest(self) -> None:
        base = {
            "llamacpp": {
                "cpu": "b1",
                "cuda": "b2",
                "metal": "b1",
                "rocm-nightly": "b3",
                "rocm-stable": "b2",
                "vulkan": "b1",
            }
        }
        candidate = copy.deepcopy(base)
        candidate["llamacpp"]["vulkan"] = "b1234"
        release_manifest = manifest.build_release_asset_manifest(
            [
                ("ggml-org/llama.cpp", "b1234", self.ggml),
                (
                    "lemonade-sdk/llamacpp-rocm",
                    "b3",
                    release_payload(
                        "b3",
                        20,
                        200,
                        publisher_claim_type="immutable-source-manifest",
                    ),
                ),
                (
                    "lemonade-sdk/llama.cpp",
                    "b2",
                    release_payload(
                        "b2",
                        30,
                        300,
                        publisher_claim_type="immutable-source-manifest",
                    ),
                ),
            ]
        )

        with self.assertRaisesRegex(
            manifest.ReleaseManifestError,
            "changed managed llama.cpp pins",
        ):
            manifest.require_release_manifest_for_managed_pin_changes(
                base,
                candidate,
                None,
            )

        self.assertTrue(
            manifest.require_release_manifest_for_managed_pin_changes(
                base,
                candidate,
                release_manifest,
            )
        )

        mismatched = json.loads(release_manifest)
        mismatched["releases"][0]["tag_name"] = "b9999"
        with self.assertRaisesRegex(
            manifest.ReleaseManifestError,
            "does not match changed managed llama.cpp pin",
        ):
            manifest.require_release_manifest_for_managed_pin_changes(
                base,
                candidate,
                json.dumps(mismatched),
            )

    def test_manifest_binds_publisher_claim_and_upstream_reference(self) -> None:
        payload = copy.deepcopy(self.rocm)
        payload.update(
            {
                "publisher_claim_type": "immutable-source-manifest",
                "upstream_reference": "refs/heads/master",
                "upstream_reference_head": "d" * 40,
            }
        )

        parsed = json.loads(
            manifest.build_release_asset_manifest(
                [("lemonade-sdk/llamacpp-rocm", "b1235", payload)]
            )
        )["releases"][0]

        self.assertEqual(
            parsed.get("publisher_claimed_source_commit"),
            payload["publisher_claimed_source_commit"],
        )
        self.assertEqual(
            parsed.get("publisher_claim_type"), "immutable-source-manifest"
        )
        self.assertIs(parsed.get("release_immutable"), True)
        self.assertEqual(parsed.get("upstream_reference"), "refs/heads/master")
        self.assertEqual(parsed.get("upstream_reference_head"), "d" * 40)
        self.assertNotIn("release_body_digest", parsed)
        self.assertNotIn("source_commit", parsed)

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

    def test_canonical_manifest_rejects_duplicate_json_members(self) -> None:
        canonical = manifest.build_release_asset_manifest(
            [("ggml-org/llama.cpp", "b1234", self.ggml)]
        )
        duplicate = canonical[:-1] + ',"schema_version":4}'

        with self.assertRaisesRegex(
            manifest.ReleaseManifestError,
            "duplicate JSON member",
        ):
            manifest.canonicalize_release_asset_manifest(duplicate)

    def test_canonical_manifest_requires_integer_schema_version(self) -> None:
        canonical = json.loads(
            manifest.build_release_asset_manifest(
                [("ggml-org/llama.cpp", "b1234", self.ggml)]
            )
        )
        for schema_version in (True, 4.0, "4"):
            with self.subTest(schema_version=schema_version):
                changed = copy.deepcopy(canonical)
                changed["schema_version"] = schema_version

                with self.assertRaisesRegex(
                    manifest.ReleaseManifestError,
                    "schema_version must be 4",
                ):
                    manifest.canonicalize_release_asset_manifest(json.dumps(changed))

    def test_comparison_rejects_any_publisher_claim_change(self) -> None:
        release = release_payload(
            "b1234",
            10,
            100,
            publisher_claim_type="immutable-source-manifest",
            publisher_claimed_source_commit="f" * 40,
        )
        expected = manifest.build_release_asset_manifest(
            [("lemonade-sdk/llama.cpp", "b1234", release)]
        )
        cases = (
            ("release_tag_commit", "e" * 40),
            ("publisher_claimed_source_commit", "e" * 40),
        )
        for field, value in cases:
            with self.subTest(field=field):
                changed = copy.deepcopy(release)
                changed[field] = value
                actual = manifest.build_release_asset_manifest(
                    [("lemonade-sdk/llama.cpp", "b1234", changed)]
                )

                with self.assertRaisesRegex(
                    manifest.ReleaseManifestError,
                    "changed since validation",
                ):
                    manifest.require_exact_manifest_match(expected, actual)

    def test_release_note_edits_do_not_change_manifest_identity(self) -> None:
        expected = manifest.build_release_asset_manifest(
            [("ggml-org/llama.cpp", "b1234", self.ggml)]
        )
        changed = copy.deepcopy(self.ggml)
        changed["body"] = "updated release notes"

        self.assertEqual(
            manifest.build_release_asset_manifest(
                [("ggml-org/llama.cpp", "b1234", changed)]
            ),
            expected,
        )

    def test_materialized_manifest_binds_digest_and_publisher_claims(self) -> None:
        expected = manifest.build_release_asset_manifest(
            [
                ("ggml-org/llama.cpp", "b1234", self.ggml),
                ("lemonade-sdk/llamacpp-rocm", "b1235", self.rocm),
            ]
        )
        expected_digest = hashlib.sha256(expected.encode("utf-8")).hexdigest()

        self.assertTrue(
            hasattr(manifest, "materialize_release_asset_manifest"),
            "manifest materializer is required",
        )
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "release-manifest.json"
            digest, summary = manifest.materialize_release_asset_manifest(
                expected,
                output_path,
            )

            self.assertEqual(digest, expected_digest)
            self.assertEqual(output_path.read_text(encoding="utf-8"), expected + "\n")
            self.assertIn(expected_digest, summary)
            self.assertIn("Only immutable GitHub releases are accepted", summary)
            self.assertIn("immutable publisher manifest", summary)
            self.assertIn("not cryptographic build provenance", summary)
            self.assertIn(self.ggml["publisher_claimed_source_commit"], summary)
            self.assertIn(self.rocm["publisher_claimed_source_commit"], summary)


if __name__ == "__main__":
    unittest.main()
