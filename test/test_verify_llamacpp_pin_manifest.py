#!/usr/bin/env python3
"""Behavior tests for trusted llama.cpp pin-manifest verification."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.utils import llamacpp_release_assets as release_assets
from test.utils import llamacpp_release_manifest as manifest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / ".github" / "scripts" / "verify_llamacpp_pin_manifest.sh"
MANIFEST_TOOL = ROOT / "test" / "utils" / "llamacpp_release_manifest.py"
ASSET_TOOL = ROOT / "test" / "utils" / "llamacpp_release_assets.py"
VERSIONS_PATH = Path("src/cpp/resources/backend_versions.json")
RELEASE_MANIFEST_PATH = Path(".github/llamacpp_release_manifest.json")
MISSING = object()


def release_payload(
    repository: str,
    tag: str,
    release_id: int,
    asset_names: list[str] | None = None,
) -> dict:
    release_tag_commit = f"{release_id:040x}"
    claim_type = (
        "source-release-tag"
        if repository == manifest.TRUSTED_SOURCE_REPOSITORY
        else "immutable-source-manifest"
    )
    names = asset_names or [f"llama-{release_id}.zip"]
    assets = [
        {
            "digest": "sha256:" + f"{release_id * 1000 + index:064x}"[-64:],
            "id": release_id * 1000 + index,
            "name": name,
            "size": release_id,
            "state": "uploaded",
            "updated_at": "2026-09-05T12:30:00Z",
        }
        for index, name in enumerate(names, start=1)
    ]
    if claim_type == "immutable-source-manifest":
        assets.append(
            {
                "digest": "sha256:" + "a" * 64,
                "id": release_id * 1000 + len(assets) + 1,
                "name": manifest.SOURCE_MANIFEST_ASSET_NAME,
                "size": 512,
                "state": "uploaded",
                "updated_at": "2026-09-05T12:31:00Z",
            }
        )
    return {
        "assets": assets,
        "draft": False,
        "id": release_id,
        "immutable": True,
        "publisher_claim_type": claim_type,
        "publisher_claimed_source_commit": release_tag_commit,
        "release_tag_commit": release_tag_commit,
        "source_repository": manifest.TRUSTED_SOURCE_REPOSITORY,
        "tag_name": tag,
        "upstream_reference": "refs/heads/master",
        "upstream_reference_head": "f" * 40,
    }


class VerifyLlamaCppPinManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.tool_root = self.directory / "tools"
        scripts = self.tool_root / ".github" / "scripts"
        utilities = self.tool_root / "test" / "utils"
        scripts.mkdir(parents=True)
        utilities.mkdir(parents=True)
        shutil.copy2(VERIFIER, scripts / VERIFIER.name)
        shutil.copy2(MANIFEST_TOOL, utilities / MANIFEST_TOOL.name)
        shutil.copy2(ASSET_TOOL, utilities / ASSET_TOOL.name)
        capture = scripts / "capture_llamacpp_release_manifest.sh"
        capture.write_text(
            "#!/usr/bin/env bash\n"
            'printf \'%s\\n\' "$*" >> "$CAPTURE_LOG"\n'
            'exit "$CAPTURE_EXIT"\n',
            encoding="utf-8",
        )
        capture.chmod(0o755)
        self.verifier = scripts / VERIFIER.name
        self.case_number = 0
        self.capture_log = self.directory / "capture-0.log"
        self.base_versions = {
            "llamacpp": {
                "cpu": "b1",
                "cuda": "b2",
                "metal": "b1",
                "rocm-nightly": "b3",
                "rocm-stable": "b2",
                "vulkan": "b1",
            },
            "rocm_asset_families": {},
            "therock": {"version": "7.14.0"},
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def initialize_repository(
        path: Path,
        versions: dict,
        release_manifest: object = MISSING,
    ) -> str:
        path.mkdir()
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(
            ["git", "-C", str(path), "config", "user.name", "Test"], check=True
        )
        subprocess.run(
            ["git", "-C", str(path), "config", "user.email", "test@example.com"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "config", "commit.gpgsign", "false"],
            check=True,
        )
        versions_path = path / VERSIONS_PATH
        versions_path.parent.mkdir(parents=True)
        versions_path.write_text(json.dumps(versions) + "\n", encoding="utf-8")
        if release_manifest is not MISSING:
            manifest_path = path / RELEASE_MANIFEST_PATH
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(str(release_manifest), encoding="utf-8")
        subprocess.run(["git", "-C", str(path), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(path), "commit", "-q", "-m", "fixture"],
            check=True,
        )
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def matching_manifest(
        self,
        ggml_tag: str = "b9",
        rocm_tag: str = "b3",
        lemonade_tag: str = "b2",
        *,
        missing_asset: str | None = None,
        missing_attested_target: tuple[str, str] | None = None,
        zero_size_asset: str | None = None,
        versions: dict | None = None,
    ) -> str:
        selected_versions = versions or self.base_versions
        requirements = release_assets.build_asset_requirements(
            ggml_release=ggml_tag,
            rocm_release=rocm_tag,
            lemonade_release=lemonade_tag,
            backend_versions=selected_versions,
        )
        rocm_target_requirements = release_assets.build_rocm_asset_target_requirements(
            rocm_release=rocm_tag,
            backend_versions=selected_versions,
        )
        release_inputs = (
            ("ggml-org/llama.cpp", ggml_tag, 9, "ggml"),
            ("lemonade-sdk/llama.cpp", lemonade_tag, 2, "lemonade"),
            ("lemonade-sdk/llamacpp-rocm", rocm_tag, 3, "rocm"),
        )
        releases = []
        for repository, tag, release_id, requirement_group in release_inputs:
            names = [
                name
                for family in requirements[requirement_group].values()
                for name in family
            ]
            payload = release_payload(repository, tag, release_id, names)
            if missing_asset is not None:
                payload["assets"] = [
                    asset
                    for asset in payload["assets"]
                    if asset["name"] != missing_asset
                ]
            if zero_size_asset is not None:
                for asset in payload["assets"]:
                    if asset["name"] == zero_size_asset:
                        asset["size"] = 0
            if repository == "lemonade-sdk/llamacpp-rocm":
                payload["build_target_attestations"] = []
                for asset in payload["assets"]:
                    required_targets = rocm_target_requirements.get(asset["name"])
                    if required_targets is None:
                        continue
                    build_targets = list(required_targets)
                    if (
                        missing_attested_target is not None
                        and asset["name"] == missing_attested_target[0]
                    ):
                        build_targets.remove(missing_attested_target[1])
                    payload["build_target_attestations"].append(
                        {
                            "build_targets": build_targets,
                            "digest": asset["digest"],
                            "name": asset["name"],
                            "size": asset["size"],
                        }
                    )
            releases.append((repository, tag, payload))
        return manifest.build_release_asset_manifest(releases)

    def run_verifier(
        self,
        candidate_versions: dict,
        *,
        base_manifest: object = MISSING,
        candidate_manifest: object = MISSING,
        capture_exit: int = 37,
    ) -> subprocess.CompletedProcess[str]:
        self.case_number += 1
        trusted = self.directory / f"trusted-{self.case_number}"
        candidate = self.directory / f"candidate-{self.case_number}"
        self.capture_log = self.directory / f"capture-{self.case_number}.log"
        base_sha = self.initialize_repository(
            trusted, self.base_versions, base_manifest
        )
        candidate_sha = self.initialize_repository(
            candidate, candidate_versions, candidate_manifest
        )
        env = os.environ.copy()
        env.update(
            {
                "CAPTURE_EXIT": str(capture_exit),
                "CAPTURE_LOG": str(self.capture_log),
            }
        )
        return subprocess.run(
            [
                "bash",
                str(self.verifier),
                str(trusted),
                str(candidate),
                base_sha,
                candidate_sha,
            ],
            capture_output=True,
            check=False,
            env=env,
            text=True,
        )

    def test_unchanged_pins_skip_capture(self) -> None:
        result = self.run_verifier(copy.deepcopy(self.base_versions))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No changed managed llama.cpp pins", result.stdout)
        self.assertFalse(self.capture_log.exists())

    def test_changed_pin_rejects_missing_and_mismatched_manifests(self) -> None:
        candidate = copy.deepcopy(self.base_versions)
        candidate["llamacpp"]["vulkan"] = "b9"

        missing = self.run_verifier(candidate)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("require a committed release manifest", missing.stderr)
        self.assertFalse(self.capture_log.exists())

        mismatched = self.run_verifier(
            candidate,
            candidate_manifest=self.matching_manifest("b8"),
        )
        self.assertNotEqual(mismatched.returncode, 0)
        self.assertIn("does not match changed managed llama.cpp pin", mismatched.stderr)
        self.assertFalse(self.capture_log.exists())

    def test_sidecar_only_change_is_rejected(self) -> None:
        result = self.run_verifier(
            copy.deepcopy(self.base_versions),
            candidate_manifest="{}\n",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "Release manifest changes require changed managed llama.cpp pins",
            result.stderr,
        )
        self.assertFalse(self.capture_log.exists())

    def test_capture_failure_is_propagated(self) -> None:
        candidate = copy.deepcopy(self.base_versions)
        candidate["llamacpp"]["vulkan"] = "b9"

        result = self.run_verifier(
            candidate,
            candidate_manifest=self.matching_manifest(),
            capture_exit=37,
        )

        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertTrue(self.capture_log.exists())

    def test_changed_pins_require_complete_positive_cross_platform_assets(
        self,
    ) -> None:
        cases = (
            (
                "cpu",
                {"ggml_tag": "b9"},
                "llama-b9-bin-win-cpu-x64.zip",
            ),
            (
                "vulkan",
                {"ggml_tag": "b9"},
                "llama-b9-bin-ubuntu-vulkan-arm64.tar.gz",
            ),
            (
                "metal",
                {"ggml_tag": "b9"},
                "llama-b9-bin-macos-arm64.tar.gz",
            ),
            (
                "cuda",
                {"ggml_tag": "b1", "lemonade_tag": "b9"},
                "llama-b9-ubuntu-cuda-sm_121-arm64.tar.xz",
            ),
            (
                "rocm-stable",
                {"ggml_tag": "b1", "lemonade_tag": "b9"},
                "llama-b9-bin-win-rocm-7.14-x64.zip",
            ),
            (
                "rocm-nightly",
                {"ggml_tag": "b1", "rocm_tag": "b9"},
                "llama-b9-ubuntu-rocm-gfx942-x64.zip",
            ),
        )
        for backend, tags, required_asset in cases:
            candidate = copy.deepcopy(self.base_versions)
            candidate["llamacpp"][backend] = "b9"
            for defect in ("missing", "zero-size"):
                with self.subTest(backend=backend, defect=defect):
                    manifest_json = self.matching_manifest(
                        **tags,
                        missing_asset=required_asset if defect == "missing" else None,
                        zero_size_asset=(
                            required_asset if defect == "zero-size" else None
                        ),
                    )
                    result = self.run_verifier(
                        candidate,
                        candidate_manifest=manifest_json,
                        capture_exit=0,
                    )

                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertIn(f"llamacpp.{backend}", result.stderr)
                    self.assertFalse(self.capture_log.exists())

    def test_rocm_nightly_pin_rejects_incomplete_concrete_target_attestation(
        self,
    ) -> None:
        candidate = copy.deepcopy(self.base_versions)
        candidate["llamacpp"]["rocm-nightly"] = "b9"
        candidate["rocm_asset_families"] = {
            "gfx1033": "gfx103X",
            "gfx1035": "gfx103X",
            "gfx1036": "gfx103X",
        }
        target_asset = "llama-b9-windows-rocm-gfx103X-x64.zip"

        result = self.run_verifier(
            candidate,
            candidate_manifest=self.matching_manifest(
                ggml_tag="b1",
                rocm_tag="b9",
                versions=candidate,
                missing_attested_target=(target_asset, "gfx1035"),
            ),
            capture_exit=0,
        )

        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("incomplete build-target attestation", result.stderr)
        self.assertIn(target_asset, result.stderr)
        self.assertFalse(self.capture_log.exists())

    def test_selector_only_changes_require_live_manifests(self) -> None:
        mapping_candidate = copy.deepcopy(self.base_versions)
        mapping_candidate["rocm_asset_families"] = {"gfx1033": "gfx103X"}
        missing_mapping_manifest = self.run_verifier(mapping_candidate)
        self.assertNotEqual(missing_mapping_manifest.returncode, 0)
        self.assertIn(
            "require a committed release manifest", missing_mapping_manifest.stderr
        )
        self.assertFalse(self.capture_log.exists())

        therock_candidates = []
        for field, value in (
            ("version", "7.14.1"),
            ("architectures", ["gfx103X", "gfx110X"]),
            ("url_mapping", {"windows": "runtime.zip"}),
        ):
            candidate = copy.deepcopy(self.base_versions)
            candidate["therock"][field] = value
            therock_candidates.append(candidate)
            with self.subTest(therock_selector=field):
                missing_therock_manifest = self.run_verifier(candidate)
                self.assertNotEqual(missing_therock_manifest.returncode, 0)
                self.assertIn(
                    "require a committed release manifest",
                    missing_therock_manifest.stderr,
                )
                self.assertFalse(self.capture_log.exists())

        therock_candidate = therock_candidates[-1]
        accepted = self.run_verifier(
            therock_candidate,
            candidate_manifest=self.matching_manifest(versions=therock_candidate),
            capture_exit=0,
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertTrue(self.capture_log.exists())


if __name__ == "__main__":
    unittest.main()
