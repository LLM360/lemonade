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

from test.utils import llamacpp_release_manifest as manifest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / ".github" / "scripts" / "verify_llamacpp_pin_manifest.sh"
MANIFEST_TOOL = ROOT / "test" / "utils" / "llamacpp_release_manifest.py"
VERSIONS_PATH = Path("src/cpp/resources/backend_versions.json")
RELEASE_MANIFEST_PATH = Path(".github/llamacpp_release_manifest.json")
MISSING = object()


def release_payload(repository: str, tag: str, release_id: int) -> dict:
    release_tag_commit = f"{release_id:040x}"
    claim_type = (
        "source-release-tag"
        if repository == manifest.TRUSTED_SOURCE_REPOSITORY
        else "immutable-source-manifest"
    )
    assets = [
        {
            "digest": "sha256:" + f"{release_id:064x}"[-64:],
            "id": release_id * 10,
            "name": f"llama-{release_id}.zip",
            "size": release_id,
            "state": "uploaded",
            "updated_at": "2026-09-05T12:30:00Z",
        }
    ]
    if claim_type == "immutable-source-manifest":
        assets.append(
            {
                "digest": "sha256:" + "a" * 64,
                "id": release_id * 10 + 1,
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
            }
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

    def matching_manifest(self, ggml_tag: str = "b9") -> str:
        return manifest.build_release_asset_manifest(
            [
                (
                    "ggml-org/llama.cpp",
                    ggml_tag,
                    release_payload("ggml-org/llama.cpp", ggml_tag, 9),
                ),
                (
                    "lemonade-sdk/llama.cpp",
                    "b2",
                    release_payload("lemonade-sdk/llama.cpp", "b2", 2),
                ),
                (
                    "lemonade-sdk/llamacpp-rocm",
                    "b3",
                    release_payload("lemonade-sdk/llamacpp-rocm", "b3", 3),
                ),
            ]
        )

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


if __name__ == "__main__":
    unittest.main()
