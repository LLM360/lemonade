import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test.utils import llamacpp_validation_artifacts as artifacts
from test.utils.llamacpp_validation_artifacts import (
    ValidationArtifactError,
    verify_validation_artifacts,
)

MODEL_ID = "K2-Horizon-Test-GGUF"
REPOSITORY = "IFM/K2-Horizon-Test-GGUF"
REVISION = "a" * 40
FILENAME = "model.gguf"


def gguf_string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def minimal_gguf(architecture="k2-horizon", context_length=131072):
    metadata = [
        gguf_string("general.architecture")
        + struct.pack("<I", 8)
        + gguf_string(architecture),
        gguf_string(f"{architecture}.context_length")
        + struct.pack("<IQ", 10, context_length),
    ]
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata)) + b"".join(metadata)


class LlamaCppValidationArtifactTests(unittest.TestCase):
    def test_direct_help_invocation_imports_repository_modules(self):
        result = subprocess.run(
            [sys.executable, str(Path(artifacts.__file__).resolve()), "--help"],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = self.root / "cache"
        self.catalog_path = self.root / "catalog.json"
        self.lock_path = self.root / "artifacts.json"
        self.model_path = (
            self.cache
            / "models--IFM--K2-Horizon-Test-GGUF"
            / "snapshots"
            / REVISION
            / FILENAME
        )
        self.model_path.parent.mkdir(parents=True)
        self.model_path.write_bytes(minimal_gguf())
        reference = self.model_path.parents[2] / "refs" / "main"
        reference.parent.mkdir(parents=True)
        reference.write_text(f"{REVISION}\n", encoding="ascii")
        self.catalog_path.write_text(
            json.dumps(
                {
                    MODEL_ID: {
                        "checkpoint": f"{REPOSITORY}:{FILENAME}",
                        "recipe": "llamacpp",
                        "labels": ["chat"],
                    }
                }
            ),
            encoding="utf-8",
        )
        self.write_lock()

    def lock(self):
        contents = self.model_path.read_bytes()
        return {
            "revision": REVISION,
            "filename": FILENAME,
            "size_bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
            "gguf_architecture": "k2-horizon",
            "max_context_window": 131072,
        }

    def write_lock(self, **changes):
        lock = self.lock()
        lock.update(changes)
        self.lock_path.write_text(
            json.dumps({MODEL_ID: lock}),
            encoding="utf-8",
        )

    def verify(self):
        return verify_validation_artifacts(
            self.cache,
            self.catalog_path,
            self.lock_path,
            [MODEL_ID],
        )

    def test_accepts_exact_isolated_cache_artifact(self):
        verified = self.verify()

        self.assertEqual(list(verified), [MODEL_ID])
        self.assertEqual(verified[MODEL_ID].model_id, MODEL_ID)
        self.assertEqual(verified[MODEL_ID].resolved_path, self.model_path.resolve())
        self.assertEqual(verified[MODEL_ID].size_bytes, self.lock()["size_bytes"])
        self.assertEqual(verified[MODEL_ID].sha256, self.lock()["sha256"])

    def test_ignores_selected_models_without_validation_locks(self):
        try:
            verified = verify_validation_artifacts(
                self.cache,
                self.catalog_path,
                self.lock_path,
                ["Other-Model"],
            )
        except ValidationArtifactError as exc:
            self.fail(f"unlocked selected model was rejected: {exc}")
        self.assertEqual(verified, {})

    def test_rejects_checkpoint_filename_traversal(self):
        self.write_lock(filename="../model.gguf")

        with self.assertRaisesRegex(ValidationArtifactError, "safe basename"):
            self.verify()

        self.write_lock(filename=None)
        with self.assertRaisesRegex(ValidationArtifactError, "safe basename"):
            self.verify()

    def test_rejects_invalid_lock_types_and_bounds(self):
        for field, value in (("size_bytes", True), ("max_context_window", 0)):
            with self.subTest(field=field):
                self.write_lock(**{field: value})
                with self.assertRaises(ValidationArtifactError):
                    self.verify()

    def test_rejects_ref_mismatch(self):
        reference = self.model_path.parents[2] / "refs" / "main"
        reference.write_text(f"{'b' * 40}\n", encoding="ascii")

        with self.assertRaisesRegex(ValidationArtifactError, "refs/main"):
            self.verify()

    def test_refs_main_accepts_only_a_canonical_revision_and_optional_newline(self):
        reference = self.model_path.parents[2] / "refs" / "main"
        for suffix in (b" ", b"\t", b"\n\n", b"\r\n", b"junk"):
            with self.subTest(suffix=suffix):
                reference.write_bytes(REVISION.encode("ascii") + suffix)

                with self.assertRaisesRegex(ValidationArtifactError, "refs/main"):
                    self.verify()

    def test_rejects_ref_replacement_during_artifact_hash(self):
        reference = self.model_path.parents[2] / "refs" / "main"
        replacement = self.root / "replacement-ref"
        replacement.write_text(f"{'b' * 40}\n", encoding="ascii")
        original_sha256 = artifacts._sha256

        def hash_then_swap(open_artifact):
            digest = original_sha256(open_artifact)
            os.replace(replacement, reference)
            return digest

        with (
            mock.patch.object(artifacts, "_sha256", side_effect=hash_then_swap),
            self.assertRaisesRegex(ValidationArtifactError, "refs/main.*changed"),
        ):
            self.verify()

    def test_rejects_ref_symlink_target_swap_during_artifact_hash(self):
        reference = self.model_path.parents[2] / "refs" / "main"
        references = self.cache / "reference-objects"
        references.mkdir()
        original = references / "original"
        replacement = references / "replacement"
        original.write_text(f"{REVISION}\n", encoding="ascii")
        replacement.write_text(f"{'b' * 40}\n", encoding="ascii")
        reference.unlink()
        try:
            reference.symlink_to(original)
        except OSError as exc:
            self.skipTest(f"file symlinks are unavailable: {exc}")
        original_sha256 = artifacts._sha256

        def hash_then_swap(open_artifact):
            digest = original_sha256(open_artifact)
            reference.unlink()
            reference.symlink_to(replacement)
            return digest

        with (
            mock.patch.object(artifacts, "_sha256", side_effect=hash_then_swap),
            self.assertRaisesRegex(ValidationArtifactError, "refs/main.*changed"),
        ):
            self.verify()

    def test_rejects_an_artifact_symlink_outside_the_cache(self):
        outside = self.root / "outside.gguf"
        outside.write_bytes(self.model_path.read_bytes())
        self.model_path.unlink()
        self.model_path.symlink_to(outside)
        self.write_lock()

        with self.assertRaisesRegex(ValidationArtifactError, "escapes"):
            self.verify()

    def test_rejects_size_and_hash_mismatch(self):
        for field, value, message in (
            ("size_bytes", self.model_path.stat().st_size + 1, "size"),
            ("sha256", "0" * 64, "SHA-256"),
        ):
            with self.subTest(field=field):
                self.write_lock(**{field: value})
                with self.assertRaisesRegex(ValidationArtifactError, message):
                    self.verify()

    def test_hash_mismatch_is_rejected_before_gguf_metadata_parsing(self):
        self.write_lock(sha256="0" * 64)

        with (
            mock.patch.object(artifacts, "_read_gguf_identity") as read_metadata,
            self.assertRaisesRegex(ValidationArtifactError, "SHA-256"),
        ):
            self.verify()

        read_metadata.assert_not_called()

    def test_rejects_same_size_path_swap_after_hash(self):
        original = minimal_gguf() + b"A"
        replacement = minimal_gguf() + b"B"
        self.assertEqual(len(original), len(replacement))
        self.model_path.write_bytes(original)
        self.write_lock()
        replacement_path = self.root / "replacement.gguf"
        replacement_path.write_bytes(replacement)
        original_sha256 = artifacts._sha256

        def hash_then_swap(open_artifact):
            digest = original_sha256(open_artifact)
            os.replace(replacement_path, self.model_path)
            return digest

        with (
            mock.patch.object(artifacts, "_sha256", side_effect=hash_then_swap),
            self.assertRaisesRegex(
                ValidationArtifactError, "changed during verification"
            ),
        ):
            self.verify()

    def test_rejects_snapshot_symlink_target_swap_after_hash(self):
        objects = self.cache / "objects"
        objects.mkdir()
        original = objects / "original.gguf"
        replacement = objects / "replacement.gguf"
        original.write_bytes(minimal_gguf() + b"A")
        replacement.write_bytes(minimal_gguf() + b"B")
        self.model_path.unlink()
        try:
            self.model_path.symlink_to(original)
        except OSError as exc:
            self.skipTest(f"file symlinks are unavailable: {exc}")
        self.write_lock()
        original_sha256 = artifacts._sha256

        def hash_then_swap(open_artifact):
            digest = original_sha256(open_artifact)
            self.model_path.unlink()
            self.model_path.symlink_to(replacement)
            return digest

        with (
            mock.patch.object(artifacts, "_sha256", side_effect=hash_then_swap),
            self.assertRaisesRegex(
                ValidationArtifactError, "changed during verification"
            ),
        ):
            self.verify()

    def test_rejects_gguf_architecture_and_context_mismatch(self):
        for field, value, message in (
            ("gguf_architecture", "other", "architecture"),
            ("max_context_window", 42, "context"),
        ):
            with self.subTest(field=field):
                self.write_lock(**{field: value})
                with self.assertRaisesRegex(ValidationArtifactError, message):
                    self.verify()

    def test_rejects_out_of_bounds_gguf_metadata(self):
        self.model_path.write_bytes(
            b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**63)
        )
        self.write_lock()

        with self.assertRaisesRegex(ValidationArtifactError, "bounds"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
