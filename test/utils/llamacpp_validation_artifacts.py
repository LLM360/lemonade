#!/usr/bin/env python3
"""Verify validation-only K2-Horizon artifacts in an isolated HF cache."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test.utils.validation_model_catalog import load_model_catalog


class ValidationArtifactError(ValueError):
    """Raised when a validation artifact does not match its immutable lock."""


@dataclass(frozen=True)
class VerifiedValidationArtifact:
    model_id: str
    resolved_path: Path
    size_bytes: int
    sha256: str


LOCK_FIELDS = {
    "filename",
    "gguf_architecture",
    "max_context_window",
    "revision",
    "sha256",
    "size_bytes",
}
LOWER_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
LOWER_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
REPOSITORY_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ARCHITECTURE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
MAX_METADATA_ITEMS = 1_000_000
MAX_STRING_BYTES = 64 * 1024 * 1024
SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<B",
    10: "<Q",
    11: "<q",
    12: "<d",
}


class _Reader:
    def __init__(self, stream, path: Path):
        self.path = path
        self.stream = stream
        self.size = os.fstat(stream.fileno()).st_size

    def read(self, count: int) -> bytes:
        if count < 0 or count > self.size - self.stream.tell():
            raise ValidationArtifactError(
                f"{self.path}: GGUF metadata exceeds file bounds"
            )
        value = self.stream.read(count)
        if len(value) != count:
            raise ValidationArtifactError(
                f"{self.path}: GGUF metadata exceeds file bounds"
            )
        return value

    def unpack(self, format_string: str):
        size = struct.calcsize(format_string)
        return struct.unpack(format_string, self.read(size))[0]

    def string(self, *, decode: bool) -> str | None:
        length = self.unpack("<Q")
        if length > MAX_STRING_BYTES:
            raise ValidationArtifactError(
                f"{self.path}: GGUF string length exceeds bounds"
            )
        value = self.read(length)
        if not decode:
            return None
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationArtifactError(
                f"{self.path}: GGUF metadata is not UTF-8"
            ) from exc


def _read_scalar(reader: _Reader, value_type: int):
    format_string = SCALAR_FORMATS.get(value_type)
    if format_string is None:
        raise ValidationArtifactError(
            f"{reader.path}: unsupported GGUF metadata type {value_type}"
        )
    return reader.unpack(format_string)


def _skip_value(reader: _Reader, value_type: int) -> None:
    if value_type == 8:
        reader.string(decode=False)
        return
    if value_type != 9:
        _read_scalar(reader, value_type)
        return

    element_type = reader.unpack("<I")
    count = reader.unpack("<Q")
    if count > MAX_METADATA_ITEMS or element_type == 9:
        raise ValidationArtifactError(
            f"{reader.path}: GGUF array count or type exceeds bounds"
        )
    if element_type == 8:
        for _ in range(count):
            reader.string(decode=False)
        return
    format_string = SCALAR_FORMATS.get(element_type)
    if format_string is None:
        raise ValidationArtifactError(
            f"{reader.path}: unsupported GGUF array type {element_type}"
        )
    reader.read(struct.calcsize(format_string) * count)


def _read_gguf_identity(open_artifact, path: Path) -> tuple[str, int]:
    open_artifact.seek(0)
    reader = _Reader(open_artifact, path)
    if reader.read(4) != b"GGUF":
        raise ValidationArtifactError(f"{path}: invalid GGUF magic")
    version = reader.unpack("<I")
    if version not in {2, 3}:
        raise ValidationArtifactError(f"{path}: unsupported GGUF version {version}")
    reader.unpack("<Q")
    metadata_count = reader.unpack("<Q")
    if metadata_count > MAX_METADATA_ITEMS:
        raise ValidationArtifactError(f"{path}: GGUF metadata count exceeds bounds")

    metadata = {}
    for _ in range(metadata_count):
        key = reader.string(decode=True)
        if key in metadata:
            raise ValidationArtifactError(f"{path}: duplicate GGUF key {key}")
        value_type = reader.unpack("<I")
        if key == "general.architecture":
            if value_type != 8:
                raise ValidationArtifactError(
                    f"{path}: general.architecture must be a string"
                )
            metadata[key] = reader.string(decode=True)
        elif key.endswith(".context_length"):
            if value_type not in {4, 10}:
                raise ValidationArtifactError(
                    f"{path}: context length must be an unsigned integer"
                )
            metadata[key] = _read_scalar(reader, value_type)
        else:
            _skip_value(reader, value_type)

    architecture = metadata.get("general.architecture")
    if not isinstance(architecture, str):
        raise ValidationArtifactError(f"{path}: missing GGUF architecture")
    context = metadata.get(f"{architecture}.context_length")
    if not isinstance(context, int) or isinstance(context, bool):
        raise ValidationArtifactError(f"{path}: missing GGUF context length")
    return architecture, context


def read_gguf_identity(path: Path) -> tuple[str, int]:
    with path.open("rb") as artifact:
        return _read_gguf_identity(artifact, path)


def _parse_checkpoint(model_id: str, checkpoint: object) -> tuple[str, str]:
    if not isinstance(checkpoint, str) or checkpoint.count(":") != 1:
        raise ValidationArtifactError(
            f"{model_id}: validation checkpoint must be repository:filename"
        )
    repository, filename = checkpoint.split(":", 1)
    components = repository.split("/")
    if (
        len(components) != 2
        or any(not REPOSITORY_COMPONENT.fullmatch(part) for part in components)
        or any(part in {".", ".."} or "--" in part for part in components)
    ):
        raise ValidationArtifactError(f"{model_id}: unsafe HF repository name")
    if (
        not filename
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or ":" in filename
        or "\0" in filename
        or filename in {".", ".."}
    ):
        raise ValidationArtifactError(f"{model_id}: filename must be a safe basename")
    return repository, filename


def _validate_lock(model_id: str, lock: dict) -> None:
    if set(lock) != LOCK_FIELDS:
        raise ValidationArtifactError(f"{model_id}: artifact lock fields are invalid")
    if not isinstance(lock["revision"], str) or not LOWER_HEX_40.fullmatch(
        lock["revision"]
    ):
        raise ValidationArtifactError(f"{model_id}: revision must be lowercase hex")
    if not isinstance(lock["sha256"], str) or not LOWER_HEX_64.fullmatch(
        lock["sha256"]
    ):
        raise ValidationArtifactError(f"{model_id}: sha256 must be lowercase hex")
    for field in ("size_bytes", "max_context_window"):
        if type(lock[field]) is not int or not 0 < lock[field] <= 2**63 - 1:
            raise ValidationArtifactError(f"{model_id}: {field} is out of bounds")
    if not isinstance(lock["gguf_architecture"], str) or not ARCHITECTURE.fullmatch(
        lock["gguf_architecture"]
    ):
        raise ValidationArtifactError(f"{model_id}: invalid GGUF architecture")
    if not isinstance(lock["filename"], str):
        raise ValidationArtifactError(f"{model_id}: filename must be a safe basename")
    _parse_checkpoint(model_id, f"x/y:{lock['filename']}")


def _sha256(open_artifact) -> str:
    digest = hashlib.sha256()
    open_artifact.seek(0)
    for chunk in iter(lambda: open_artifact.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _artifact_state(file_status: os.stat_result) -> tuple[int, ...]:
    return (
        file_status.st_dev,
        file_status.st_ino,
        file_status.st_mode,
        file_status.st_size,
        file_status.st_mtime_ns,
        file_status.st_ctime_ns,
    )


def _verify_locked_artifact(
    cache_root: Path,
    repository_path: Path,
    model_id: str,
    lock: dict,
) -> VerifiedValidationArtifact:
    artifact_path = repository_path / "snapshots" / lock["revision"] / lock["filename"]
    try:
        resolved_artifact = artifact_path.resolve(strict=True)
        resolved_artifact.relative_to(cache_root)
    except (OSError, ValueError) as exc:
        raise ValidationArtifactError(
            f"{model_id}: artifact path escapes or is absent from the HF cache"
        ) from exc
    try:
        with resolved_artifact.open("rb") as artifact:
            status_before = os.fstat(artifact.fileno())
            if not stat.S_ISREG(status_before.st_mode):
                raise ValidationArtifactError(f"{model_id}: artifact is not a file")
            if status_before.st_size != lock["size_bytes"]:
                raise ValidationArtifactError(
                    f"{model_id}: artifact size {status_before.st_size} != "
                    f"{lock['size_bytes']}"
                )
            artifact_sha256 = _sha256(artifact)
            if artifact_sha256 != lock["sha256"]:
                raise ValidationArtifactError(f"{model_id}: artifact SHA-256 mismatch")
            architecture, context = _read_gguf_identity(artifact, resolved_artifact)
            status_after = os.fstat(artifact.fileno())

        path_after = artifact_path.resolve(strict=True)
        path_after.relative_to(cache_root)
        path_status = path_after.stat()
    except ValidationArtifactError:
        raise
    except (OSError, ValueError) as exc:
        raise ValidationArtifactError(
            f"{model_id}: artifact changed during verification"
        ) from exc
    if (
        path_after != resolved_artifact
        or _artifact_state(status_before) != _artifact_state(status_after)
        or _artifact_state(status_after) != _artifact_state(path_status)
    ):
        raise ValidationArtifactError(
            f"{model_id}: artifact changed during verification"
        )
    if architecture != lock["gguf_architecture"]:
        raise ValidationArtifactError(
            f"{model_id}: GGUF architecture {architecture} != "
            f"{lock['gguf_architecture']}"
        )
    if context != lock["max_context_window"]:
        raise ValidationArtifactError(
            f"{model_id}: GGUF context {context} != {lock['max_context_window']}"
        )
    return VerifiedValidationArtifact(
        model_id=model_id,
        resolved_path=resolved_artifact,
        size_bytes=status_after.st_size,
        sha256=artifact_sha256,
    )


def _verify_reference_and_artifact(
    cache_root: Path,
    repository_path: Path,
    model_id: str,
    lock: dict,
) -> VerifiedValidationArtifact:
    reference_path = repository_path / "refs" / "main"
    try:
        resolved_reference = reference_path.resolve(strict=True)
        resolved_reference.relative_to(cache_root)
    except OSError as exc:
        raise ValidationArtifactError(
            f"{model_id}: could not read HF refs/main"
        ) from exc
    except ValueError as exc:
        raise ValidationArtifactError(
            f"{model_id}: HF refs/main escapes the cache"
        ) from exc

    expected_reference = lock["revision"].encode("ascii")
    canonical_references = {expected_reference, expected_reference + b"\n"}
    try:
        with resolved_reference.open("rb") as reference_file:
            status_before = os.fstat(reference_file.fileno())
            if not stat.S_ISREG(status_before.st_mode):
                raise ValidationArtifactError(f"{model_id}: HF refs/main is not a file")
            reference_bytes = reference_file.read(42)
            if reference_bytes not in canonical_references:
                raise ValidationArtifactError(
                    f"{model_id}: HF refs/main does not match locked revision"
                )

            verified_artifact = _verify_locked_artifact(
                cache_root,
                repository_path,
                model_id,
                lock,
            )

            reference_file.seek(0)
            reference_bytes_after = reference_file.read(42)
            status_after = os.fstat(reference_file.fileno())

        path_after = reference_path.resolve(strict=True)
        path_after.relative_to(cache_root)
        path_status = path_after.stat()
    except ValidationArtifactError:
        raise
    except (OSError, ValueError) as exc:
        raise ValidationArtifactError(
            f"{model_id}: HF refs/main changed during verification"
        ) from exc
    if (
        reference_bytes_after not in canonical_references
        or reference_bytes_after != reference_bytes
        or path_after != resolved_reference
        or _artifact_state(status_before) != _artifact_state(status_after)
        or _artifact_state(status_after) != _artifact_state(path_status)
    ):
        raise ValidationArtifactError(
            f"{model_id}: HF refs/main changed during verification"
        )
    return verified_artifact


def verify_validation_artifacts(
    cache_path: Path | str,
    catalog_path: Path | str,
    lock_path: Path | str,
    model_ids: list[str],
) -> dict[str, VerifiedValidationArtifact]:
    cache_path = Path(cache_path)
    catalog = load_model_catalog(catalog_path)
    locks = load_model_catalog(lock_path)
    if set(catalog) != set(locks):
        raise ValidationArtifactError(
            "validation catalog and artifact lock IDs must match exactly"
        )

    requested = list(
        dict.fromkeys(value.removeprefix("builtin.") for value in model_ids)
    )
    locked_requested = [model_id for model_id in requested if model_id in locks]
    if not locked_requested:
        return {}

    try:
        cache_root = cache_path.resolve(strict=True)
    except OSError as exc:
        raise ValidationArtifactError("HF cache does not exist") from exc
    verified_artifacts = {}
    for model_id in locked_requested:
        model = catalog[model_id]
        lock = locks[model_id]
        _validate_lock(model_id, lock)
        repository, checkpoint_filename = _parse_checkpoint(
            model_id, model.get("checkpoint")
        )
        if checkpoint_filename != lock["filename"]:
            raise ValidationArtifactError(
                f"{model_id}: checkpoint and artifact filenames differ"
            )

        repository_path = cache_path / ("models--" + repository.replace("/", "--"))
        verified_artifacts[model_id] = _verify_reference_and_artifact(
            cache_root,
            repository_path,
            model_id,
            lock,
        )

    return verified_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--locks", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True)
    args = parser.parse_args()
    verify_validation_artifacts(
        args.cache,
        args.catalog,
        args.locks,
        args.model,
    )


if __name__ == "__main__":
    main()
