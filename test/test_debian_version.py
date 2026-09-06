#!/usr/bin/env python3
"""Regression tests for Debian package version resolution."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESOLVER = ROOT / ".github" / "scripts" / "resolve_debian_version.sh"


class DebianVersionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.repo = Path(self.temporary_directory.name) / "repo"
        self.repo.mkdir()
        self.git_env = os.environ.copy()
        self.git_env.update(
            {
                "GIT_AUTHOR_DATE": "2001-01-01T00:00:00+0000",
                "GIT_AUTHOR_EMAIL": "test@example.com",
                "GIT_AUTHOR_NAME": "Test Author",
                "GIT_COMMITTER_DATE": "2001-01-01T00:00:00+0000",
                "GIT_COMMITTER_EMAIL": "test@example.com",
                "GIT_COMMITTER_NAME": "Test Committer",
            }
        )
        self.run_git("init", "--quiet")

    def run_git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=self.repo,
            env=self.git_env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def write_cmake_version(self, contents: str) -> None:
        (self.repo / "CMakeLists.txt").write_text(contents, encoding="utf-8")

    def commit_all(self, message: str) -> None:
        self.run_git("add", "--all")
        self.run_git("commit", "--quiet", "--allow-empty", "--message", message)

    def move_to_sha_prefix(self, prefix_kind: str) -> str:
        for attempt in range(256):
            self.run_git(
                "commit",
                "--quiet",
                "--allow-empty",
                "--message",
                f"sha-prefix-{prefix_kind}-{attempt}",
            )
            short_sha = self.run_git("rev-parse", "--short", "HEAD")
            if prefix_kind == "alpha" and short_sha[0] in "abcdef":
                return short_sha
            if prefix_kind == "numeric" and short_sha[0].isdigit():
                return short_sha
        self.fail(f"Could not create a {prefix_kind}-leading commit SHA")

    def run_resolver(
        self, *, dpkg_exit: int = 0
    ) -> tuple[subprocess.CompletedProcess, Path]:
        fake_bin = Path(self.temporary_directory.name) / "fake-bin"
        fake_bin.mkdir(exist_ok=True)
        validation_log = Path(self.temporary_directory.name) / "dpkg-arguments"
        fake_dpkg = fake_bin / "dpkg"
        fake_dpkg.write_text(
            "#!/bin/sh\n"
            'printf "%s\\n" "$@" > "$DPKG_VALIDATION_LOG"\n'
            'exit "$DPKG_EXIT"\n',
            encoding="utf-8",
        )
        fake_dpkg.chmod(fake_dpkg.stat().st_mode | stat.S_IXUSR)

        env = os.environ.copy()
        env.update(
            {
                "DPKG_EXIT": str(dpkg_exit),
                "DPKG_VALIDATION_LOG": str(validation_log),
                "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            }
        )
        result = subprocess.run(
            ["bash", str(RESOLVER), "24.04"],
            cwd=self.repo,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        return result, validation_log

    def assert_tagless_version(self, prefix_kind: str) -> None:
        self.write_cmake_version("project(lemon_cpp VERSION 11.9.0)\n")
        self.commit_all("initial")
        short_sha = self.move_to_sha_prefix(prefix_kind)
        commit_count = int(self.run_git("rev-list", "--count", "HEAD"))

        result, validation_log = self.run_resolver()

        expected = f"11.9.0+git{commit_count}.g{short_sha}"
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), expected)
        self.assertEqual(
            validation_log.read_text(encoding="utf-8").splitlines(),
            ["--validate-version", f"{expected}~24.04"],
        )

    def test_tagless_alpha_leading_sha_uses_cmake_version_prefix(self) -> None:
        self.assert_tagless_version("alpha")

    def test_tagless_numeric_leading_sha_uses_cmake_version_prefix(self) -> None:
        self.assert_tagless_version("numeric")

    def test_tagless_versions_increase_with_commit_ancestry(self) -> None:
        self.write_cmake_version("project(lemon_cpp VERSION 11.9.0)\n")
        self.commit_all("parent")
        parent_sha = self.run_git("rev-parse", "--short", "HEAD")
        parent_count = int(self.run_git("rev-list", "--count", "HEAD"))
        parent_result, _validation_log = self.run_resolver()

        self.run_git("commit", "--quiet", "--allow-empty", "--message", "child")
        child_sha = self.run_git("rev-parse", "--short", "HEAD")
        child_count = int(self.run_git("rev-list", "--count", "HEAD"))
        child_result, _validation_log = self.run_resolver()

        parent_version = f"11.9.0+git{parent_count}.g{parent_sha}"
        child_version = f"11.9.0+git{child_count}.g{child_sha}"
        self.assertEqual(parent_result.returncode, 0, parent_result.stderr)
        self.assertEqual(child_result.returncode, 0, child_result.stderr)
        self.assertEqual(parent_result.stdout.strip(), parent_version)
        self.assertEqual(child_result.stdout.strip(), child_version)
        self.assertGreater(child_count, parent_count)

        dpkg = shutil.which("dpkg")
        if dpkg:
            subprocess.run(
                [dpkg, "--compare-versions", parent_version, "lt", child_version],
                check=True,
            )
            subprocess.run(
                [dpkg, "--compare-versions", "11.9.0", "lt", parent_version],
                check=True,
            )

    def test_reachable_tag_preserves_git_describe_version(self) -> None:
        self.write_cmake_version("project(lemon_cpp VERSION 11.9.0)\n")
        self.commit_all("tagged release")
        self.run_git("tag", "--no-sign", "v4.5.6")
        self.run_git("commit", "--quiet", "--allow-empty", "--message", "after tag")
        described = self.run_git("describe", "--tags", "--always")

        result, validation_log = self.run_resolver()

        expected = described.removeprefix("v")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), expected)
        self.assertEqual(
            validation_log.read_text(encoding="utf-8").splitlines(),
            ["--validate-version", f"{expected}~24.04"],
        )

    def test_tagless_repo_rejects_missing_or_malformed_cmake_version(self) -> None:
        for name, cmake_contents in (
            ("missing", None),
            ("malformed", "project(lemon_cpp VERSION next)\n"),
        ):
            with self.subTest(name=name):
                if cmake_contents is None:
                    (self.repo / "README.md").write_text("test\n", encoding="utf-8")
                else:
                    self.write_cmake_version(cmake_contents)
                self.commit_all(name)

                result, validation_log = self.run_resolver()

                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                self.assertFalse(validation_log.exists())

    def test_dpkg_validation_failure_is_propagated(self) -> None:
        self.write_cmake_version("project(lemon_cpp VERSION 11.9.0)\n")
        self.commit_all("initial")

        result, validation_log = self.run_resolver(dpkg_exit=42)

        self.assertEqual(result.returncode, 42)
        self.assertEqual(result.stdout, "")
        self.assertTrue(validation_log.exists())


if __name__ == "__main__":
    unittest.main()
