#!/usr/bin/env python3
"""Regression tests for llama.cpp update publication recovery."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / ".github" / "scripts" / "publish_llamacpp_update.sh"
VERSIONS_PATH = Path("src/cpp/resources/backend_versions.json")


class PublishLlamaCppUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temp_root = Path(self.temporary_directory.name)
        self.repo = self.temp_root / "repo"
        self.remote = self.temp_root / "remote.git"
        self.fake_bin = self.temp_root / "fake-bin"
        self.event_log = self.temp_root / "events.log"
        self.real_git = shutil.which("git")
        if not self.real_git:
            self.skipTest("git is required")

        self.run_command([self.real_git, "init", "--quiet", "--bare", self.remote])
        self.repo.mkdir()
        self.run_git("init", "--quiet")
        self.run_git("config", "user.name", "Test Author")
        self.run_git("config", "user.email", "test@example.com")
        versions = self.repo / VERSIONS_PATH
        versions.parent.mkdir(parents=True)
        versions.write_text(
            '{"llamacpp":{"vulkan":"b1","cpu":"b1"}}\n', encoding="utf-8"
        )
        (self.repo / "pr_body.md").write_text("initial body\n", encoding="utf-8")
        self.run_git("add", "--all")
        self.run_git("commit", "--quiet", "--message", "base")
        self.run_git("branch", "-M", "main")
        self.run_git("remote", "add", "origin", str(self.remote))
        self.run_git("push", "--quiet", "--set-upstream", "origin", "main")
        self.install_command_wrappers()

    @property
    def branch_name(self) -> str:
        base_sha = self.run_git("rev-parse", "HEAD")
        versions_blob = self.run_git("hash-object", str(VERSIONS_PATH))
        return f"auto/llamacpp-update-b100-b200-b300-{base_sha}-{versions_blob}"

    def run_command(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=check,
            capture_output=True,
            text=True,
        )

    def run_git(self, *args: str) -> str:
        return self.run_command(
            [self.real_git, *args],
            cwd=self.repo,
        ).stdout.strip()

    def install_command_wrappers(self) -> None:
        self.fake_bin.mkdir()
        git_wrapper = self.fake_bin / "git"
        git_wrapper.write_text(
            "#!/bin/sh\n"
            "printf 'git' >> \"$EVENT_LOG\"\n"
            'printf \' <%s>\' "$@" >> "$EVENT_LOG"\n'
            "printf '\\n' >> \"$EVENT_LOG\"\n"
            'exec "$REAL_GIT" "$@"\n',
            encoding="utf-8",
        )
        git_wrapper.chmod(git_wrapper.stat().st_mode | stat.S_IXUSR)

        gh_wrapper = self.fake_bin / "gh"
        gh_wrapper.write_text(
            "#!/bin/sh\n"
            "printf 'gh' >> \"$EVENT_LOG\"\n"
            'printf \' <%s>\' "$@" >> "$EVENT_LOG"\n'
            "printf '\\n' >> \"$EVENT_LOG\"\n"
            'if [ "$1" = api ]; then\n'
            '    case "$2" in\n'
            "        repos/ggml-org/llama.cpp/releases) "
            "printf '%s\\n' \"$FRESH_GGML_RELEASE\" ;;\n"
            "        repos/lemonade-sdk/llamacpp-rocm/releases/latest) "
            "printf '%s\\n' \"$FRESH_ROCM_RELEASE\" ;;\n"
            "        repos/lemonade-sdk/llama.cpp/releases/latest) "
            "printf '%s\\n' \"$FRESH_LEMONADE_RELEASE\" ;;\n"
            "        repos/LLM360/lemonade/pulls) "
            "printf '%s\\n' \"$OPEN_PR_RECORDS\" ;;\n"
            "    esac\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        gh_wrapper.chmod(gh_wrapper.stat().st_mode | stat.S_IXUSR)

    def publisher_env(
        self,
        *,
        open_pr_records: str = "",
        fresh_ggml_release: str = "b100",
        validated_base_sha: str | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "EVENT_LOG": str(self.event_log),
                "FRESH_GGML_RELEASE": fresh_ggml_release,
                "FRESH_LEMONADE_RELEASE": "b200",
                "FRESH_ROCM_RELEASE": "b300",
                "GITHUB_REPOSITORY": "LLM360/lemonade",
                "GITHUB_REPOSITORY_OWNER": "LLM360",
                "LLAMACPP_LEMONADE_RELEASE": "b200",
                "LLAMACPP_RELEASE": "b100",
                "LLAMACPP_ROCM_RELEASE": "b300",
                "OPEN_PR_RECORDS": open_pr_records,
                "PATH": f"{self.fake_bin}{os.pathsep}{env['PATH']}",
                "PR_BODY_FILE": "pr_body.md",
                "REAL_GIT": self.real_git,
                "VALIDATED_BASE_SHA": validated_base_sha
                or self.run_git("rev-parse", "HEAD"),
            }
        )
        return env

    def run_publisher(
        self,
        *,
        open_pr_records: str = "",
        fresh_ggml_release: str = "b100",
        validated_base_sha: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self.run_command(
            ["bash", str(PUBLISHER)],
            cwd=self.repo,
            env=self.publisher_env(
                open_pr_records=open_pr_records,
                fresh_ggml_release=fresh_ggml_release,
                validated_base_sha=validated_base_sha,
            ),
            check=False,
        )

    def change_versions(self) -> None:
        (self.repo / VERSIONS_PATH).write_text(
            '{"llamacpp":{"vulkan":"b100","cpu":"b100"}}\n',
            encoding="utf-8",
        )

    def trusted_open_pr_record(self, number: int = 17) -> str:
        return (
            f"{number}\t{self.branch_name}\tLLM360/lemonade\t"
            "github-actions[bot]\tmain"
        )

    def create_remote_update_branch(
        self,
        versions_contents: str,
        *,
        unrelated_contents: str | None = None,
    ) -> tuple[str, str, str]:
        base_oid = self.run_git("rev-parse", "HEAD")
        (self.repo / VERSIONS_PATH).write_text(versions_contents, encoding="utf-8")
        if unrelated_contents is not None:
            (self.repo / "unrelated.txt").write_text(
                unrelated_contents, encoding="utf-8"
            )
        branch_name = self.branch_name
        self.run_git("add", "--all")
        self.run_git("commit", "--quiet", "--message", "automated update")
        branch_oid = self.run_git("rev-parse", "HEAD")
        self.run_git(
            "push",
            "--quiet",
            "origin",
            f"HEAD:refs/heads/{branch_name}",
        )
        self.run_git("reset", "--quiet", "--hard", base_oid)
        return base_oid, branch_name, branch_oid

    def events(self) -> list[str]:
        if not self.event_log.exists():
            return []
        return self.event_log.read_text(encoding="utf-8").splitlines()

    def test_unchanged_versions_reconcile_before_exiting(self) -> None:
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\tgithub-actions[bot]\tmain"
        )

        result = self.run_publisher(open_pr_records=stale_record)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nothing to update", result.stdout)
        events = self.events()
        self.assertTrue(
            any(
                event.startswith("gh <api> <repos/LLM360/lemonade/pulls/8>")
                and "<-f> <state=closed>" in event
                for event in events
            )
        )
        self.assertFalse(any(event.startswith("git <push>") for event in events))

    def test_untrusted_stale_pull_request_is_not_closed(self) -> None:
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\toctocat\tmain"
        )

        result = self.run_publisher(open_pr_records=stale_record)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(
            any("<repos/LLM360/lemonade/pulls/8>" in event for event in self.events())
        )

    def test_wrong_base_stale_pull_request_is_not_closed(self) -> None:
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\tgithub-actions[bot]\trelease"
        )

        result = self.run_publisher(open_pr_records=stale_record)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(
            any("<repos/LLM360/lemonade/pulls/8>" in event for event in self.events())
        )

    def test_stale_candidate_fails_before_query_or_push(self) -> None:
        self.change_versions()

        result = self.run_publisher(fresh_ggml_release="b101")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("candidate releases changed", result.stderr)
        events = self.events()
        self.assertFalse(
            any("<repos/LLM360/lemonade/pulls>" in event for event in events)
        )
        self.assertFalse(any(event.startswith("git <push>") for event in events))

    def test_mismatched_validated_base_is_rejected_before_publication(self) -> None:
        self.change_versions()

        result = self.run_publisher(validated_base_sha="0" * 40)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("validated base", result.stderr.lower())
        self.assertFalse(
            any("<repos/LLM360/lemonade/pulls>" in event for event in self.events())
        )

    def test_remote_base_advance_is_rejected_before_publication(self) -> None:
        validated_base = self.run_git("rev-parse", "HEAD")
        (self.repo / "main.txt").write_text("new main content\n", encoding="utf-8")
        self.run_git("add", "main.txt")
        self.run_git("commit", "--quiet", "--message", "advance main")
        self.run_git("push", "--quiet", "origin", "main")
        self.run_git("reset", "--quiet", "--hard", validated_base)
        self.change_versions()

        result = self.run_publisher(validated_base_sha=validated_base)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("advanced after validation", result.stderr)
        self.assertFalse(
            any("<repos/LLM360/lemonade/pulls>" in event for event in self.events())
        )

    def test_trusted_open_pr_is_refreshed_on_the_same_validated_base(self) -> None:
        _base_oid, branch_name, observed_oid = self.create_remote_update_branch(
            '{"llamacpp":{"vulkan":"b100","cpu":"b100"}}\n'
        )
        self.change_versions()

        result = self.run_publisher(
            open_pr_records=self.trusted_open_pr_record(),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        query_index = next(
            index
            for index, event in enumerate(events)
            if "<repos/LLM360/lemonade/pulls>" in event
        )
        push_index = next(
            index
            for index, event in enumerate(events)
            if event.startswith("git <push>")
        )
        edit_index = next(
            index
            for index, event in enumerate(events)
            if event.startswith("gh <pr> <edit>")
        )
        self.assertLess(query_index, push_index)
        self.assertLess(push_index, edit_index)
        self.assertIn("<state=open>", events[query_index])
        self.assertNotIn("<base=main>", events[query_index])
        self.assertIn(".base.ref", events[query_index])
        self.assertIn(
            f"<--force-with-lease=refs/heads/{branch_name}:{observed_oid}>",
            events[push_index],
        )
        self.assertIn(
            f"<{observed_oid}:refs/heads/{branch_name}>",
            events[push_index],
        )
        self.assertNotIn(f"<HEAD:refs/heads/{branch_name}>", events[push_index])
        self.assertIn("<17>", events[edit_index])
        self.assertIn("<--repo> <LLM360/lemonade>", events[edit_index])
        self.assertIn("<--body-file> <pr_body.md>", events[edit_index])
        self.assertFalse(any(event.startswith("gh <pr> <create>") for event in events))

    def test_trusted_open_pr_with_unrelated_changes_is_not_overwritten(self) -> None:
        _base_oid, branch_name, _branch_oid = self.create_remote_update_branch(
            '{"llamacpp":{"vulkan":"b100","cpu":"b100"}}\n',
            unrelated_contents="do not overwrite\n",
        )
        self.change_versions()

        record = f"17\t{branch_name}\tLLM360/lemonade\tgithub-actions[bot]\tmain"
        result = self.run_publisher(open_pr_records=record)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("changes unexpected path unrelated.txt", result.stderr)
        self.assertFalse(any(event.startswith("git <push>") for event in self.events()))

    def test_untrusted_open_pr_does_not_authorize_branch_overwrite(self) -> None:
        _base_oid, branch_name, branch_oid = self.create_remote_update_branch(
            '{"llamacpp":{"vulkan":"b100","cpu":"b100"}}\n'
        )
        self.change_versions()
        untrusted_record = f"17\t{branch_name}\tLLM360/lemonade\toctocat\tmain"

        result = self.run_publisher(open_pr_records=untrusted_record)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("untrusted open pull request", result.stderr)
        self.assertFalse(any(event.startswith("git <push>") for event in self.events()))

    def test_cross_repository_same_named_pr_cannot_block_publication(self) -> None:
        _base_oid, branch_name, branch_oid = self.create_remote_update_branch(
            '{"llamacpp":{"vulkan":"b100","cpu":"b100"}}\n'
        )
        self.change_versions()
        cross_repo_record = (
            f"17\t{branch_name}\tattacker/lemonade\tgithub-actions[bot]\tmain"
        )

        result = self.run_publisher(open_pr_records=cross_repo_record)

        self.assertEqual(result.returncode, 0, result.stderr)
        push = next(event for event in self.events() if event.startswith("git <push>"))
        self.assertIn(
            f"<--force-with-lease=refs/heads/{branch_name}:{branch_oid}>",
            push,
        )
        self.assertIn(f"<{branch_oid}:refs/heads/{branch_name}>", push)
        self.assertNotIn(f"<HEAD:refs/heads/{branch_name}>", push)
        create = next(
            event for event in self.events() if event.startswith("gh <pr> <create>")
        )
        self.assertIn("<--repo> <LLM360/lemonade>", create)
        self.assertIn(f"<--head> <{branch_name}>", create)
        self.assertNotIn(f"<LLM360:{branch_name}>", create)

    def test_wrong_base_pull_request_is_not_trusted(self) -> None:
        _base_oid, branch_name, branch_oid = self.create_remote_update_branch(
            '{"llamacpp":{"vulkan":"b100","cpu":"b100"}}\n'
        )
        self.change_versions()
        wrong_base_record = (
            f"17\t{branch_name}\tLLM360/lemonade\t" "github-actions[bot]\trelease"
        )

        result = self.run_publisher(open_pr_records=wrong_base_record)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("untrusted open pull request", result.stderr)
        events = self.events()
        self.assertFalse(any(event.startswith("git <push>") for event in events))

    def test_evolved_orphan_uses_a_new_content_addressed_branch(self) -> None:
        _base_oid, old_branch, _branch_oid = self.create_remote_update_branch(
            '{"llamacpp":{"vulkan":"b100","cpu":"b1"}}\n'
        )
        self.change_versions()
        new_branch = self.branch_name

        result = self.run_publisher()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(old_branch, new_branch)
        push = next(event for event in self.events() if event.startswith("git <push>"))
        self.assertIn(f"<HEAD:refs/heads/{new_branch}>", push)
        self.assertTrue(
            any(event.startswith("gh <pr> <create>") for event in self.events())
        )

    def test_base_advance_uses_a_new_branch_and_closes_the_old_pull_request(
        self,
    ) -> None:
        _base_oid, old_branch, _branch_oid = self.create_remote_update_branch(
            '{"llamacpp":{"vulkan":"b100","cpu":"b100"}}\n'
        )
        (self.repo / "main.txt").write_text("new main content\n", encoding="utf-8")
        self.run_git("add", "main.txt")
        self.run_git("commit", "--quiet", "--message", "advance main")
        self.run_git("push", "--quiet", "origin", "main")
        self.change_versions()
        new_branch = self.branch_name
        old_record = f"17\t{old_branch}\tLLM360/lemonade\tgithub-actions[bot]\tmain"

        result = self.run_publisher(open_pr_records=old_record)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(old_branch, new_branch)
        events = self.events()
        self.assertTrue(
            any("<repos/LLM360/lemonade/pulls/17>" in event for event in events)
        )
        push = next(event for event in events if event.startswith("git <push>"))
        self.assertIn(f"<HEAD:refs/heads/{new_branch}>", push)

    def test_base_advance_does_not_leave_an_orphan_branch_blocker(self) -> None:
        _base_oid, old_branch, _branch_oid = self.create_remote_update_branch(
            '{"llamacpp":{"vulkan":"b100","cpu":"b100"}}\n'
        )
        (self.repo / "main.txt").write_text("new main content\n", encoding="utf-8")
        self.run_git("add", "main.txt")
        self.run_git("commit", "--quiet", "--message", "advance main")
        self.run_git("push", "--quiet", "origin", "main")
        self.change_versions()
        new_branch = self.branch_name

        result = self.run_publisher()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(old_branch, new_branch)
        push = next(event for event in self.events() if event.startswith("git <push>"))
        self.assertIn(f"<HEAD:refs/heads/{new_branch}>", push)

    def test_matching_historical_branch_is_reused_for_a_new_pr(self) -> None:
        _base_oid, branch_name, branch_oid = self.create_remote_update_branch(
            '{"llamacpp":{"vulkan":"b100","cpu":"b100"}}\n'
        )
        self.change_versions()

        result = self.run_publisher()

        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        query = next(
            event for event in events if "<repos/LLM360/lemonade/pulls>" in event
        )
        self.assertIn("<state=open>", query)
        self.assertNotIn("<base=main>", query)
        push = next(event for event in events if event.startswith("git <push>"))
        self.assertIn(
            f"<--force-with-lease=refs/heads/{branch_name}:{branch_oid}>",
            push,
        )
        self.assertTrue(any(event.startswith("gh <pr> <create>") for event in events))
        self.assertFalse(any(event.startswith("gh <pr> <edit>") for event in events))

    def test_absent_branch_uses_an_explicit_empty_lease(self) -> None:
        self.change_versions()
        branch_name = self.branch_name

        result = self.run_publisher()

        self.assertEqual(result.returncode, 0, result.stderr)
        push = next(event for event in self.events() if event.startswith("git <push>"))
        self.assertIn(
            f"<--force-with-lease=refs/heads/{branch_name}:>",
            push,
        )
        self.assertTrue(
            any(event.startswith("gh <pr> <create>") for event in self.events())
        )

    def test_shallow_checkout_can_refresh_an_existing_pull_request(self) -> None:
        _base_oid, branch_name, _branch_oid = self.create_remote_update_branch(
            '{"llamacpp":{"vulkan":"b100","cpu":"b100"}}\n'
        )

        shallow_repo = self.temp_root / "shallow-repo"
        self.run_command(
            [
                self.real_git,
                "clone",
                "--quiet",
                "--depth=1",
                "--branch",
                "main",
                f"file://{self.remote}",
                shallow_repo,
            ]
        )
        self.repo = shallow_repo
        self.change_versions()
        record = f"17\t{branch_name}\tLLM360/lemonade\tgithub-actions[bot]\tmain"

        result = self.run_publisher(open_pr_records=record)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.run_git("rev-parse", "--is-shallow-repository"), "false")
        self.assertTrue(
            any(event.startswith("gh <pr> <edit>") for event in self.events())
        )


if __name__ == "__main__":
    unittest.main()
