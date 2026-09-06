#!/usr/bin/env python3
"""Regression tests for llama.cpp update publication recovery."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from test.utils import llamacpp_release_manifest as release_manifest

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / ".github" / "scripts" / "publish_llamacpp_update.sh"
VERSIONS_PATH = Path("src/cpp/resources/backend_versions.json")
MANIFEST_PATH = Path(".github/llamacpp_release_manifest.json")


class PublishLlamaCppUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temp_root = Path(self.temporary_directory.name)
        self.repo = self.temp_root / "repo"
        self.remote = self.temp_root / "remote.git"
        self.fake_bin = self.temp_root / "fake-bin"
        self.event_log = self.temp_root / "events.log"
        self.snapshot_count = self.temp_root / "snapshot-count.txt"
        self.closed_prs = self.temp_root / "closed-prs.txt"
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
        self.upstream_head = "f" * 40
        self.live_upstream_head = "9" * 40
        self.release_payloads = {
            "ggml": self.release_payload(
                repository="ggml-org/llama.cpp",
                tag="b100",
                release_id=100,
                release_tag_commit="a" * 40,
                claimed_source_commit="a" * 40,
            ),
            "rocm": self.release_payload(
                repository="lemonade-sdk/llamacpp-rocm",
                tag="b300",
                release_id=300,
                release_tag_commit="b" * 40,
                claimed_source_commit="c" * 40,
            ),
            "lemonade": self.release_payload(
                repository="lemonade-sdk/llama.cpp",
                tag="b200",
                release_id=200,
                release_tag_commit="d" * 40,
                claimed_source_commit="e" * 40,
            ),
        }
        self.expected_manifest = self.build_expected_manifest()
        manifest_path = self.repo / MANIFEST_PATH
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(self.expected_manifest + "\n", encoding="utf-8")
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
        manifest_digest = hashlib.sha256(self.expected_manifest.encode()).hexdigest()
        return (
            f"auto/llamacpp-update-b100-b200-b300-{base_sha}-{versions_blob}-"
            f"{manifest_digest}"
        )

    def release_payload(
        self,
        *,
        repository: str,
        tag: str,
        release_id: int,
        release_tag_commit: str,
        claimed_source_commit: str,
    ) -> dict:
        binary_asset = {
            "digest": "sha256:" + f"{release_id:064x}"[-64:],
            "id": release_id * 10,
            "name": f"asset-{release_id}.zip",
            "size": release_id,
            "state": "uploaded",
            "updated_at": "2026-09-06T12:00:00Z",
        }
        assets = [binary_asset]
        source_manifest = None
        if repository != "ggml-org/llama.cpp":
            source_manifest = json.dumps(
                {
                    "assets": [
                        {
                            "digest": binary_asset["digest"],
                            "name": binary_asset["name"],
                            "size": binary_asset["size"],
                        }
                    ],
                    "release_repository": repository,
                    "release_tag": tag,
                    "release_tag_commit": release_tag_commit,
                    "schema_version": 1,
                    "source_commit": claimed_source_commit,
                    "source_repository": "ggml-org/llama.cpp",
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            source_bytes = source_manifest.encode("utf-8")
            assets.append(
                {
                    "digest": "sha256:" + hashlib.sha256(source_bytes).hexdigest(),
                    "id": release_id * 10 + 1,
                    "name": ".llamacpp-source.json",
                    "size": len(source_bytes),
                    "state": "uploaded",
                    "updated_at": "2026-09-06T12:01:00Z",
                }
            )
        return {
            "assets": assets,
            "body": "release notes",
            "draft": False,
            "id": release_id,
            "immutable": True,
            "tag_name": tag,
            "_claimed_source_commit": claimed_source_commit,
            "_release_tag_commit": release_tag_commit,
            "_source_manifest": source_manifest,
        }

    def build_expected_manifest(self) -> str:
        releases = []
        for key, repository, tag in (
            ("ggml", "ggml-org/llama.cpp", "b100"),
            ("rocm", "lemonade-sdk/llamacpp-rocm", "b300"),
            ("lemonade", "lemonade-sdk/llama.cpp", "b200"),
        ):
            payload = dict(self.release_payloads[key])
            claimed_commit = payload.pop("_claimed_source_commit")
            release_tag_commit = payload.pop("_release_tag_commit")
            payload.pop("_source_manifest")
            payload.update(
                {
                    "publisher_claim_type": (
                        "source-release-tag"
                        if repository == "ggml-org/llama.cpp"
                        else "immutable-source-manifest"
                    ),
                    "publisher_claimed_source_commit": claimed_commit,
                    "release_tag_commit": release_tag_commit,
                    "source_repository": "ggml-org/llama.cpp",
                    "upstream_reference": "refs/heads/master",
                    "upstream_reference_head": self.upstream_head,
                }
            )
            releases.append((repository, tag, payload))
        return release_manifest.build_release_asset_manifest(releases)

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
            'if [ "$1" = push ] && '
            '[ "$MUTATE_BRANCH_AFTER_PUSH" = true ]; then\n'
            '    "$REAL_GIT" "$@" || exit $?\n'
            '    "$REAL_GIT" --git-dir="$REMOTE_REPOSITORY_PATH" update-ref '
            '"refs/heads/$EXPECTED_BRANCH" "$VALIDATED_BASE_SHA"\n'
            "    exit $?\n"
            "fi\n"
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
            'if [ "$1" = pr ] && [ "$2" = create ]; then\n'
            "    printf 'https://github.com/LLM360/lemonade/pull/%s\\n' "
            '"$CREATED_PR_NUMBER"\n'
            "    exit 0\n"
            "fi\n"
            'if [ "$1" = api ]; then\n'
            '    case "$2" in\n'
            "        repos/ggml-org/llama.cpp) printf '%s\\n' master ;;\n"
            "        repos/ggml-org/llama.cpp/releases) "
            "printf '%s\\n' \"$FRESH_GGML_RELEASE\" ;;\n"
            "        repos/lemonade-sdk/llamacpp-rocm/releases/latest) "
            "printf '%s\\n' \"$FRESH_ROCM_RELEASE\" ;;\n"
            "        repos/lemonade-sdk/llama.cpp/releases/latest) "
            "printf '%s\\n' \"$FRESH_LEMONADE_RELEASE\" ;;\n"
            "        repos/ggml-org/llama.cpp/releases/tags/b100)\n"
            "            snapshot=0\n"
            '            if [ -f "$SNAPSHOT_COUNT_FILE" ]; then '
            'snapshot=$(cat "$SNAPSHOT_COUNT_FILE"); fi\n'
            "            snapshot=$((snapshot + 1))\n"
            '            printf \'%s\\n\' "$snapshot" > "$SNAPSHOT_COUNT_FILE"\n'
            "            printf '%s\\n' \"$GGML_RELEASE_PAYLOAD\" ;;\n"
            "        repos/lemonade-sdk/llamacpp-rocm/releases/tags/b300)\n"
            '            if [ "$MUTATE_ON_SNAPSHOT" -gt 0 ] && '
            '[ "$(cat "$SNAPSHOT_COUNT_FILE")" -ge "$MUTATE_ON_SNAPSHOT" ]; then\n'
            "                printf '%s\\n' \"$MUTATED_ROCM_RELEASE_PAYLOAD\"\n"
            "            else\n"
            "                printf '%s\\n' \"$ROCM_RELEASE_PAYLOAD\"\n"
            "            fi ;;\n"
            "        repos/lemonade-sdk/llama.cpp/releases/tags/b200) "
            "printf '%s\\n' \"$LEMONADE_RELEASE_PAYLOAD\" ;;\n"
            "        repos/lemonade-sdk/llamacpp-rocm/releases/assets/3001) "
            "printf '%s' \"$ROCM_SOURCE_MANIFEST\" ;;\n"
            "        repos/lemonade-sdk/llama.cpp/releases/assets/2001) "
            "printf '%s' \"$LEMONADE_SOURCE_MANIFEST\" ;;\n"
            "        repos/ggml-org/llama.cpp/commits/b100) "
            "printf '%s\\n' \"$GGML_RELEASE_TAG_COMMIT\" ;;\n"
            "        repos/lemonade-sdk/llamacpp-rocm/commits/b300) "
            "printf '%s\\n' \"$ROCM_RELEASE_TAG_COMMIT\" ;;\n"
            "        repos/lemonade-sdk/llama.cpp/commits/b200) "
            "printf '%s\\n' \"$LEMONADE_RELEASE_TAG_COMMIT\" ;;\n"
            "        repos/ggml-org/llama.cpp/commits/master) "
            "printf '%s\\n' \"$LIVE_UPSTREAM_HEAD\" ;;\n"
            "        repos/ggml-org/llama.cpp/compare/*)\n"
            "            comparison=${2##*/}; base=${comparison%%...*}; "
            "head=${comparison#*...}\n"
            "            status=ahead; merge_base=$base\n"
            '            if [ -n "$DIVERGENT_COMMIT" ] && '
            '[ "$base" = "$DIVERGENT_COMMIT" ]; then\n'
            "                status=diverged; merge_base=$LIVE_UPSTREAM_HEAD\n"
            "            fi\n"
            "            printf "
            '\'{"status":"%s","base_commit":{"sha":"%s"},'
            '"merge_base_commit":{"sha":"%s"},'
            '"url":"https://api.github.com/repos/ggml-org/llama.cpp/compare/%s...%s"}\\n\' '
            '"$status" "$base" "$merge_base" "$base" "$head" ;;\n'
            "        repos/LLM360/lemonade/pulls) "
            "printf '%s\\n' \"$OPEN_PR_RECORDS\" ;;\n"
            "        repos/LLM360/lemonade/pulls/[0-9]*)\n"
            "            number=${2##*/}\n"
            '            case " $* " in\n'
            '                *" --method PATCH "*" state=closed "*)\n'
            '                    printf \'%s\\n\' "$number" >> "$CLOSED_PRS_FILE"\n'
            "                    printf '{}\\n'\n"
            "                    exit 0 ;;\n"
            "            esac\n"
            "            record=$(printf '%s\\n' \"$OPEN_PR_RECORDS\" | "
            "awk -F '\\t' -v number=\"$number\" '$1 == number { print; exit }')\n"
            "            head_ref=$(printf '%s' \"$record\" | cut -f2)\n"
            "            head_repository=$(printf '%s' \"$record\" | cut -f3)\n"
            "            author=$(printf '%s' \"$record\" | cut -f4)\n"
            "            base_ref=$(printf '%s' \"$record\" | cut -f5)\n"
            "            head_ref=${head_ref:-$EXPECTED_BRANCH}\n"
            "            head_repository=${head_repository:-LLM360/lemonade}\n"
            "            author=${author:-github-actions[bot]}\n"
            "            base_ref=${base_ref:-main}\n"
            "            head_ref=${PR_DETAIL_HEAD_REF:-$head_ref}\n"
            "            head_repository=${PR_DETAIL_HEAD_REPOSITORY:-$head_repository}\n"
            "            author=${PR_DETAIL_AUTHOR:-$author}\n"
            "            base_ref=${PR_DETAIL_BASE_REF:-$base_ref}\n"
            "            state=open\n"
            '            if [ -f "$CLOSED_PRS_FILE" ] && '
            'grep -qx "$number" "$CLOSED_PRS_FILE"; then state=closed; fi\n'
            '            head_sha=$("$REAL_GIT" --git-dir="$REMOTE_REPOSITORY_PATH" '
            'show-ref --verify --hash "refs/heads/$head_ref" 2>/dev/null || '
            "printf '%040d' 0)\n"
            "            head_sha=${PR_DETAIL_HEAD_SHA:-$head_sha}\n"
            '            base_sha=$("$REAL_GIT" --git-dir="$REMOTE_REPOSITORY_PATH" '
            'show-ref --verify --hash "refs/heads/$base_ref" 2>/dev/null || '
            "printf '%040d' 0)\n"
            "            printf "
            '\'{"state":"%s","head":{"ref":"%s",'
            '"sha":"%s","repo":{"full_name":"%s"}},'
            '"user":{"login":"%s"},"base":{"ref":"%s",'
            '"sha":"%s"}}\\n\' '
            '"$state" "$head_ref" "$head_sha" "$head_repository" '
            '"$author" "$base_ref" "$base_sha" ;;\n'
            "    esac\n"
            "fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        gh_wrapper.chmod(gh_wrapper.stat().st_mode | stat.S_IXUSR)

    def publisher_env(
        self,
        *,
        divergent_commit: str = "",
        include_expected_manifest: bool = True,
        mutate_on_snapshot: int = 0,
        open_pr_records: str = "",
        fresh_ggml_release: str = "b100",
        mutate_branch_after_push: bool = False,
        pr_detail_base_ref: str = "",
        pr_detail_head_ref: str = "",
        validated_base_sha: str | None = None,
    ) -> dict[str, str]:
        api_payloads = {}
        for name, payload in self.release_payloads.items():
            api_payloads[name] = {
                key: value for key, value in payload.items() if not key.startswith("_")
            }
        mutated_rocm = dict(api_payloads["rocm"])
        mutated_rocm["assets"] = [dict(asset) for asset in mutated_rocm["assets"]]
        mutated_rocm["assets"][0]["updated_at"] = "2026-09-06T12:02:00Z"
        env = os.environ.copy()
        env.update(
            {
                "DIVERGENT_COMMIT": divergent_commit,
                "CLOSED_PRS_FILE": str(self.closed_prs),
                "CREATED_PR_NUMBER": "41",
                "EVENT_LOG": str(self.event_log),
                "EXPECTED_BRANCH": self.branch_name,
                "FRESH_GGML_RELEASE": fresh_ggml_release,
                "FRESH_LEMONADE_RELEASE": "b200",
                "FRESH_ROCM_RELEASE": "b300",
                "GGML_RELEASE_PAYLOAD": json.dumps(api_payloads["ggml"]),
                "GGML_RELEASE_TAG_COMMIT": self.release_payloads["ggml"][
                    "_release_tag_commit"
                ],
                "GITHUB_REPOSITORY": "LLM360/lemonade",
                "GITHUB_REPOSITORY_OWNER": "LLM360",
                "LEMONADE_RELEASE_PAYLOAD": json.dumps(api_payloads["lemonade"]),
                "LEMONADE_SOURCE_MANIFEST": self.release_payloads["lemonade"][
                    "_source_manifest"
                ],
                "LEMONADE_RELEASE_TAG_COMMIT": self.release_payloads["lemonade"][
                    "_release_tag_commit"
                ],
                "LLAMACPP_LEMONADE_RELEASE": "b200",
                "LLAMACPP_RELEASE": "b100",
                "LLAMACPP_ROCM_RELEASE": "b300",
                "LIVE_UPSTREAM_HEAD": self.live_upstream_head,
                "MUTATED_ROCM_RELEASE_PAYLOAD": json.dumps(mutated_rocm),
                "MUTATE_ON_SNAPSHOT": str(mutate_on_snapshot),
                "MUTATE_BRANCH_AFTER_PUSH": (
                    "true" if mutate_branch_after_push else "false"
                ),
                "OPEN_PR_RECORDS": open_pr_records,
                "PATH": f"{self.fake_bin}{os.pathsep}{env['PATH']}",
                "PR_BODY_FILE": "pr_body.md",
                "PR_DETAIL_AUTHOR": "",
                "PR_DETAIL_BASE_REF": pr_detail_base_ref,
                "PR_DETAIL_HEAD_REF": pr_detail_head_ref,
                "PR_DETAIL_HEAD_REPOSITORY": "",
                "PR_DETAIL_HEAD_SHA": "",
                "REAL_GIT": self.real_git,
                "REMOTE_REPOSITORY_PATH": str(self.remote),
                "ROCM_RELEASE_PAYLOAD": json.dumps(api_payloads["rocm"]),
                "ROCM_SOURCE_MANIFEST": self.release_payloads["rocm"][
                    "_source_manifest"
                ],
                "ROCM_RELEASE_TAG_COMMIT": self.release_payloads["rocm"][
                    "_release_tag_commit"
                ],
                "SNAPSHOT_COUNT_FILE": str(self.snapshot_count),
                "VALIDATED_BASE_SHA": validated_base_sha
                or self.run_git("rev-parse", "HEAD"),
            }
        )
        if include_expected_manifest:
            env["EXPECTED_RELEASE_ASSET_MANIFEST"] = self.expected_manifest
        return env

    def run_publisher(
        self,
        *,
        divergent_commit: str = "",
        include_expected_manifest: bool = True,
        mutate_on_snapshot: int = 0,
        open_pr_records: str = "",
        fresh_ggml_release: str = "b100",
        mutate_branch_after_push: bool = False,
        pr_detail_base_ref: str = "",
        pr_detail_head_ref: str = "",
        validated_base_sha: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self.run_command(
            ["bash", str(PUBLISHER)],
            cwd=self.repo,
            env=self.publisher_env(
                divergent_commit=divergent_commit,
                include_expected_manifest=include_expected_manifest,
                mutate_on_snapshot=mutate_on_snapshot,
                open_pr_records=open_pr_records,
                fresh_ggml_release=fresh_ggml_release,
                mutate_branch_after_push=mutate_branch_after_push,
                pr_detail_base_ref=pr_detail_base_ref,
                pr_detail_head_ref=pr_detail_head_ref,
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

    def test_missing_release_manifest_fails_before_external_queries(self) -> None:
        self.change_versions()

        result = self.run_publisher(include_expected_manifest=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EXPECTED_RELEASE_ASSET_MANIFEST", result.stderr)
        self.assertEqual(self.events(), [])

    def test_release_mutation_at_final_recheck_prevents_remote_mutation(self) -> None:
        self.change_versions()
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\tgithub-actions[bot]\tmain"
        )

        result = self.run_publisher(
            mutate_on_snapshot=3,
            open_pr_records=stale_record,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("changed since validation", result.stderr)
        events = self.events()
        self.assertFalse(any(event.startswith("git <push>") for event in events))
        self.assertFalse(any(event.startswith("gh <pr>") for event in events))
        self.assertFalse(any("<-f> <state=closed>" in event for event in events))

    def test_mutable_release_prevents_remote_mutation(self) -> None:
        self.change_versions()
        self.release_payloads["rocm"]["immutable"] = False

        result = self.run_publisher()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("release immutable must be true", result.stderr)
        events = self.events()
        self.assertFalse(any(event.startswith("git <push>") for event in events))
        self.assertFalse(any(event.startswith("gh <pr>") for event in events))

    def test_corrupt_source_manifest_prevents_remote_mutation(self) -> None:
        self.change_versions()
        self.release_payloads["rocm"]["_source_manifest"] += " "

        result = self.run_publisher()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source manifest size does not match", result.stderr)
        events = self.events()
        self.assertFalse(any(event.startswith("git <push>") for event in events))
        self.assertFalse(any(event.startswith("gh <pr>") for event in events))

    def test_fork_network_manifest_anchor_fails_against_live_upstream(self) -> None:
        self.change_versions()

        result = self.run_publisher(divergent_commit=self.upstream_head)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not reachable", result.stderr)
        events = self.events()
        self.assertTrue(
            any(
                f"{self.upstream_head}...{self.live_upstream_head}>" in event
                for event in events
            )
        )
        self.assertFalse(any(event.startswith("git <push>") for event in events))
        self.assertFalse(any(event.startswith("gh <pr>") for event in events))

    def test_publication_binds_manifest_to_branch_commit_and_pr_body(self) -> None:
        self.change_versions()
        (self.repo / MANIFEST_PATH).write_text("tampered\n", encoding="utf-8")
        expected_branch = self.branch_name
        expected_digest = hashlib.sha256(
            self.expected_manifest.encode("utf-8")
        ).hexdigest()

        result = self.run_publisher()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.run_git("branch", "--show-current"), expected_branch)
        self.assertEqual(
            self.run_git("show", f"HEAD:{MANIFEST_PATH}"),
            self.expected_manifest,
        )
        self.assertIn(
            f"LlamaCpp-Release-Manifest-SHA256: {expected_digest}",
            self.run_git("log", "-1", "--pretty=%B"),
        )
        body = (self.repo / "pr_body.md").read_text(encoding="utf-8")
        self.assertIn(f"Canonical manifest SHA-256: `{expected_digest}`", body)
        self.assertIn(self.upstream_head, body)
        self.assertIn("`true`", body)
        self.assertIn("Only immutable GitHub releases are accepted", body)
        self.assertIn("immutable publisher manifest", body)
        self.assertIn("not cryptographic build provenance", body)
        for payload in self.release_payloads.values():
            self.assertIn(payload["_release_tag_commit"], body)
            self.assertIn(payload["_claimed_source_commit"], body)
        self.assertGreaterEqual(int(self.snapshot_count.read_text()), 3)

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

    def test_retargeted_open_pr_fails_before_refresh(self) -> None:
        self.create_remote_update_branch(
            '{"llamacpp":{"vulkan":"b100","cpu":"b100"}}\n'
        )
        self.change_versions()

        result = self.run_publisher(
            open_pr_records=self.trusted_open_pr_record(),
            pr_detail_base_ref="release",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("identity changed before publication", result.stderr)
        self.assertFalse(
            any(event.startswith("gh <pr> <edit>") for event in self.events())
        )

    def test_remote_branch_move_after_push_fails_before_pr_mutation(self) -> None:
        self.change_versions()

        result = self.run_publisher(mutate_branch_after_push=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("publication branch changed after push", result.stderr)
        self.assertFalse(
            any(
                event.startswith("gh <pr> <create>")
                or event.startswith("gh <pr> <edit>")
                for event in self.events()
            )
        )

    def test_retargeted_stale_pr_is_not_closed(self) -> None:
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\tgithub-actions[bot]\tmain"
        )

        result = self.run_publisher(
            open_pr_records=stale_record,
            pr_detail_head_ref="retargeted-branch",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("identity changed before publication", result.stderr)
        self.assertFalse(any("<-f> <state=closed>" in event for event in self.events()))

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
