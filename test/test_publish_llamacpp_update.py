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

from test.utils import llamacpp_release_assets as release_assets
from test.utils import llamacpp_release_manifest as release_manifest

ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = ROOT / ".github" / "scripts" / "publish_llamacpp_update.sh"
SCHEDULE_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_schedule.yml"
CHECKOUT_SHA = "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"
DOWNLOAD_ARTIFACT_SHA = "37930b1c2abaa49bbe596cd826c3c89aef350131"
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
        self.dynamic_open_prs = self.temp_root / "dynamic-open-prs.txt"
        self.pr_create_failed = self.temp_root / "pr-create-failed.txt"
        self.pr_draft_state = self.temp_root / "pr-draft-state.txt"
        self.rocm_latest_query_count = self.temp_root / "rocm-latest-count.txt"
        self.real_git = shutil.which("git")
        if not self.real_git:
            self.skipTest("git is required")

        self.run_command([self.real_git, "init", "--quiet", "--bare", self.remote])
        self.repo.mkdir()
        self.run_git("init", "--quiet")
        self.run_git("config", "user.name", "Test Author")
        self.run_git("config", "user.email", "test@example.com")
        self.base_versions = {
            "llamacpp": {
                "cpu": "b1",
                "cuda": "b1",
                "metal": "b1",
                "rocm-nightly": "b1",
                "rocm-stable": "b1",
                "vulkan": "b1",
            },
            "rocm_asset_families": {},
            "therock": {"version": "7.14.0"},
        }
        self.candidate_versions = {
            **self.base_versions,
            "llamacpp": {
                "cpu": "b100",
                "cuda": "b200",
                "metal": "b100",
                "rocm-nightly": "b300",
                "rocm-stable": "b200",
                "vulkan": "b100",
            },
        }
        versions = self.repo / VERSIONS_PATH
        versions.parent.mkdir(parents=True)
        versions.write_text(
            json.dumps(self.base_versions, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        self.upstream_head = "f" * 40
        self.live_upstream_head = "9" * 40
        requirements = release_assets.build_asset_requirements(
            ggml_release="b100",
            rocm_release="b300",
            lemonade_release="b200",
            backend_versions=self.candidate_versions,
        )
        rocm_target_requirements = release_assets.build_rocm_asset_target_requirements(
            rocm_release="b300",
            backend_versions=self.candidate_versions,
        )
        self.release_payloads = {
            "ggml": self.release_payload(
                repository="ggml-org/llama.cpp",
                tag="b100",
                release_id=100,
                release_tag_commit="a" * 40,
                claimed_source_commit="a" * 40,
                asset_names=self.asset_names(requirements["ggml"]),
            ),
            "rocm": self.release_payload(
                repository="lemonade-sdk/llamacpp-rocm",
                tag="b300",
                release_id=300,
                release_tag_commit="b" * 40,
                claimed_source_commit="c" * 40,
                asset_names=self.asset_names(requirements["rocm"]),
                build_targets_by_asset=rocm_target_requirements,
            ),
            "lemonade": self.release_payload(
                repository="lemonade-sdk/llama.cpp",
                tag="b200",
                release_id=200,
                release_tag_commit="d" * 40,
                claimed_source_commit="e" * 40,
                asset_names=self.asset_names(requirements["lemonade"]),
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
        asset_names: list[str],
        build_targets_by_asset: dict[str, tuple[str, ...]] | None = None,
    ) -> dict:
        assets = [
            {
                "digest": "sha256:"
                + hashlib.sha256(
                    f"{repository}@{tag}:{asset_name}".encode("utf-8")
                ).hexdigest(),
                "id": release_id * 1000 + index,
                "name": asset_name,
                "size": release_id + index,
                "state": "uploaded",
                "updated_at": "2026-09-06T12:00:00Z",
            }
            for index, asset_name in enumerate(asset_names, start=1)
        ]
        source_manifest = None
        build_target_attestations = []
        if repository != "ggml-org/llama.cpp":
            source_bindings = []
            for asset in assets:
                binding = {
                    "digest": asset["digest"],
                    "name": asset["name"],
                    "size": asset["size"],
                }
                if build_targets_by_asset is not None:
                    binding["build_targets"] = list(
                        build_targets_by_asset.get(asset["name"], ())
                    )
                    build_target_attestations.append(dict(binding))
                source_bindings.append(binding)
            source_manifest = json.dumps(
                {
                    "assets": source_bindings,
                    "release_repository": repository,
                    "release_tag": tag,
                    "release_tag_commit": release_tag_commit,
                    "schema_version": 2 if build_targets_by_asset is not None else 1,
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
            "_build_target_attestations": build_target_attestations,
            "_release_tag_commit": release_tag_commit,
            "_source_manifest": source_manifest,
        }

    @staticmethod
    def asset_names(requirements: dict[str, list[str]]) -> list[str]:
        return [
            asset_name
            for backend_assets in requirements.values()
            for asset_name in backend_assets
        ]

    def build_expected_manifest(self) -> str:
        releases = []
        for key, repository, tag in (
            ("ggml", "ggml-org/llama.cpp", "b100"),
            ("rocm", "lemonade-sdk/llamacpp-rocm", "b300"),
            ("lemonade", "lemonade-sdk/llama.cpp", "b200"),
        ):
            payload = dict(self.release_payloads[key])
            claimed_commit = payload.pop("_claimed_source_commit")
            build_target_attestations = payload.pop("_build_target_attestations")
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
                    "build_target_attestations": build_target_attestations,
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
            'if [ "$1" = push ]; then\n'
            '    if [ "${GIT_TERMINAL_PROMPT:-}" != 0 ] || '
            '[ "${GIT_CONFIG_KEY_0:-}" != credential.helper ] || '
            '[ "${GIT_CONFIG_VALUE_0+x}" != x ] || '
            '[ "${GIT_CONFIG_VALUE_0}" != "" ] || '
            '[ ! -x "${GIT_ASKPASS:-}" ]; then\n'
            "        printf 'missing ephemeral authenticated push configuration\\n' >&2\n"
            "        exit 96\n"
            "    fi\n"
            '    username=$("$GIT_ASKPASS" "Username for https://github.com")\n'
            '    password=$("$GIT_ASKPASS" "Password for https://github.com")\n'
            '    if [ "$username" != x-access-token ] || '
            '[ "$password" != "$GH_TOKEN" ]; then\n'
            "        printf 'invalid ephemeral authenticated push configuration\\n' >&2\n"
            "        exit 97\n"
            "    fi\n"
            "    printf 'authenticated-push\\n' >> \"$EVENT_LOG\"\n"
            "fi\n"
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
            'if [ "$1" = api ] && [ "$2" = graphql ]; then\n'
            "    printf '%s\\n' \"$AUTHENTICATED_ACTOR\"\n"
            "    exit 0\n"
            "fi\n"
            'if [ "$1" = pr ] && [ "$2" = create ]; then\n'
            "    requested_draft=false\n"
            '    case " $* " in *" --draft "*) requested_draft=true ;; esac\n'
            '    if [ "$MUTATE_BRANCH_BEFORE_PR_CREATE_FAILURE" = true ]; then\n'
            '        "$REAL_GIT" --git-dir="$REMOTE_REPOSITORY_PATH" update-ref '
            '"refs/heads/$EXPECTED_BRANCH" "$VALIDATED_BASE_SHA"\n'
            "    fi\n"
            '    if [ "$PR_CREATE_EXIT_CODE" -ne 0 ]; then\n'
            '        if [ "$CREATE_PR_BEFORE_FAILURE" = true ]; then\n'
            '            head_sha=$("$REAL_GIT" '
            '--git-dir="$REMOTE_REPOSITORY_PATH" show-ref --verify --hash '
            '"refs/heads/$EXPECTED_BRANCH")\n'
            "            printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "
            '"$CREATED_PR_NUMBER" "$EXPECTED_BRANCH" "LLM360/lemonade" '
            '"$EXPECTED_PUBLICATION_ACTOR" "main" "$head_sha" '
            '> "$DYNAMIC_OPEN_PRS_FILE"\n'
            '            if [ "$requested_draft" = true ]; then\n'
            "                printf 'draft\\n' > \"$PR_DRAFT_STATE_FILE\"\n"
            "            fi\n"
            '            if [ "$DUPLICATE_PR_BEFORE_FAILURE" = true ]; then\n'
            "                printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "
            '"42" "$EXPECTED_BRANCH" "LLM360/lemonade" '
            '"$EXPECTED_PUBLICATION_ACTOR" "main" "$head_sha" '
            '>> "$DYNAMIC_OPEN_PRS_FILE"\n'
            "            fi\n"
            "        fi\n"
            "        printf 'failed\\n' > \"$PR_CREATE_FAILED_FILE\"\n"
            "        printf 'simulated pull request creation failure\\n' >&2\n"
            '        exit "$PR_CREATE_EXIT_CODE"\n'
            "    fi\n"
            '    if [ "$MUTATE_BRANCH_AFTER_PR_CREATE" = true ]; then\n'
            '        "$REAL_GIT" --git-dir="$REMOTE_REPOSITORY_PATH" update-ref '
            '"refs/heads/$EXPECTED_BRANCH" "$VALIDATED_BASE_SHA"\n'
            "    fi\n"
            '    if [ "$requested_draft" = true ]; then\n'
            "        printf 'draft\\n' > \"$PR_DRAFT_STATE_FILE\"\n"
            "    fi\n"
            "    printf 'https://github.com/LLM360/lemonade/pull/%s\\n' "
            '"$CREATED_PR_NUMBER"\n'
            "    exit 0\n"
            "fi\n"
            'if [ "$1" = pr ] && [ "$2" = ready ]; then\n'
            '    if [ "$PR_READY_EXIT_CODE" -ne 0 ]; then\n'
            "        printf 'simulated pull request ready failure\\n' >&2\n"
            '        exit "$PR_READY_EXIT_CODE"\n'
            "    fi\n"
            '    if [ "$PR_READY_CLOSES_CURRENT" = true ]; then\n'
            '        printf \'%s\\n\' "$3" >> "$CLOSED_PRS_FILE"\n'
            "    fi\n"
            '    if [ "$PR_READY_NOOP" != true ]; then\n'
            '        rm -f "$PR_DRAFT_STATE_FILE"\n'
            "    fi\n"
            "    exit 0\n"
            "fi\n"
            'if [ "$1" = api ]; then\n'
            '    case "$2" in\n'
            "        repos/ggml-org/llama.cpp) printf '%s\\n' master ;;\n"
            "        repos/ggml-org/llama.cpp/releases) "
            "printf '%s\\n' \"$FRESH_GGML_RELEASE\" ;;\n"
            "        repos/lemonade-sdk/llamacpp-rocm/releases/latest)\n"
            "            count=0\n"
            '            if [ -f "$ROCM_LATEST_QUERY_COUNT_FILE" ]; then '
            'count=$(cat "$ROCM_LATEST_QUERY_COUNT_FILE"); fi\n'
            "            count=$((count + 1))\n"
            "            printf '%s\\n' \"$count\" > "
            '"$ROCM_LATEST_QUERY_COUNT_FILE"\n'
            '            if [ "$ROCM_RELEASE_DRIFT_AT" -gt 0 ] && '
            '[ "$count" -ge "$ROCM_RELEASE_DRIFT_AT" ]; then\n'
            "                printf 'b301\\n'\n"
            "            else\n"
            "                printf '%s\\n' \"$FRESH_ROCM_RELEASE\"\n"
            "            fi ;;\n"
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
            "        repos/ggml-org/llama.cpp/commits/tags/b100) "
            "printf '%s\\n' \"$GGML_RELEASE_TAG_COMMIT\" ;;\n"
            "        repos/lemonade-sdk/llamacpp-rocm/commits/tags/b300) "
            "printf '%s\\n' \"$ROCM_RELEASE_TAG_COMMIT\" ;;\n"
            "        repos/lemonade-sdk/llama.cpp/commits/tags/b200) "
            "printf '%s\\n' \"$LEMONADE_RELEASE_TAG_COMMIT\" ;;\n"
            "        repos/ggml-org/llama.cpp/commits/heads/master) "
            "printf '%s\\n' \"$LIVE_UPSTREAM_HEAD\" ;;\n"
            "        repos/*/commits/b*|repos/*/commits/master) "
            "printf '%040d\\n' 7 ;;\n"
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
            "        repos/LLM360/lemonade/pulls)\n"
            '            if [ -f "$PR_CREATE_FAILED_FILE" ] && '
            '[ "$FAIL_POST_CREATE_PULLS_QUERY" = true ]; then exit 92; fi\n'
            "            printf '%s\\n' \"$OPEN_PR_RECORDS\"\n"
            '            if [ -f "$DYNAMIC_OPEN_PRS_FILE" ]; then '
            "sed -n '1,$p' \"$DYNAMIC_OPEN_PRS_FILE\"; fi ;;\n"
            "        repos/LLM360/lemonade/actions/permissions/workflow) exit 91 ;;\n"
            "        repos/LLM360/lemonade/pulls/[0-9]*)\n"
            "            number=${2##*/}\n"
            '            case " $* " in\n'
            '                *" --method PATCH "*" state=closed "*)\n'
            '                    printf \'%s\\n\' "$number" >> "$CLOSED_PRS_FILE"\n'
            "                    printf '{}\\n'\n"
            "                    exit 0 ;;\n"
            "            esac\n"
            "            records=$(printf '%s\\n' \"$OPEN_PR_RECORDS\")\n"
            '            if [ -f "$DYNAMIC_OPEN_PRS_FILE" ]; then\n'
            "                records=$(printf '%s\\n%s\\n' \"$records\" "
            '"$(sed -n \'1,$p\' "$DYNAMIC_OPEN_PRS_FILE")")\n'
            "            fi\n"
            "            record=$(printf '%s\\n' \"$records\" | "
            "awk -F '\\t' -v number=\"$number\" '$1 == number { print; exit }')\n"
            "            head_ref=$(printf '%s' \"$record\" | cut -f2)\n"
            "            head_repository=$(printf '%s' \"$record\" | cut -f3)\n"
            "            author=$(printf '%s' \"$record\" | cut -f4)\n"
            "            base_ref=$(printf '%s' \"$record\" | cut -f5)\n"
            "            head_ref=${head_ref:-$EXPECTED_BRANCH}\n"
            "            head_repository=${head_repository:-LLM360/lemonade}\n"
            "            author=${author:-$EXPECTED_PUBLICATION_ACTOR}\n"
            "            base_ref=${base_ref:-main}\n"
            "            head_ref=${PR_DETAIL_HEAD_REF:-$head_ref}\n"
            "            head_repository=${PR_DETAIL_HEAD_REPOSITORY:-$head_repository}\n"
            "            author=${PR_DETAIL_AUTHOR:-$author}\n"
            "            base_ref=${PR_DETAIL_BASE_REF:-$base_ref}\n"
            "            state=open\n"
            "            draft=false\n"
            '            if [ "$number" = "$CREATED_PR_NUMBER" ] && '
            '[ -f "$PR_DRAFT_STATE_FILE" ]; then draft=true; fi\n'
            "            draft=${PR_DETAIL_DRAFT:-$draft}\n"
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
            '\'{"state":"%s","draft":%s,"head":{"ref":"%s",'
            '"sha":"%s","repo":{"full_name":"%s"}},'
            '"user":{"login":"%s"},"base":{"ref":"%s",'
            '"sha":"%s"}}\\n\' '
            '"$state" "$draft" "$head_ref" "$head_sha" "$head_repository" '
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
        authenticated_actor: str = "github-actions[bot]",
        create_pr_before_failure: bool = False,
        divergent_commit: str = "",
        duplicate_pr_before_failure: bool = False,
        fail_post_create_pulls_query: bool = False,
        include_expected_manifest: bool = True,
        include_publication_token: bool = True,
        mutate_on_snapshot: int = 0,
        open_pr_records: str = "",
        fresh_ggml_release: str = "b100",
        mutate_branch_after_push: bool = False,
        mutate_branch_after_pr_create: bool = False,
        mutate_branch_before_pr_create_failure: bool = False,
        pr_create_exit_code: int = 0,
        pr_detail_base_ref: str = "",
        pr_detail_head_ref: str = "",
        pr_detail_draft: str = "",
        pr_ready_closes_current: bool = False,
        pr_ready_exit_code: int = 0,
        pr_ready_noop: bool = False,
        rocm_release_drift_at: int = 0,
        expected_publication_actor: str | None = "github-actions[bot]",
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
                "AUTHENTICATED_ACTOR": authenticated_actor,
                "CREATE_PR_BEFORE_FAILURE": (
                    "true" if create_pr_before_failure else "false"
                ),
                "DIVERGENT_COMMIT": divergent_commit,
                "DYNAMIC_OPEN_PRS_FILE": str(self.dynamic_open_prs),
                "DUPLICATE_PR_BEFORE_FAILURE": (
                    "true" if duplicate_pr_before_failure else "false"
                ),
                "FAIL_POST_CREATE_PULLS_QUERY": (
                    "true" if fail_post_create_pulls_query else "false"
                ),
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
                "MUTATE_BRANCH_AFTER_PR_CREATE": (
                    "true" if mutate_branch_after_pr_create else "false"
                ),
                "MUTATE_BRANCH_BEFORE_PR_CREATE_FAILURE": (
                    "true" if mutate_branch_before_pr_create_failure else "false"
                ),
                "OPEN_PR_RECORDS": open_pr_records,
                "PATH": f"{self.fake_bin}{os.pathsep}{env['PATH']}",
                "PR_BODY_FILE": "pr_body.md",
                "PR_CREATE_EXIT_CODE": str(pr_create_exit_code),
                "PR_CREATE_FAILED_FILE": str(self.pr_create_failed),
                "PR_DRAFT_STATE_FILE": str(self.pr_draft_state),
                "PR_DETAIL_AUTHOR": "",
                "PR_DETAIL_BASE_REF": pr_detail_base_ref,
                "PR_DETAIL_HEAD_REF": pr_detail_head_ref,
                "PR_DETAIL_DRAFT": pr_detail_draft,
                "PR_DETAIL_HEAD_REPOSITORY": "",
                "PR_DETAIL_HEAD_SHA": "",
                "PR_READY_CLOSES_CURRENT": (
                    "true" if pr_ready_closes_current else "false"
                ),
                "PR_READY_EXIT_CODE": str(pr_ready_exit_code),
                "PR_READY_NOOP": "true" if pr_ready_noop else "false",
                "REAL_GIT": self.real_git,
                "REMOTE_REPOSITORY_PATH": str(self.remote),
                "ROCM_RELEASE_PAYLOAD": json.dumps(api_payloads["rocm"]),
                "ROCM_LATEST_QUERY_COUNT_FILE": str(self.rocm_latest_query_count),
                "ROCM_RELEASE_DRIFT_AT": str(rocm_release_drift_at),
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
        if expected_publication_actor is None:
            env.pop("EXPECTED_PUBLICATION_ACTOR", None)
        else:
            env["EXPECTED_PUBLICATION_ACTOR"] = expected_publication_actor
        if include_publication_token:
            env["GH_TOKEN"] = "dedicated-publication-token"
        else:
            env.pop("GH_TOKEN", None)
        if include_expected_manifest:
            env["EXPECTED_RELEASE_ASSET_MANIFEST"] = self.expected_manifest
        return env

    def run_publisher(
        self,
        *,
        authenticated_actor: str = "github-actions[bot]",
        create_pr_before_failure: bool = False,
        divergent_commit: str = "",
        duplicate_pr_before_failure: bool = False,
        fail_post_create_pulls_query: bool = False,
        include_expected_manifest: bool = True,
        include_publication_token: bool = True,
        mutate_on_snapshot: int = 0,
        open_pr_records: str = "",
        fresh_ggml_release: str = "b100",
        mutate_branch_after_push: bool = False,
        mutate_branch_after_pr_create: bool = False,
        mutate_branch_before_pr_create_failure: bool = False,
        pr_create_exit_code: int = 0,
        pr_detail_base_ref: str = "",
        pr_detail_head_ref: str = "",
        pr_detail_draft: str = "",
        pr_ready_closes_current: bool = False,
        pr_ready_exit_code: int = 0,
        pr_ready_noop: bool = False,
        rocm_release_drift_at: int = 0,
        expected_publication_actor: str | None = "github-actions[bot]",
        validated_base_sha: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self.run_command(
            ["bash", str(PUBLISHER)],
            cwd=self.repo,
            env=self.publisher_env(
                authenticated_actor=authenticated_actor,
                create_pr_before_failure=create_pr_before_failure,
                divergent_commit=divergent_commit,
                duplicate_pr_before_failure=duplicate_pr_before_failure,
                fail_post_create_pulls_query=fail_post_create_pulls_query,
                include_expected_manifest=include_expected_manifest,
                include_publication_token=include_publication_token,
                mutate_on_snapshot=mutate_on_snapshot,
                open_pr_records=open_pr_records,
                fresh_ggml_release=fresh_ggml_release,
                mutate_branch_after_push=mutate_branch_after_push,
                mutate_branch_after_pr_create=mutate_branch_after_pr_create,
                mutate_branch_before_pr_create_failure=(
                    mutate_branch_before_pr_create_failure
                ),
                pr_create_exit_code=pr_create_exit_code,
                pr_detail_base_ref=pr_detail_base_ref,
                pr_detail_head_ref=pr_detail_head_ref,
                pr_detail_draft=pr_detail_draft,
                pr_ready_closes_current=pr_ready_closes_current,
                pr_ready_exit_code=pr_ready_exit_code,
                pr_ready_noop=pr_ready_noop,
                rocm_release_drift_at=rocm_release_drift_at,
                expected_publication_actor=expected_publication_actor,
                validated_base_sha=validated_base_sha,
            ),
            check=False,
        )

    def change_versions(self) -> None:
        (self.repo / VERSIONS_PATH).write_text(
            json.dumps(self.candidate_versions, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def advance_base_versions(self, versions: dict[str, str]) -> None:
        document = {
            **self.base_versions,
            "llamacpp": {**self.base_versions["llamacpp"], **versions},
        }
        (self.repo / VERSIONS_PATH).write_text(
            json.dumps(document, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        self.run_git("add", str(VERSIONS_PATH))
        self.run_git("commit", "--quiet", "--message", "advance backend pins")
        self.run_git("push", "--quiet", "origin", "main")

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
        overrides = json.loads(versions_contents)
        document = {
            **self.candidate_versions,
            "llamacpp": {
                **self.candidate_versions["llamacpp"],
                **overrides["llamacpp"],
            },
        }
        (self.repo / VERSIONS_PATH).write_text(
            json.dumps(document, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
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

    def assert_no_remote_mutation(self) -> None:
        events = self.events()
        self.assertFalse(any(event.startswith("git <push>") for event in events))
        self.assertFalse(any(event.startswith("gh <pr>") for event in events))
        self.assertFalse(
            any(
                event.startswith("gh <api>")
                and any(
                    marker in event
                    for marker in (
                        "<--method> <DELETE>",
                        "<--method> <PATCH>",
                        "<--method> <POST>",
                        "<-f> <state=closed>",
                    )
                )
                for event in events
            )
        )

    def test_missing_publication_token_fails_before_any_mutation(self) -> None:
        self.change_versions()
        body_before = (self.repo / "pr_body.md").read_text(encoding="utf-8")
        manifest_before = (self.repo / MANIFEST_PATH).read_text(encoding="utf-8")

        result = self.run_publisher(include_publication_token=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GH_TOKEN", result.stderr)
        self.assertEqual(self.events(), [])
        self.assertEqual(
            (self.repo / "pr_body.md").read_text(encoding="utf-8"), body_before
        )
        self.assertEqual(
            (self.repo / MANIFEST_PATH).read_text(encoding="utf-8"), manifest_before
        )

    def test_missing_expected_publication_actor_fails_before_any_mutation(
        self,
    ) -> None:
        self.change_versions()
        body_before = (self.repo / "pr_body.md").read_text(encoding="utf-8")

        result = self.run_publisher(expected_publication_actor=None)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EXPECTED_PUBLICATION_ACTOR", result.stderr)
        self.assertEqual(self.events(), [])
        self.assertEqual(
            (self.repo / "pr_body.md").read_text(encoding="utf-8"), body_before
        )

    def test_authenticated_actor_mismatch_fails_before_any_mutation(self) -> None:
        self.change_versions()
        body_before = (self.repo / "pr_body.md").read_text(encoding="utf-8")

        result = self.run_publisher(
            authenticated_actor="octocat",
            expected_publication_actor="lemonade-release[bot]",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("authenticated publication actor", result.stderr)
        self.assertEqual(
            self.events(),
            [
                "gh <api> <graphql> <-f> <query=query { viewer { login } }> "
                "<--jq> <.data.viewer.login>"
            ],
        )
        self.assert_no_remote_mutation()
        self.assertEqual(
            (self.repo / "pr_body.md").read_text(encoding="utf-8"), body_before
        )

    def test_configured_publication_actor_owns_commit_and_pull_request(
        self,
    ) -> None:
        actor = "lemonade-release[bot]"
        self.change_versions()

        result = self.run_publisher(
            authenticated_actor=actor,
            expected_publication_actor=actor,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.run_git("show", "-s", "--format=%an", "HEAD"), actor)
        self.assertEqual(
            self.run_git("show", "-s", "--format=%ae", "HEAD"),
            f"{actor}@users.noreply.github.com",
        )
        self.assertTrue(
            any(event.startswith("gh <pr> <create>") for event in self.events())
        )
        self.assertTrue(
            any(event.startswith("gh <pr> <ready>") for event in self.events())
        )

    def test_push_uses_ephemeral_askpass_without_persisting_the_token(self) -> None:
        self.change_versions()

        result = self.run_publisher()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("authenticated-push", self.events())

    def test_unchanged_versions_report_stale_pr_for_manual_cleanup(self) -> None:
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\tgithub-actions[bot]\tmain"
        )

        result = self.run_publisher(open_pr_records=stale_record)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nothing to update", result.stdout)
        self.assertIn(
            "Superseded pull request #8 "
            "(auto/llamacpp-update-b90-b190-b290-deadbeef1234) "
            "requires manual cleanup.",
            result.stdout,
        )
        events = self.events()
        self.assertFalse(
            any("<repos/LLM360/lemonade/pulls/8>" in event for event in events)
        )
        self.assertFalse(any("<-f> <state=closed>" in event for event in events))
        self.assertFalse(any(event.startswith("git <push>") for event in events))

    def test_unchanged_versions_preserve_stale_pr_when_release_tags_drift(
        self,
    ) -> None:
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\tgithub-actions[bot]\tmain"
        )

        result = self.run_publisher(
            open_pr_records=stale_record,
            rocm_release_drift_at=2,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("candidate releases changed", result.stderr)
        self.assertFalse(any("<-f> <state=closed>" in event for event in self.events()))

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

    def test_scheduled_publication_rejects_all_managed_pin_downgrades(self) -> None:
        self.advance_base_versions({"vulkan": "b6000", "cpu": "b6000"})
        self.change_versions()

        result = self.run_publisher()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("would downgrade llamacpp.cpu from b6000 to b100", result.stderr)
        events = self.events()
        self.assertFalse(any(event.startswith("git <push>") for event in events))
        self.assertFalse(any(event.startswith("gh <pr>") for event in events))

    def test_scheduled_publication_rejects_a_mixed_backend_downgrade(self) -> None:
        self.advance_base_versions({"vulkan": "b50", "cpu": "b150"})
        self.change_versions()

        result = self.run_publisher()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("would downgrade llamacpp.cpu from b150 to b100", result.stderr)
        events = self.events()
        self.assertFalse(any(event.startswith("git <push>") for event in events))
        self.assertFalse(any(event.startswith("gh <pr>") for event in events))

    def test_scheduled_publication_rejects_arbitrary_precision_downgrade(
        self,
    ) -> None:
        self.advance_base_versions({"cpu": "b9223372036854775808"})
        candidate = {
            **self.candidate_versions,
            "llamacpp": {
                **self.candidate_versions["llamacpp"],
                "cpu": "b9223372036854775807",
            },
        }
        (self.repo / VERSIONS_PATH).write_text(
            json.dumps(candidate, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        result = self.run_publisher()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "would downgrade llamacpp.cpu from b9223372036854775808 "
            "to b9223372036854775807",
            result.stderr,
        )
        self.assert_no_remote_mutation()

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

    def test_missing_required_asset_prevents_remote_mutation(self) -> None:
        self.change_versions()
        missing_asset = "llama-b100-bin-win-vulkan-x64.zip"
        self.release_payloads["ggml"]["assets"] = [
            asset
            for asset in self.release_payloads["ggml"]["assets"]
            if asset["name"] != missing_asset
        ]
        self.expected_manifest = self.build_expected_manifest()

        result = self.run_publisher()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("llamacpp.vulkan", result.stderr)
        self.assertIn(missing_asset, result.stderr)
        self.assert_no_remote_mutation()

    def test_zero_size_required_asset_prevents_remote_mutation(self) -> None:
        self.change_versions()
        zero_size_asset = "llama-b100-bin-ubuntu-arm64.tar.gz"
        for asset in self.release_payloads["ggml"]["assets"]:
            if asset["name"] == zero_size_asset:
                asset["size"] = 0
                break
        else:
            self.fail(f"fixture is missing required asset {zero_size_asset}")
        self.expected_manifest = self.build_expected_manifest()

        result = self.run_publisher()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("llamacpp.cpu", result.stderr)
        self.assertIn("positive size", result.stderr)
        self.assertIn(zero_size_asset, result.stderr)
        self.assert_no_remote_mutation()

    def test_capture_uses_namespaced_refs_when_branch_and_tag_names_collide(
        self,
    ) -> None:
        result = self.run_publisher()

        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        self.assertTrue(
            any(
                "<repos/ggml-org/llama.cpp/commits/heads/master>" in event
                for event in events
            )
        )
        for repository, tag in (
            ("ggml-org/llama.cpp", "b100"),
            ("lemonade-sdk/llamacpp-rocm", "b300"),
            ("lemonade-sdk/llama.cpp", "b200"),
        ):
            self.assertTrue(
                any(
                    f"<repos/{repository}/commits/tags/{tag}>" in event
                    for event in events
                )
            )
            self.assertFalse(
                any(f"<repos/{repository}/commits/{tag}>" in event for event in events)
            )
        self.assertFalse(
            any(
                "<repos/ggml-org/llama.cpp/commits/master>" in event for event in events
            )
        )

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

    def test_new_pull_request_does_not_depend_on_actions_admin_permission(
        self,
    ) -> None:
        self.change_versions()

        result = self.run_publisher()

        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        self.assertFalse(
            any("actions/permissions/workflow" in event for event in events)
        )
        self.assertTrue(any(event.startswith("gh <pr> <create>") for event in events))

    def test_new_pull_request_is_promoted_only_after_final_identity_checks(
        self,
    ) -> None:
        self.change_versions()

        result = self.run_publisher()

        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events()
        create_index = next(
            index
            for index, event in enumerate(events)
            if event.startswith("gh <pr> <create>")
        )
        ready_index = next(
            index
            for index, event in enumerate(events)
            if event.startswith("gh <pr> <ready>")
        )
        self.assertIn("<--draft>", events[create_index])
        self.assertGreater(ready_index, create_index)
        self.assertTrue(
            any(
                create_index < index < ready_index
                and "<repos/LLM360/lemonade/pulls/41>" in event
                for index, event in enumerate(events)
            )
        )
        self.assertFalse(self.pr_draft_state.exists())

    def test_branch_move_after_draft_creation_leaves_no_ready_pull_request(
        self,
    ) -> None:
        self.change_versions()

        result = self.run_publisher(mutate_branch_after_pr_create=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("identity changed before publication", result.stderr)
        events = self.events()
        create = next(event for event in events if event.startswith("gh <pr> <create>"))
        self.assertIn("<--draft>", create)
        self.assertFalse(any(event.startswith("gh <pr> <ready>") for event in events))
        self.assertTrue(self.pr_draft_state.exists())

    def test_latest_release_drift_before_push_prevents_remote_mutation(self) -> None:
        self.change_versions()

        result = self.run_publisher(rocm_release_drift_at=2)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("candidate releases changed", result.stderr)
        self.assert_no_remote_mutation()

    def test_latest_release_drift_after_draft_creation_prevents_promotion(
        self,
    ) -> None:
        self.change_versions()

        result = self.run_publisher(rocm_release_drift_at=3)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("candidate releases changed", result.stderr)
        events = self.events()
        self.assertTrue(any(event.startswith("gh <pr> <create>") for event in events))
        self.assertFalse(any(event.startswith("gh <pr> <ready>") for event in events))
        self.assertTrue(self.pr_draft_state.exists())

    def test_late_release_tag_drift_preserves_superseded_pull_request(self) -> None:
        self.change_versions()
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\tgithub-actions[bot]\tmain"
        )

        result = self.run_publisher(
            open_pr_records=stale_record,
            rocm_release_drift_at=3,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("candidate releases changed", result.stderr)
        events = self.events()
        self.assertFalse(any("<-f> <state=closed>" in event for event in events))
        self.assertFalse(any(event.startswith("gh <pr> <ready>") for event in events))
        self.assertTrue(self.pr_draft_state.exists())

    def test_late_release_manifest_drift_preserves_superseded_pull_request(
        self,
    ) -> None:
        self.change_versions()
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\tgithub-actions[bot]\tmain"
        )

        result = self.run_publisher(
            mutate_on_snapshot=4,
            open_pr_records=stale_record,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("changed since validation", result.stderr)
        events = self.events()
        self.assertFalse(any("<-f> <state=closed>" in event for event in events))
        self.assertFalse(any(event.startswith("gh <pr> <ready>") for event in events))
        self.assertTrue(self.pr_draft_state.exists())

    def test_ready_api_failure_preserves_superseded_pull_request(self) -> None:
        self.change_versions()
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\tgithub-actions[bot]\tmain"
        )

        result = self.run_publisher(
            open_pr_records=stale_record,
            pr_ready_exit_code=92,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("simulated pull request ready failure", result.stderr)
        events = self.events()
        self.assertTrue(any(event.startswith("gh <pr> <ready>") for event in events))
        self.assertFalse(any("<-f> <state=closed>" in event for event in events))
        self.assertTrue(self.pr_draft_state.exists())

    def test_ready_noop_preserves_superseded_pull_request(self) -> None:
        self.change_versions()
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\tgithub-actions[bot]\tmain"
        )

        result = self.run_publisher(
            open_pr_records=stale_record,
            pr_ready_noop=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("remained a draft", result.stderr)
        self.assertFalse(any("<-f> <state=closed>" in event for event in self.events()))
        self.assertTrue(self.pr_draft_state.exists())

    def test_concurrent_current_pr_closure_preserves_superseded_pull_request(
        self,
    ) -> None:
        self.change_versions()
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\tgithub-actions[bot]\tmain"
        )

        result = self.run_publisher(
            open_pr_records=stale_record,
            pr_ready_closes_current=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("identity changed before publication", result.stderr)
        close_events = [
            event for event in self.events() if "<-f> <state=closed>" in event
        ]
        self.assertFalse(
            any("<repos/LLM360/lemonade/pulls/8>" in event for event in close_events)
        )

    def test_latest_release_drift_before_failed_create_reconciliation_is_safe(
        self,
    ) -> None:
        self.change_versions()

        result = self.run_publisher(
            create_pr_before_failure=True,
            pr_create_exit_code=42,
            rocm_release_drift_at=3,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("candidate releases changed", result.stderr)
        events = self.events()
        self.assertFalse(any(event.startswith("gh <pr> <ready>") for event in events))
        self.assertTrue(self.pr_draft_state.exists())

    def test_failed_pull_request_creation_preserves_new_branch_after_zero_recheck(
        self,
    ) -> None:
        self.change_versions()
        branch_name = self.branch_name

        result = self.run_publisher(pr_create_exit_code=42)

        self.assertNotEqual(result.returncode, 0)
        published_oid = self.run_git("rev-parse", "HEAD")
        branch_lookup = self.run_command(
            [
                self.real_git,
                "--git-dir",
                str(self.remote),
                "show-ref",
                "--verify",
                f"refs/heads/{branch_name}",
            ],
            check=False,
        )
        self.assertEqual(branch_lookup.returncode, 0)
        self.assertEqual(branch_lookup.stdout.split()[0], published_oid)
        events = self.events()
        create_index = next(
            index
            for index, event in enumerate(events)
            if event.startswith("gh <pr> <create>")
        )
        recheck_index = next(
            index
            for index, event in enumerate(events)
            if index > create_index
            and event.startswith("gh <api> <repos/LLM360/lemonade/pulls>")
        )
        self.assertIn(f"<-f> <head=LLM360:{branch_name}>", events[recheck_index])
        self.assertIn("<-f> <base=main>", events[recheck_index])
        self.assertLess(create_index, recheck_index)
        self.assertFalse(
            any(
                event.startswith("git <push>")
                and f"<:refs/heads/{branch_name}>" in event
                for event in events
            )
        )
        self.assertIn("preserving", result.stderr)

    def test_failed_pull_request_creation_preserves_preexisting_branch(self) -> None:
        _base_oid, branch_name, branch_oid = self.create_remote_update_branch(
            '{"llamacpp":{"vulkan":"b100","cpu":"b100"}}\n'
        )
        self.change_versions()

        result = self.run_publisher(pr_create_exit_code=42)

        self.assertNotEqual(result.returncode, 0)
        remote_oid = self.run_command(
            [
                self.real_git,
                "--git-dir",
                str(self.remote),
                "show-ref",
                "--verify",
                "--hash",
                f"refs/heads/{branch_name}",
            ]
        ).stdout.strip()
        self.assertEqual(remote_oid, branch_oid)
        self.assertFalse(
            any(
                event.startswith("git <push>")
                and f"<:refs/heads/{branch_name}>" in event
                for event in self.events()
            )
        )

    def test_failed_pull_request_creation_adopts_trusted_concurrent_pr(self) -> None:
        self.change_versions()
        branch_name = self.branch_name

        result = self.run_publisher(
            create_pr_before_failure=True,
            pr_create_exit_code=42,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.run_command(
                [
                    self.real_git,
                    "--git-dir",
                    str(self.remote),
                    "show-ref",
                    "--verify",
                    "--hash",
                    f"refs/heads/{branch_name}",
                ]
            ).stdout.strip(),
            self.run_git("rev-parse", "HEAD"),
        )
        events = self.events()
        self.assertFalse(
            any(
                event.startswith("git <push>")
                and f"<:refs/heads/{branch_name}>" in event
                for event in events
            )
        )
        self.assertTrue(
            any("<repos/LLM360/lemonade/pulls/41>" in event for event in events)
        )
        create = next(event for event in events if event.startswith("gh <pr> <create>"))
        self.assertIn("<--draft>", create)
        self.assertTrue(any(event.startswith("gh <pr> <ready>") for event in events))
        self.assertFalse(self.pr_draft_state.exists())

    def test_failed_pull_request_creation_preserves_branch_when_recheck_fails(
        self,
    ) -> None:
        self.change_versions()
        branch_name = self.branch_name

        result = self.run_publisher(
            fail_post_create_pulls_query=True,
            pr_create_exit_code=42,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            self.run_command(
                [
                    self.real_git,
                    "--git-dir",
                    str(self.remote),
                    "show-ref",
                    "--verify",
                    f"refs/heads/{branch_name}",
                ],
                check=False,
            ).returncode,
            0,
        )
        self.assertFalse(
            any(
                event.startswith("git <push>")
                and f"<:refs/heads/{branch_name}>" in event
                for event in self.events()
            )
        )

    def test_failed_pull_request_creation_preserves_branch_when_recheck_is_ambiguous(
        self,
    ) -> None:
        self.change_versions()
        branch_name = self.branch_name

        result = self.run_publisher(
            create_pr_before_failure=True,
            duplicate_pr_before_failure=True,
            pr_create_exit_code=42,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("multiple matching pull requests", result.stderr)
        self.assertEqual(
            self.run_command(
                [
                    self.real_git,
                    "--git-dir",
                    str(self.remote),
                    "show-ref",
                    "--verify",
                    "--hash",
                    f"refs/heads/{branch_name}",
                ]
            ).stdout.strip(),
            self.run_git("rev-parse", "HEAD"),
        )
        self.assertFalse(
            any(
                event.startswith("git <push>")
                and f"<:refs/heads/{branch_name}>" in event
                for event in self.events()
            )
        )

    def test_failed_pull_request_creation_preserves_concurrently_moved_branch(
        self,
    ) -> None:
        validated_base = self.run_git("rev-parse", "HEAD")
        self.change_versions()
        branch_name = self.branch_name

        result = self.run_publisher(
            mutate_branch_before_pr_create_failure=True,
            pr_create_exit_code=42,
        )

        self.assertNotEqual(result.returncode, 0)
        remote_oid = self.run_command(
            [
                self.real_git,
                "--git-dir",
                str(self.remote),
                "show-ref",
                "--verify",
                "--hash",
                f"refs/heads/{branch_name}",
            ]
        ).stdout.strip()
        self.assertEqual(remote_oid, validated_base)
        self.assertFalse(
            any(
                event.startswith("git <push>")
                and f"<:refs/heads/{branch_name}>" in event
                for event in self.events()
            )
        )

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
        self.assertFalse(
            any("actions/permissions/workflow" in event for event in events)
        )

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

    def test_concurrently_retargeted_stale_pr_is_only_reported(self) -> None:
        stale_record = (
            "8\tauto/llamacpp-update-b90-b190-b290-deadbeef1234\t"
            "LLM360/lemonade\tgithub-actions[bot]\tmain"
        )

        result = self.run_publisher(
            open_pr_records=stale_record,
            pr_detail_head_ref="retargeted-branch",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "Superseded pull request #8 "
            "(auto/llamacpp-update-b90-b190-b290-deadbeef1234) "
            "requires manual cleanup.",
            result.stdout,
        )
        self.assertFalse(
            any("<repos/LLM360/lemonade/pulls/8>" in event for event in self.events())
        )
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

    def test_base_advance_uses_a_new_branch_and_reports_the_old_pull_request(
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
        self.assertFalse(
            any("<repos/LLM360/lemonade/pulls/17>" in event for event in events)
        )
        self.assertFalse(any("<-f> <state=closed>" in event for event in events))
        ready_message = "Marked pull request #41 ready for review."
        report_message = (
            f"Superseded pull request #17 ({old_branch}) requires manual cleanup."
        )
        self.assertIn(ready_message, result.stdout)
        self.assertIn(report_message, result.stdout)
        self.assertLess(
            result.stdout.index(ready_message), result.stdout.index(report_message)
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


class PublishLlamaCppWorkflowTests(unittest.TestCase):
    def test_publication_uses_only_a_dedicated_token_and_expected_actor(
        self,
    ) -> None:
        schedule = SCHEDULE_WORKFLOW.read_text(encoding="utf-8")
        publication = schedule.split("  publish:\n", 1)[1]
        preflight_marker = "      - name: Require dedicated publication identity\n"
        checkout_marker = f"      - uses: actions/checkout@{CHECKOUT_SHA} # v5.1.0\n"
        self.assertIn(preflight_marker, publication)
        self.assertIn(checkout_marker, publication)
        preflight = publication.split(preflight_marker, 1)[1].split(checkout_marker, 1)[
            0
        ]
        checkout = publication.split(checkout_marker, 1)[1].split("\n      - name:", 1)[
            0
        ]
        manifest = publication.split(
            "      - name: Verify release asset manifest\n", 1
        )[1].split("      - name: Publish update pull request\n", 1)[0]
        publisher = publication.split("      - name: Publish update pull request\n", 1)[
            1
        ]

        self.assertLess(
            publication.index("      - name: Require dedicated publication identity"),
            publication.index(f"      - uses: actions/checkout@{CHECKOUT_SHA}"),
        )
        self.assertIn("GH_TOKEN: ${{ secrets.LLAMACPP_UPDATE_TOKEN }}", preflight)
        self.assertIn(
            "EXPECTED_PUBLICATION_ACTOR: ${{ vars.LLAMACPP_UPDATE_ACTOR }}",
            preflight,
        )
        self.assertIn("gh api graphql", preflight)
        self.assertIn(".data.viewer.login", preflight)
        self.assertIn("token: ${{ secrets.LLAMACPP_UPDATE_TOKEN }}", checkout)
        self.assertIn("persist-credentials: false", checkout)
        self.assertEqual(
            publication.count(
                f"uses: actions/download-artifact@{DOWNLOAD_ARTIFACT_SHA} # v7.0.0"
            ),
            2,
        )
        self.assertIn("GH_TOKEN: ${{ secrets.LLAMACPP_UPDATE_TOKEN }}", manifest)
        self.assertIn("GH_TOKEN: ${{ secrets.LLAMACPP_UPDATE_TOKEN }}", publisher)
        self.assertIn(
            "EXPECTED_PUBLICATION_ACTOR: ${{ vars.LLAMACPP_UPDATE_ACTOR }}",
            publisher,
        )
        self.assertNotIn("secrets.GITHUB_TOKEN", publication)
        self.assertNotIn("contents: write", publication)
        self.assertNotIn("pull-requests: write", publication)


if __name__ == "__main__":
    unittest.main()
