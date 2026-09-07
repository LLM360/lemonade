#!/usr/bin/env python3
"""Tests for exact-commit llama.cpp pull-request file classification."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_HELPER = ROOT / ".github" / "scripts" / "llamacpp_required_check_policy.py"
PR_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_pr.yml"
BASE_SHA = "a" * 40
CANDIDATE_A_SHA = "b" * 40
CANDIDATE_B_SHA = "c" * 40


def load_policy_helper():
    spec = importlib.util.spec_from_file_location(
        "llamacpp_required_check_policy", POLICY_HELPER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load llama.cpp required-check policy")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_response(
    *,
    target_sha: str,
    files: list[dict[str, object]],
    base_sha: str = BASE_SHA,
    merge_base_sha: str = BASE_SHA,
) -> dict[str, object]:
    return {
        "ahead_by": 1,
        "base_commit": {"sha": base_sha},
        "behind_by": 0,
        "commits": [{"sha": target_sha}],
        "files": files,
        "merge_base_commit": {"sha": merge_base_sha},
        "status": "ahead",
        "total_commits": 1,
    }


def extract_classify_step_script() -> str:
    workflow = PR_WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split(
        "      - name: Classify the complete pull request file set\n", 1
    )[1].split("      - name:", 1)[0]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


class TestPullRequestFileClassification(unittest.TestCase):
    def test_exact_compare_identity_can_be_classified_as_neutral(self) -> None:
        policy = load_policy_helper()

        self.assertFalse(
            policy.classify_compare_response(
                compare_response(
                    target_sha=CANDIDATE_B_SHA,
                    files=[{"filename": "docs/dev/testing.md", "status": "modified"}],
                ),
                expected_base_sha=BASE_SHA,
                expected_target_sha=CANDIDATE_B_SHA,
                expected_count=1,
            )
        )

    def test_same_count_neutral_payload_for_other_candidate_is_rejected(
        self,
    ) -> None:
        policy = load_policy_helper()

        with self.assertRaises(ValueError):
            policy.classify_compare_response(
                compare_response(
                    target_sha=CANDIDATE_A_SHA,
                    files=[{"filename": "docs/dev/testing.md", "status": "modified"}],
                ),
                expected_base_sha=BASE_SHA,
                expected_target_sha=CANDIDATE_B_SHA,
                expected_count=1,
            )

    def test_compare_file_cap_is_conservatively_relevant(self) -> None:
        policy = load_policy_helper()
        neutral_files = [
            {"filename": f"docs/file-{index}.md", "status": "modified"}
            for index in range(300)
        ]

        self.assertFalse(
            policy.classify_compare_response(
                compare_response(
                    target_sha=CANDIDATE_B_SHA,
                    files=neutral_files,
                ),
                expected_base_sha=BASE_SHA,
                expected_target_sha=CANDIDATE_B_SHA,
                expected_count=300,
            )
        )
        self.assertTrue(
            policy.classify_compare_response(
                {},
                expected_base_sha=BASE_SHA,
                expected_target_sha=CANDIDATE_B_SHA,
                expected_count=301,
            )
        )

    def test_incomplete_compare_file_list_is_conservatively_relevant(self) -> None:
        policy = load_policy_helper()

        self.assertTrue(
            policy.classify_compare_response(
                compare_response(
                    target_sha=CANDIDATE_B_SHA,
                    files=[{"filename": "docs/dev/testing.md", "status": "modified"}],
                ),
                expected_base_sha=BASE_SHA,
                expected_target_sha=CANDIDATE_B_SHA,
                expected_count=2,
            )
        )

    def test_compare_identity_and_file_schema_errors_fail_closed(self) -> None:
        policy = load_policy_helper()
        valid = compare_response(
            target_sha=CANDIDATE_B_SHA,
            files=[{"filename": "docs/dev/testing.md", "status": "modified"}],
        )
        invalid_responses = {
            "base": {**valid, "base_commit": {"sha": CANDIDATE_A_SHA}},
            "merge-base": {
                **valid,
                "merge_base_commit": {"sha": CANDIDATE_A_SHA},
            },
            "commit-count": {**valid, "total_commits": 2},
            "commit-list": {**valid, "commits": []},
            "file-path": {
                **valid,
                "files": [{"filename": "/docs/testing.md", "status": "modified"}],
            },
            "file-status": {
                **valid,
                "files": [{"filename": "docs/testing.md", "status": "moved"}],
            },
        }

        for name, response in invalid_responses.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                policy.classify_compare_response(
                    response,
                    expected_base_sha=BASE_SHA,
                    expected_target_sha=CANDIDATE_B_SHA,
                    expected_count=1,
                )

    def test_compare_rename_from_protected_path_is_relevant(self) -> None:
        policy = load_policy_helper()

        self.assertTrue(
            policy.classify_compare_response(
                compare_response(
                    target_sha=CANDIDATE_B_SHA,
                    files=[
                        {
                            "filename": "docs/old-build-input.md",
                            "previous_filename": "CMakeLists.txt",
                            "status": "renamed",
                        }
                    ],
                ),
                expected_base_sha=BASE_SHA,
                expected_target_sha=CANDIDATE_B_SHA,
                expected_count=1,
            )
        )

    def test_identical_comparison_cannot_contain_changed_files(self) -> None:
        policy = load_policy_helper()
        invalid = {
            "ahead_by": 0,
            "base_commit": {"sha": BASE_SHA},
            "behind_by": 0,
            "commits": [],
            "files": [{"filename": "docs/testing.md", "status": "modified"}],
            "merge_base_commit": {"sha": BASE_SHA},
            "status": "identical",
            "total_commits": 0,
        }

        with self.assertRaises(ValueError):
            policy.classify_compare_response(
                invalid,
                expected_base_sha=BASE_SHA,
                expected_target_sha=BASE_SHA,
                expected_count=1,
            )

    def test_workflow_does_not_accept_stale_same_count_neutral_payload(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            trusted_helper = (
                temporary
                / "trusted"
                / ".github"
                / "scripts"
                / "llamacpp_required_check_policy.py"
            )
            trusted_helper.parent.mkdir(parents=True)
            shutil.copyfile(POLICY_HELPER, trusted_helper)
            response_path = temporary / "response.json"
            response_path.write_text(
                json.dumps(
                    compare_response(
                        target_sha=CANDIDATE_A_SHA,
                        files=[
                            {
                                "filename": "docs/dev/testing.md",
                                "status": "modified",
                            }
                        ],
                    )
                ),
                encoding="utf-8",
            )
            calls_path = temporary / "gh-calls.txt"
            output_path = temporary / "github-output.txt"

            fake_git = temporary / "git"
            fake_git.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$BASE_SHA\"\n", encoding="utf-8"
            )
            fake_git.chmod(0o755)
            fake_gh = temporary / "gh"
            fake_gh.write_text(
                """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

Path(os.environ["FAKE_GH_CALLS"]).write_text("\\n".join(sys.argv[1:]))
response = json.loads(Path(os.environ["FAKE_GH_RESPONSE"]).read_text())
if any("/pulls/7/files" in argument for argument in sys.argv):
    response = [response["files"]]
print(json.dumps(response))
""",
                encoding="utf-8",
            )
            fake_gh.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "BASE_SHA": BASE_SHA,
                    "EXPECTED_COUNT": "1",
                    "FAKE_GH_CALLS": str(calls_path),
                    "FAKE_GH_RESPONSE": str(response_path),
                    "GH_TOKEN": "test-token",
                    "GITHUB_OUTPUT": str(output_path),
                    "GITHUB_REPOSITORY": "LLM360/lemonade",
                    "MERGE_SHA": CANDIDATE_B_SHA,
                    "PATH": f"{temporary}{os.pathsep}{environment['PATH']}",
                    "PR_NUMBER": "7",
                    "RUNNER_TEMP": str(temporary),
                }
            )

            completed = subprocess.run(
                ["bash", "-c", extract_classify_step_script()],
                cwd=temporary,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            outputs = (
                output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            )
            calls = calls_path.read_text(encoding="utf-8")

        self.assertNotEqual(completed.returncode, 0)
        self.assertNotIn("relevant=false", outputs)
        self.assertIn(f"compare/{BASE_SHA}...{CANDIDATE_B_SHA}", calls)


if __name__ == "__main__":
    unittest.main()
