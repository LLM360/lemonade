#!/usr/bin/env python3
"""Unit tests for the llama.cpp validation workflow plan."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

from test.utils import llamacpp_release_assets as release_assets
from test.utils import llamacpp_validation_plan as planning

K2_SMALL = "K2-Horizon-0.9B-GGUF"
K2_MEDIUM = "K2-Horizon-3.7B-GGUF"
K2_LARGE = "K2-Horizon-7B-GGUF"
CHECKOUT_SHA = "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"
DOWNLOAD_ARTIFACT_SHA = "37930b1c2abaa49bbe596cd826c3c89aef350131"
ROOT = Path(__file__).resolve().parents[1]
LEGACY_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp.yml"
VALIDATION_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_core.yml"
PR_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_pr.yml"
MANUAL_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_manual.yml"
SCHEDULE_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_schedule.yml"
DOCS_AND_STYLE_WORKFLOW = ROOT / ".github" / "workflows" / "docs_and_style.yml"
VALIDATION_RUNNER = ROOT / ".github" / "scripts" / "run_llamacpp_validation.py"
REQUIRED_CHECK_POLICY = (
    ROOT / ".github" / "scripts" / "llamacpp_required_check_policy.py"
)
TESTING_GUIDE = ROOT / "docs" / "dev" / "testing.md"


def load_validation_runner():
    spec = importlib.util.spec_from_file_location(
        "llamacpp_validation_runner", VALIDATION_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load llama.cpp validation runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_required_check_policy():
    spec = importlib.util.spec_from_file_location(
        "llamacpp_required_check_policy", REQUIRED_CHECK_POLICY
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load llama.cpp required-check policy")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_workflow(path: Path) -> dict:
    workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    if not isinstance(workflow, dict):
        raise ValueError(f"workflow must be a mapping: {path}")
    return workflow


def workflow_step(path: Path, job_name: str, step_name: str) -> dict:
    workflow = load_workflow(path)
    steps = workflow["jobs"][job_name]["steps"]
    matches = [step for step in steps if step.get("name") == step_name]
    if len(matches) != 1:
        raise ValueError(f"workflow step must be unique: {job_name}/{step_name}")
    return matches[0]


def extract_workflow_step_script(step_name: str) -> str:
    workflow = PR_WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split(f"      - name: {step_name}\n", 1)[1].split(
        "      - name:", 1
    )[0]
    return textwrap.dedent(step.split("        run: |\n", 1)[1])


def run_identity_step(
    snapshots: list[dict[str, object]],
    *,
    event_merge_sha: str,
    event_base_ref: str = "main",
) -> tuple[subprocess.CompletedProcess[str], dict[str, str], int]:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        snapshots_path = temporary / "snapshots.json"
        call_count_path = temporary / "call-count.txt"
        output_path = temporary / "github-output.txt"
        normalized_snapshots = json.loads(json.dumps(snapshots))
        for snapshot in normalized_snapshots:
            snapshot["base"].setdefault("ref", "main")
        snapshots_path.write_text(json.dumps(normalized_snapshots), encoding="utf-8")

        fake_gh = temporary / "gh"
        fake_gh.write_text(
            """#!/usr/bin/env python3
import json
import os
from pathlib import Path

snapshots = json.loads(Path(os.environ["FAKE_GH_SNAPSHOTS"]).read_text())
count_path = Path(os.environ["FAKE_GH_CALL_COUNT"])
count = int(count_path.read_text()) if count_path.exists() else 0
count_path.write_text(str(count + 1))
print(json.dumps(snapshots[min(count, len(snapshots) - 1)]))
""",
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)
        fake_sleep = temporary / "sleep"
        fake_sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake_sleep.chmod(0o755)

        environment = os.environ.copy()
        base = snapshots[0]["base"]
        head = snapshots[0]["head"]
        if not isinstance(base, dict) or not isinstance(head, dict):
            raise ValueError("fake pull request base and head must be objects")
        environment.update(
            {
                "EVENT_BASE_SHA": str(base["sha"]),
                "EVENT_BASE_REF": event_base_ref,
                "EVENT_HEAD_SHA": str(head["sha"]),
                "EVENT_MERGE_SHA": event_merge_sha,
                "EVENT_NAME": "pull_request_target",
                "FAKE_GH_CALL_COUNT": str(call_count_path),
                "FAKE_GH_SNAPSHOTS": str(snapshots_path),
                "GH_TOKEN": "test-token",
                "GITHUB_OUTPUT": str(output_path),
                "GITHUB_REPOSITORY": "LLM360/lemonade",
                "MERGE_GROUP_BASE_SHA": "",
                "MERGE_GROUP_HEAD_SHA": "",
                "PATH": f"{temporary}{os.pathsep}{environment['PATH']}",
                "PR_NUMBER": "7",
            }
        )
        completed = subprocess.run(
            [
                "bash",
                "-c",
                extract_workflow_step_script("Resolve the live candidate identity"),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        outputs = {}
        if output_path.exists():
            outputs = dict(
                line.split("=", 1)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            )
        calls = int(call_count_path.read_text()) if call_count_path.exists() else 0
        return completed, outputs, calls


def issue_event(
    event_id: int,
    event: str,
    *,
    actor: str = "maintainer",
    actor_id: int = 100,
    label: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "actor": {
            "id": actor_id,
            "login": actor,
            "node_id": f"USER_{actor_id}",
            "type": "User",
        },
        "created_at": f"2026-01-01T00:00:{event_id:02d}Z",
        "event": event,
        "id": event_id,
        "node_id": f"EVENT_{event_id}",
        "performed_via_github_app": None,
    }
    if label is not None:
        record["label"] = {"name": label}
    return record


def run_output_step(
    step_name: str, environment_values: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    with tempfile.TemporaryDirectory() as directory:
        output_path = Path(directory) / "github-output.txt"
        environment = os.environ.copy()
        environment.update(environment_values)
        environment["GITHUB_OUTPUT"] = str(output_path)
        completed = subprocess.run(
            ["bash", "-c", extract_workflow_step_script(step_name)],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        outputs = {}
        if output_path.exists():
            outputs = dict(
                line.split("=", 1)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            )
        return completed, outputs


def run_failed_closed_assessment(
    pull_request: dict[str, object],
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        output_path = temporary / "github-output.txt"
        fake_gh = temporary / "gh"
        fake_gh.write_text(
            f"#!/bin/sh\nprintf '%s\\n' '{json.dumps(pull_request)}'\n",
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "EVENT_ACTION": "closed",
                "EVENT_BASE_REF": "main",
                "EVENT_BASE_SHA": "a" * 40,
                "EVENT_HEAD_SHA": "b" * 40,
                "EVENT_MERGE_SHA": "c" * 40,
                "GITHUB_OUTPUT": str(output_path),
                "GITHUB_REPOSITORY": "LLM360/lemonade",
                "MODE": "",
                "PATH": f"{temporary}{os.pathsep}{environment['PATH']}",
                "POLICY_CONCLUSION": "",
                "POLICY_RESULT": "failure",
                "POLICY_SUMMARY": "",
                "PR_NUMBER": "7",
                "VALIDATION_RESULT": "skipped",
            }
        )
        completed = subprocess.run(
            [
                "bash",
                "-c",
                extract_workflow_step_script("Assess policy and validation"),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        outputs = {}
        if output_path.exists():
            outputs = dict(
                line.split("=", 1)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            )
        return completed, outputs


def run_policy_decision_step(
    *,
    action: str,
    event_label: str,
    event_actor: str,
    event_actor_id: int,
    event_updated_at: str,
    issue_events: list[list[dict[str, object]]],
    check_pages: list[dict[str, object]],
    authorization_label_present: bool = True,
    dependabot_pr: bool = False,
    fail_check_api: bool = False,
    permission: str = "write",
    permission_actor_id: int = 100,
    run_attempt: int = 1,
    run_conclusion: str = "failure",
    run_created_at: str = "2026-01-01T00:00:00Z",
    run_event: str = "pull_request_target",
    run_id: int = 500,
    run_number: int = 50,
    run_path: str = ".github/workflows/validate_llamacpp_pr.yml@main",
    run_repository: str = "LLM360/lemonade",
    run_response_attempt: int = 1,
    run_response_head_branch: str = "feature/k2-horizon",
    run_response_head_repository: str = "LLM360/lemonade",
    run_response_head_sha: str = "b" * 40,
    run_response_id: int | None = None,
    run_response_number: int | None = None,
    run_response_pr_base_ref: str = "main",
    run_response_pr_base_sha: str = "a" * 40,
    run_response_pr_head_ref: str = "feature/k2-horizon",
    run_response_pr_head_repository: str = "LLM360/lemonade",
    run_response_pr_head_sha: str = "b" * 40,
    run_response_pr_number: int = 7,
    run_response_pull_request_count: int = 1,
    run_response_pull_requests_json: str | None = None,
    run_started_at: str | None = None,
    run_status: str = "completed",
    triggering_actor: str = "maintainer",
    identity_matches: bool = True,
    relevant: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        events_path = temporary / "events.json"
        checks_path = temporary / "checks.json"
        output_path = temporary / "github-output.txt"
        events_path.write_text(json.dumps(issue_events), encoding="utf-8")
        checks_path.write_text(json.dumps(check_pages), encoding="utf-8")
        fake_gh = temporary / "gh"
        fake_gh.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

endpoint = next(
    (argument for argument in sys.argv[1:] if argument.startswith("repos/")), ""
)
if "/issues/" in endpoint:
    print(Path(os.environ["FAKE_GH_EVENTS"]).read_text(encoding="utf-8"))
elif "/actions/runs/" in endpoint:
    pull_request = {
        "base": {
            "ref": os.environ["FAKE_RUN_PR_BASE_REF"],
            "repo": {"url": os.environ["FAKE_RUN_REPOSITORY_URL"]},
            "sha": os.environ["FAKE_RUN_PR_BASE_SHA"],
        },
        "head": {
            "ref": os.environ["FAKE_RUN_PR_HEAD_REF"],
            "repo": {"url": os.environ["FAKE_RUN_PR_HEAD_REPOSITORY_URL"]},
            "sha": os.environ["FAKE_RUN_PR_HEAD_SHA"],
        },
        "id": 4453422506,
        "number": int(os.environ["FAKE_RUN_PR_NUMBER"]),
        "url": os.environ["FAKE_RUN_PR_URL"],
    }
    pull_requests = [pull_request] * int(os.environ["FAKE_RUN_PR_COUNT"])
    if os.environ["FAKE_RUN_PULL_REQUESTS_JSON"]:
        pull_requests = json.loads(os.environ["FAKE_RUN_PULL_REQUESTS_JSON"])
    print(json.dumps({
        "conclusion": os.environ["FAKE_RUN_CONCLUSION"],
        "created_at": os.environ["FAKE_RUN_CREATED_AT"],
        "event": os.environ["FAKE_RUN_EVENT"],
        "head_branch": os.environ["FAKE_RUN_RESPONSE_HEAD_BRANCH"],
        "head_repository": {"full_name": os.environ["FAKE_RUN_RESPONSE_HEAD_REPOSITORY"]},
        "head_sha": os.environ["FAKE_RUN_RESPONSE_HEAD_SHA"],
        "id": int(os.environ["FAKE_RUN_RESPONSE_ID"]),
        "path": os.environ["FAKE_RUN_PATH"],
        "pull_requests": pull_requests,
        "repository": {"full_name": os.environ["FAKE_RUN_REPOSITORY"], "id": 1356276914},
        "run_attempt": int(os.environ["FAKE_RUN_RESPONSE_ATTEMPT"]),
        "run_number": int(os.environ["FAKE_RUN_RESPONSE_NUMBER"]),
        "run_started_at": os.environ["FAKE_RUN_STARTED_AT"],
        "status": os.environ["FAKE_RUN_STATUS"],
    }))
elif "/collaborators/" in endpoint:
    print(json.dumps({
        "permission": os.environ["FAKE_GH_PERMISSION"],
        "user": {"id": int(os.environ["FAKE_GH_PERMISSION_ACTOR_ID"])},
    }))
elif "/check-runs" in endpoint:
    if os.environ["FAKE_FAIL_CHECK_API"] == "true":
        raise SystemExit("Checks API is unavailable")
    print(Path(os.environ["FAKE_GH_CHECKS"]).read_text(encoding="utf-8"))
else:
    raise SystemExit(f"unexpected endpoint: {endpoint}")
""",
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "AUTHORIZATION_LABEL_PRESENT": str(authorization_label_present).lower(),
                "BASE_REF_CHANGED": "false",
                "CHECK_NAME": "llama.cpp authorized validation v2",
                "CURRENT_RUN_ATTEMPT": str(run_attempt),
                "CURRENT_RUN_ID": str(run_id),
                "CURRENT_RUN_NUMBER": str(run_number),
                "DEPENDABOT_PR": str(dependabot_pr).lower(),
                "EVENT_ACTION": action,
                "EVENT_ACTOR": event_actor,
                "EVENT_ACTOR_ID": str(event_actor_id),
                "EVENT_BASE_REF": "main",
                "EVENT_BASE_SHA": "a" * 40,
                "EVENT_LABEL": event_label,
                "EVENT_HEAD_SHA": "b" * 40,
                "EVENT_NAME": "pull_request_target",
                "EVENT_UPDATED_AT": event_updated_at,
                "FAKE_GH_CHECKS": str(checks_path),
                "FAKE_GH_EVENTS": str(events_path),
                "FAKE_GH_PERMISSION": permission,
                "FAKE_GH_PERMISSION_ACTOR_ID": str(permission_actor_id),
                "FAKE_FAIL_CHECK_API": str(fail_check_api).lower(),
                "FAKE_RUN_CONCLUSION": run_conclusion,
                "FAKE_RUN_CREATED_AT": run_created_at,
                "FAKE_RUN_EVENT": run_event,
                "FAKE_RUN_PATH": run_path,
                "FAKE_RUN_REPOSITORY": run_repository,
                "FAKE_RUN_REPOSITORY_URL": (
                    f"https://api.github.com/repos/{run_repository}"
                ),
                "FAKE_RUN_RESPONSE_ATTEMPT": str(run_response_attempt),
                "FAKE_RUN_RESPONSE_HEAD_BRANCH": run_response_head_branch,
                "FAKE_RUN_RESPONSE_HEAD_REPOSITORY": run_response_head_repository,
                "FAKE_RUN_RESPONSE_HEAD_SHA": run_response_head_sha,
                "FAKE_RUN_RESPONSE_ID": str(
                    run_id if run_response_id is None else run_response_id
                ),
                "FAKE_RUN_RESPONSE_NUMBER": str(
                    run_number if run_response_number is None else run_response_number
                ),
                "FAKE_RUN_PR_BASE_REF": run_response_pr_base_ref,
                "FAKE_RUN_PR_BASE_SHA": run_response_pr_base_sha,
                "FAKE_RUN_PR_COUNT": str(run_response_pull_request_count),
                "FAKE_RUN_PR_HEAD_REF": run_response_pr_head_ref,
                "FAKE_RUN_PR_HEAD_REPOSITORY_URL": (
                    "https://api.github.com/repos/" f"{run_response_pr_head_repository}"
                ),
                "FAKE_RUN_PR_HEAD_SHA": run_response_pr_head_sha,
                "FAKE_RUN_PR_NUMBER": str(run_response_pr_number),
                "FAKE_RUN_PULL_REQUESTS_JSON": (run_response_pull_requests_json or ""),
                "FAKE_RUN_PR_URL": (
                    f"https://api.github.com/repos/{run_repository}/pulls/"
                    f"{run_response_pr_number}"
                ),
                "FAKE_RUN_STARTED_AT": (
                    run_created_at if run_started_at is None else run_started_at
                ),
                "FAKE_RUN_STATUS": run_status,
                "GH_TOKEN": "test-token",
                "GITHUB_OUTPUT": str(output_path),
                "GITHUB_API_URL": "https://api.github.com",
                "GITHUB_REPOSITORY": "LLM360/lemonade",
                "IDENTITY_MATCHES": str(identity_matches).lower(),
                "MERGED": "false",
                "MERGE_CANDIDATE_AVAILABLE": "true",
                "MERGE_SHA": "c" * 40,
                "PATH": f"{temporary}{os.pathsep}{environment['PATH']}",
                "PR_NUMBER": "7",
                "RELEVANT": str(relevant).lower(),
                "RUNNER_TEMP": str(temporary),
                "STATE": "open",
                "TRIGGERING_ACTOR": triggering_actor,
            }
        )
        script = extract_workflow_step_script(
            "Decide whether protected validation may run"
        ).split("\nvalidate:\n", 1)[0]
        script = script.replace(
            "trusted/.github/scripts/llamacpp_required_check_policy.py",
            str(REQUIRED_CHECK_POLICY),
        )
        completed = subprocess.run(
            ["bash", "-c", script],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        outputs = {}
        if output_path.exists():
            outputs = dict(
                line.split("=", 1)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            )
        return completed, outputs


def run_observed_merge_invalidation(
    check_pages: list[dict[str, object]],
    *,
    observed_merge_sha: str,
    secondary_observed_merge_sha: str = "",
    authorization_epoch: int = 1,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        pages_path = temporary / "check-pages.json"
        calls_path = temporary / "gh-calls.jsonl"
        pages_path.write_text(json.dumps(check_pages), encoding="utf-8")

        fake_gh = temporary / "gh"
        fake_gh.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
with Path(os.environ["FAKE_GH_CALLS"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(arguments) + "\\n")
method = arguments[arguments.index("--method") + 1] if "--method" in arguments else "GET"
if method == "GET":
    print(Path(os.environ["FAKE_GH_CHECK_PAGES"]).read_text(encoding="utf-8"))
elif method == "POST":
    print('{"id": 80}')
else:
    print('{}')
""",
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "CHECK_NAME": "llama.cpp authorization diagnostics v2",
                "AUTHORIZATION_EPOCH": str(authorization_epoch),
                "FAKE_GH_CALLS": str(calls_path),
                "FAKE_GH_CHECK_PAGES": str(pages_path),
                "GH_TOKEN": "test-token",
                "GITHUB_REPOSITORY": "LLM360/lemonade",
                "GITHUB_RUN_ID": "500",
                "GITHUB_SERVER_URL": "https://github.com",
                "OBSERVED_MERGE_SHA": observed_merge_sha,
                "PATH": f"{temporary}{os.pathsep}{environment['PATH']}",
                "PR_NUMBER": "7",
                "RUN_ATTEMPT": "1",
                "RUN_ID": "500",
                "RUN_NUMBER": "50",
                "SECONDARY_OBSERVED_MERGE_SHA": secondary_observed_merge_sha,
            }
        )
        completed = subprocess.run(
            [
                "bash",
                "-c",
                extract_workflow_step_script(
                    "Invalidate an unconfirmed observed merge commit"
                ),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        calls = []
        if calls_path.exists():
            calls = [
                json.loads(line)
                for line in calls_path.read_text(encoding="utf-8").splitlines()
            ]
        return completed, calls


def run_preserve_step(
    *,
    pull_request: dict[str, object],
    issue_events: list[list[dict[str, object]]],
    check_pages: list[dict[str, object]],
    run_attempt: int = 1,
    run_number: int = 50,
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        output_path = temporary / "github-output.txt"
        inputs_path = temporary / "inputs.json"
        inputs_path.write_text(
            json.dumps(
                {
                    "checks": check_pages,
                    "events": issue_events,
                    "pull_request": pull_request,
                }
            ),
            encoding="utf-8",
        )
        fake_gh = temporary / "gh"
        fake_gh.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

inputs = json.loads(Path(os.environ["FAKE_GH_INPUTS"]).read_text())
endpoint = next(
    (argument for argument in sys.argv[1:] if argument.startswith("repos/")), ""
)
if "/pulls/" in endpoint:
    print(json.dumps(inputs["pull_request"]))
elif "/issues/" in endpoint:
    print(json.dumps(inputs["events"]))
elif "/collaborators/" in endpoint:
    print('{"permission":"write","user":{"id":100}}')
elif "/check-runs" in endpoint:
    print(json.dumps(inputs["checks"]))
else:
    raise SystemExit(f"unexpected endpoint: {endpoint}")
""",
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "AUTHORIZATION_ACTOR": "maintainer",
                "AUTHORIZATION_ACTOR_ID": "100",
                "AUTHORIZATION_BOUNDARY": "2025-12-31T23:59:59Z",
                "AUTHORIZATION_CREATED_AT": "2026-01-01T00:00:01Z",
                "AUTHORIZATION_EPOCH": "1",
                "BASE_REF": "main",
                "BASE_SHA": "a" * 40,
                "CHECK_NAME": "llama.cpp authorized validation v2",
                "FAKE_GH_INPUTS": str(inputs_path),
                "GH_TOKEN": "test-token",
                "GITHUB_OUTPUT": str(output_path),
                "GITHUB_REPOSITORY": "LLM360/lemonade",
                "HEAD_SHA": "b" * 40,
                "MERGE_SHA": "c" * 40,
                "PATH": f"{temporary}{os.pathsep}{environment['PATH']}",
                "POLICY_HELPER": str(REQUIRED_CHECK_POLICY),
                "PR_NUMBER": "7",
                "RUN_ATTEMPT": str(run_attempt),
                "RUN_NUMBER": str(run_number),
                "RUNNER_TEMP": str(temporary),
            }
        )
        completed = subprocess.run(
            [
                "bash",
                "-c",
                extract_workflow_step_script(
                    "Preserve the exact-commit gate for unrelated metadata"
                ),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        outputs = {}
        if output_path.exists():
            outputs = dict(
                line.split("=", 1)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            )
        return completed, outputs


def run_report_step(
    check_page_snapshots: list[list[dict[str, object]]],
    *,
    pull_request_snapshots: list[dict[str, object]] | None = None,
    issue_event_snapshots: list[list[list[dict[str, object]]]] | None = None,
    permission: str = "write",
    permission_snapshots: list[str] | None = None,
    run_attempt: int = 1,
    run_id: int = 500,
    run_number: int = 50,
    mode: str = "validate",
    conclusion: str = "success",
    preserve_valid: bool = False,
    event_action: str = "synchronize",
    event_name: str | None = None,
    authorization_created_at: str = "2026-01-01T00:00:01Z",
    authorization_epoch: int = 1,
    step_name: str = "Report result on the exact candidate commit",
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        snapshots_path = temporary / "check-snapshots.json"
        check_count_path = temporary / "check-count.txt"
        event_count_path = temporary / "event-count.txt"
        pr_count_path = temporary / "pr-count.txt"
        permission_count_path = temporary / "permission-count.txt"
        events_path = temporary / "event-snapshots.json"
        pull_requests_path = temporary / "pr-snapshots.json"
        calls_path = temporary / "gh-calls.jsonl"
        snapshots_path.write_text(json.dumps(check_page_snapshots), encoding="utf-8")
        events_path.write_text(
            json.dumps(issue_event_snapshots or []), encoding="utf-8"
        )
        permission_path = temporary / "permission-snapshots.json"
        permission_path.write_text(
            json.dumps(permission_snapshots or [permission]), encoding="utf-8"
        )
        normalized_pull_requests = json.loads(json.dumps(pull_request_snapshots or []))
        for pull_request in normalized_pull_requests:
            pull_request["base"].setdefault("ref", "main")
        pull_requests_path.write_text(
            json.dumps(normalized_pull_requests), encoding="utf-8"
        )

        fake_gh = temporary / "gh"
        fake_gh.write_text(
            """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
with Path(os.environ["FAKE_GH_CALLS"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(arguments) + "\\n")
method = arguments[arguments.index("--method") + 1] if "--method" in arguments else "GET"
endpoint = next((argument for argument in arguments if argument.startswith("repos/")), "")
if method == "GET" and "/check-runs" in endpoint and "/commits/" in endpoint:
    count_path = Path(os.environ["FAKE_GH_CHECK_COUNT"])
    count = int(count_path.read_text()) if count_path.exists() else 0
    count_path.write_text(str(count + 1))
    snapshots = json.loads(Path(os.environ["FAKE_GH_SNAPSHOTS"]).read_text())
    print(json.dumps(snapshots[min(count, len(snapshots) - 1)]))
elif method == "GET" and "/pulls/" in endpoint:
    count_path = Path(os.environ["FAKE_GH_PR_COUNT"])
    count = int(count_path.read_text()) if count_path.exists() else 0
    count_path.write_text(str(count + 1))
    snapshots = json.loads(Path(os.environ["FAKE_GH_PULL_REQUESTS"]).read_text())
    print(json.dumps(snapshots[min(count, len(snapshots) - 1)]))
elif method == "GET" and "/issues/" in endpoint and endpoint.endswith("/events?per_page=100"):
    count_path = Path(os.environ["FAKE_GH_EVENT_COUNT"])
    count = int(count_path.read_text()) if count_path.exists() else 0
    count_path.write_text(str(count + 1))
    snapshots = json.loads(Path(os.environ["FAKE_GH_EVENTS"]).read_text())
    print(json.dumps(snapshots[min(count, len(snapshots) - 1)]))
elif method == "GET" and "/collaborators/" in endpoint:
    count_path = Path(os.environ["FAKE_GH_PERMISSION_COUNT"])
    count = int(count_path.read_text()) if count_path.exists() else 0
    count_path.write_text(str(count + 1))
    permissions = json.loads(Path(os.environ["FAKE_GH_PERMISSIONS"]).read_text())
    permission = permissions[min(count, len(permissions) - 1)]
    if "--jq" in arguments:
        print(permission)
    else:
        print(json.dumps({
            "permission": permission,
            "user": {"id": 100},
        }))
elif method == "POST":
    print('{"id": 80}')
else:
    print('{}')
""",
            encoding="utf-8",
        )
        fake_gh.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "BASE_SHA": "a" * 40,
                "BASE_REF": "main",
                "CHECK_NAME": "llama.cpp authorized validation v2",
                "CHECK_RUN_ID": "",
                "CONCLUSION": conclusion,
                "EVENT_ACTION": event_action,
                "EVENT_NAME": (
                    event_name
                    if event_name is not None
                    else (
                        "pull_request_target"
                        if normalized_pull_requests
                        else "merge_group"
                    )
                ),
                "FAKE_GH_CALLS": str(calls_path),
                "FAKE_GH_CHECK_COUNT": str(check_count_path),
                "FAKE_GH_EVENT_COUNT": str(event_count_path),
                "FAKE_GH_EVENTS": str(events_path),
                "FAKE_GH_PERMISSION_COUNT": str(permission_count_path),
                "FAKE_GH_PERMISSIONS": str(permission_path),
                "FAKE_GH_PR_COUNT": str(pr_count_path),
                "FAKE_GH_PULL_REQUESTS": str(pull_requests_path),
                "FAKE_GH_SNAPSHOTS": str(snapshots_path),
                "GH_TOKEN": "test-token",
                "GITHUB_REPOSITORY": "LLM360/lemonade",
                "GITHUB_RUN_ID": str(run_id),
                "GITHUB_SERVER_URL": "https://github.com",
                "HEAD_SHA": "b" * 40,
                "MERGE_SHA": "c" * 40,
                "METADATA_CHECK_NAME": "llama.cpp authorization diagnostics v2",
                "MODE": mode,
                "PATH": f"{temporary}{os.pathsep}{environment['PATH']}",
                "POLICY_HELPER": str(REQUIRED_CHECK_POLICY),
                "PR_NUMBER": "7",
                "PRESERVE_VALID": str(preserve_valid).lower(),
                "RUN_ATTEMPT": str(run_attempt),
                "RUN_ID": str(run_id),
                "RUN_NUMBER": str(run_number),
                "SUMMARY": "Authorized validation passed.",
                "TARGET_KEY": "pr-7",
                "AUTHORIZATION_ACTOR": "maintainer",
                "AUTHORIZATION_ACTOR_ID": "100",
                "AUTHORIZATION_BOUNDARY": "2025-12-31T23:59:59Z",
                "AUTHORIZATION_CREATED_AT": authorization_created_at,
                "AUTHORIZATION_EPOCH": str(authorization_epoch),
            }
        )
        completed = subprocess.run(
            [
                "bash",
                "-c",
                extract_workflow_step_script(step_name),
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        calls = []
        if calls_path.exists():
            calls = [
                json.loads(line)
                for line in calls_path.read_text(encoding="utf-8").splitlines()
            ]
        return completed, calls


class LlamaCppValidationPlanTests(unittest.TestCase):
    def test_authorization_provenance_is_a_fully_validated_epoch(self) -> None:
        policy = load_required_check_policy()
        valid = policy.classify_authorization_event_pages(
            [
                [
                    issue_event(1, "reopened"),
                    issue_event(2, "labeled", label="ci:upgrades"),
                ]
            ]
        )
        self.assertTrue(valid.epoch_valid)
        self.assertEqual(valid.authorization_event_id, 2)
        self.assertEqual(valid.authorization_actor, "maintainer")
        self.assertEqual(valid.authorization_actor_id, 100)
        self.assertEqual(valid.authorization_created_at, "2026-01-01T00:00:02Z")
        self.assertTrue(
            policy.classify_authorization_event_pages(
                [[issue_event(2, "labeled", label="ci:upgrades")]],
                authorization_after="2026-01-01T00:00:01Z",
            ).epoch_valid
        )
        for boundary in (
            "2026-01-01T00:00:02Z",
            "2026-01-01T00:00:03Z",
        ):
            with self.subTest(boundary=boundary):
                stale_for_run = policy.classify_authorization_event_pages(
                    [[issue_event(2, "labeled", label="ci:upgrades")]],
                    authorization_after=boundary,
                )
                self.assertFalse(stale_for_run.epoch_valid)
        for invalid_boundary in ("not-a-time", "2026-01-01T00:00:00"):
            with self.subTest(invalid_boundary=invalid_boundary):
                with self.assertRaises(ValueError):
                    policy.classify_authorization_event_pages(
                        [[issue_event(2, "labeled", label="ci:upgrades")]],
                        authorization_after=invalid_boundary,
                    )

        for invalidator in ("base_ref_changed", "reopened", "head_ref_force_pushed"):
            with self.subTest(invalidator=invalidator):
                invalid = policy.classify_authorization_event_pages(
                    [
                        [
                            issue_event(1, "labeled", label="ci:upgrades"),
                            issue_event(2, invalidator),
                        ]
                    ]
                )
                self.assertFalse(invalid.epoch_valid)

        untrusted_readd = policy.classify_authorization_event_pages(
            [
                [
                    issue_event(1, "labeled", actor="trusted", label="ci:upgrades"),
                    issue_event(2, "unlabeled", label="ci:upgrades"),
                    issue_event(3, "labeled", actor="triage", label="ci:upgrades"),
                ]
            ]
        )
        self.assertTrue(untrusted_readd.epoch_valid)
        self.assertEqual(untrusted_readd.authorization_actor, "triage")
        unrelated_bot_label = issue_event(
            1,
            "labeled",
            actor="github-actions[bot]",
            actor_id=41898282,
            label="automation:classified",
        )
        unrelated_bot_label["actor"]["type"] = "Bot"
        unrelated_bot_label["performed_via_github_app"] = {"id": 15368}
        bot_label_before_authorization = policy.classify_authorization_event_pages(
            [
                [
                    unrelated_bot_label,
                    issue_event(2, "labeled", label="ci:upgrades"),
                ]
            ]
        )
        self.assertTrue(bot_label_before_authorization.epoch_valid)
        self.assertEqual(
            bot_label_before_authorization.authorization_actor,
            "maintainer",
        )
        stale_epoch = policy.decide_pull_request_policy(
            action="labeled",
            event_label="ci:upgrades",
            base_ref_changed=False,
            state="open",
            merged=False,
            merge_candidate_available=True,
            relevant=True,
            identity_matches=True,
            authorization_label_present=True,
            authorization_epoch_valid=False,
            actor_permission="write",
            has_existing_check=True,
            has_newer_check=False,
        )
        self.assertEqual(stale_epoch.mode, "awaiting_rerun")
        self.assertEqual(stale_epoch.conclusion, "failure")
        delayed_delivery = policy.decide_pull_request_policy(
            action="labeled",
            event_label="ci:upgrades",
            base_ref_changed=False,
            state="open",
            merged=False,
            merge_candidate_available=True,
            relevant=True,
            identity_matches=True,
            authorization_label_present=True,
            authorization_epoch_valid=True,
            authorization_rerun_matches=False,
            actor_permission="write",
            has_existing_check=False,
            has_newer_check=False,
        )
        self.assertEqual(delayed_delivery.mode, "awaiting_rerun")
        with self.assertRaises(ValueError):
            policy.classify_authorization_event_pages(
                [
                    [
                        issue_event(1, "reopened"),
                        issue_event(1, "labeled", label="ci:upgrades"),
                    ]
                ]
            )
        malformed_actor = issue_event(1, "labeled", label="ci:upgrades")
        malformed_actor["actor"] = {
            "id": 100,
            "login": "maintainer",
            "node_id": "USER_100",
            "type": "Bot",
        }
        duplicate_node = issue_event(2, "reopened")
        duplicate_node["node_id"] = "EVENT_1"
        for pages in (
            [[], [issue_event(1, "reopened")]],
            [[issue_event(2, "reopened"), issue_event(1, "closed")]],
            [[issue_event(1, "reopened"), duplicate_node]],
            [[malformed_actor]],
        ):
            with self.subTest(invalid_pages=pages):
                with self.assertRaises(ValueError):
                    policy.classify_authorization_event_pages(pages)

    def test_live_authorization_epoch_binding_is_enforced(self) -> None:
        events = [
            [
                issue_event(1, "labeled", label="ci:upgrades"),
                issue_event(2, "unlabeled", label="ci:upgrades"),
                issue_event(3, "labeled", label="ci:upgrades"),
            ]
        ]
        current_epoch_check = {
            "app": {"slug": "github-actions"},
            "conclusion": "success",
            "external_id": "lemonade-llamacpp-required-pr-7-3-40-1-400",
            "id": 40,
            "name": "llama.cpp authorized validation v2",
            "status": "completed",
        }
        old_epoch_check = {
            **current_epoch_check,
            "external_id": "lemonade-llamacpp-required-pr-7-1-40-1-400",
        }

        authorized, authorized_outputs = run_policy_decision_step(
            action="synchronize",
            event_label="",
            event_actor="contributor",
            event_actor_id=200,
            event_updated_at="2026-01-01T00:00:00Z",
            issue_events=events,
            check_pages=[{"check_runs": []}],
            run_attempt=2,
        )
        self.assertEqual(authorized.returncode, 0, authorized.stderr)
        self.assertEqual(authorized_outputs["mode"], "validate")
        self.assertEqual(authorized_outputs["authorization_event_id"], "3")

        for run_path in (
            ".github/workflows/validate_llamacpp_pr.yml",
            ".github/workflows/validate_llamacpp_pr.yml@refs/heads/main",
        ):
            with self.subTest(allowed_workflow_path=run_path):
                allowed, allowed_outputs = run_policy_decision_step(
                    action="synchronize",
                    event_label="",
                    event_actor="contributor",
                    event_actor_id=200,
                    event_updated_at="2026-01-01T00:00:00Z",
                    issue_events=events,
                    check_pages=[{"check_runs": []}],
                    run_attempt=2,
                    run_path=run_path,
                )
                self.assertEqual(allowed.returncode, 0, allowed.stderr)
                self.assertEqual(allowed_outputs["mode"], "validate")

        fork, fork_outputs = run_policy_decision_step(
            action="synchronize",
            event_label="",
            event_actor="contributor",
            event_actor_id=200,
            event_updated_at="2026-01-01T00:00:00Z",
            issue_events=events,
            check_pages=[{"check_runs": []}],
            run_attempt=2,
            run_response_head_branch="fork/k2-horizon",
            run_response_head_repository="contributor/lemonade",
            run_response_pr_head_ref="fork/k2-horizon",
            run_response_pr_head_repository="contributor/lemonade",
            run_response_pull_request_count=0,
        )
        self.assertEqual(fork.returncode, 0, fork.stderr)
        self.assertEqual(fork_outputs["mode"], "validate")

        multiple, multiple_outputs = run_policy_decision_step(
            action="synchronize",
            event_label="",
            event_actor="contributor",
            event_actor_id=200,
            event_updated_at="2026-01-01T00:00:00Z",
            issue_events=events,
            check_pages=[{"check_runs": []}],
            run_attempt=2,
            run_response_pr_base_ref="release",
            run_response_pr_base_sha="d" * 40,
            run_response_pr_head_sha="e" * 40,
            run_response_pr_number=99,
            run_response_pull_request_count=3,
        )
        self.assertEqual(multiple.returncode, 0, multiple.stderr)
        self.assertEqual(multiple_outputs["mode"], "validate")

        for name, overrides, expected_mode in (
            ("first-attempt", {"run_attempt": 1}, "authorization_required"),
            (
                "label-not-after-run",
                {"run_created_at": "2026-01-01T00:00:03Z"},
                "authorization_required",
            ),
            (
                "label-after-queue-but-before-start",
                {
                    "run_created_at": "2025-12-31T23:59:59Z",
                    "run_started_at": "2026-01-01T00:00:04Z",
                },
                "authorization_required",
            ),
            (
                "missing-run-start",
                {"run_started_at": ""},
                "authorization_required",
            ),
            (
                "wrong-permission-identity",
                {"permission_actor_id": 999},
                "authorization_required",
            ),
            (
                "untrusted-rerun-actor",
                {"triggering_actor": "triage"},
                "authorization_required",
            ),
            (
                "first-attempt-succeeded",
                {"run_conclusion": "success"},
                "authorization_required",
            ),
            (
                "wrong-run-event",
                {"run_event": "pull_request"},
                "authorization_required",
            ),
            (
                "wrong-workflow",
                {"run_path": ".github/workflows/other.yml"},
                "authorization_required",
            ),
            (
                "wrong-workflow-ref",
                {"run_path": (".github/workflows/validate_llamacpp_pr.yml@release")},
                "authorization_required",
            ),
            (
                "malformed-workflow-ref",
                {
                    "run_path": (
                        ".github/workflows/validate_llamacpp_pr.yml@main@release"
                    )
                },
                "authorization_required",
            ),
            (
                "wrong-repository",
                {"run_repository": "attacker/lemonade"},
                "authorization_required",
            ),
            (
                "wrong-top-level-head-sha",
                {"run_response_head_sha": "d" * 40},
                "authorization_required",
            ),
            (
                "malformed-pull-request-associations",
                {"run_response_pull_requests_json": '"not-an-array"'},
                "authorization_required",
            ),
            (
                "live-identity-drift",
                {"identity_matches": False},
                "stale_required",
            ),
            (
                "wrong-run-id",
                {"run_response_id": 501},
                "authorization_required",
            ),
            (
                "wrong-run-number",
                {"run_response_number": 51},
                "authorization_required",
            ),
            (
                "wrong-attempt-record",
                {"run_response_attempt": 2},
                "authorization_required",
            ),
            (
                "incomplete-first-attempt",
                {"run_status": "in_progress"},
                "authorization_required",
            ),
            (
                "label-event-rerun",
                {"action": "labeled", "event_label": "ci:upgrades"},
                "awaiting_rerun",
            ),
        ):
            arguments = {
                "action": "synchronize",
                "event_label": "",
                "event_actor": "contributor",
                "event_actor_id": 200,
                "event_updated_at": "2026-01-01T00:00:00Z",
                "issue_events": events,
                "check_pages": [{"check_runs": []}],
                "run_attempt": 2,
            }
            arguments.update(overrides)
            with self.subTest(name=name):
                denied, denied_outputs = run_policy_decision_step(**arguments)
                self.assertEqual(denied.returncode, 0, denied.stderr)
                self.assertEqual(denied_outputs["mode"], expected_mode)
                self.assertEqual(denied_outputs["conclusion"], "failure")

        for name, checks, expected_mode in (
            ("same-epoch", [current_epoch_check], "preserve"),
            ("old-epoch", [old_epoch_check], "authorization_required"),
            (
                "in-progress",
                [{**current_epoch_check, "conclusion": None, "status": "in_progress"}],
                "authorization_required",
            ),
            (
                "failed",
                [{**current_epoch_check, "conclusion": "failure"}],
                "authorization_required",
            ),
            (
                "later-failed-decision",
                [
                    current_epoch_check,
                    {
                        **current_epoch_check,
                        "conclusion": "failure",
                        "external_id": ("lemonade-llamacpp-required-pr-7-3-41-1-401"),
                        "id": 41,
                    },
                ],
                "authorization_required",
            ),
            ("missing", [], "authorization_required"),
        ):
            with self.subTest(name=name):
                metadata, metadata_outputs = run_policy_decision_step(
                    action="edited",
                    event_label="",
                    event_actor="maintainer",
                    event_actor_id=100,
                    event_updated_at="2026-01-01T00:00:04Z",
                    issue_events=events,
                    check_pages=[{"check_runs": checks}],
                )
                self.assertEqual(metadata.returncode, 0, metadata.stderr)
                self.assertEqual(metadata_outputs["mode"], expected_mode)
                if expected_mode != "preserve":
                    self.assertEqual(metadata_outputs["conclusion"], "failure")

    def test_dependabot_policy_does_not_require_the_checks_api(self) -> None:
        authorization = [[issue_event(3, "labeled", label="ci:upgrades")]]
        cases = (
            (
                "neutral",
                {
                    "action": "synchronize",
                    "authorization_label_present": False,
                    "issue_events": [[]],
                    "relevant": False,
                },
                "neutral",
                "success",
            ),
            (
                "unauthorized",
                {
                    "action": "synchronize",
                    "authorization_label_present": False,
                    "issue_events": [[]],
                },
                "authorization_required",
                "failure",
            ),
            (
                "awaiting-rerun",
                {
                    "action": "labeled",
                    "event_label": "ci:upgrades",
                    "issue_events": authorization,
                },
                "awaiting_rerun",
                "failure",
            ),
            (
                "metadata-preserve",
                {"action": "edited", "issue_events": authorization},
                "preserve",
                "",
            ),
            (
                "authorized-rerun",
                {
                    "action": "synchronize",
                    "issue_events": authorization,
                    "run_attempt": 2,
                },
                "validate",
                "",
            ),
        )
        for name, overrides, expected_mode, expected_conclusion in cases:
            arguments = {
                "action": "synchronize",
                "event_label": "",
                "event_actor": "dependabot[bot]",
                "event_actor_id": 49699333,
                "event_updated_at": "2026-01-01T00:00:00Z",
                "issue_events": authorization,
                "check_pages": [{"check_runs": []}],
                "dependabot_pr": True,
                "fail_check_api": True,
            }
            arguments.update(overrides)
            with self.subTest(name=name):
                completed, outputs = run_policy_decision_step(**arguments)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(outputs["mode"], expected_mode)
                self.assertEqual(outputs["conclusion"], expected_conclusion)

    def test_dependabot_pr_uses_only_the_required_workflow_conclusion(self) -> None:
        expected_identity = (
            "${{ github.event_name == 'pull_request_target' && "
            "github.event.pull_request.user.login == 'dependabot[bot]' && "
            "github.event.pull_request.user.id == 49699333 && "
            "github.event.pull_request.user.type == 'Bot' }}"
        )
        self.assertEqual(
            load_workflow(PR_WORKFLOW)["env"]["DEPENDABOT_PR"], expected_identity
        )
        self.assertEqual(
            load_workflow(VALIDATION_WORKFLOW)["env"]["DEPENDABOT_PR"],
            expected_identity,
        )

        decision = workflow_step(
            PR_WORKFLOW, "policy", "Decide whether protected validation may run"
        )
        self.assertEqual(decision["env"]["DEPENDABOT_PR"], "${{ env.DEPENDABOT_PR }}")
        self.assertIn('if [ "$DEPENDABOT_PR" != "true" ]', decision["run"])
        self.assertIn('--required-workflow-only "$DEPENDABOT_PR"', decision["run"])

        for step_name in (
            "Preserve the exact-commit gate for unrelated metadata",
            "Invalidate an unconfirmed observed merge commit",
            "Report result on the exact candidate commit",
        ):
            with self.subTest(step=step_name):
                condition = workflow_step(PR_WORKFLOW, "reconcile", step_name)["if"]
                self.assertIn("env.DEPENDABOT_PR != 'true'", condition)

        start = workflow_step(
            VALIDATION_WORKFLOW, "validate-invocation", "Start authorized merge check"
        )
        self.assertIn("env.DEPENDABOT_PR != 'true'", start["if"])
        enforce = workflow_step(PR_WORKFLOW, "reconcile", "Enforce policy result")["if"]
        self.assertIn("env.DEPENDABOT_PR != 'true'", enforce)

        guide = TESTING_GUIDE.read_text(encoding="utf-8")
        self.assertIn("Dependabot", guide)
        self.assertIn("read-only `GITHUB_TOKEN`", guide)
        self.assertIn("required-workflow conclusion", guide)
        self.assertIn("omit custom diagnostic Checks API runs", guide)

    def test_required_workflow_only_metadata_preserves_without_a_custom_check(
        self,
    ) -> None:
        policy = load_required_check_policy()
        common = {
            "action": "edited",
            "event_label": "",
            "base_ref_changed": False,
            "state": "open",
            "merged": False,
            "merge_candidate_available": True,
            "relevant": True,
            "identity_matches": True,
            "authorization_label_present": True,
            "actor_permission": "write",
            "has_existing_check": False,
            "has_newer_check": False,
            "authorization_epoch_valid": True,
            "authorization_rerun_matches": False,
            "required_workflow_only": True,
        }
        preserved = policy.decide_pull_request_policy(**common)
        self.assertEqual(preserved.mode, "preserve")
        self.assertFalse(preserved.run_validation)
        self.assertEqual(preserved.conclusion, "")

        for name, overrides, expected_mode in (
            (
                "missing-authorization",
                {"authorization_label_present": False},
                "authorization_required",
            ),
            (
                "untrusted-authorization",
                {"actor_permission": "triage"},
                "authorization_required",
            ),
            (
                "stale-identity",
                {"identity_matches": False},
                "stale",
            ),
            (
                "base-ref-change",
                {"base_ref_changed": True},
                "authorization_required",
            ),
            (
                "protected-synchronize",
                {"action": "synchronize"},
                "authorization_required",
            ),
        ):
            arguments = dict(common)
            arguments.update(overrides)
            with self.subTest(name=name):
                self.assertEqual(
                    policy.decide_pull_request_policy(**arguments).mode,
                    expected_mode,
                )

    def test_dependabot_authorized_result_is_revalidated_without_writes(self) -> None:
        pull_request = {
            "base": {"ref": "main", "sha": "a" * 40},
            "head": {"sha": "b" * 40},
            "labels": [{"name": "ci:upgrades"}],
            "merge_commit_sha": "c" * 40,
            "mergeable": True,
            "merged": False,
            "number": 7,
            "state": "open",
        }
        events = [[issue_event(1, "labeled", label="ci:upgrades")]]

        valid, valid_calls = run_report_step(
            [[]],
            pull_request_snapshots=[pull_request],
            issue_event_snapshots=[events],
            step_name="Revalidate an authorized Dependabot result",
        )
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertFalse(
            any(
                "--method" in call
                and call[call.index("--method") + 1] in {"POST", "PATCH"}
                for call in valid_calls
            )
        )

        invalid_cases = (
            (
                "unlabeled-live-pr",
                {**pull_request, "labels": []},
                events,
                "write",
            ),
            (
                "invalidated-epoch",
                pull_request,
                [
                    [
                        *events[0],
                        issue_event(2, "unlabeled", label="ci:upgrades"),
                    ]
                ],
                "write",
            ),
            ("permission-loss", pull_request, events, "read"),
            (
                "identity-drift",
                {**pull_request, "head": {"sha": "d" * 40}},
                events,
                "write",
            ),
        )
        for name, live_pull_request, live_events, permission in invalid_cases:
            with self.subTest(name=name):
                invalid, calls = run_report_step(
                    [[]],
                    pull_request_snapshots=[live_pull_request],
                    issue_event_snapshots=[live_events],
                    permission_snapshots=[permission],
                    step_name="Revalidate an authorized Dependabot result",
                )
                self.assertNotEqual(invalid.returncode, 0)
                self.assertFalse(
                    any(
                        "--method" in call
                        and call[call.index("--method") + 1] in {"POST", "PATCH"}
                        for call in calls
                    )
                )

    def test_dependabot_unauthorized_result_still_fails_reconciliation(self) -> None:
        assessed, outputs = run_output_step(
            "Assess policy and validation",
            {
                "EVENT_ACTION": "synchronize",
                "EVENT_BASE_REF": "main",
                "EVENT_BASE_SHA": "a" * 40,
                "EVENT_HEAD_SHA": "b" * 40,
                "EVENT_MERGE_SHA": "c" * 40,
                "GH_TOKEN": "test-token",
                "MODE": "authorization_required",
                "POLICY_CONCLUSION": "failure",
                "POLICY_RESULT": "success",
                "POLICY_SUMMARY": "Maintainer authorization is required.",
                "PR_NUMBER": "7",
                "VALIDATION_RESULT": "skipped",
            },
        )
        self.assertEqual(assessed.returncode, 0, assessed.stderr)
        self.assertEqual(outputs["mode"], "authorization_required")
        self.assertEqual(outputs["passed"], "false")

        dependabot_pr = True
        preserve_failed = (
            outputs["mode"] == "preserve"
            and not dependabot_pr
            and outputs.get("preserve_valid") != "true"
        )
        self.assertTrue(outputs["passed"] != "true" or preserve_failed)
        enforce_script = extract_workflow_step_script("Enforce policy result").replace(
            "${{ steps.assess-policy.outputs.summary || "
            "'Validation assessment failed.' }}",
            "Maintainer authorization is required.",
        )
        enforced = subprocess.run(
            ["bash", "-c", enforce_script],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(enforced.returncode, 0)

    def test_required_workflow_rerun_consumes_only_a_fresh_label(self) -> None:
        events = [[issue_event(1, "labeled", label="ci:upgrades")]]

        authorized, outputs = run_policy_decision_step(
            action="synchronize",
            event_label="",
            event_actor="contributor",
            event_actor_id=200,
            event_updated_at="2026-01-01T00:00:00Z",
            issue_events=events,
            check_pages=[{"check_runs": []}],
            run_attempt=2,
            run_created_at="2025-12-31T23:59:59Z",
        )

        self.assertEqual(authorized.returncode, 0, authorized.stderr)
        self.assertEqual(outputs["mode"], "validate")
        self.assertEqual(outputs["run_validation"], "true")
        self.assertEqual(outputs["authorization_event_id"], "1")

        later_attempt, later_outputs = run_policy_decision_step(
            action="synchronize",
            event_label="",
            event_actor="contributor",
            event_actor_id=200,
            event_updated_at="2026-01-01T00:00:00Z",
            issue_events=events,
            check_pages=[{"check_runs": []}],
            run_attempt=10,
            run_created_at="2025-12-31T23:59:59Z",
        )
        self.assertEqual(later_attempt.returncode, 0, later_attempt.stderr)
        self.assertEqual(later_outputs["mode"], "validate")

    def test_diagnostic_check_cannot_make_a_required_workflow_pass(self) -> None:
        policy = load_required_check_policy()

        decision = policy.decide_pull_request_policy(
            action="synchronize",
            event_label="",
            base_ref_changed=False,
            state="open",
            merged=False,
            merge_candidate_available=True,
            relevant=True,
            identity_matches=True,
            authorization_label_present=True,
            actor_permission="write",
            has_existing_check=True,
            has_newer_check=True,
            authorization_epoch_valid=True,
            authorization_rerun_matches=True,
        )

        self.assertEqual(decision.mode, "superseded")
        self.assertFalse(decision.run_validation)
        self.assertEqual(decision.conclusion, "failure")

    def test_required_workflow_rerun_rejects_identity_and_lifecycle_races(
        self,
    ) -> None:
        newer_check = {
            "app": {"slug": "github-actions"},
            "external_id": "lemonade-llamacpp-required-pr-7-2-51-1-501",
            "id": 51,
            "name": "llama.cpp authorized validation v2",
        }
        common = {
            "action": "synchronize",
            "event_label": "",
            "event_actor": "contributor",
            "event_actor_id": 200,
            "event_updated_at": "2026-01-01T00:00:00Z",
            "issue_events": [[issue_event(2, "labeled", label="ci:upgrades")]],
            "check_pages": [{"check_runs": []}],
            "run_attempt": 2,
            "run_created_at": "2026-01-01T00:00:01Z",
        }
        cases = (
            (
                "label-at-run-boundary",
                {
                    "issue_events": [[issue_event(1, "labeled", label="ci:upgrades")]],
                },
                "authorization_required",
            ),
            (
                "base-retarget-after-label",
                {
                    "issue_events": [
                        [
                            issue_event(1, "labeled", label="ci:upgrades"),
                            issue_event(2, "base_ref_changed"),
                        ]
                    ],
                    "run_created_at": "2026-01-01T00:00:00Z",
                },
                "authorization_required",
            ),
            (
                "reopen-after-label",
                {
                    "issue_events": [
                        [
                            issue_event(1, "labeled", label="ci:upgrades"),
                            issue_event(2, "reopened"),
                        ]
                    ],
                    "run_created_at": "2026-01-01T00:00:00Z",
                },
                "authorization_required",
            ),
            (
                "remove-after-label",
                {
                    "issue_events": [
                        [
                            issue_event(1, "labeled", label="ci:upgrades"),
                            issue_event(2, "unlabeled", label="ci:upgrades"),
                        ]
                    ],
                    "run_created_at": "2026-01-01T00:00:00Z",
                },
                "authorization_required",
            ),
            (
                "push-after-original-run",
                {"identity_matches": False},
                "stale_required",
            ),
            (
                "newer-default-event-run",
                {"check_pages": [{"check_runs": [newer_check]}]},
                "superseded",
            ),
        )
        for name, overrides, expected_mode in cases:
            arguments = dict(common)
            arguments.update(overrides)
            with self.subTest(name=name):
                completed, outputs = run_policy_decision_step(**arguments)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(outputs["mode"], expected_mode)
                self.assertEqual(outputs["run_validation"], "false")
                self.assertEqual(outputs["conclusion"], "failure")

        for action in ("opened", "reopened", "synchronize"):
            with self.subTest(valid_action=action):
                completed, outputs = run_policy_decision_step(
                    **{
                        **common,
                        "action": action,
                        "issue_events": [
                            [
                                issue_event(1, "reopened"),
                                issue_event(2, "labeled", label="ci:upgrades"),
                            ]
                        ],
                    }
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(outputs["mode"], "validate")
                self.assertEqual(outputs["run_validation"], "true")

    def test_reauthorization_metadata_does_not_deadlock_the_required_rerun(
        self,
    ) -> None:
        events = [
            [
                issue_event(1, "labeled", label="ci:upgrades"),
                issue_event(2, "unlabeled", label="ci:upgrades"),
                issue_event(3, "labeled", label="ci:upgrades"),
            ]
        ]
        common = {
            "action": "synchronize",
            "event_label": "",
            "event_actor": "contributor",
            "event_actor_id": 200,
            "event_updated_at": "2026-01-01T00:00:00Z",
            "issue_events": events,
            "run_attempt": 3,
            "run_created_at": "2026-01-01T00:00:00Z",
        }
        prior_required = {
            "app": {"slug": "github-actions"},
            "external_id": "lemonade-llamacpp-required-pr-7-1-50-2-500",
            "id": 50,
            "name": "llama.cpp authorized validation v2",
        }
        metadata_revocation = {
            "app": {"slug": "github-actions"},
            "external_id": "lemonade-llamacpp-metadata-pr-7-2-51-1-501",
            "id": 51,
            "name": "llama.cpp authorization diagnostics v2",
        }
        legacy_unscoped_revocation = {
            **metadata_revocation,
            "external_id": "lemonade-llamacpp-pr-7-2-51-1-501",
            "name": "llama.cpp authorized validation v2",
        }

        initial_recovery, outputs = run_policy_decision_step(
            check_pages=[{"check_runs": [metadata_revocation]}],
            **{**common, "run_attempt": 2},
        )
        self.assertEqual(initial_recovery.returncode, 0, initial_recovery.stderr)
        self.assertEqual(outputs["mode"], "validate")
        self.assertEqual(outputs["run_validation"], "true")

        for name, checks in (
            ("scoped-metadata", [prior_required, metadata_revocation]),
            ("legacy-unscoped-metadata", [legacy_unscoped_revocation]),
        ):
            with self.subTest(name=name):
                completed, outputs = run_policy_decision_step(
                    check_pages=[{"check_runs": checks}],
                    **common,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(outputs["mode"], "validate")
                self.assertEqual(outputs["run_validation"], "true")

        newer_required = {
            **prior_required,
            "external_id": "lemonade-llamacpp-required-pr-7-3-51-1-501",
            "id": 52,
        }
        superseded, outputs = run_policy_decision_step(
            check_pages=[{"check_runs": [newer_required]}],
            **common,
        )
        self.assertEqual(superseded.returncode, 0, superseded.stderr)
        self.assertEqual(outputs["mode"], "superseded")
        self.assertEqual(outputs["conclusion"], "failure")

    def test_recovered_required_result_is_distinct_from_metadata_diagnostics(
        self,
    ) -> None:
        required_name = "llama.cpp authorized validation v2"
        metadata_name = "llama.cpp authorization diagnostics v2"
        metadata_report, metadata_calls = run_report_step(
            [[]],
            authorization_epoch=2,
            conclusion="failure",
            event_action="unlabeled",
            event_name="pull_request_target",
            mode="authorization_required",
            run_id=501,
            run_number=51,
        )
        self.assertEqual(metadata_report.returncode, 0, metadata_report.stderr)
        metadata_writes = [
            call
            for call in metadata_calls
            if "--method" in call
            and call[call.index("--method") + 1] in {"POST", "PATCH"}
        ]
        self.assertEqual(len(metadata_writes), 1)
        self.assertIn(f"name={metadata_name}", metadata_writes[0])
        self.assertNotIn(f"name={required_name}", metadata_writes[0])

        metadata_check = {
            "app": {"slug": "github-actions"},
            "external_id": "lemonade-llamacpp-metadata-pr-7-2-51-1-501",
            "id": 90,
            "name": metadata_name,
        }
        recovered_check = {
            "app": {"slug": "github-actions"},
            "external_id": "lemonade-llamacpp-required-pr-7-3-50-3-500",
            "id": 80,
            "name": required_name,
        }
        live_pull_request = {
            "base": {"ref": "main", "sha": "a" * 40},
            "head": {"sha": "b" * 40},
            "labels": [{"name": "ci:upgrades"}],
            "merge_commit_sha": "c" * 40,
            "mergeable": True,
            "merged": False,
            "number": 7,
            "state": "open",
        }
        authorization_events = [
            [
                issue_event(1, "labeled", label="ci:upgrades"),
                issue_event(2, "unlabeled", label="ci:upgrades"),
                issue_event(3, "labeled", label="ci:upgrades"),
            ]
        ]
        recovered, recovered_calls = run_report_step(
            [
                [{"check_runs": [metadata_check]}],
                [{"check_runs": [metadata_check, recovered_check]}],
            ],
            authorization_created_at="2026-01-01T00:00:03Z",
            authorization_epoch=3,
            issue_event_snapshots=[authorization_events, authorization_events],
            pull_request_snapshots=[live_pull_request, live_pull_request],
            run_attempt=3,
        )
        self.assertEqual(recovered.returncode, 0, recovered.stderr)
        recovered_writes = [
            call
            for call in recovered_calls
            if "--method" in call
            and call[call.index("--method") + 1] in {"POST", "PATCH"}
        ]
        self.assertTrue(
            any(f"name={required_name}" in call for call in recovered_writes)
        )
        self.assertFalse(
            any(f"name={metadata_name}" in call for call in recovered_writes)
        )
        self.assertFalse(
            any(
                "repos/LLM360/lemonade/check-runs/90" in call
                for call in recovered_writes
            )
        )

    def test_policy_failure_without_outputs_still_reports_and_enforces_failure(
        self,
    ) -> None:
        completed, outputs = run_output_step(
            "Assess policy and validation",
            {
                "EVENT_ACTION": "labeled",
                "MODE": "",
                "POLICY_CONCLUSION": "",
                "POLICY_RESULT": "failure",
                "POLICY_SUMMARY": "",
                "VALIDATION_RESULT": "skipped",
            },
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(outputs["mode"], "policy_error")
        self.assertEqual(outputs["conclusion"], "failure")
        self.assertEqual(outputs["passed"], "false")

        workflow = PR_WORKFLOW.read_text(encoding="utf-8")
        assess = workflow.split("      - name: Assess policy and validation\n", 1)[
            1
        ].split("      - name:", 1)[0]
        report = workflow.split(
            "      - name: Report result on the exact candidate commit\n", 1
        )[1].split("      - name: Enforce policy result\n", 1)[0]
        enforce = workflow.split("      - name: Enforce policy result\n", 1)[1]
        self.assertIn("        if: always()\n", assess)
        self.assertIn("          always() &&\n", report)
        self.assertIn("          always() &&\n", enforce)

    def test_failed_closed_event_uses_live_lifecycle_before_skipping(self) -> None:
        reopened = {
            "base": {"ref": "main", "sha": "a" * 40},
            "head": {"sha": "b" * 40},
            "labels": [{"name": "ci:upgrades"}],
            "merge_commit_sha": "c" * 40,
            "mergeable": True,
            "merged": False,
            "number": 7,
            "state": "open",
        }
        completed, outputs = run_failed_closed_assessment(reopened)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(outputs["mode"], "policy_error")
        self.assertEqual(outputs["conclusion"], "failure")
        self.assertEqual(outputs["passed"], "false")
        self.assertEqual(outputs["fallback_merge_sha"], "c" * 40)

    def test_unconfirmed_observed_merge_gate_is_failed_with_run_ordering(
        self,
    ) -> None:
        workflow = PR_WORKFLOW.read_text(encoding="utf-8")
        step_name = "Invalidate an unconfirmed observed merge commit"
        self.assertIn(f"      - name: {step_name}\n", workflow)
        observed_merge_sha = "c" * 40
        prefix = "lemonade-llamacpp-metadata-pr-7-"
        old_check = {
            "id": 11,
            "name": "llama.cpp authorization diagnostics v2",
            "app": {"slug": "github-actions"},
            "external_id": f"{prefix}1-10-1-9000",
        }
        foreign_check = {
            "id": 12,
            "name": "llama.cpp authorization diagnostics v2",
            "app": {"slug": "foreign-app"},
            "external_id": f"{prefix}1-9-1-8000",
        }

        completed, calls = run_observed_merge_invalidation(
            [{"check_runs": [old_check, foreign_check]}],
            observed_merge_sha=observed_merge_sha,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        mutations = [
            call for call in calls if call[call.index("--method") + 1] != "GET"
        ]
        posts = [
            call for call in mutations if call[call.index("--method") + 1] == "POST"
        ]
        self.assertEqual(len(posts), 1)
        self.assertIn(f"head_sha={observed_merge_sha}", posts[0])
        self.assertIn("conclusion=failure", posts[0])
        patched_endpoints = {
            next(argument for argument in call if argument.startswith("repos/"))
            for call in mutations
            if call[call.index("--method") + 1] == "PATCH"
        }
        self.assertIn("repos/LLM360/lemonade/check-runs/11", patched_endpoints)
        self.assertNotIn("repos/LLM360/lemonade/check-runs/12", patched_endpoints)

        newer_check = dict(old_check, id=13, external_id=f"{prefix}1-60-1-10")
        superseded, superseded_calls = run_observed_merge_invalidation(
            [{"check_runs": [newer_check]}],
            observed_merge_sha=observed_merge_sha,
        )
        self.assertEqual(superseded.returncode, 0, superseded.stderr)
        self.assertTrue(
            all(call[call.index("--method") + 1] == "GET" for call in superseded_calls)
        )

        second_observed_merge_sha = "d" * 40
        both, both_calls = run_observed_merge_invalidation(
            [{"check_runs": []}],
            observed_merge_sha=observed_merge_sha,
            secondary_observed_merge_sha=second_observed_merge_sha,
            authorization_epoch=0,
        )
        self.assertEqual(both.returncode, 0, both.stderr)
        observed_posts = [
            call
            for call in both_calls
            if "--method" in call and call[call.index("--method") + 1] == "POST"
        ]
        self.assertEqual(len(observed_posts), 2)
        self.assertEqual(
            {
                next(value for value in call if value.startswith("head_sha="))
                for call in observed_posts
            },
            {f"head_sha={observed_merge_sha}", f"head_sha={second_observed_merge_sha}"},
        )
        self.assertTrue(
            all(
                "external_id=lemonade-llamacpp-metadata-pr-7-0-50-1-500" in call
                for call in observed_posts
            )
        )

    def test_post_write_recheck_closes_concurrent_check_ordering_races(
        self,
    ) -> None:
        prefix = "lemonade-llamacpp-required-pr-7-"
        current = {
            "id": 80,
            "name": "llama.cpp authorized validation v2",
            "app": {"slug": "github-actions"},
            "external_id": f"{prefix}1-50-1-500",
        }
        newer_revocation = {
            "id": 90,
            "name": "llama.cpp authorized validation v2",
            "app": {"slug": "github-actions"},
            "external_id": f"{prefix}0-51-1-10",
        }

        completed, calls = run_report_step(
            [[], [{"check_runs": [current, newer_revocation]}]]
        )
        self.assertNotEqual(completed.returncode, 0)
        mutations = [
            call for call in calls if call[call.index("--method") + 1] != "GET"
        ]
        self.assertIn("conclusion=success", mutations[0])
        self.assertTrue(
            any(
                "repos/LLM360/lemonade/check-runs/80" in call
                and "conclusion=cancelled" in call
                for call in mutations
            )
        )
        self.assertFalse(
            any("repos/LLM360/lemonade/check-runs/90" in call for call in mutations)
        )

        superseded_before_write, pre_write_calls = run_report_step(
            [[{"check_runs": [newer_revocation]}]]
        )
        self.assertNotEqual(superseded_before_write.returncode, 0)
        self.assertTrue(
            all(call[call.index("--method") + 1] == "GET" for call in pre_write_calls)
        )

        prior_attempt = dict(current, id=70, external_id=f"{prefix}9-50-1-9000")
        retry = dict(current, external_id=f"{prefix}1-50-2-500")
        retried, retry_calls = run_report_step(
            [
                [{"check_runs": [prior_attempt]}],
                [{"check_runs": [prior_attempt, retry]}],
            ],
            run_attempt=2,
        )
        self.assertEqual(retried.returncode, 0, retried.stderr)
        self.assertTrue(
            any(
                "repos/LLM360/lemonade/check-runs/70" in call
                and "conclusion=cancelled" in call
                for call in retry_calls
            )
        )

    def test_success_is_demoted_when_live_identity_changes_after_write(self) -> None:
        valid_pull_request = {
            "base": {"ref": "main", "sha": "a" * 40},
            "head": {"sha": "b" * 40},
            "labels": [{"name": "ci:upgrades"}],
            "merge_commit_sha": "c" * 40,
            "mergeable": True,
            "merged": False,
            "number": 7,
            "state": "open",
        }
        current_check = {
            "app": {"slug": "github-actions"},
            "external_id": "lemonade-llamacpp-required-pr-7-1-50-1-500",
            "id": 80,
            "name": "llama.cpp authorized validation v2",
        }
        valid_events = [[issue_event(1, "labeled", label="ci:upgrades")]]
        changed_snapshots = {
            "authorization": {
                **valid_pull_request,
                "labels": [],
            },
            "base-ref": {
                **valid_pull_request,
                "base": {"ref": "release", "sha": "a" * 40},
            },
            "head": {
                **valid_pull_request,
                "head": {"sha": "d" * 40},
            },
            "merge": {
                **valid_pull_request,
                "merge_commit_sha": "d" * 40,
            },
            "base-epoch": valid_pull_request,
            "reopened-epoch": valid_pull_request,
            "permission": valid_pull_request,
        }
        for name, changed_pull_request in changed_snapshots.items():
            with self.subTest(name=name):
                post_events = (
                    [
                        [
                            *valid_events[0],
                            issue_event(2, "unlabeled", label="ci:upgrades"),
                        ]
                    ]
                    if name == "authorization"
                    else (
                        [[*valid_events[0], issue_event(2, "base_ref_changed")]]
                        if name == "base-epoch"
                        else (
                            [
                                [
                                    *valid_events[0],
                                    issue_event(2, "closed"),
                                    issue_event(3, "reopened"),
                                ]
                            ]
                            if name == "reopened-epoch"
                            else valid_events
                        )
                    )
                )
                completed, calls = run_report_step(
                    [[], [{"check_runs": [current_check]}]],
                    pull_request_snapshots=[valid_pull_request, changed_pull_request],
                    issue_event_snapshots=[valid_events, post_events],
                    permission_snapshots=(
                        ["write", "read"] if name == "permission" else None
                    ),
                )
                self.assertNotEqual(completed.returncode, 0)
                pull_reads = [
                    call
                    for call in calls
                    if any("/pulls/7" in argument for argument in call)
                ]
                self.assertEqual(len(pull_reads), 2)
                self.assertTrue(
                    any(
                        "repos/LLM360/lemonade/check-runs/80" in call
                        and "conclusion=failure" in call
                        for call in calls
                    )
                )

    def test_missing_live_authorization_invalidates_a_prior_protected_gate(
        self,
    ) -> None:
        policy = load_required_check_policy()
        common = {
            "base_ref_changed": False,
            "state": "open",
            "merged": False,
            "merge_candidate_available": True,
            "identity_matches": True,
            "actor_permission": "write",
            "has_newer_check": False,
        }

        authorized = policy.decide_pull_request_policy(
            action="synchronize",
            event_label="",
            relevant=True,
            authorization_label_present=True,
            authorization_rerun_matches=True,
            has_existing_check=False,
            **common,
        )
        self.assertEqual(authorized.mode, "validate")
        self.assertTrue(authorized.run_validation)

        for action, event_label in (("edited", ""), ("labeled", "ci:macos")):
            with self.subTest(action=action, event_label=event_label):
                missing_live_label = policy.decide_pull_request_policy(
                    action=action,
                    event_label=event_label,
                    relevant=True,
                    authorization_label_present=False,
                    has_existing_check=True,
                    **common,
                )
                self.assertEqual(missing_live_label.mode, "authorization_required")
                self.assertFalse(missing_live_label.run_validation)
                self.assertEqual(missing_live_label.conclusion, "failure")

        neutral = policy.decide_pull_request_policy(
            action="edited",
            event_label="",
            relevant=False,
            authorization_label_present=False,
            has_existing_check=True,
            **common,
        )
        self.assertEqual(neutral.mode, "neutral")
        self.assertEqual(neutral.conclusion, "success")

    def test_authorization_lifecycle_events_cannot_be_replaced_or_bypassed(
        self,
    ) -> None:
        policy = load_required_check_policy()
        common = {
            "base_ref_changed": False,
            "state": "open",
            "merged": False,
            "merge_candidate_available": True,
            "relevant": True,
            "identity_matches": True,
            "authorization_label_present": True,
            "has_existing_check": True,
            "has_newer_check": False,
        }

        removed_before_fast_readd = policy.decide_pull_request_policy(
            action="unlabeled",
            event_label="ci:upgrades",
            actor_permission="",
            **common,
        )
        self.assertEqual(removed_before_fast_readd.mode, "authorization_required")
        self.assertEqual(removed_before_fast_readd.conclusion, "failure")

        triage_readd = policy.decide_pull_request_policy(
            action="labeled",
            event_label="ci:upgrades",
            actor_permission="triage",
            **common,
        )
        self.assertEqual(triage_readd.mode, "awaiting_rerun")
        self.assertFalse(triage_readd.run_validation)

        metadata_replacement = policy.decide_pull_request_policy(
            action="labeled",
            event_label="ci:macos",
            actor_permission="write",
            **common,
        )
        self.assertEqual(metadata_replacement.mode, "preserve")
        self.assertFalse(metadata_replacement.run_validation)

        metadata_after_untrusted_readd = policy.decide_pull_request_policy(
            action="edited",
            event_label="",
            actor_permission="triage",
            authorization_epoch_valid=True,
            **common,
        )
        self.assertEqual(metadata_after_untrusted_readd.mode, "authorization_required")
        self.assertEqual(metadata_after_untrusted_readd.conclusion, "failure")

        metadata_before_readd = policy.decide_pull_request_policy(
            action="edited",
            event_label="",
            actor_permission="",
            has_existing_check=False,
            **{
                key: value
                for key, value in common.items()
                if key != "has_existing_check"
            },
        )
        self.assertEqual(metadata_before_readd.mode, "authorization_required")
        self.assertEqual(metadata_before_readd.conclusion, "failure")
        self.assertFalse(metadata_before_readd.run_validation)

        trusted_readd = policy.decide_pull_request_policy(
            action="labeled",
            event_label="ci:upgrades",
            actor_permission="write",
            **common,
        )
        self.assertEqual(trusted_readd.mode, "awaiting_rerun")
        self.assertFalse(trusted_readd.run_validation)

        trusted_rerun = policy.decide_pull_request_policy(
            action="synchronize",
            event_label="",
            actor_permission="write",
            authorization_rerun_matches=True,
            **common,
        )
        self.assertEqual(trusted_rerun.mode, "validate")
        self.assertTrue(trusted_rerun.run_validation)

        delayed_removal = policy.decide_pull_request_policy(
            action="unlabeled",
            event_label="ci:upgrades",
            actor_permission="",
            has_newer_check=True,
            **{key: value for key, value in common.items() if key != "has_newer_check"},
        )
        self.assertEqual(delayed_removal.mode, "preserve")

        workflow = PR_WORKFLOW.read_text(encoding="utf-8")
        decision_environment = workflow.split(
            "      - name: Decide whether protected validation may run\n", 1
        )[1].split("        run: |\n", 1)[0]
        concurrency = workflow.split("concurrency:\n", 1)[1].split("\njobs:\n", 1)[0]
        self.assertIn("github.event.action == 'labeled'", concurrency)
        self.assertIn("github.event.action == 'unlabeled'", concurrency)
        self.assertIn("github.run_number", concurrency)
        self.assertIn("github.run_attempt", concurrency)
        self.assertNotIn("github.run_id", concurrency)
        for action in (
            "opened",
            "reopened",
            "synchronize",
            "edited",
            "labeled",
            "unlabeled",
            "closed",
        ):
            self.assertIn(action, concurrency)
        self.assertNotIn("github.event.pull_request.merge_commit_sha", concurrency)
        self.assertIn("issues: read", workflow)
        self.assertIn("issues/${PR_NUMBER}/events?per_page=100", workflow)
        self.assertIn("--paginate --slurp", workflow)
        self.assertIn("authorization_event_id", workflow)
        self.assertIn("authorization_created_at", workflow)
        self.assertIn("actions: read", workflow)
        self.assertIn("CURRENT_RUN_ID: ${{ github.run_id }}", workflow)
        self.assertIn(
            "EVENT_BASE_REF: ${{ github.event.pull_request.base.ref || '' }}",
            decision_environment,
        )
        self.assertIn(
            "EVENT_BASE_SHA: ${{ github.event.pull_request.base.sha || '' }}",
            decision_environment,
        )
        self.assertIn(
            "EVENT_HEAD_SHA: ${{ github.event.pull_request.head.sha || '' }}",
            decision_environment,
        )
        self.assertIn(
            "TRIGGERING_ACTOR: ${{ github.triggering_actor || '' }}", workflow
        )
        self.assertIn("actions/runs/${CURRENT_RUN_ID}/attempts/1", workflow)
        self.assertIn(".github/workflows/validate_llamacpp_pr.yml", workflow)
        self.assertNotIn(".head_branch == $base_ref", workflow)
        self.assertNotIn(".head_repository.full_name == $repository", workflow)
        self.assertNotIn(".head_sha == $base_sha", workflow)
        self.assertIn('$path + "@" + $base_ref', workflow)
        self.assertIn('$path + "@refs/heads/" + $base_ref', workflow)
        self.assertIn('((.pull_requests | type) == "array")', workflow)
        self.assertNotIn(".pull_requests | length", workflow)
        self.assertNotIn(".pull_requests[0]", workflow)
        self.assertIn(".head_sha == $head_sha", workflow)
        self.assertIn('.conclusion == "failure"', workflow)
        self.assertIn(".run_started_at", workflow)
        self.assertIn("authorization_boundary", workflow)
        self.assertIn("authorization_rerun_matches", workflow)
        self.assertIn(
            '--authorization-rerun-matches "$authorization_rerun_matches"',
            workflow,
        )
        self.assertIn('[[ "$CURRENT_RUN_ATTEMPT" =~ ^([2-9]|[1-9][0-9]+)$ ]]', workflow)
        self.assertIn(".user.id == $actor_id", workflow)
        report = workflow.split(
            "      - name: Report result on the exact candidate commit\n", 1
        )[1].split("      - name: Enforce policy result\n", 1)[0]
        self.assertLess(
            report.index('newest_number" -gt "$RUN_NUMBER'),
            report.index("--method PATCH"),
        )
        self.assertIn(".number < $current_number", report)
        self.assertIn("RUN_NUMBER: ${{ github.run_number }}", report)

    def test_preserve_rechecks_the_live_epoch_or_reports_terminal_failure(
        self,
    ) -> None:
        pull_request = {
            "base": {"ref": "main", "sha": "a" * 40},
            "head": {"sha": "b" * 40},
            "labels": [{"name": "ci:upgrades"}],
            "merge_commit_sha": "c" * 40,
            "mergeable": True,
            "merged": False,
            "number": 7,
            "state": "open",
        }
        current_epoch_check = {
            "app": {"slug": "github-actions"},
            "conclusion": "success",
            "external_id": "lemonade-llamacpp-required-pr-7-1-40-1-400",
            "id": 40,
            "name": "llama.cpp authorized validation v2",
            "status": "completed",
        }
        valid, valid_outputs = run_preserve_step(
            pull_request=pull_request,
            issue_events=[[issue_event(1, "labeled", label="ci:upgrades")]],
            check_pages=[{"check_runs": [current_epoch_check]}],
        )
        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertEqual(valid_outputs["valid"], "true")

        for name, events, checks in (
            (
                "unlabeled-after-policy",
                [
                    [
                        issue_event(1, "labeled", label="ci:upgrades"),
                        issue_event(2, "unlabeled", label="ci:upgrades"),
                    ]
                ],
                [current_epoch_check],
            ),
            (
                "old-epoch-only",
                [[issue_event(1, "labeled", label="ci:upgrades")]],
                [
                    {
                        **current_epoch_check,
                        "external_id": "lemonade-llamacpp-required-pr-7-9-40-1-400",
                    }
                ],
            ),
            (
                "malformed-final-provenance",
                [
                    [
                        {
                            "actor": {"id": 100, "login": "maintainer"},
                            "created_at": "2026-01-01T00:00:01Z",
                            "event": "labeled",
                            "id": 1,
                            "label": {"name": "ci:upgrades"},
                        }
                    ]
                ],
                [current_epoch_check],
            ),
            (
                "in-progress-prior-validation",
                [[issue_event(1, "labeled", label="ci:upgrades")]],
                [{**current_epoch_check, "conclusion": None, "status": "in_progress"}],
            ),
            (
                "failed-prior-validation",
                [[issue_event(1, "labeled", label="ci:upgrades")]],
                [{**current_epoch_check, "conclusion": "failure"}],
            ),
            (
                "newer-required-decision",
                [[issue_event(1, "labeled", label="ci:upgrades")]],
                [
                    current_epoch_check,
                    {
                        **current_epoch_check,
                        "conclusion": None,
                        "external_id": ("lemonade-llamacpp-required-pr-7-1-51-1-600"),
                        "id": 51,
                        "status": "in_progress",
                    },
                ],
            ),
            (
                "later-failed-prior-decision",
                [[issue_event(1, "labeled", label="ci:upgrades")]],
                [
                    current_epoch_check,
                    {
                        **current_epoch_check,
                        "conclusion": "failure",
                        "external_id": ("lemonade-llamacpp-required-pr-7-1-49-1-499"),
                        "id": 49,
                    },
                ],
            ),
        ):
            with self.subTest(name=name):
                invalid, invalid_outputs = run_preserve_step(
                    pull_request=pull_request,
                    issue_events=events,
                    check_pages=[{"check_runs": checks}],
                )
                self.assertEqual(invalid.returncode, 0, invalid.stderr)
                self.assertEqual(invalid_outputs["valid"], "false")

        reported_check = {
            **current_epoch_check,
            "external_id": "lemonade-llamacpp-required-pr-7-1-50-1-500",
            "id": 80,
        }
        reported, calls = run_report_step(
            [[], [{"check_runs": [reported_check]}]],
            mode="preserve",
            conclusion="failure",
            preserve_valid=invalid_outputs["valid"] == "true",
        )
        self.assertEqual(reported.returncode, 0, reported.stderr)
        self.assertTrue(
            any(
                "--method" in call
                and call[call.index("--method") + 1] in {"POST", "PATCH"}
                and "conclusion=failure" in call
                for call in calls
            )
        )

    def test_live_identity_rejects_stale_merge_sha_until_mergeability_is_known(
        self,
    ) -> None:
        base_sha = "a" * 40
        head_sha = "b" * 40
        stale_merge_sha = "c" * 40

        def snapshot(mergeable: object) -> dict[str, object]:
            return {
                "number": 7,
                "state": "open",
                "merged": False,
                "mergeable": mergeable,
                "merge_commit_sha": stale_merge_sha,
                "base": {"sha": base_sha},
                "head": {"sha": head_sha},
                "changed_files": 1,
                "labels": [{"name": "ci:upgrades"}],
            }

        cases = (
            ("unknown-through-timeout", [snapshot(None)] * 3, 3),
            ("known-unmergeable", [snapshot(False)], 1),
        )
        policy = load_required_check_policy()
        for name, snapshots, expected_calls in cases:
            with self.subTest(name=name):
                completed, outputs, calls = run_identity_step(
                    snapshots,
                    event_merge_sha=stale_merge_sha,
                )

                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(calls, expected_calls)
                self.assertEqual(outputs["merge_candidate_available"], "false")
                self.assertEqual(outputs["merge_sha"], head_sha)
                self.assertEqual(outputs["observed_merge_sha"], stale_merge_sha)
                self.assertEqual(outputs["identity_matches"], "true")

                decision = policy.decide_pull_request_policy(
                    action="synchronize",
                    event_label="",
                    base_ref_changed=False,
                    state=outputs["state"],
                    merged=outputs["merged"] == "true",
                    merge_candidate_available=(
                        outputs["merge_candidate_available"] == "true"
                    ),
                    relevant=True,
                    identity_matches=outputs["identity_matches"] == "true",
                    authorization_label_present=True,
                    authorization_rerun_matches=True,
                    actor_permission="write",
                    has_existing_check=False,
                    has_newer_check=False,
                )
                self.assertEqual(decision.mode, "authorization_required")
                self.assertFalse(decision.run_validation)
                self.assertEqual(decision.conclusion, "failure")

    def test_identity_tracks_event_and_live_unconfirmed_merge_commits(self) -> None:
        base_sha = "a" * 40
        head_sha = "b" * 40
        event_merge_sha = "c" * 40
        live_merge_sha = "d" * 40

        def snapshot(merge_commit_sha: str | None) -> dict[str, object]:
            return {
                "base": {"ref": "main", "sha": base_sha},
                "changed_files": 1,
                "head": {"sha": head_sha},
                "labels": [{"name": "ci:upgrades"}],
                "merge_commit_sha": merge_commit_sha,
                "mergeable": None,
                "merged": False,
                "number": 7,
                "state": "open",
            }

        for name, live_merge, expected_primary, expected_secondary in (
            ("event-only", None, event_merge_sha, ""),
            ("live-and-event", live_merge_sha, live_merge_sha, event_merge_sha),
        ):
            with self.subTest(name=name):
                completed, outputs, _ = run_identity_step(
                    [snapshot(live_merge)] * 3,
                    event_merge_sha=event_merge_sha,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(outputs["merge_sha"], head_sha)
                self.assertEqual(outputs["observed_merge_sha"], expected_primary)
                self.assertEqual(
                    outputs["secondary_observed_merge_sha"], expected_secondary
                )

        retargeted = snapshot(live_merge_sha)
        retargeted["base"] = {"ref": "release", "sha": base_sha}
        completed, outputs, _ = run_identity_step(
            [retargeted] * 3,
            event_base_ref="main",
            event_merge_sha=event_merge_sha,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(outputs["base_ref"], "release")
        self.assertEqual(outputs["identity_matches"], "false")

    def test_required_check_classifier_covers_protected_inputs(self) -> None:
        policy = load_required_check_policy()

        protected_paths = (
            ".github/llamacpp_release_manifest.json",
            ".github/workflows/validate_llamacpp_pr.yml",
            ".github/workflows/validate_llamacpp_core.yml",
            ".github/scripts/run_llamacpp_validation.py",
            ".github/scripts/llamacpp_required_check_policy.py",
            ".github/actions/setup-venv/action.yml",
            ".github/actions/setup-python/action.yml",
            ".github/actions/cleanup-processes-linux/action.yml",
            ".github/actions/cleanup-processes-windows/action.yml",
            "CMakeLists.txt",
            "CMakePresets.json",
            "cmake/DetectSystemHttplib.cmake",
            "setup.sh",
            "src/cpp/resources/backend_versions.json",
            "src/cpp/resources/server_models.json",
            "src/app/src/renderer/utils/toolDefinitions.json",
            "src/cpp/server/backends/llamacpp/llamacpp_server.cpp",
            "src/cpp/server/backends/vllm/vllm_server.cpp",
            "src/cpp/server/backends/fastflowlm/fastflowlm_models.cpp",
            "src/cpp/include/lemon/backend_manager.h",
            "src/cpp/server/backend_manager.cpp",
            "src/cpp/include/lemon/runtime_config.h",
            "src/cpp/server/runtime_config.cpp",
            "src/cpp/include/lemon/backends/llamacpp/llamacpp_server.h",
            "src/cpp/server/backends/backend_utils.cpp",
            "src/cpp/include/lemon/backends/backend_utils.h",
            "src/cpp/server/streaming_proxy.cpp",
            "src/cpp/include/lemon/streaming_proxy.h",
            "src/cpp/server/wrapped_server.cpp",
            "src/cpp/include/lemon/wrapped_server.h",
            "src/cpp/server/system_info.cpp",
            "src/cpp/include/lemon/system_info.h",
            "src/cpp/server/utils/platform/process_linux.cpp",
            "src/cpp/include/lemon/utils/process_manager.h",
            "test/requirements.txt",
            "test/validate_llamacpp.py",
            "test/utils/llamacpp_capability_validation.py",
            "test/utils/capabilities.py",
            "test/utils/server_base.py",
            "test/utils/test_models.py",
            "test/utils/validation_model_catalog.py",
            "test/utils/validation_model_selection.py",
            "test/fixtures/llamacpp_validation_models.json",
            "new-root-build-input.txt",
            ".github/actions/future-validation-action/action.yml",
            ".github/scripts/future-validation-helper.py",
        )
        for path in protected_paths:
            with self.subTest(path=path):
                self.assertTrue(policy.is_protected_path(path))

        unrelated_paths = (
            "docs/dev/testing.md",
            "src/app/src/renderer/ChatWindow.tsx",
            "src/web-app/src/index.ts",
        )
        for path in unrelated_paths:
            with self.subTest(path=path):
                self.assertFalse(policy.is_protected_path(path))

    def test_required_check_classifier_fails_closed_on_incomplete_file_lists(
        self,
    ) -> None:
        policy = load_required_check_policy()
        base_sha = "a" * 40
        stale_target_sha = "b" * 40
        candidate_sha = "c" * 40
        neutral_comparison = {
            "ahead_by": 1,
            "base_commit": {"sha": base_sha},
            "behind_by": 0,
            "commits": [{"sha": candidate_sha}],
            "files": [{"filename": "docs/dev/testing.md", "status": "modified"}],
            "merge_base_commit": {"sha": base_sha},
            "status": "ahead",
            "total_commits": 1,
        }

        self.assertFalse(
            policy.classify_compare_response(
                neutral_comparison,
                expected_base_sha=base_sha,
                expected_target_sha=candidate_sha,
                expected_count=1,
            )
        )
        stale_comparison = {
            **neutral_comparison,
            "commits": [{"sha": stale_target_sha}],
        }
        with self.assertRaises(ValueError):
            policy.classify_compare_response(
                stale_comparison,
                expected_base_sha=base_sha,
                expected_target_sha=candidate_sha,
                expected_count=1,
            )
        self.assertTrue(
            policy.classify_compare_response(
                {},
                expected_base_sha=base_sha,
                expected_target_sha=candidate_sha,
                expected_count=policy.MAX_COMPARE_FILES + 1,
            )
        )

        self.assertTrue(
            policy.classify_changed_file_pages(
                [
                    [
                        {
                            "filename": "docs/renamed.md",
                            "previous_filename": "CMakeLists.txt",
                            "status": "renamed",
                        }
                    ]
                ],
                expected_count=1,
            )
        )
        self.assertTrue(
            policy.classify_changed_file_pages(
                [
                    [
                        {
                            "filename": "src/cpp/resources/backend_versions.json",
                            "status": "modified",
                        }
                    ]
                ],
                expected_count=1,
            )
        )
        self.assertFalse(
            policy.classify_changed_file_pages(
                [[{"filename": "docs/dev/testing.md", "status": "modified"}]],
                expected_count=1,
            )
        )

        for status in (
            "added",
            "removed",
            "modified",
            "copied",
            "changed",
            "unchanged",
        ):
            with self.subTest(valid_status=status):
                self.assertFalse(
                    policy.classify_changed_file_pages(
                        [[{"filename": "docs/dev/testing.md", "status": status}]],
                        expected_count=1,
                    )
                )

                with self.assertRaises(ValueError):
                    policy.classify_changed_file_pages(
                        [
                            [
                                {
                                    "filename": "docs/dev/testing.md",
                                    "previous_filename": "docs/dev/old.md",
                                    "status": status,
                                }
                            ]
                        ],
                        expected_count=1,
                    )

        for previous_filename in (None, "", 3, "/docs/old.md", "docs/../old.md"):
            with self.subTest(invalid_previous_filename=previous_filename):
                with self.assertRaises(ValueError):
                    policy.classify_changed_file_pages(
                        [
                            [
                                {
                                    "filename": "docs/new.md",
                                    "previous_filename": previous_filename,
                                    "status": "renamed",
                                }
                            ]
                        ],
                        expected_count=1,
                    )

        invalid_cases = (
            ([], 1),
            ([[{"filename": "docs/one.md", "status": "modified"}]], 2),
            (
                [
                    [
                        {"filename": "docs/one.md", "status": "modified"},
                        {"filename": "docs/one.md", "status": "modified"},
                    ]
                ],
                2,
            ),
            ([[{"filename": "/absolute/path", "status": "modified"}]], 1),
            ([[{"filename": "docs/one.md"}]], 1),
            ([[{"filename": "docs/one.md", "status": "moved"}]], 1),
            ([[{"filename": "docs/one.md", "status": 3}]], 1),
            ([[{"filename": "docs/renamed.md", "status": "renamed"}]], 1),
            (
                [
                    [
                        {
                            "filename": "docs/one.md",
                            "previous_filename": "docs/old.md",
                            "status": "modified",
                        }
                    ]
                ],
                1,
            ),
            (
                [
                    [
                        {
                            "filename": "docs/one.md",
                            "previous_filename": None,
                            "status": "modified",
                        }
                    ]
                ],
                1,
            ),
            (
                [[{"filename": "docs/one.md", "status": "modified"}]],
                policy.MAX_CHANGED_FILES + 1,
            ),
        )
        for pages, expected_count in invalid_cases:
            with self.subTest(pages=pages, expected_count=expected_count):
                with self.assertRaises(ValueError):
                    policy.classify_changed_file_pages(
                        pages,
                        expected_count=expected_count,
                    )

    def test_required_check_policy_requires_fresh_maintainer_authorization(
        self,
    ) -> None:
        policy = load_required_check_policy()

        def decide(**overrides):
            arguments = {
                "action": "synchronize",
                "event_label": "",
                "base_ref_changed": False,
                "state": "open",
                "merged": False,
                "merge_candidate_available": True,
                "relevant": True,
                "identity_matches": True,
                "authorization_label_present": True,
                "actor_permission": "write",
                "authorization_rerun_matches": False,
                "has_existing_check": True,
                "has_newer_check": False,
            }
            arguments.update(overrides)
            return policy.decide_pull_request_policy(**arguments)

        for action in ("opened", "reopened", "synchronize"):
            with self.subTest(action=action):
                decision = decide(action=action)
                self.assertEqual(decision.mode, "authorization_required")
                self.assertFalse(decision.run_validation)
                self.assertIn("remove and reapply", decision.summary)

        base_edit = decide(action="edited", base_ref_changed=True)
        self.assertEqual(base_edit.mode, "authorization_required")

        removed = decide(
            action="unlabeled",
            event_label="ci:upgrades",
            authorization_label_present=False,
        )
        self.assertEqual(removed.mode, "authorization_required")

        authorized = decide(
            action="synchronize",
            authorization_rerun_matches=True,
        )
        self.assertEqual(authorized.mode, "validate")
        self.assertTrue(authorized.run_validation)

        authorized_override = decide(
            action="synchronize",
            authorization_rerun_matches=True,
            relevant=False,
            has_existing_check=False,
        )
        self.assertEqual(authorized_override.mode, "validate")
        self.assertTrue(authorized_override.run_validation)

        unauthorized_override = decide(
            action="synchronize",
            authorization_rerun_matches=True,
            relevant=False,
            actor_permission="read",
            has_existing_check=False,
        )
        self.assertEqual(unauthorized_override.mode, "neutral")
        self.assertFalse(unauthorized_override.run_validation)

        for permission in ("read", "none", ""):
            with self.subTest(permission=permission):
                denied = decide(
                    action="synchronize",
                    authorization_rerun_matches=True,
                    actor_permission=permission,
                )
                self.assertEqual(denied.mode, "authorization_required")
                self.assertFalse(denied.run_validation)

        stale = decide(
            action="synchronize",
            authorization_rerun_matches=True,
            identity_matches=False,
        )
        self.assertEqual(stale.mode, "stale_required")

        stale_without_gate = decide(
            action="synchronize",
            authorization_rerun_matches=True,
            identity_matches=False,
            has_existing_check=False,
        )
        self.assertEqual(stale_without_gate.mode, "stale_required")

        neutral_without_merge = decide(
            action="opened",
            merge_candidate_available=False,
            relevant=False,
            has_existing_check=False,
        )
        self.assertEqual(neutral_without_merge.mode, "neutral")

        protected_without_merge = decide(
            action="synchronize",
            authorization_rerun_matches=True,
            merge_candidate_available=False,
            has_existing_check=False,
        )
        self.assertEqual(protected_without_merge.mode, "authorization_required")
        self.assertFalse(protected_without_merge.run_validation)

        preserved = decide(action="labeled", event_label="ci:macos")
        self.assertEqual(preserved.mode, "preserve")

        missing_prior = decide(
            action="labeled",
            event_label="ci:macos",
            has_existing_check=False,
        )
        self.assertEqual(missing_prior.mode, "authorization_required")
        self.assertEqual(missing_prior.conclusion, "failure")

        neutral = decide(relevant=False, has_existing_check=False)
        self.assertEqual(neutral.mode, "neutral")
        self.assertEqual(neutral.conclusion, "success")

        closed = decide(action="closed", state="closed")
        self.assertEqual(closed.mode, "closed")
        self.assertEqual(closed.conclusion, "cancelled")

        merged = decide(action="closed", state="closed", merged=True)
        self.assertEqual(merged.mode, "merged")
        self.assertEqual(merged.conclusion, "")

        stale_after_merge = decide(action="labeled", state="closed", merged=True)
        self.assertEqual(stale_after_merge.mode, "merged")
        self.assertEqual(stale_after_merge.conclusion, "")

        stale_close_after_reopen = decide(
            action="closed",
            state="open",
            merged=False,
        )
        self.assertEqual(stale_close_after_reopen.mode, "stale")

        removal_before_reapply_is_processed = decide(
            action="unlabeled",
            event_label="ci:upgrades",
            authorization_label_present=True,
        )
        self.assertEqual(
            removal_before_reapply_is_processed.mode, "authorization_required"
        )
        self.assertEqual(removal_before_reapply_is_processed.conclusion, "failure")

        delayed_synchronize = decide(
            action="synchronize",
            has_newer_check=True,
        )
        self.assertEqual(delayed_synchronize.mode, "superseded")
        self.assertEqual(delayed_synchronize.conclusion, "failure")

    def test_protected_validation_bootstrap_requires_the_trusted_workflow(
        self,
    ) -> None:
        guide = TESTING_GUIDE.read_text(encoding="utf-8")
        pull_request = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn(
            "| llama.cpp, vLLM, stable-diffusion.cpp validation | "
            "Required workflow `Validate llama.cpp protected change`;",
            guide,
        )
        self.assertNotIn(
            "| llama.cpp, vLLM, stable-diffusion.cpp validation | "
            "`llama.cpp authorized validation v2`,",
            guide,
        )
        self.assertIn("Do not require `llama.cpp validation`", guide)
        self.assertIn("`ci:upgrades` label", guide)
        self.assertIn("protected self-hosted runners", guide)
        self.assertIn("CODEOWNERS", guide)
        self.assertIn("separate bootstrap merge", guide)
        self.assertIn(
            "Enforcement therefore starts with later protected pull requests", guide
        )
        self.assertIn("current `.github/CODEOWNERS` covers only", guide)
        self.assertIn("explicit entries for every trusted validation", guide)
        self.assertIn("enforce code-owner review", guide)
        self.assertIn("Missing exact CODEOWNERS coverage", guide)
        self.assertIn("external activation **NO-GO**", guide)
        self.assertIn("merge queue", guide)
        self.assertIn(
            "Allow GitHub Actions to create and approve pull requests",
            guide,
        )
        self.assertIn("`LLAMACPP_UPDATE_TOKEN`", guide)
        self.assertIn("`LLAMACPP_UPDATE_ACTOR`", guide)
        self.assertIn("repository-scoped fine-grained PAT", guide)
        self.assertNotIn("short-lived dedicated GitHub App installation token", guide)
        self.assertIn("installation tokens expire after one hour", guide)
        self.assertIn("mint a fresh token inside each publication job", guide)
        self.assertIn("Contents: read and write", guide)
        self.assertIn("Pull requests: read and write", guide)
        self.assertIn("Never substitute the repository's `GITHUB_TOKEN`", guide)
        self.assertIn("does not trigger the required workflow", guide)
        self.assertIn("deliberately creates a draft first", guide)
        self.assertIn("missing secret or variable configuration", guide)
        self.assertIn("neutral success", guide)
        self.assertIn("synthetic test-merge commit", guide)
        self.assertIn("falls back to the immutable current head", guide)
        self.assertIn(
            "protected validation never executes without a test-merge commit", guide
        )
        self.assertIn("GitHub ignores configured activity filters", guide)
        self.assertIn("`opened`, `synchronize`, and `reopened`", guide)
        self.assertIn("**Re-run all jobs**", guide)
        self.assertIn("The same maintainer must start the rerun", guide)
        self.assertIn("a label at or before the original run start time", guide)
        self.assertIn("not its queued creation time", guide)
        self.assertIn("while the run was queued", guide)
        self.assertIn("base-branch retarget", guide)
        self.assertIn("supported default-event run", guide)
        self.assertIn("Metadata diagnostic runs do not supersede", guide)
        self.assertIn("`llama.cpp authorization diagnostics v2`", guide)
        self.assertIn("uses a distinct visible name", guide)
        self.assertIn("does not conflict with a recovered required result", guide)
        self.assertIn("newer `opened`, `synchronize`, or `reopened` run", guide)
        self.assertIn("rerun the same selected default-event run", guide)
        self.assertIn("Removal after a completed green rerun is diagnostic only", guide)
        self.assertIn("one exact default-event run and candidate", guide)
        self.assertNotIn("The label authorizes one exact run attempt", guide)
        self.assertIn("Activation is **NO-GO** unless", guide)
        self.assertIn("the default branch advances after a green rerun", guide)
        self.assertIn(
            "a merge queue, strict up-to-date enforcement, or a live-proven equivalent",
            guide,
        )
        self.assertIn("the exact validated base-plus-head candidate", guide)
        self.assertIn("exclusive GitHub App", guide)
        self.assertIn(
            "Do **not** require `llama.cpp authorized validation v2` during bootstrap",
            guide,
        )
        self.assertIn("immutable K2-capable backend releases", guide)
        self.assertIn("all three pinned release repositories", guide)
        self.assertIn("controlled immutable mirror", guide)
        self.assertIn("all three `sm_121` artifacts", guide)
        self.assertIn("source sidecars", guide)
        self.assertIn("Require workflows to pass before merging", guide)
        self.assertIn("organization-level", guide)
        self.assertIn("`LLM360/lemonade`", guide)
        self.assertIn("`main`", guide)
        self.assertIn("`.github/workflows/validate_llamacpp_pr.yml`", guide)
        self.assertIn("organization owner", guide)
        self.assertIn("After the bootstrap merge is on the default branch", guide)
        self.assertIn("separate, unmerged live fork PRs", guide)
        self.assertIn("Do not combine validation-infrastructure bootstrap", guide)
        self.assertNotIn("Keep the integration PR in draft until", guide)
        self.assertIn("shared `github-actions` App", guide)
        self.assertIn("does not identify the workflow", guide)
        self.assertIn("diagnostic only", guide)
        self.assertIn(
            "Never configure `llama.cpp authorized validation v2` as a required "
            "status check",
            guide,
        )
        self.assertNotIn(
            "select **GitHub Actions** as the expected check source", guide
        )
        self.assertIn("  pull_request_target:\n", pull_request)
        self.assertIn("  merge_group:\n", pull_request)

        reconcile = pull_request.split("  reconcile:\n", 1)[1]
        enforce = reconcile.split("      - name: Enforce policy result\n", 1)[1]
        self.assertIn("    if: always()", reconcile)
        self.assertIn("if: >-\n          always()", enforce)
        self.assertIn("exit 1", enforce)

    def test_pr_required_check_wrapper_covers_every_policy_lifecycle(self) -> None:
        pull_request = PR_WORKFLOW.read_text(encoding="utf-8")
        classify_step = extract_workflow_step_script(
            "Classify the complete pull request file set"
        )
        triggers = pull_request.split("permissions:", 1)[0]
        report = pull_request.split(
            "      - name: Report result on the exact candidate commit\n", 1
        )[1].split("      - name: Enforce policy result\n", 1)[0]

        self.assertIn(
            "types: [opened, reopened, synchronize, edited, labeled, unlabeled, closed]",
            triggers,
        )
        self.assertIn("  merge_group:\n", triggers)
        self.assertIn(
            "compare/${BASE_SHA}...${MERGE_SHA}",
            classify_step,
        )
        self.assertNotIn("pulls/${PR_NUMBER}/files?per_page=100", classify_step)
        self.assertNotIn("--paginate --slurp", classify_step)
        self.assertIn("classify-compare", classify_step)
        self.assertIn(
            "MERGE_SHA: ${{ steps.identity.outputs.merge_sha }}", pull_request
        )
        self.assertIn("current_changed_files", pull_request)
        self.assertIn("for attempt in 1 2 3; do", pull_request)
        self.assertIn("snapshot_mergeable", pull_request)
        self.assertIn("sleep 2", pull_request)
        self.assertIn("llamacpp_required_check_policy.py", pull_request)
        self.assertIn(
            "previous_filename", REQUIRED_CHECK_POLICY.read_text(encoding="utf-8")
        )
        self.assertIn(
            "trusted/.github/scripts/llamacpp_required_check_policy.py", pull_request
        )
        self.assertIn("ref: ${{ steps.identity.outputs.base_sha }}", pull_request)
        self.assertIn("persist-credentials: false", pull_request)
        self.assertIn("collaborators/${authorization_actor}/permission", pull_request)
        self.assertIn("EVENT_ACTION: ${{ github.event.action || '' }}", pull_request)
        self.assertIn("github.event.changes.base.ref.from", pull_request)
        self.assertIn("llama.cpp authorized validation v2", pull_request)
        self.assertIn("needs.policy.outputs.run_validation == 'true'", pull_request)
        self.assertIn("needs.policy.outputs.mode == 'preserve'", pull_request)
        self.assertIn("needs.policy.outputs.conclusion", pull_request)
        self.assertIn("github.event.pull_request.merge_commit_sha", pull_request)
        self.assertIn("github.event.merge_group.head_sha", pull_request)
        self.assertIn("current_merge", pull_request)
        self.assertIn("current_mergeable", report)
        self.assertIn(
            '[ "$current_mergeable" = "true" ] && [[ "$current_merge" =~ $oid_pattern ]]',
            report,
        )
        self.assertIn("current_base", pull_request)
        self.assertIn("current_head", pull_request)
        self.assertIn(
            "current_merge=$(jq -r '.merge_commit_sha // empty'", pull_request
        )
        self.assertIn('candidate_sha="$current_head"', pull_request)
        self.assertIn(
            '--merge-candidate-available "$MERGE_CANDIDATE_AVAILABLE"',
            pull_request,
        )
        self.assertIn("steps.assess-policy.outputs.mode != 'stale'", pull_request)
        self.assertIn('.app.slug == "github-actions"', pull_request)
        self.assertNotIn("HUGGINGFACE_ACCESS_TOKEN", pull_request)

    def test_protected_candidate_checkouts_opt_in_without_weakening_trusted_checkouts(
        self,
    ) -> None:
        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        unsafe_opt_in = (
            "allow-unsafe-pr-checkout: "
            "${{ github.event_name == 'pull_request_target' }}"
        )
        checkout_v5_1 = (
            "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09 " "# v5.1.0"
        )

        self.assertEqual(validation.count(unsafe_opt_in), 4)
        self.assertEqual(
            validation.count(checkout_v5_1),
            validation.count("uses: actions/checkout@"),
        )
        for trusted_checkout in validation.split(
            "      - name: Check out trusted validation tools\n"
        )[1:]:
            trusted_block = trusted_checkout.split("      - name:", 1)[0]
            self.assertNotIn("allow-unsafe-pr-checkout", trusted_block)

        wrapper = PR_WORKFLOW.read_text(encoding="utf-8")
        policy_checkout = wrapper.split(
            "      - name: Check out trusted policy code\n", 1
        )[1].split("      - name:", 1)[0]
        self.assertNotIn("allow-unsafe-pr-checkout", policy_checkout)

    def test_required_check_lookups_are_complete_source_bound_and_race_safe(
        self,
    ) -> None:
        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        pull_request = PR_WORKFLOW.read_text(encoding="utf-8")
        invocation = validation.split("  validate-invocation:\n", 1)[1].split(
            "  plan:\n", 1
        )[0]
        report = pull_request.split(
            "      - name: Report result on the exact candidate commit\n", 1
        )[1].split("      - name: Enforce policy result\n", 1)[0]

        self.assertIn(
            'EXTERNAL_PREFIX="lemonade-llamacpp-required-${TARGET_KEY}-"',
            invocation,
        )
        self.assertIn("decision_scope=metadata", report)
        self.assertIn("decision_scope=required", report)
        self.assertIn(
            'external_prefix="lemonade-llamacpp-${decision_scope}-${TARGET_KEY}-"',
            report,
        )

        for lookup in (invocation, report):
            with self.subTest(workflow="core" if lookup is invocation else "wrapper"):
                self.assertIn("--paginate --slurp", lookup)
                self.assertIn('.app.slug == "github-actions"', lookup)
                self.assertIn('newest_number" -gt "$RUN_NUMBER', lookup)
                self.assertIn(".number < $current_number", lookup)
                self.assertLess(
                    lookup.index('newest_number" -gt "$RUN_NUMBER'),
                    lookup.index("--method PATCH"),
                )
                self.assertGreaterEqual(lookup.count("--paginate --slurp"), 2)
                self.assertIn("RUN_NUMBER: ${{ github.run_number }}", lookup)

    def test_required_check_filters_ignore_foreign_and_newer_runs(self) -> None:
        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        pull_request = PR_WORKFLOW.read_text(encoding="utf-8")
        lookups = (
            validation.split("  validate-invocation:\n", 1)[1].split("  plan:\n", 1)[0],
            pull_request.split(
                "      - name: Report result on the exact candidate commit\n", 1
            )[1].split("      - name: Enforce policy result\n", 1)[0],
        )
        check_name = "llama.cpp authorized validation v2"
        external_prefix = "lemonade-llamacpp-required-pr-7-"
        check_pages = [
            {
                "check_runs": [
                    {
                        "id": 11,
                        "name": check_name,
                        "app": {"slug": "github-actions"},
                        "conclusion": "success",
                        "external_id": f"{external_prefix}1-10-1-9000",
                        "status": "completed",
                    },
                    {
                        "id": 12,
                        "name": check_name,
                        "app": {"slug": "github-actions"},
                        "conclusion": "success",
                        "external_id": f"{external_prefix}1-20-1-8000",
                        "status": "completed",
                    },
                    {
                        "id": 13,
                        "name": check_name,
                        "app": {"slug": "github-actions"},
                        "conclusion": "success",
                        "external_id": f"{external_prefix}1-30-2-7000",
                        "status": "completed",
                    },
                    {
                        "id": 14,
                        "name": check_name,
                        "app": {"slug": "foreign-app"},
                        "conclusion": "success",
                        "external_id": f"{external_prefix}1-99-1-6000",
                        "status": "completed",
                    },
                    {
                        "id": 15,
                        "name": "foreign check",
                        "app": {"slug": "github-actions"},
                        "conclusion": "success",
                        "external_id": f"{external_prefix}1-100-1-5000",
                        "status": "completed",
                    },
                ]
            }
        ]

        def jq_program(block: str, anchor: str) -> str:
            tail = block.split(anchor, 1)[1]
            starts = [
                position
                for token in ("'[.[]", "'.[]")
                if (position := tail.find(token)) >= 0
            ]
            self.assertTrue(starts)
            start = min(starts)
            end = tail.index('\' <<<"$check_runs"', start)
            return tail[start + 1 : end]

        policy_job = pull_request.split("  policy:\n", 1)[1].split("  validate:\n", 1)[
            0
        ]
        preserve_step = pull_request.split(
            "      - name: Preserve the exact-commit gate for unrelated metadata\n",
            1,
        )[1].split("      - name:", 1)[0]
        for lookup in (policy_job, preserve_step):
            latest_program = jq_program(
                lookup, "read -r newest_check_epoch newest_check_number"
            )
            base_command = [
                "jq",
                "-r",
                "--arg",
                "check_name",
                check_name,
                "--arg",
                "external_prefix",
                external_prefix,
                latest_program,
            ]
            trusted = subprocess.run(
                base_command,
                input=json.dumps(check_pages),
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(
                trusted.stdout.rstrip("\n").split("\t"),
                ["1", "30", "2", "completed", "success"],
            )
            foreign_only = [{"check_runs": [check_pages[0]["check_runs"][3]]}]
            foreign = subprocess.run(
                base_command,
                input=json.dumps(foreign_only),
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(
                foreign.stdout.rstrip("\n").split("\t"),
                ["0", "0", "0", "", ""],
            )
            old_epoch_only = [
                {
                    "check_runs": [
                        {
                            **check_pages[0]["check_runs"][0],
                            "external_id": f"{external_prefix}9-10-1-9000",
                        }
                    ]
                }
            ]
            old_epoch = subprocess.run(
                base_command,
                input=json.dumps(old_epoch_only),
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertEqual(
                old_epoch.stdout.rstrip("\n").split("\t"),
                ["9", "10", "1", "completed", "success"],
            )
            for status, conclusion in (
                ("in_progress", None),
                ("completed", "failure"),
            ):
                unfinished = [
                    {
                        "check_runs": [
                            {
                                **check_pages[0]["check_runs"][0],
                                "conclusion": conclusion,
                                "status": status,
                            }
                        ]
                    }
                ]
                rejected = subprocess.run(
                    base_command,
                    input=json.dumps(unfinished),
                    text=True,
                    capture_output=True,
                    check=True,
                )
                self.assertEqual(
                    rejected.stdout.rstrip("\n").split("\t"),
                    ["1", "10", "1", status, conclusion or ""],
                )

        for lookup in lookups:
            with self.subTest(workflow="core" if lookup is lookups[0] else "wrapper"):
                newest = subprocess.run(
                    [
                        "jq",
                        "-r",
                        "--arg",
                        "check_name",
                        check_name,
                        "--arg",
                        "external_prefix",
                        external_prefix,
                        jq_program(lookup, "read -r newest_number newest_attempt"),
                    ],
                    input=json.dumps(check_pages),
                    text=True,
                    capture_output=True,
                    check=True,
                )
                self.assertEqual(newest.stdout.strip(), "30\t2")

                older = subprocess.run(
                    [
                        "jq",
                        "-r",
                        "--arg",
                        "check_name",
                        check_name,
                        "--arg",
                        "external_prefix",
                        external_prefix,
                        "--argjson",
                        "current_number",
                        "20",
                        "--argjson",
                        "current_attempt",
                        "1",
                        jq_program(lookup, "prior_check_ids=$(jq -r"),
                    ],
                    input=json.dumps(check_pages),
                    text=True,
                    capture_output=True,
                    check=True,
                )
                self.assertEqual(older.stdout.splitlines(), ["11"])

    def test_portable_runner_builds_platform_neutral_validation_arguments(self) -> None:
        runner = load_validation_runner()

        command = runner.build_validation_command(
            python_executable=Path("python"),
            backend="rocm",
            channel="nightly",
            target="windows-rocm-nightly",
            models_csv=f"{K2_SMALL},{K2_MEDIUM}",
            lite=False,
            capability_profile="k2-horizon-v1",
            capability_models_csv=K2_SMALL,
            port=13305,
        )

        self.assertEqual(
            command[0:2],
            ["python", str(ROOT / "test" / "validate_llamacpp.py")],
        )
        self.assertIn("--channel", command)
        self.assertIn("nightly", command)
        self.assertEqual(command.count("--model"), 2)
        self.assertIn("--capability-profile", command)
        self.assertIn("llamacpp_validation_windows-rocm-nightly.json", command)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            runner.build_validation_command(
                python_executable=Path("python"),
                backend="cpu",
                channel="",
                target="linux-cpu",
                models_csv=K2_SMALL,
                lite=True,
                capability_profile="k2-horizon-v1",
                capability_models_csv=K2_SMALL,
                port=13305,
            )

        inherited = {"HF_HUB_CACHE": "shared-cache", "PRESERVED": "yes"}
        environment = runner.build_server_environment(
            Path("target-isolated-cache"),
            inherited=inherited,
            private_temp_directory=Path("target-private-temp"),
            candidate_defaults_path=Path("candidate-defaults.json"),
        )
        self.assertEqual(
            environment["HF_HUB_CACHE"],
            str(Path("target-isolated-cache").resolve()),
        )
        self.assertNotIn("PRESERVED", environment)
        self.assertEqual(inherited["HF_HUB_CACHE"], "shared-cache")

        builtin_small_command = runner.build_validation_command(
            python_executable=Path("python"),
            backend="metal",
            channel="",
            target="macos-metal",
            models_csv=f"builtin.{K2_SMALL}",
            lite=False,
            capability_profile="k2-horizon-v1",
            capability_models_csv=K2_SMALL,
            port=13305,
        )
        self.assertEqual(builtin_small_command.count("--model"), 1)

    def test_validation_plan_supports_direct_script_invocation(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "test/utils/llamacpp_validation_plan.py"),
                "--help",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_validation_runner_supports_direct_script_invocation(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(VALIDATION_RUNNER), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def assert_active_lanes(self, rows):
        self.assertEqual(
            [
                (
                    row["target"],
                    row["backend"],
                    row["channel"],
                    row["managed_pin"],
                    row["build_platform"],
                )
                for row in rows
            ],
            [
                ("windows-vulkan", "vulkan", "", "vulkan", "windows"),
                ("linux-vulkan", "vulkan", "", "vulkan", "linux"),
                ("linux-cpu", "cpu", "", "cpu", "linux"),
                ("macos-metal", "metal", "", "metal", "macos"),
                ("windows-cuda", "cuda", "", "cuda", "windows"),
                (
                    "windows-rocm-stable",
                    "rocm",
                    "stable",
                    "rocm-stable",
                    "windows",
                ),
                (
                    "windows-rocm-nightly",
                    "rocm",
                    "nightly",
                    "rocm-nightly",
                    "windows",
                ),
            ],
        )
        self.assertEqual(
            {row["managed_pin"] for row in rows},
            {"cpu", "cuda", "metal", "rocm-nightly", "rocm-stable", "vulkan"},
        )
        vulkan_platforms = {
            row["build_platform"] for row in rows if row["backend"] == "vulkan"
        }
        self.assertEqual(vulkan_platforms, {"linux", "windows"})
        self.assertTrue(
            all(row["capability_profile"] == "k2-horizon-v1" for row in rows)
        )
        self.assertTrue(all(row["capability_models"] == [K2_SMALL] for row in rows))
        self.assertTrue(all(K2_SMALL in row["models"] for row in rows if row["models"]))

        for row in rows:
            if K2_MEDIUM in row["models"] or K2_LARGE in row["models"]:
                self.assertEqual(row["target"], "windows-vulkan")
                self.assertIn("128gb", row["runner"])

    def test_pull_request_and_merge_group_require_k2(self) -> None:
        for event_name in ("pull_request", "merge_group"):
            with self.subTest(event_name=event_name):
                rows = planning.create_validation_plan(event_name)["include"]

                self.assert_active_lanes(rows)
                self.assertEqual(rows[0]["models"], [K2_SMALL, K2_MEDIUM, K2_LARGE])
                self.assertTrue(all(row["models"] == [K2_SMALL] for row in rows[1:]))
                self.assertTrue(all(not row["lite"] for row in rows))
                self.assertIn("128gb", rows[0]["runner"])

    def test_schedule_captures_exact_validation_candidate_contract(self) -> None:
        rows = planning.create_validation_plan("schedule")["include"]

        self.assert_active_lanes(rows)
        expected = planning.load_validation_candidate_model_ids()
        self.assertIn(K2_SMALL, expected)
        self.assertIn(K2_MEDIUM, expected)
        self.assertIn(K2_LARGE, expected)
        self.assertEqual(rows[0]["models"], expected)
        self.assertEqual(rows[0]["expected_models"], expected)
        self.assertTrue(all(row["models"] == [K2_SMALL] for row in rows[1:]))
        self.assertTrue(all(row["expected_models"] == [K2_SMALL] for row in rows[1:]))
        self.assertTrue(all(not row["lite"] for row in rows))
        self.assertIn("128gb", rows[0]["runner"])

    def test_validation_candidate_set_reads_overlay_outside_product_catalog(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "server_models.json"
            overlay = Path(directory) / "validation_models.json"
            registry.write_text(
                json.dumps(
                    {
                        "Existing-Hot-Model": {
                            "recipe": "llamacpp",
                            "labels": ["chat", "hot"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            overlay.write_text(
                json.dumps(
                    {
                        "Candidate-Not-Hot": {
                            "recipe": "llamacpp",
                            "labels": ["chat"],
                        }
                    }
                ),
                encoding="utf-8",
            )

            models = planning.load_validation_candidate_model_ids(registry, overlay)

        self.assertEqual(
            models,
            ["Candidate-Not-Hot", "Existing-Hot-Model"],
        )

    def test_default_dispatch_selects_hot_models_and_its_capability_model(
        self,
    ) -> None:
        rows = planning.create_validation_plan("workflow_dispatch")["include"]

        self.assert_active_lanes(rows)
        self.assertEqual(
            set(rows[0]["models"]),
            {K2_SMALL, *planning.load_hot_llamacpp_model_ids()},
        )
        self.assertTrue(all(row["models"] == [K2_SMALL] for row in rows[1:]))
        self.assertTrue(
            all(
                set(row["capability_models"]).issubset(
                    {model.removeprefix("builtin.") for model in row["models"]}
                )
                for row in rows
            )
        )
        self.assertTrue(all(not row["lite"] for row in rows))
        self.assertIn("128gb", rows[0]["runner"])

    def test_manual_lite_selection_runs_only_k2_small(self) -> None:
        rows = planning.create_validation_plan("workflow_dispatch", lite=True)[
            "include"
        ]

        self.assert_active_lanes(rows)
        self.assertTrue(all(row["models"] == [K2_SMALL] for row in rows))
        self.assertTrue(all(not row["lite"] for row in rows))

    def test_manual_models_are_trimmed_and_keep_order_and_duplicates(self) -> None:
        rows = planning.create_validation_plan(
            "workflow_dispatch",
            models_csv=" Zulu, ,Alpha,Zulu ",
        )["include"]

        self.assert_active_lanes(rows)
        self.assertEqual(
            rows[0]["models"],
            [K2_SMALL, "Zulu", "Alpha", "Zulu"],
        )
        self.assertTrue(
            all(
                row["models"] == [K2_SMALL, "Zulu", "Alpha", "Zulu"] for row in rows[1:]
            )
        )
        self.assertTrue(all(not row["lite"] for row in rows))
        self.assertIn("128gb", rows[0]["runner"])

    def test_manual_large_k2_models_are_confined_to_large_vulkan_lane(self) -> None:
        rows = planning.create_validation_plan(
            "workflow_dispatch",
            models_csv=f"{K2_MEDIUM},{K2_LARGE}",
        )["include"]

        self.assert_active_lanes(rows)
        self.assertEqual(rows[0]["models"], [K2_SMALL, K2_MEDIUM, K2_LARGE])
        self.assertTrue(all(row["models"] == [K2_SMALL] for row in rows[1:]))

    def test_manual_builtin_large_k2_aliases_are_confined_to_large_lane(self) -> None:
        builtin_medium = f"builtin.{K2_MEDIUM}"
        builtin_large = f"builtin.{K2_LARGE}"
        rows = planning.create_validation_plan(
            "workflow_dispatch",
            models_csv=f"{builtin_medium},{builtin_large}",
        )["include"]

        self.assert_active_lanes(rows)
        self.assertEqual(rows[0]["models"], [K2_SMALL, builtin_medium, builtin_large])
        self.assertTrue(all(row["models"] == [K2_SMALL] for row in rows[1:]))

    def test_manual_models_and_lite_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            planning.create_validation_plan(
                "workflow_dispatch",
                models_csv=K2_SMALL,
                lite=True,
            )

    def test_non_manual_events_reject_manual_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "workflow_dispatch"):
            planning.create_validation_plan("schedule", models_csv=K2_SMALL)

    def test_unknown_event_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            planning.create_validation_plan("push")

    def test_only_scheduled_runs_are_promotion_eligible(self) -> None:
        cases = (
            ("schedule", "", False, True),
            ("workflow_dispatch", "", False, False),
            ("workflow_dispatch", K2_SMALL, False, False),
            ("workflow_dispatch", "", True, False),
            ("pull_request", "", False, False),
            ("merge_group", "", False, False),
        )

        for event_name, models_csv, lite, expected in cases:
            with self.subTest(
                event_name=event_name,
                models_csv=models_csv,
                lite=lite,
            ):
                self.assertEqual(
                    planning.is_promotion_eligible(event_name, models_csv, lite),
                    expected,
                )

    def test_promotion_coverage_rejects_missing_or_mapped_wrong_lanes(self) -> None:
        complete = planning.create_validation_plan("schedule")
        planning.require_complete_promotion_coverage(complete)

        missing = json.loads(json.dumps(complete))
        missing["include"].pop()
        with self.assertRaisesRegex(ValueError, "exactly"):
            planning.require_complete_promotion_coverage(missing)

        wrong_mapping = json.loads(json.dumps(complete))
        wrong_mapping["include"][2]["managed_pin"] = "cuda"
        with self.assertRaisesRegex(ValueError, "mapping"):
            planning.require_complete_promotion_coverage(wrong_mapping)

        wrong_vulkan_platform = json.loads(json.dumps(complete))
        wrong_vulkan_platform["include"][1]["build_platform"] = "windows"
        with self.assertRaisesRegex(ValueError, "mapping"):
            planning.require_complete_promotion_coverage(wrong_vulkan_platform)

        unexecuted_models = json.loads(json.dumps(complete))
        unexecuted_models["include"][0]["models"] = []
        with self.assertRaisesRegex(ValueError, "selected models"):
            planning.require_complete_promotion_coverage(unexecuted_models)

        lite_lane = json.loads(json.dumps(complete))
        lite_lane["include"][0]["lite"] = True
        with self.assertRaisesRegex(ValueError, "lite"):
            planning.require_complete_promotion_coverage(lite_lane)

        missing_candidate = json.loads(json.dumps(complete))
        for field in ("models", "expected_models"):
            missing_candidate["include"][0][field].remove(K2_MEDIUM)
        with self.assertRaisesRegex(ValueError, "exact candidate model set"):
            planning.require_complete_promotion_coverage(missing_candidate)

        unexpected_small_lane_model = json.loads(json.dumps(complete))
        for field in ("models", "expected_models"):
            unexpected_small_lane_model["include"][1][field].append("Other-Model")
        with self.assertRaisesRegex(ValueError, "exact K2 small model set"):
            planning.require_complete_promotion_coverage(unexpected_small_lane_model)

    def test_github_output_contains_compact_matrix_json(self) -> None:
        plan = planning.create_validation_plan("merge_group")
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "github-output.txt"

            planning.write_github_output(
                output_path,
                plan,
                promotion_eligible=False,
            )

            outputs = dict(
                line.split("=", 1)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(json.loads(outputs["matrix"]), plan)
            self.assertNotIn("\n", outputs["matrix"])
            self.assertEqual(outputs["promotion_eligible"], "false")

    def test_validation_core_has_constrained_permissions_and_no_legacy_entrypoint(
        self,
    ) -> None:
        self.assertFalse(LEGACY_WORKFLOW.exists())

        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        validation_triggers = validation.split("permissions:", 1)[0]
        workflow_scope = validation.split("jobs:", 1)[0]
        invocation_job = validation.split("  validate-invocation:\n", 1)[1].split(
            "  plan:\n", 1
        )[0]
        plan_job = validation.split("  plan:\n", 1)[1].split(
            "  get-latest-releases:\n", 1
        )[0]
        build_job = validation.split("  build:\n", 1)[1].split("  validate:\n", 1)[0]
        validate_job = validation.split("  validate:\n", 1)[1].split(
            "  validation-gate:\n", 1
        )[0]
        validation_step = validate_job.split(
            "      - name: Run validation with lemond\n", 1
        )[1].split("      - name: Upload results\n", 1)[0]
        result_upload = validate_job.split("      - name: Upload results\n", 1)[
            1
        ].split("      - name: Upload restart results\n", 1)[0]
        restart_result_upload = validate_job.split(
            "      - name: Upload restart results\n", 1
        )[1].split("      - name: Upload server logs\n", 1)[0]
        server_log_upload = validate_job.split("      - name: Upload server logs\n", 1)[
            1
        ].split("      - name: Cleanup\n", 1)[0]

        self.assertIn("  workflow_call:\n", validation_triggers)
        self.assertNotIn("  pull_request_target:\n", validation_triggers)
        self.assertNotIn("  pull_request:\n", validation_triggers)
        self.assertNotIn("  merge_group:\n", validation_triggers)
        self.assertNotIn("  workflow_dispatch:\n", validation_triggers)
        self.assertNotIn("  schedule:\n", validation_triggers)
        self.assertIn("permissions:\n  contents: read\n", validation)
        self.assertNotIn("contents: write", validation)
        self.assertNotIn("pull-requests: write", validation)
        self.assertEqual(validation.count("checks: write"), 1)
        self.assertIn("checks: write", invocation_job)
        self.assertNotIn("concurrency:", workflow_scope)
        self.assertNotIn("  create-pr:\n", validation)
        self.assertIn("      HUGGINGFACE_ACCESS_TOKEN:\n", validation_triggers)
        self.assertNotIn("HF_TOKEN:", workflow_scope)
        self.assertNotIn("secrets.HUGGINGFACE_ACCESS_TOKEN", workflow_scope)
        self.assertNotIn("HUGGINGFACE_ACCESS_TOKEN", plan_job)
        self.assertNotIn("HUGGINGFACE_ACCESS_TOKEN", build_job)
        self.assertEqual(validate_job.count("HF_TOKEN:"), 1)
        self.assertIn(
            "          HF_TOKEN: ${{ github.event_name == 'schedule' && secrets.HUGGINGFACE_ACCESS_TOKEN || '' }}\n",
            validation_step,
        )
        self.assertNotIn(
            "          HF_TOKEN: ${{ secrets.HUGGINGFACE_ACCESS_TOKEN }}\n",
            validation_step,
        )
        self.assertIn(".github/scripts/run_llamacpp_validation.py", validation_step)
        self.assertIn("        id: run-validation\n", validation_step)
        self.assertIn("steps.validation-venv.outputs.python-path", validation_step)
        self.assertIn("--lemond", validation_step)
        self.assertIn(
            "${{ github.event_name == 'schedule' && '--allow-huggingface-download-credentials' || '' }}",
            validation_step,
        )
        self.assertEqual(validation_step.count("LEMONADE_PROTECTED_RUNNER_CONTEXT:"), 1)
        self.assertEqual(
            validation_step.count("--allow-darwin-github-hosted-ephemeral-runner"),
            1,
        )
        self.assertIn("matrix.target == 'macos-metal'", validation_step)
        self.assertIn("runner.environment == 'github-hosted'", validation_step)
        self.assertIn("join(matrix.runner, ',') == 'macos-latest'", validation_step)
        self.assertIn("github-hosted:macos-latest:macos-metal", validation_step)
        self.assertNotIn("shell: PowerShell", validation_step)
        protected_hosted_mac_exclusion = (
            "!((github.event_name == 'pull_request_target' || "
            "github.event_name == 'merge_group') &&"
        )
        for upload in (result_upload, restart_result_upload, server_log_upload):
            self.assertIn("        if: >-\n", upload)
            self.assertIn(
                "          steps.run-validation.outcome == 'success' &&\n", upload
            )
            self.assertIn(protected_hosted_mac_exclusion, upload)
            self.assertIn("matrix.target == 'macos-metal'", upload)
            self.assertIn("runner.environment == 'github-hosted'", upload)
            self.assertIn("join(matrix.runner, ',') == 'macos-latest'", upload)
        self.assertIn(
            "path: ${{ steps.run-validation.outputs.validation_result }}",
            result_upload,
        )
        self.assertIn("          if-no-files-found: error\n", result_upload)
        self.assertNotIn("llamacpp_restart_validation_", restart_result_upload)
        self.assertIn("          if-no-files-found: error\n", restart_result_upload)
        self.assertIn(
            "path: ${{ steps.run-validation.outputs.restart_result }}",
            restart_result_upload,
        )
        self.assertNotIn("server-restart-logs", server_log_upload)
        for output_name in (
            "prepare_stdout_log",
            "prepare_stderr_log",
            "validation_stdout_log",
            "validation_stderr_log",
            "restart_stdout_log",
            "restart_stderr_log",
        ):
            self.assertIn(
                f"steps.run-validation.outputs.{output_name}",
                server_log_upload,
            )
        self.assertIn("          if-no-files-found: error\n", server_log_upload)
        self.assertNotIn("if: always()", result_upload)
        self.assertNotIn("if: always()", restart_result_upload)
        self.assertNotIn("if: always()", server_log_upload)

        runner = VALIDATION_RUNNER.read_text(encoding="utf-8")
        self.assertIn("verify_validation_artifacts", runner)
        self.assertIn('cache_directory / "hf-hub"', runner)
        self.assertIn('environment["HF_HUB_CACHE"]', runner)
        self.assertIn("VALIDATION_ARTIFACT_LOCKS", runner)

    def test_build_and_validation_orchestration_are_native_per_platform(self) -> None:
        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        build_job = validation.split("  build:\n", 1)[1].split("  validate:\n", 1)[0]
        validate_job = validation.split("  validate:\n", 1)[1].split(
            "  reverify-protected-release-manifest:\n", 1
        )[0]

        for platform, runner in (
            ("windows", "windows-latest"),
            ("linux", "ubuntu-latest"),
            ("macos", "macos-latest"),
        ):
            with self.subTest(platform=platform):
                self.assertIn(f"platform: {platform}", build_job)
                self.assertIn(f"runner: {runner}", build_job)
        self.assertIn("runs-on: ${{ matrix.runner }}", build_job)
        self.assertIn("llamacpp-build-${{ matrix.platform }}", build_job)
        self.assertIn("if: runner.os == 'Windows'", build_job)
        self.assertIn("if: runner.os != 'Windows'", build_job)
        self.assertIn("llamacpp-build-${{ matrix.build_platform }}", validate_job)
        self.assertIn("if: runner.os == 'Windows'", validate_job)
        self.assertIn("if: runner.os == 'Linux'", validate_job)
        self.assertIn("if: runner.os != 'Windows'", validate_job)

        protected_overlay_step = build_job.split(
            "      - name: Stage protected validation-only model catalog\n", 1
        )[1].split("      - name: Stage branch validation-only model catalog\n", 1)[0]
        branch_overlay_step = build_job.split(
            "      - name: Stage branch validation-only model catalog\n", 1
        )[1].split("      - name: Install Linux build dependencies\n", 1)[0]
        self.assertIn(
            "$GITHUB_WORKSPACE/test/utils/validation_model_catalog.py",
            protected_overlay_step,
        )
        self.assertIn("$GITHUB_WORKSPACE/candidate", protected_overlay_step)
        self.assertIn(
            "$GITHUB_WORKSPACE/test/fixtures/llamacpp_validation_models.json",
            protected_overlay_step,
        )
        self.assertIn(
            "python -m test.utils.validation_model_catalog", branch_overlay_step
        )
        self.assertIn(
            "--base src/cpp/resources/server_models.json", branch_overlay_step
        )
        self.assertIn(
            "--overlay test/fixtures/llamacpp_validation_models.json",
            branch_overlay_step,
        )
        self.assertIn(
            "--output src/cpp/resources/server_models.json", branch_overlay_step
        )
        self.assertLess(
            build_job.index(
                "      - name: Stage protected validation-only model catalog\n"
            ),
            build_job.index("          cmake --preset vs18"),
        )
        self.assertLess(
            build_job.index(
                "      - name: Stage branch validation-only model catalog\n"
            ),
            build_job.index("          cmake --preset default"),
        )

    def test_focused_ci_runs_the_capability_contract_tests(self) -> None:
        workflow = DOCS_AND_STYLE_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("test.test_llamacpp_capabilities", workflow)
        self.assertIn("test.test_llamacpp_validation_artifacts", workflow)
        self.assertIn("test.test_llamacpp_validation_evidence", workflow)
        self.assertIn("test.test_llamacpp_pr_file_classification", workflow)
        self.assertIn("test.test_run_llamacpp_validation", workflow)
        self.assertIn("test.test_validation_model_catalog", workflow)
        self.assertIn("test.test_validate_llamacpp_action_pins", workflow)
        self.assertIn("test.test_validate_llamacpp_selection", workflow)
        self.assertIn("test.test_validate_llamacpp_trusted_execution", workflow)
        self.assertIn("test.test_verify_llamacpp_pin_manifest", workflow)

    def test_focused_ci_installs_process_tracking_dependency(self) -> None:
        workflow = DOCS_AND_STYLE_WORKFLOW.read_text(encoding="utf-8")
        focused_tests = workflow.split(
            "      - name: Run focused Python unit tests\n", 1
        )[0]

        self.assertIn("pip install pre-commit psutil", focused_tests)

    def test_release_asset_manifest_is_captured_and_exported(self) -> None:
        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        release_job = validation.split("  get-latest-releases:\n", 1)[1].split(
            "  verify-release-assets:\n", 1
        )[0]
        asset_job = validation.split("  verify-release-assets:\n", 1)[1].split(
            "  build:\n", 1
        )[0]

        self.assertIn("      release_asset_manifest:\n", validation)
        self.assertIn(
            "jobs.verify-release-assets.outputs.release_asset_manifest",
            validation,
        )
        self.assertIn("release_asset_manifest:", asset_job)
        self.assertIn("llamacpp_release_manifest", asset_job)
        self.assertIn('--github-output "$GITHUB_OUTPUT"', asset_job)
        self.assertIn("jq -r '.assets[].name'", asset_job)
        self.assertIn("attach_release_claims", asset_job)
        self.assertIn("--source-manifest-asset-id", asset_job)
        self.assertIn("--validate-source-manifest", asset_job)
        self.assertIn("--build-target-attestation-output", asset_job)
        self.assertIn(".build_target_attestations", asset_job)
        self.assertIn("Accept: application/octet-stream", asset_job)
        self.assertIn("source_manifest_max_bytes=65536", asset_job)
        self.assertIn("source_manifest_directory=$(mktemp -d)", asset_job)
        self.assertIn(
            'source_manifest_path="${source_manifest_directory}/source-'
            '${source_manifest_asset_id}.json"',
            asset_job,
        )
        self.assertIn('head -c "$((source_manifest_max_bytes + 1))"', asset_job)
        self.assertIn('publisher_claim_type="immutable-source-manifest"', asset_job)
        self.assertNotIn("--extract-source-commit", asset_job)
        self.assertIn("--require-upstream-ancestry", asset_job)
        self.assertIn("/compare/", asset_job)
        self.assertRegex(
            asset_job,
            r"attach_release_claims lemonade-sdk/llamacpp-rocm "
            r'\s*\\?\s*"\$ROCM_RELEASE" rocm_release\.json',
        )
        self.assertRegex(
            asset_job,
            r"attach_release_claims lemonade-sdk/llama\.cpp "
            r'\s*\\?\s*"\$LEMONADE_RELEASE" lemonade_release\.json',
        )
        self.assertIn("source_repository=ggml-org/llama.cpp", asset_job)
        self.assertIn("upstream_reference_head", asset_job)
        self.assertIn("publisher_claimed_source_commit", asset_job)
        self.assertIn("publisher_claim_type", asset_job)
        self.assertIn("release_tag_commit", asset_job)
        self.assertIn("source_repository", asset_job)
        self.assertNotIn("attach_source_commit", asset_job)
        self.assertNotIn(
            'source_commit=$(gh api "repos/${repository}/commits/${tag}"',
            asset_job,
        )
        self.assertNotIn("release_asset_manifest", release_job)

    def test_protected_pin_changes_require_live_committed_manifest(self) -> None:
        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("  verify-protected-release-manifest:\n", validation)
        manifest_job = validation.split("  verify-protected-release-manifest:\n", 1)[
            1
        ].split("  build:\n", 1)[0]
        build_job = validation.split("  build:\n", 1)[1].split("  validate:\n", 1)[0]
        verifier = (
            ROOT / ".github" / "scripts" / "verify_llamacpp_pin_manifest.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("pull_request_target", manifest_job)
        self.assertIn("merge_group", manifest_job)
        self.assertIn("base.sha", manifest_job)
        self.assertIn("merge_group.base_sha", manifest_job)
        self.assertIn("verify_llamacpp_pin_manifest.sh", manifest_job)
        self.assertNotIn("id: managed-pin-changes", manifest_job)
        self.assertNotIn("steps.managed-pin-changes.outputs.changed", manifest_job)
        self.assertIn(".github/llamacpp_release_manifest.json", verifier)
        self.assertIn("capture_llamacpp_release_manifest.sh", verifier)
        self.assertIn("changed managed llama.cpp pins", verifier)
        self.assertIn("verify-protected-release-manifest", build_job)
        self.assertIn("  reverify-protected-release-manifest:\n", validation)
        final_manifest_job = validation.split(
            "  reverify-protected-release-manifest:\n", 1
        )[1].split("  validation-gate:\n", 1)[0]
        validation_gate = validation.split("  validation-gate:\n", 1)[1]
        self.assertIn("needs: [validate", final_manifest_job)
        self.assertIn("verify_llamacpp_pin_manifest.sh", final_manifest_job)
        self.assertIn("reverify-protected-release-manifest", validation_gate)

    def test_protected_sidecar_only_changes_fail_closed(self) -> None:
        verifier = (
            ROOT / ".github" / "scripts" / "verify_llamacpp_pin_manifest.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("manifest_path=.github/llamacpp_release_manifest.json", verifier)
        self.assertIn(
            'base_manifest_entry=$(git -C "$trusted_repository" ls-tree', verifier
        )
        self.assertIn(
            'candidate_manifest_entry=$(git -C "$candidate_repository" ls-tree',
            verifier,
        )
        self.assertIn("if [[ \"$changes\" == '{}' ]]", verifier)
        self.assertIn(
            '[[ "$base_manifest_entry" != "$candidate_manifest_entry" ]]', verifier
        )
        self.assertIn(
            "Release manifest changes require changed managed llama.cpp pins.",
            verifier,
        )

    def test_manifest_recheck_anchors_to_live_trusted_upstream(self) -> None:
        capture = (
            ROOT / ".github" / "scripts" / "capture_llamacpp_release_manifest.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("trusted_source_repository=ggml-org/llama.cpp", capture)
        self.assertIn(".default_branch", capture)
        self.assertIn("live_upstream_head", capture)
        self.assertIn("upstream-anchor-comparison", capture)
        self.assertIn("--source-manifest-asset-id", capture)
        self.assertIn("--validate-source-manifest", capture)
        self.assertIn("Accept: application/octet-stream", capture)
        self.assertIn("source_manifest_max_bytes=65536", capture)
        self.assertIn("source_manifest_directory=$(mktemp -d)", capture)
        self.assertIn(
            'source_manifest_path="${source_manifest_directory}/source-'
            '${source_manifest_asset_id}.json"',
            capture,
        )
        self.assertIn('head -c "$((source_manifest_max_bytes + 1))"', capture)
        self.assertIn('publisher_claim_type="immutable-source-manifest"', capture)
        self.assertIn(
            "${upstream_reference_head}...${live_upstream_head}",
            capture,
        )
        self.assertIn("release_pattern='^b[0-9]+$'", capture)

    def test_release_commit_resolution_uses_explicit_ref_namespaces(self) -> None:
        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        asset_job = validation.split("  verify-release-assets:\n", 1)[1].split(
            "  build:\n", 1
        )[0]
        capture = (
            ROOT / ".github" / "scripts" / "capture_llamacpp_release_manifest.sh"
        ).read_text(encoding="utf-8")

        for source in (asset_job, capture):
            with self.subTest(source="workflow" if source is asset_job else "capture"):
                self.assertIn(
                    "commits/heads/${source_default_branch}",
                    source,
                )
                self.assertNotIn(
                    "commits/${source_default_branch}",
                    source,
                )
                self.assertIn(
                    '"repos/${repository}/commits/tags/${tag}"',
                    source,
                )
                self.assertNotIn(
                    '"repos/${repository}/commits/${tag}"',
                    source,
                )

    def test_release_asset_eligibility_uses_size_and_rocm_target_metadata(self) -> None:
        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        asset_job = validation.split("  verify-release-assets:\n", 1)[1].split(
            "  build:\n", 1
        )[0]

        self.assertIn("def read_asset_metadata(path):", asset_job)
        self.assertIn('asset_sizes[asset["name"]] = asset["size"]', asset_job)
        self.assertIn('read_asset_metadata("ggml_release.json")', asset_job)
        self.assertIn('read_asset_metadata("rocm_release.json")', asset_job)
        self.assertIn('read_asset_metadata("lemonade_release.json")', asset_job)
        self.assertIn("build_rocm_asset_target_requirements", asset_job)
        self.assertIn("required_asset_targets=rocm_target_requirements", asset_job)
        self.assertIn("attested_asset_targets=rocm_attested_targets", asset_job)

        versions = {
            "therock": {"version": "7.14.0"},
            "rocm_asset_families": {
                "gfx1033": "gfx103X",
                "gfx1035": "gfx103X",
                "gfx1036": "gfx103X",
            },
        }
        requirements = release_assets.build_asset_requirements(
            ggml_release="b100",
            rocm_release="b300",
            lemonade_release="b200",
            backend_versions=versions,
        )
        required_targets = release_assets.build_rocm_asset_target_requirements(
            rocm_release="b300",
            backend_versions=versions,
        )
        available = {
            asset_name: 1 for asset_name in requirements["rocm"]["rocm-nightly"]
        }
        for platform in ("windows", "ubuntu"):
            target_asset = f"llama-b300-{platform}-rocm-gfx103X-x64.zip"
            for missing_target in ("gfx1033", "gfx1035", "gfx1036"):
                with self.subTest(
                    platform=platform,
                    missing_target=missing_target,
                ):
                    attested_targets = dict(required_targets)
                    attested_targets[target_asset] = tuple(
                        target
                        for target in required_targets[target_asset]
                        if target != missing_target
                    )
                    eligible, missing = release_assets.evaluate_asset_group(
                        available,
                        requirements["rocm"],
                        required_asset_targets=required_targets,
                        attested_asset_targets=attested_targets,
                    )
                    self.assertEqual(eligible, [])
                    self.assertEqual(missing["rocm-nightly"], [target_asset])

    def test_pr_validation_uses_a_trusted_merge_check_gate(self) -> None:
        pull_request = PR_WORKFLOW.read_text(encoding="utf-8")
        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        pull_request_triggers = pull_request.split("permissions:", 1)[0]
        concurrency = pull_request.split("concurrency:\n", 1)[1].split("\njobs:\n", 1)[
            0
        ]
        policy_job = pull_request.split("  policy:\n", 1)[1].split("  validate:\n", 1)[
            0
        ]
        invocation_job = validation.split("  validate-invocation:\n", 1)[1].split(
            "  plan:\n", 1
        )[0]
        validation_job = pull_request.split("  validate:\n", 1)[1].split(
            "  reconcile:\n", 1
        )[0]
        report_job = pull_request.split("  reconcile:\n", 1)[1]
        report_step = report_job.split(
            "      - name: Report result on the exact candidate commit\n", 1
        )[1].split("      - name: Enforce policy result\n", 1)[0]

        self.assertIn("  pull_request_target:\n", pull_request_triggers)
        self.assertIn("  merge_group:\n", pull_request_triggers)
        self.assertNotIn("  pull_request:\n", pull_request_triggers)
        self.assertNotIn("  workflow_call:\n", pull_request_triggers)
        self.assertIn(
            "types: [opened, reopened, synchronize, edited, labeled, unlabeled, closed]",
            pull_request_triggers,
        )
        self.assertIn("github.event.pull_request.number", concurrency)
        self.assertIn("github.event.merge_group.head_sha", concurrency)
        self.assertIn("github.run_number", concurrency)
        self.assertIn("github.run_attempt", concurrency)
        self.assertNotIn("github.run_id", concurrency)
        self.assertNotIn("cancel-in-progress: true", concurrency)

        self.assertIn('if [ "$EVENT_NAME" = "merge_group" ]', policy_job)
        self.assertIn("collaborators/${authorization_actor}/permission", policy_job)
        self.assertIn("merge_candidate_available", policy_job)
        self.assertIn('candidate_sha="$current_head"', policy_job)
        self.assertIn("--paginate --slurp", policy_job)
        self.assertIn('.app.slug == "github-actions"', policy_job)

        self.assertNotIn("  start-check:\n", pull_request)
        self.assertIn("checks: write", invocation_job)
        self.assertIn("github.event.pull_request.merge_commit_sha", invocation_job)
        self.assertIn("github.event.merge_group.head_sha", invocation_job)
        self.assertNotIn("github.event.pull_request.head.sha", invocation_job)
        self.assertIn('head_sha="$MERGE_SHA"', invocation_job)
        self.assertIn("CHECK_NAME: llama.cpp authorized validation v2", invocation_job)
        self.assertIn('name="$CHECK_NAME"', invocation_job)
        self.assertIn('external_id="$EXTERNAL_ID"', invocation_job)
        self.assertIn("github.run_id", invocation_job)
        self.assertIn("github.run_number", invocation_job)
        self.assertIn("github.run_attempt", invocation_job)
        self.assertIn("status=in_progress", invocation_job)
        self.assertIn("check_run_id", invocation_job)
        self.assertIn("--paginate --slurp", invocation_job)
        self.assertIn('.app.slug == "github-actions"', invocation_job)
        self.assertIn("newest_number", invocation_job)
        self.assertIn(".number < $current_number", invocation_job)
        self.assertIn(
            "value: ${{ jobs.validate-invocation.outputs.check_run_id }}", validation
        )

        self.assertIn("needs: policy", validation_job)
        self.assertNotIn("start-check", validation_job)
        self.assertIn(
            "needs.policy.outputs.run_validation == 'true'",
            validation_job,
        )
        self.assertIn(
            "uses: ./.github/workflows/validate_llamacpp_core.yml", validation_job
        )
        self.assertIn(
            "permissions:\n      checks: write\n      contents: read", validation_job
        )
        self.assertNotIn("secrets:", validation_job)
        self.assertNotIn("HUGGINGFACE_ACCESS_TOKEN", pull_request)

        self.assertIn("needs: [policy, validate]", report_job)
        self.assertIn("if: always()", report_job)
        self.assertIn("checks: write", report_job)
        self.assertIn("repos/${GITHUB_REPOSITORY}/check-runs", report_job)
        self.assertIn("github.event.pull_request.head.sha", report_job)
        self.assertIn("needs.validate.outputs.check_run_id", report_job)
        self.assertIn("commits/${MERGE_SHA}/check-runs", report_step)
        self.assertIn(".external_id", report_step)
        self.assertIn("--arg external_id", report_step)
        self.assertIn('head_sha="$MERGE_SHA"', report_step)
        self.assertIn("CHECK_NAME: llama.cpp authorized validation v2", report_job)
        self.assertIn('name="$CHECK_NAME"', report_step)
        self.assertIn('external_id="$external_id"', report_step)
        self.assertNotIn('head_sha="$HEAD_SHA"', report_step)
        self.assertNotIn('-f name="llama.cpp validation"', pull_request)
        self.assertNotIn("    name: llama.cpp validation\n", pull_request)
        self.assertIn("--method PATCH", report_step)
        self.assertIn("check-runs/${CHECK_RUN_ID}", report_step)
        self.assertIn("pull-requests: read", report_job)
        self.assertIn("repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}", report_job)
        self.assertIn("github.event.pull_request.base.sha", report_job)
        self.assertIn("github.event.pull_request.merge_commit_sha", report_job)
        self.assertIn("github.event.merge_group.head_sha", report_job)
        self.assertIn(".base.sha", report_job)
        self.assertIn(".head.sha", report_job)
        self.assertIn(".merge_commit_sha", report_job)
        self.assertIn('current_candidate="$current_head"', report_step)
        self.assertIn('.app.slug == "github-actions"', report_step)
        self.assertIn("newest_number", report_step)
        self.assertIn(".number < $current_number", report_step)
        self.assertLess(
            report_step.index("repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}"),
            report_step.index("--method PATCH"),
        )
        self.assertIn("needs.validate.result", report_job)

        self.assertIn("  validate-invocation:\n", validation)
        plan_job = validation.split("  plan:\n", 1)[1].split(
            "  get-latest-releases:\n", 1
        )[0]
        build_job_header = validation.split("  build:\n", 1)[1].split(
            "    runs-on:", 1
        )[0]
        validation_gate = validation.split("  validation-gate:\n", 1)[1]

        self.assertIn(
            "merge_group|pull_request_target|schedule|workflow_dispatch",
            invocation_job,
        )
        self.assertIn("Unsupported invocation event", invocation_job)
        self.assertNotIn("actions/checkout", invocation_job)
        self.assertIn("needs: validate-invocation", plan_job)
        self.assertIn("validate-invocation", build_job_header)
        self.assertIn(
            "needs: [validate-invocation, build, validate, "
            "reverify-protected-release-manifest, plan]",
            validation_gate,
        )
        self.assertNotIn("checks: write", validation_gate)
        self.assertNotIn("check-runs", validation_gate)

        merge_repository = "github.repository"
        merge_sha = "github.event.pull_request.merge_commit_sha"
        self.assertGreaterEqual(validation.count(merge_repository), 4)
        self.assertGreaterEqual(validation.count(merge_sha), 2)
        self.assertGreaterEqual(
            validation.count("github.event.merge_group.head_sha"), 4
        )
        self.assertIn("github.event.pull_request.base.sha", validation)
        self.assertIn("github.event.pull_request.head.sha", validation)
        self.assertEqual(validation.count("persist-credentials: false"), 10)

    def test_authorization_removal_reconciles_from_fresh_live_state(self) -> None:
        pull_request = PR_WORKFLOW.read_text(encoding="utf-8")
        policy_job = pull_request.split("  policy:\n", 1)[1].split("  validate:\n", 1)[
            0
        ]
        helper = REQUIRED_CHECK_POLICY.read_text(encoding="utf-8")

        self.assertIn("current_authorized", policy_job)
        self.assertIn("authorization_label_present=$current_authorized", policy_job)
        self.assertIn("EVENT_ACTION", policy_job)
        self.assertIn("EVENT_LABEL", policy_job)
        self.assertIn("AUTHORIZATION_LABEL_PRESENT", policy_job)
        self.assertIn("not authorization_label_present", helper)
        self.assertIn("conclusion=failure", pull_request)
        self.assertNotIn("  revoke-check:\n", pull_request)

    def test_closed_pull_requests_use_live_merged_state_before_mutating(self) -> None:
        pull_request = PR_WORKFLOW.read_text(encoding="utf-8")
        helper = REQUIRED_CHECK_POLICY.read_text(encoding="utf-8")

        self.assertIn("current_state", pull_request)
        self.assertIn("current_merged", pull_request)
        self.assertIn("if merged:", helper)
        self.assertIn('if state == "closed":', helper)
        self.assertIn("mode != 'merged'", pull_request)
        self.assertIn("mode != 'skip_closed'", pull_request)
        self.assertNotIn("  authorize-invocation:\n", pull_request)
        self.assertNotIn("  revoke-check:\n", pull_request)

    def test_protected_change_runner_matrix_is_defined_by_trusted_yaml(self) -> None:
        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        plan_job = validation.split("  plan:\n", 1)[1].split(
            "  get-latest-releases:\n", 1
        )[0]

        self.assertIn("      - name: Build trusted protected-change matrix\n", plan_job)
        trusted_step = plan_job.split(
            "      - name: Build trusted protected-change matrix\n", 1
        )[1].split("      - name: Build branch validation matrix\n", 1)[0]
        branch_step = plan_job.split(
            "      - name: Build branch validation matrix\n", 1
        )[1]

        self.assertIn("github.event_name == 'pull_request_target'", trusted_step)
        self.assertIn("github.event_name == 'merge_group'", trusted_step)
        self.assertIn('"lemon-prod"', trusted_step)
        self.assertIn('"K2-Horizon-0.9B-GGUF"', trusted_step)
        self.assertIn('"K2-Horizon-3.7B-GGUF"', trusted_step)
        self.assertIn('"K2-Horizon-7B-GGUF"', trusted_step)
        self.assertIn('target: "windows-vulkan"', trusted_step)
        for target in (
            "windows-vulkan",
            "linux-vulkan",
            "linux-cpu",
            "macos-metal",
            "windows-cuda",
            "windows-rocm-stable",
            "windows-rocm-nightly",
        ):
            with self.subTest(target=target):
                self.assertIn(f'target: "{target}"', trusted_step)
        self.assertEqual(trusted_step.count('capability_profile: "k2-horizon-v1"'), 7)
        self.assertEqual(
            trusted_step.count('capability_models: ["K2-Horizon-0.9B-GGUF"]'),
            7,
        )
        self.assertEqual(trusted_step.count('managed_pin: "vulkan"'), 2)
        for managed_pin in (
            "cpu",
            "cuda",
            "metal",
            "rocm-stable",
            "rocm-nightly",
        ):
            self.assertEqual(trusted_step.count(f'managed_pin: "{managed_pin}"'), 1)
        self.assertEqual(trusted_step.count('build_platform: "windows"'), 4)
        self.assertEqual(trusted_step.count('build_platform: "linux"'), 2)
        self.assertEqual(trusted_step.count('build_platform: "macos"'), 1)
        self.assertEqual(trusted_step.count('runner: ["macos-latest"]'), 1)
        self.assertNotIn("python -m test.utils", trusted_step)
        self.assertIn("github.event_name != 'pull_request_target'", branch_step)
        self.assertIn("github.event_name != 'merge_group'", branch_step)
        self.assertIn("python -m test.utils.llamacpp_validation_plan", branch_step)
        self.assertIn("steps.protected-plan.outputs.matrix", plan_job)
        self.assertIn("steps.branch-plan.outputs.matrix", plan_job)

        validate_job = validation.split("  validate:\n", 1)[1].split(
            "  validation-gate:\n", 1
        )[0]
        self.assertIn(".github/scripts/run_llamacpp_validation.py", validate_job)
        self.assertIn('          TARGET: "${{ matrix.target }}"\n', validate_job)
        self.assertIn(
            "          CAPABILITY_PROFILE: ${{ matrix.capability_profile || '' }}\n",
            validate_job,
        )
        self.assertIn("          CAPABILITY_MODELS:", validate_job)
        self.assertIn("          LLAMACPP_BACKEND: ${{ matrix.backend }}", validate_job)
        self.assertIn("          LLAMACPP_CHANNEL: ${{ matrix.channel }}", validate_job)
        self.assertIn("validation-results-${{ matrix.target }}", validate_job)
        self.assertIn("steps.run-validation.outputs.validation_result", validate_job)

    def test_manual_validation_calls_the_read_only_core(self) -> None:
        manual = MANUAL_WORKFLOW.read_text(encoding="utf-8")
        manual_triggers = manual.split("permissions:", 1)[0]

        self.assertIn("  workflow_dispatch:\n", manual_triggers)
        self.assertNotIn("  schedule:\n", manual_triggers)
        self.assertIn("permissions:\n  contents: read\n", manual)
        self.assertNotIn("contents: write", manual)
        self.assertIn("permissions:\n      checks: write\n      contents: read", manual)
        self.assertNotIn("pull-requests: write", manual)
        self.assertNotIn("secrets: inherit", manual)
        self.assertNotIn("HUGGINGFACE_ACCESS_TOKEN", manual)
        self.assertIn(
            "uses: ./.github/workflows/validate_llamacpp_core.yml",
            manual,
        )

    def test_validation_credential_aliases_are_scrubbed_by_trusted_control_plane(
        self,
    ) -> None:
        step = workflow_step(
            VALIDATION_WORKFLOW,
            "validate",
            "Run validation with lemond",
        )
        environment = step["env"]
        run = step["run"]
        credential_aliases = {
            "HF_TOKEN",
            "HUGGINGFACE_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        }

        self.assertEqual(
            {key for key in environment if key.upper() in credential_aliases},
            credential_aliases,
        )
        self.assertEqual(
            environment["HF_TOKEN"],
            "${{ github.event_name == 'schedule' && secrets.HUGGINGFACE_ACCESS_TOKEN || '' }}",
        )
        self.assertEqual(environment["HUGGINGFACE_TOKEN"], "")
        self.assertEqual(environment["HUGGING_FACE_HUB_TOKEN"], "")
        self.assertEqual(run.count("--allow-huggingface-download-credentials"), 1)
        self.assertIn(
            "${{ github.event_name == 'schedule' && '--allow-huggingface-download-credentials' || '' }}",
            run,
        )
        self.assertIn(
            'python "$VALIDATION_ROOT/.github/scripts/run_llamacpp_validation.py"',
            run,
        )
        self.assertNotIn("VALIDATION_CANDIDATE_ROOT/.github/scripts", run)

        manual_job = load_workflow(MANUAL_WORKFLOW)["jobs"]["validate"]
        schedule_job = load_workflow(SCHEDULE_WORKFLOW)["jobs"]["validate"]
        self.assertNotIn("secrets", manual_job)
        self.assertEqual(
            schedule_job["secrets"],
            {"HUGGINGFACE_ACCESS_TOKEN": ("${{ secrets.HUGGINGFACE_ACCESS_TOKEN }}")},
        )

    def test_scheduled_publication_is_isolated_and_serialized(self) -> None:
        schedule = SCHEDULE_WORKFLOW.read_text(encoding="utf-8")
        schedule_triggers = schedule.split("permissions:", 1)[0]
        validation_job = schedule.split("  validate:\n", 1)[1].split("  publish:\n", 1)[
            0
        ]
        publication = schedule.split("  publish:\n", 1)[1]
        checkout_marker = f"      - uses: actions/checkout@{CHECKOUT_SHA} # v5.1.0\n"
        publication_identity_step = publication.split(
            "      - name: Require dedicated publication identity\n", 1
        )[1].split(checkout_marker, 1)[0]
        checkout_step = publication.split(checkout_marker, 1)[1].split(
            "\n      - name:", 1
        )[0]

        self.assertIn("  schedule:\n", schedule_triggers)
        self.assertNotIn("  workflow_dispatch:\n", schedule_triggers)
        self.assertNotIn("  workflow_call:\n", schedule_triggers)
        self.assertNotIn("  pull_request:\n", schedule_triggers)
        self.assertNotIn("  merge_group:\n", schedule_triggers)
        self.assertIn("group: llamacpp-auto-update\n", schedule)
        self.assertIn(
            "uses: ./.github/workflows/validate_llamacpp_core.yml",
            validation_job,
        )
        self.assertIn(
            "permissions:\n      checks: write\n      contents: read", validation_job
        )
        self.assertNotIn("contents: write", validation_job)
        self.assertIn(
            "permissions:\n      checks: write\n      contents: read", validation_job
        )
        self.assertNotIn("secrets: inherit", schedule)
        self.assertIn(
            "HUGGINGFACE_ACCESS_TOKEN: ${{ secrets.HUGGINGFACE_ACCESS_TOKEN }}",
            validation_job,
        )
        self.assertIn("group: llamacpp-auto-update-publication", publication)
        self.assertIn("cancel-in-progress: false", publication)
        self.assertIn(
            "needs.validate.outputs.promotion_eligible == 'true'", publication
        )
        self.assertIn("permissions:\n      contents: read", publication)
        self.assertNotIn("contents: write", publication)
        self.assertNotIn("pull-requests: write", publication)
        self.assertNotIn("secrets.GITHUB_TOKEN", publication)
        self.assertIn("publish_llamacpp_update.sh", publication)
        self.assertIn("fetch-depth: 0", checkout_step)
        self.assertIn("persist-credentials: false", checkout_step)
        self.assertIn("token: ${{ secrets.LLAMACPP_UPDATE_TOKEN }}", checkout_step)
        self.assertEqual(
            publication.count(
                f"uses: actions/download-artifact@{DOWNLOAD_ARTIFACT_SHA} # v7.0.0"
            ),
            2,
        )
        self.assertNotIn("actions/checkout@v5", publication)
        self.assertNotIn("actions/download-artifact@v7", publication)
        self.assertIn("gh api graphql", publication_identity_step)
        self.assertIn(".data.viewer.login", publication_identity_step)
        self.assertIn(
            "GH_TOKEN: ${{ secrets.LLAMACPP_UPDATE_TOKEN }}",
            publication_identity_step,
        )
        self.assertIn(
            "EXPECTED_PUBLICATION_ACTOR: ${{ vars.LLAMACPP_UPDATE_ACTOR }}",
            publication_identity_step,
        )
        self.assertLess(
            publication.index("      - name: Require dedicated publication identity"),
            publication.index(f"      - uses: actions/checkout@{CHECKOUT_SHA}"),
        )
        self.assertIn("VALIDATED_BASE_SHA: ${{ github.sha }}", publication)
        self.assertIn(
            "BASE_BRANCH: ${{ github.event.repository.default_branch }}",
            publication,
        )
        self.assertIn("pattern: validation-results-*", publication)
        self.assertIn("pattern: restart-validation-results-*", publication)
        evidence_step = publication.split(
            "      - name: Validate complete validation evidence\n", 1
        )[1].split(
            "      - name: Update backend_versions.json from release manifest\n", 1
        )[
            0
        ]
        update_step = publication.split(
            "      - name: Update backend_versions.json from release manifest\n", 1
        )[1].split("      - name: Generate PR body\n", 1)[0]
        manifest_step = publication.split(
            "      - name: Verify release asset manifest\n", 1
        )[1].split("      - name: Publish update pull request\n", 1)[0]
        self.assertIn("llamacpp_validation_evidence", evidence_step)
        self.assertIn("--expected-matrix-json", evidence_step)
        self.assertIn(
            "needs.validate.outputs.validation_matrix",
            evidence_step,
        )
        self.assertEqual(update_step.count("set -euo pipefail"), 1)
        self.assertIn("llamacpp_release_manifest", manifest_step)
        self.assertIn(
            "needs.validate.outputs.release_asset_manifest",
            manifest_step,
        )
        self.assertIn("capture_llamacpp_release_manifest.sh", manifest_step)
        self.assertIn("--materialize-manifest", manifest_step)
        self.assertIn("GH_TOKEN: ${{ secrets.LLAMACPP_UPDATE_TOKEN }}", manifest_step)
        self.assertIn("EXPECTED_RELEASE_ASSET_MANIFEST", publication)
        self.assertNotIn("\n      - name:", manifest_step)
        self.assertLess(
            publication.index("      - name: Generate PR body\n"),
            publication.index("      - name: Verify release asset manifest\n"),
        )
        self.assertLess(
            publication.index("      - name: Verify release asset manifest\n"),
            publication.index("      - name: Publish update pull request\n"),
        )
        publisher_step = publication.split(
            "      - name: Publish update pull request\n", 1
        )[1]
        self.assertIn("GH_TOKEN: ${{ secrets.LLAMACPP_UPDATE_TOKEN }}", publisher_step)
        self.assertIn(
            "EXPECTED_PUBLICATION_ACTOR: ${{ vars.LLAMACPP_UPDATE_ACTOR }}",
            publisher_step,
        )

        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        output_names = (
            "promotion_eligible",
            "validation_matrix",
            "llamacpp_release",
            "llamacpp_rocm_release",
            "llamacpp_lemonade_release",
            "ggml_update_backends",
            "lemonade_update_backends",
            "rocm_update_backends",
            "ggml_missing_count",
            "rocm_nightly_missing_count",
            "rocm_stable_missing_count",
            "cuda_missing_count",
            "vulkan_available",
            "cpu_available",
            "metal_available",
            "rocm_nightly_available",
            "rocm_stable_available",
            "cuda_available",
            "release_asset_manifest",
        )
        for output_name in output_names:
            with self.subTest(output_name=output_name):
                self.assertIn(f"      {output_name}:\n", validation)
                self.assertIn(f"needs.validate.outputs.{output_name}", schedule)


if __name__ == "__main__":
    unittest.main()
