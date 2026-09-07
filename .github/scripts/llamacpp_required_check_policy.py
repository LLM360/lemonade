#!/usr/bin/env python3
"""Trusted path and lifecycle policy for the llama.cpp required check."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import re
from typing import NamedTuple

MAX_CHANGED_FILES = 3000
MAX_COMPARE_COMMITS = 250
MAX_COMPARE_FILES = 300
MAX_PAGE_SIZE = 100
MAX_EVENT_PAGES = 100
CHANGED_FILE_STATUSES = {
    "added",
    "changed",
    "copied",
    "modified",
    "removed",
    "renamed",
    "unchanged",
}
AUTHORIZATION_LABEL = "ci:upgrades"
TRUSTED_PERMISSIONS = {"admin", "maintain", "push", "write"}
AUTHORIZATION_INVALIDATING_EVENTS = {
    "automatic_base_change_succeeded",
    "base_ref_changed",
    "closed",
    "head_ref_deleted",
    "head_ref_force_pushed",
    "head_ref_restored",
    "merged",
    "reopened",
}
IDENTITY_CHANGING_ACTIONS = {"opened", "reopened", "synchronize"}
REQUIRED_WORKFLOW_ACTIONS = {"opened", "reopened", "synchronize"}
SUPPORTED_ACTIONS = {
    "closed",
    "edited",
    "labeled",
    "opened",
    "reopened",
    "synchronize",
    "unlabeled",
}
COMPARE_STATUSES = {"ahead", "behind", "diverged", "identical"}
FULL_OID_PATTERN = re.compile(r"[0-9a-f]{40}")

NEUTRAL_PREFIXES = (
    "docs/",
    "src/app/",
    "src/web-app/",
)
PROTECTED_OVERRIDES = frozenset({"src/app/src/renderer/utils/toolDefinitions.json"})


class PolicyDecision(NamedTuple):
    mode: str
    run_validation: bool
    conclusion: str
    summary: str


class AuthorizationProvenance(NamedTuple):
    authorization_event_id: int
    authorization_event: str
    authorization_actor: str
    authorization_actor_id: int
    authorization_created_at: str
    latest_invalidator_event_id: int
    epoch_valid: bool


def _validated_path(path: object, field: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError(f"{field} must be a non-empty string")
    if "\\" in path or path.startswith("/"):
        raise ValueError(f"{field} must be a repository-relative POSIX path")
    parsed = PurePosixPath(path)
    if any(part in {"", ".", ".."} for part in parsed.parts) or str(parsed) != path:
        raise ValueError(f"{field} is not a normalized repository path")
    return path


def is_protected_path(path: str) -> bool:
    normalized = _validated_path(path, "filename")
    return normalized in PROTECTED_OVERRIDES or not normalized.startswith(
        NEUTRAL_PREFIXES
    )


def _classify_file_records(records: list[object], *, source: str) -> bool:
    filenames: set[str] = set()
    relevant = False
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{source} file record {index} must be an object")
        filename = _validated_path(record.get("filename"), "filename")
        if filename in filenames:
            raise ValueError(f"duplicate {source} filename: {filename}")
        filenames.add(filename)
        relevant = relevant or is_protected_path(filename)

        status = record.get("status")
        if not isinstance(status, str) or status not in CHANGED_FILE_STATUSES:
            raise ValueError(f"{source} file has an invalid status: {filename}")

        has_previous_filename = "previous_filename" in record
        if status == "renamed":
            if not has_previous_filename:
                raise ValueError(
                    f"renamed file is missing previous_filename: {filename}"
                )
            previous = _validated_path(record["previous_filename"], "previous_filename")
            relevant = relevant or is_protected_path(previous)
        elif has_previous_filename:
            raise ValueError(
                f"previous_filename is only valid for renamed files: {filename}"
            )

    return relevant


def classify_changed_file_pages(pages: object, *, expected_count: int) -> bool:
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ValueError("expected_count must be an integer")
    if expected_count < 0 or expected_count > MAX_CHANGED_FILES:
        raise ValueError(f"changed_files must be between 0 and {MAX_CHANGED_FILES}")
    if not isinstance(pages, list):
        raise ValueError("paginated pull-request files must be a JSON array")
    if len(pages) > MAX_CHANGED_FILES // MAX_PAGE_SIZE:
        raise ValueError("pull-request files exceeded the API pagination limit")

    records: list[object] = []
    for page in pages:
        if not isinstance(page, list):
            raise ValueError("each pull-request files page must be a JSON array")
        if len(page) > MAX_PAGE_SIZE:
            raise ValueError("a pull-request files page exceeded per_page=100")
        records.extend(page)

    if len(records) != expected_count:
        raise ValueError(
            "paginated pull-request file count does not match changed_files"
        )

    return _classify_file_records(records, source="pull-request")


def _validated_oid(value: object, field: str) -> str:
    if not isinstance(value, str) or FULL_OID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase full Git OID")
    return value


def _validated_count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _nested_oid(response: dict[str, object], field: str) -> str:
    value = response.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"compare response {field} must be an object")
    return _validated_oid(value.get("sha"), f"compare response {field}.sha")


def classify_compare_response(
    response: object,
    *,
    expected_base_sha: str,
    expected_target_sha: str,
    expected_count: int,
) -> bool:
    expected_base = _validated_oid(expected_base_sha, "expected base SHA")
    expected_target = _validated_oid(expected_target_sha, "expected target SHA")
    expected_files = _validated_count(expected_count, "expected file count")
    if expected_files > MAX_COMPARE_FILES:
        return True
    if not isinstance(response, dict):
        raise ValueError("compare response must be a JSON object")

    base_commit = _nested_oid(response, "base_commit")
    merge_base_commit = _nested_oid(response, "merge_base_commit")
    if base_commit != expected_base:
        raise ValueError("compare response does not identify the expected base commit")

    status = response.get("status")
    if not isinstance(status, str) or status not in COMPARE_STATUSES:
        raise ValueError("compare response has an invalid status")
    ahead_by = _validated_count(response.get("ahead_by"), "compare response ahead_by")
    behind_by = _validated_count(
        response.get("behind_by"), "compare response behind_by"
    )
    total_commits = _validated_count(
        response.get("total_commits"), "compare response total_commits"
    )
    if total_commits != ahead_by:
        raise ValueError("compare response commit counts are inconsistent")

    expected_relationships = {
        "ahead": (True, False),
        "behind": (False, True),
        "diverged": (True, True),
        "identical": (False, False),
    }
    expects_ahead, expects_behind = expected_relationships[status]
    if (ahead_by > 0) != expects_ahead or (behind_by > 0) != expects_behind:
        raise ValueError("compare response status and commit counts are inconsistent")
    if status in {"ahead", "identical"} and merge_base_commit != expected_base:
        raise ValueError("compare response has an unexpected merge-base commit")
    if status == "behind" and merge_base_commit != expected_target:
        raise ValueError(
            "compare response does not identify the expected target commit"
        )
    if status == "diverged" and merge_base_commit in {expected_base, expected_target}:
        raise ValueError("compare response has an inconsistent divergent merge base")
    if (status == "identical") != (expected_base == expected_target):
        raise ValueError("compare response has an inconsistent identical target")

    commits = response.get("commits")
    if not isinstance(commits, list):
        raise ValueError("compare response commits must be a JSON array")
    expected_commit_records = min(total_commits, MAX_COMPARE_COMMITS)
    if len(commits) != expected_commit_records:
        raise ValueError("compare response commit list is incomplete")
    commit_oids: set[str] = set()
    for index, commit in enumerate(commits):
        if not isinstance(commit, dict):
            raise ValueError(f"compare response commit {index} must be an object")
        commit_oid = _validated_oid(
            commit.get("sha"), f"compare response commit {index}.sha"
        )
        if commit_oid in commit_oids:
            raise ValueError(f"duplicate compare response commit: {commit_oid}")
        commit_oids.add(commit_oid)
    if status in {"ahead", "diverged"} and commits[-1]["sha"] != expected_target:
        raise ValueError(
            "compare response does not identify the expected target commit"
        )

    files = response.get("files")
    if not isinstance(files, list):
        raise ValueError("compare response files must be a JSON array")
    if len(files) > MAX_COMPARE_FILES:
        raise ValueError("compare response exceeded the file-list limit")
    if status in {"behind", "identical"} and files:
        raise ValueError("compare response status cannot contain changed files")
    if len(files) != expected_files:
        return True
    return _classify_file_records(files, source="compare response")


def _parse_timestamp(value: str, field: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} has an invalid timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} has a naive timestamp")
    return timestamp


def classify_authorization_event_pages(
    pages: object,
    *,
    authorization_after: str | None = None,
) -> AuthorizationProvenance:
    if not isinstance(pages, list):
        raise ValueError("paginated issue events must be a JSON array")
    if len(pages) > MAX_EVENT_PAGES:
        raise ValueError("issue events exceeded the trusted pagination limit")

    boundary = (
        _parse_timestamp(authorization_after, "authorization boundary")
        if authorization_after is not None
        else None
    )
    records: list[object] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, list):
            raise ValueError("each issue-events page must be a JSON array")
        if len(page) > MAX_PAGE_SIZE:
            raise ValueError("an issue-events page exceeded per_page=100")
        if not page and page_index != len(pages) - 1:
            raise ValueError("an intermediate issue-events page was empty")
        records.extend(page)
    if (
        len(pages) == MAX_EVENT_PAGES
        and pages
        and isinstance(pages[-1], list)
        and len(pages[-1]) == MAX_PAGE_SIZE
    ):
        raise ValueError("issue events may have exceeded the trusted pagination limit")

    seen_ids: set[int] = set()
    seen_node_ids: set[str] = set()
    previous_event_id = 0
    previous_order: tuple[datetime, int] | None = None
    latest_authorization: tuple[int, dict[str, object]] | None = None
    latest_invalidator: tuple[int, int] | None = None
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"issue event record {index} must be an object")
        event_id = record.get("id")
        node_id = record.get("node_id")
        event = record.get("event")
        created_at = record.get("created_at")
        if (
            isinstance(event_id, bool)
            or not isinstance(event_id, int)
            or event_id <= 0
            or not isinstance(node_id, str)
            or not node_id
            or not isinstance(event, str)
            or not event
            or not isinstance(created_at, str)
        ):
            raise ValueError(f"issue event record {index} has invalid common fields")
        if event_id in seen_ids:
            raise ValueError(f"duplicate issue event ID: {event_id}")
        if event_id <= previous_event_id:
            raise ValueError("issue event IDs are not in strict API order")
        if node_id in seen_node_ids:
            raise ValueError(f"duplicate issue event node ID: {node_id}")
        seen_ids.add(event_id)
        previous_event_id = event_id
        seen_node_ids.add(node_id)
        timestamp = _parse_timestamp(
            created_at,
            f"issue event record {index} created_at",
        )
        order = (timestamp, event_id)
        if previous_order is not None and order <= previous_order:
            raise ValueError("issue events are not in strict chronological order")
        previous_order = order

        if event in {"labeled", "unlabeled"}:
            label = record.get("label")
            actor = record.get("actor")
            if (
                not isinstance(label, dict)
                or not isinstance(label.get("name"), str)
                or not label["name"]
                or not isinstance(actor, dict)
                or isinstance(actor.get("id"), bool)
                or not isinstance(actor.get("id"), int)
                or actor["id"] <= 0
                or not isinstance(actor.get("login"), str)
                or not actor["login"]
            ):
                raise ValueError(f"issue label event {event_id} is malformed")
            if label["name"] == AUTHORIZATION_LABEL:
                if (
                    not isinstance(actor.get("node_id"), str)
                    or not actor["node_id"]
                    or actor.get("type") != "User"
                    or record.get("performed_via_github_app") is not None
                    or re.fullmatch(
                        r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", actor["login"]
                    )
                    is None
                ):
                    raise ValueError(
                        f"authorization label event {event_id} has an invalid actor"
                    )
                latest_authorization = (index, record)

        if event in AUTHORIZATION_INVALIDATING_EVENTS:
            latest_invalidator = (index, event_id)

    invalidator_id = latest_invalidator[1] if latest_invalidator is not None else 0
    if latest_authorization is None:
        return AuthorizationProvenance(0, "", "", 0, "", invalidator_id, False)

    authorization_index, authorization = latest_authorization
    actor = authorization["actor"]
    assert isinstance(actor, dict)
    invalidator_index = latest_invalidator[0] if latest_invalidator is not None else -1
    authorization_created_at = str(authorization["created_at"])
    authorization_timestamp = _parse_timestamp(
        authorization_created_at,
        "authorization event created_at",
    )
    return AuthorizationProvenance(
        authorization_event_id=int(authorization["id"]),
        authorization_event=str(authorization["event"]),
        authorization_actor=str(actor["login"]),
        authorization_actor_id=int(actor["id"]),
        authorization_created_at=authorization_created_at,
        latest_invalidator_event_id=invalidator_id,
        epoch_valid=(
            authorization["event"] == "labeled"
            and authorization_index > invalidator_index
            and (boundary is None or authorization_timestamp > boundary)
        ),
    )


def decide_pull_request_policy(
    *,
    action: str,
    event_label: str,
    base_ref_changed: bool,
    state: str,
    merged: bool,
    merge_candidate_available: bool,
    relevant: bool,
    identity_matches: bool,
    authorization_label_present: bool,
    actor_permission: str,
    has_existing_check: bool,
    has_newer_check: bool,
    authorization_epoch_valid: bool = True,
    authorization_rerun_matches: bool = False,
    required_workflow_only: bool = False,
) -> PolicyDecision:
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"unsupported pull_request_target action: {action}")

    if state not in {"open", "closed"}:
        raise ValueError(f"unsupported pull request state: {state}")
    if merged and state != "closed":
        raise ValueError("a merged pull request must be closed")
    if merged:
        return PolicyDecision(
            "merged",
            False,
            "",
            "The pull request merged; the pre-merge gate is left unchanged.",
        )
    if state == "closed":
        return PolicyDecision(
            "closed",
            False,
            "cancelled",
            "The pull request was closed; protected validation was cancelled.",
        )
    if action == "labeled" and event_label == AUTHORIZATION_LABEL:
        return PolicyDecision(
            "awaiting_rerun",
            False,
            "failure",
            "The authorization label was recorded. Re-run all jobs on the "
            "matching failed required-workflow run.",
        )
    if has_newer_check:
        if action in REQUIRED_WORKFLOW_ACTIONS:
            return PolicyDecision(
                "superseded",
                False,
                "failure",
                "A newer diagnostic decision exists for this candidate. Use the "
                "newest failed default-event workflow run for authorization.",
            )
        return PolicyDecision(
            "preserve",
            False,
            "",
            "A newer exact-commit policy decision superseded this event.",
        )

    authorization_summary = (
        "Protected llama.cpp inputs changed. After the latest required-workflow "
        "run fails, a maintainer must apply ci:upgrades and then use Re-run all "
        "jobs on that same workflow run. If the label is already present, remove "
        "and reapply it after the latest candidate change."
    )
    if not identity_matches:
        if action in REQUIRED_WORKFLOW_ACTIONS:
            return PolicyDecision(
                "stale_required",
                False,
                "failure",
                "The required-workflow run no longer identifies the live candidate. "
                "Use the newest failed default-event workflow run.",
            )
        return PolicyDecision(
            "stale",
            False,
            "",
            "A stale event cannot change the live candidate's exact-commit gate.",
        )
    if action == "closed":
        return PolicyDecision(
            "stale",
            False,
            "",
            "A stale lifecycle event cannot change the open pull request's gate.",
        )

    if (
        action in REQUIRED_WORKFLOW_ACTIONS
        and identity_matches
        and merge_candidate_available
        and authorization_label_present
        and authorization_epoch_valid
        and authorization_rerun_matches
        and actor_permission in TRUSTED_PERMISSIONS
    ):
        return PolicyDecision(
            "validate",
            True,
            "",
            "A trusted maintainer authorized protected llama.cpp validation.",
        )

    if not relevant:
        return PolicyDecision(
            "neutral",
            False,
            "success",
            "No protected llama.cpp validation inputs changed; the gate is neutral.",
        )

    if action == "unlabeled" and event_label == AUTHORIZATION_LABEL:
        return PolicyDecision(
            "authorization_required", False, "failure", authorization_summary
        )

    if not authorization_label_present:
        return PolicyDecision(
            "authorization_required", False, "failure", authorization_summary
        )

    if not authorization_epoch_valid or actor_permission not in TRUSTED_PERMISSIONS:
        return PolicyDecision(
            "authorization_required", False, "failure", authorization_summary
        )

    if not merge_candidate_available:
        return PolicyDecision(
            "authorization_required",
            False,
            "failure",
            "GitHub has not produced a synthetic test merge commit. Resolve "
            "merge conflicts, establish a new default-event workflow run, then "
            "apply ci:upgrades and re-run all jobs on that run.",
        )

    invalidated = action in IDENTITY_CHANGING_ACTIONS or (
        action == "edited" and base_ref_changed
    )
    if invalidated:
        return PolicyDecision(
            "authorization_required", False, "failure", authorization_summary
        )

    if required_workflow_only:
        return PolicyDecision(
            "preserve",
            False,
            "",
            "An unrelated metadata event left the required-workflow gate unchanged.",
        )

    if not has_existing_check:
        return PolicyDecision(
            "authorization_required", False, "failure", authorization_summary
        )

    return PolicyDecision(
        "preserve",
        False,
        "",
        "An unrelated metadata event preserved the exact-commit gate.",
    )


def _parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _write_outputs(path: Path, values: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
            if "\n" in rendered or "\r" in rendered:
                raise ValueError(f"output {key} must fit on one line")
            output.write(f"{key}={rendered}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify = subparsers.add_parser("classify")
    classify.add_argument("--pages", type=Path, required=True)
    classify.add_argument("--expected-count", type=int, required=True)
    classify.add_argument("--github-output", type=Path, required=True)

    classify_compare = subparsers.add_parser("classify-compare")
    classify_compare.add_argument("--response", type=Path, required=True)
    classify_compare.add_argument("--expected-base-sha", required=True)
    classify_compare.add_argument("--expected-target-sha", required=True)
    classify_compare.add_argument("--expected-count", type=int, required=True)
    classify_compare.add_argument("--github-output", type=Path, required=True)

    provenance = subparsers.add_parser("provenance")
    provenance.add_argument("--pages", type=Path, required=True)
    provenance.add_argument("--authorization-after")

    decide = subparsers.add_parser("decide")
    decide.add_argument("--action", required=True)
    decide.add_argument("--event-label", default="")
    decide.add_argument("--base-ref-changed", type=_parse_bool, required=True)
    decide.add_argument("--state", required=True)
    decide.add_argument("--merged", type=_parse_bool, required=True)
    decide.add_argument("--merge-candidate-available", type=_parse_bool, required=True)
    decide.add_argument("--relevant", type=_parse_bool, required=True)
    decide.add_argument("--identity-matches", type=_parse_bool, required=True)
    decide.add_argument(
        "--authorization-label-present", type=_parse_bool, required=True
    )
    decide.add_argument("--authorization-epoch-valid", type=_parse_bool, required=True)
    decide.add_argument(
        "--authorization-rerun-matches", type=_parse_bool, required=True
    )
    decide.add_argument("--actor-permission", default="")
    decide.add_argument("--has-existing-check", type=_parse_bool, required=True)
    decide.add_argument("--has-newer-check", type=_parse_bool, required=True)
    decide.add_argument("--required-workflow-only", type=_parse_bool, required=True)
    decide.add_argument("--github-output", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "classify":
        pages = json.loads(args.pages.read_text(encoding="utf-8"))
        relevant = classify_changed_file_pages(
            pages,
            expected_count=args.expected_count,
        )
        _write_outputs(args.github_output, {"relevant": relevant})
        return 0
    if args.command == "classify-compare":
        response = json.loads(args.response.read_text(encoding="utf-8"))
        relevant = classify_compare_response(
            response,
            expected_base_sha=args.expected_base_sha,
            expected_target_sha=args.expected_target_sha,
            expected_count=args.expected_count,
        )
        _write_outputs(args.github_output, {"relevant": relevant})
        return 0
    if args.command == "provenance":
        pages = json.loads(args.pages.read_text(encoding="utf-8"))
        provenance = classify_authorization_event_pages(
            pages,
            authorization_after=args.authorization_after,
        )
        print(json.dumps(provenance._asdict(), separators=(",", ":")))
        return 0

    decision = decide_pull_request_policy(
        action=args.action,
        event_label=args.event_label,
        base_ref_changed=args.base_ref_changed,
        state=args.state,
        merged=args.merged,
        merge_candidate_available=args.merge_candidate_available,
        relevant=args.relevant,
        identity_matches=args.identity_matches,
        authorization_label_present=args.authorization_label_present,
        authorization_epoch_valid=args.authorization_epoch_valid,
        authorization_rerun_matches=args.authorization_rerun_matches,
        actor_permission=args.actor_permission,
        has_existing_check=args.has_existing_check,
        has_newer_check=args.has_newer_check,
        required_workflow_only=args.required_workflow_only,
    )
    _write_outputs(
        args.github_output,
        {
            "mode": decision.mode,
            "run_validation": decision.run_validation,
            "conclusion": decision.conclusion,
            "summary": decision.summary,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
