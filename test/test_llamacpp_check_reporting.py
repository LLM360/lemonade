#!/usr/bin/env python3
"""Regression tests for llama.cpp validation check reporting."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_core.yml"
PR_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_pr.yml"
MANUAL_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_manual.yml"
SCHEDULE_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_schedule.yml"


def workflow_job(workflow: Path, job_name: str, next_job_name: str | None) -> str:
    text = workflow.read_text(encoding="utf-8")
    job = text.split(f"  {job_name}:\n", 1)[1]
    if next_job_name is not None:
        job = job.split(f"  {next_job_name}:\n", 1)[0]
    return job


class LlamaCppCheckReportingTests(unittest.TestCase):
    def test_reusable_validation_callers_grant_check_write_permission(self) -> None:
        callers = (
            (PR_WORKFLOW, "validate", "reconcile"),
            (MANUAL_WORKFLOW, "validate", None),
            (SCHEDULE_WORKFLOW, "validate", "publish"),
        )

        for workflow, job_name, next_job_name in callers:
            with self.subTest(workflow=workflow.name):
                job = workflow_job(workflow, job_name, next_job_name)
                self.assertIn(
                    "      checks: write\n",
                    job,
                    "The caller must grant each permission requested by the "
                    "reusable validation workflow.",
                )

    def test_reusable_workflow_starts_or_resets_check_before_validation(self) -> None:
        core = CORE_WORKFLOW.read_text(encoding="utf-8")
        invocation = workflow_job(CORE_WORKFLOW, "validate-invocation", "plan")
        existing_branch = '          if [ -z "$check_run_id" ]; then\n'
        invalid_id_guard = '          if ! [[ "$check_run_id" =~ ^[0-9]+$ ]]; then\n'
        self.assertIn(existing_branch, invocation)
        self.assertIn(invalid_id_guard, invocation)
        existing_check = invocation.split(existing_branch, 1)[1].split(
            invalid_id_guard, 1
        )[0]

        self.assertIn(
            "      check_run_id:\n"
            "        value: ${{ jobs.validate-invocation.outputs.check_run_id }}\n",
            core.split("permissions:\n", 1)[0],
        )
        self.assertIn("      checks: write\n", invocation)
        self.assertIn(
            "      check_run_id: ${{ steps.start-check.outputs.check_run_id }}\n",
            invocation,
        )
        self.assertIn("        id: start-check\n", invocation)
        self.assertIn("github.event_name == 'pull_request_target'", invocation)
        self.assertIn("github.event_name == 'merge_group'", invocation)
        self.assertIn(
            "${AUTHORIZATION_EPOCH}-${RUN_NUMBER}-${RUN_ATTEMPT}-${RUN_ID}",
            invocation,
        )
        self.assertIn(
            "AUTHORIZATION_EPOCH: ${{ inputs.authorization_epoch }}", invocation
        )
        self.assertIn("RUN_NUMBER: ${{ github.run_number }}", invocation)
        self.assertIn("-f status=in_progress", invocation)
        self.assertIn(
            'echo "check_run_id=$check_run_id" >> "$GITHUB_OUTPUT"', invocation
        )
        self.assertIn("          else\n", existing_check)
        reset = existing_check.split("          else\n", 1)[1]
        self.assertIn('"repos/${GITHUB_REPOSITORY}/check-runs/${check_run_id}"', reset)
        self.assertIn("-f status=in_progress", reset)
        self.assertNotIn("-f conclusion=", reset)
        self.assertGreaterEqual(invocation.count("--paginate --slurp"), 2)

    def test_pr_wrapper_reconciles_the_check_started_by_reusable_validation(
        self,
    ) -> None:
        wrapper = PR_WORKFLOW.read_text(encoding="utf-8")
        policy = workflow_job(PR_WORKFLOW, "policy", "validate")
        validate = workflow_job(PR_WORKFLOW, "validate", "reconcile")
        reconcile = workflow_job(PR_WORKFLOW, "reconcile", None)

        self.assertNotIn("\n  start-check:\n", wrapper)
        self.assertNotIn("\n  authorize-invocation:\n", wrapper)
        self.assertIn(
            "      run_validation: ${{ steps.decision.outputs.run_validation }}", policy
        )
        self.assertIn("    needs: policy\n", validate)
        self.assertNotIn("start-check", validate)
        self.assertIn("    needs: [policy, validate]\n", reconcile)
        self.assertNotIn("needs.start-check", reconcile)
        self.assertIn(
            "CHECK_RUN_ID: ${{ needs.validate.outputs.check_run_id }}", reconcile
        )

        exact_check_lookup = reconcile.split(
            '          if [[ "$CHECK_RUN_ID" =~ ^[0-9]+$ ]]; then\n', 1
        )[1].split('          if [[ "$CHECK_RUN_ID" =~ ^[0-9]+$ ]]; then\n', 1)[0]
        self.assertIn("commits/${MERGE_SHA}/check-runs", reconcile)
        self.assertIn('--arg external_id "$external_id"', exact_check_lookup)
        self.assertIn('.app.slug == "github-actions"', exact_check_lookup)
        self.assertIn(".external_id == $external_id", exact_check_lookup)


if __name__ == "__main__":
    unittest.main()
