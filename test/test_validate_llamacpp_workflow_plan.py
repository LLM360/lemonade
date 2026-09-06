#!/usr/bin/env python3
"""Unit tests for the llama.cpp validation workflow plan."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test.utils import llamacpp_validation_plan as planning

K2_SMALL = "K2-Horizon-0.9B-GGUF"
K2_MEDIUM = "K2-Horizon-3.7B-GGUF"
K2_LARGE = "K2-Horizon-7B-GGUF"
ROOT = Path(__file__).resolve().parents[1]
LEGACY_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp.yml"
VALIDATION_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_core.yml"
PR_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_pr.yml"
MANUAL_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_manual.yml"
SCHEDULE_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_schedule.yml"
DOCS_AND_STYLE_WORKFLOW = ROOT / ".github" / "workflows" / "docs_and_style.yml"


class LlamaCppValidationPlanTests(unittest.TestCase):
    def assert_active_lanes(self, rows):
        self.assertEqual(
            [(row["backend"], row["channel"]) for row in rows],
            [("vulkan", ""), ("rocm", "stable"), ("rocm", "nightly")],
        )
        self.assertEqual(
            [row["target"] for row in rows],
            ["windows-vulkan", "rocm-stable", "rocm-nightly"],
        )

    def test_pull_request_and_merge_group_require_k2(self) -> None:
        for event_name in ("pull_request", "merge_group"):
            with self.subTest(event_name=event_name):
                rows = planning.create_validation_plan(event_name)["include"]

                self.assert_active_lanes(rows)
                self.assertEqual(rows[0]["models"], [K2_SMALL, K2_MEDIUM, K2_LARGE])
                self.assertEqual(rows[1]["models"], [K2_SMALL])
                self.assertEqual(rows[2]["models"], [K2_SMALL])
                self.assertTrue(all(not row["lite"] for row in rows))
                self.assertIn("128gb", rows[0]["runner"])
                self.assertTrue(
                    all(row["capability_profile"] == "k2-horizon-v1" for row in rows)
                )
                self.assertTrue(
                    all(row["capability_models"] == [K2_SMALL] for row in rows)
                )

    def test_schedule_captures_exact_hot_model_evidence_contract(self) -> None:
        rows = planning.create_validation_plan("schedule")["include"]

        self.assert_active_lanes(rows)
        expected = planning.load_hot_llamacpp_model_ids()
        self.assertIn(K2_SMALL, expected)
        self.assertTrue(all(row["models"] == [] for row in rows))
        self.assertTrue(all(row["expected_models"] == expected for row in rows))
        self.assertTrue(all(not row["lite"] for row in rows))
        self.assertTrue(all("128gb" in row["runner"] for row in rows))
        self.assertTrue(
            all(row["capability_profile"] == "k2-horizon-v1" for row in rows)
        )
        self.assertTrue(all(row["capability_models"] == [K2_SMALL] for row in rows))

    def test_default_dispatch_keeps_runtime_hot_selection(self) -> None:
        rows = planning.create_validation_plan("workflow_dispatch")["include"]

        self.assert_active_lanes(rows)
        self.assertTrue(all(row["models"] == [] for row in rows))
        self.assertTrue(all(not row["lite"] for row in rows))
        self.assertTrue(all("128gb" in row["runner"] for row in rows))
        self.assertTrue(all("capability_profile" not in row for row in rows))
        self.assertTrue(all("capability_models" not in row for row in rows))

    def test_manual_lite_selection_uses_small_runners(self) -> None:
        rows = planning.create_validation_plan("workflow_dispatch", lite=True)[
            "include"
        ]

        self.assert_active_lanes(rows)
        self.assertTrue(all(row["models"] == [] for row in rows))
        self.assertTrue(all(row["lite"] for row in rows))
        self.assertTrue(all("stx-halo" in row["runner"] for row in rows))

    def test_manual_models_are_trimmed_and_keep_order_and_duplicates(self) -> None:
        rows = planning.create_validation_plan(
            "workflow_dispatch",
            models_csv=" Zulu, ,Alpha,Zulu ",
        )["include"]

        self.assert_active_lanes(rows)
        self.assertTrue(all(row["models"] == ["Zulu", "Alpha", "Zulu"] for row in rows))
        self.assertTrue(all(not row["lite"] for row in rows))
        self.assertTrue(all("128gb" in row["runner"] for row in rows))

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

    def test_validation_core_is_read_only_and_has_no_legacy_entrypoint(self) -> None:
        self.assertFalse(LEGACY_WORKFLOW.exists())

        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        validation_triggers = validation.split("permissions:", 1)[0]
        workflow_scope = validation.split("jobs:", 1)[0]
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
        ].split("      - name: Upload server logs\n", 1)[0]

        self.assertIn("  workflow_call:\n", validation_triggers)
        self.assertNotIn("  pull_request_target:\n", validation_triggers)
        self.assertNotIn("  pull_request:\n", validation_triggers)
        self.assertNotIn("  merge_group:\n", validation_triggers)
        self.assertNotIn("  workflow_dispatch:\n", validation_triggers)
        self.assertNotIn("  schedule:\n", validation_triggers)
        self.assertIn("permissions:\n  contents: read\n", validation)
        self.assertNotIn("contents: write", validation)
        self.assertNotIn("pull-requests: write", validation)
        self.assertNotIn("checks: write", validation)
        self.assertNotIn("concurrency:", workflow_scope)
        self.assertNotIn("  create-pr:\n", validation)
        self.assertIn("      HUGGINGFACE_ACCESS_TOKEN:\n", validation_triggers)
        self.assertNotIn("HF_TOKEN:", workflow_scope)
        self.assertNotIn("secrets.HUGGINGFACE_ACCESS_TOKEN", workflow_scope)
        self.assertNotIn("HUGGINGFACE_ACCESS_TOKEN", plan_job)
        self.assertNotIn("HUGGINGFACE_ACCESS_TOKEN", build_job)
        self.assertEqual(validate_job.count("HF_TOKEN:"), 1)
        self.assertIn(
            "          HF_TOKEN: ${{ github.event_name != 'pull_request_target' && github.event_name != 'merge_group' && secrets.HUGGINGFACE_ACCESS_TOKEN || '' }}\n",
            validation_step,
        )
        self.assertNotIn(
            "          HF_TOKEN: ${{ secrets.HUGGINGFACE_ACCESS_TOKEN }}\n",
            validation_step,
        )
        self.assertIn("          $cleanupExitCode = 0\n", validation_step)
        self.assertIn("              $cleanupExitCode = 1\n", validation_step)
        self.assertIn("          if ($cleanupExitCode -ne 0) {\n", validation_step)
        self.assertIn("          if-no-files-found: error\n", result_upload)

    def test_focused_ci_runs_the_capability_contract_tests(self) -> None:
        workflow = DOCS_AND_STYLE_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("test.test_llamacpp_capabilities", workflow)
        self.assertIn("test.test_llamacpp_validation_evidence", workflow)
        self.assertIn("test.test_validate_llamacpp_selection", workflow)

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
        self.assertIn("repos/${repository}/commits/${tag}", asset_job)
        self.assertIn("source_commit", asset_job)
        self.assertNotIn("release_asset_manifest", release_job)

    def test_pr_validation_uses_a_trusted_merge_check_gate(self) -> None:
        pull_request = PR_WORKFLOW.read_text(encoding="utf-8")
        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        pull_request_triggers = pull_request.split("permissions:", 1)[0]
        concurrency = pull_request.split("concurrency:\n", 1)[1].split("\njobs:\n", 1)[
            0
        ]
        authorization_job = pull_request.split("  authorize-invocation:\n", 1)[1].split(
            "  start-check:\n", 1
        )[0]
        start_check_job = pull_request.split("  start-check:\n", 1)[1].split(
            "  validate:\n", 1
        )[0]
        validation_job = pull_request.split("  validate:\n", 1)[1].split(
            "  report:\n", 1
        )[0]
        report_job = pull_request.split("  report:\n", 1)[1]
        report_step = report_job.split(
            "      - name: Report result on the authorized merge commit\n", 1
        )[1].split("      - name: Enforce result\n", 1)[0]

        self.assertIn("  pull_request_target:\n", pull_request_triggers)
        self.assertIn("  merge_group:\n", pull_request_triggers)
        self.assertNotIn("  pull_request:\n", pull_request_triggers)
        self.assertNotIn("  workflow_call:\n", pull_request_triggers)
        self.assertIn(
            "    types: [synchronize, edited, labeled, unlabeled, closed]\n",
            pull_request_triggers,
        )
        self.assertIn("github.event.pull_request.number", concurrency)
        self.assertNotIn("github.ref", concurrency)
        self.assertIn("github.run_id", concurrency)
        self.assertIn("github.event.action == 'synchronize'", concurrency)
        self.assertIn("github.event.action == 'edited'", concurrency)
        self.assertIn("github.event.changes.base.ref.from", concurrency)
        self.assertIn("github.event.action == 'closed'", concurrency)
        self.assertIn("github.event.action == 'unlabeled'", concurrency)
        self.assertIn("github.event.label.name == 'ci:upgrades'", concurrency)
        self.assertNotIn("cancel-in-progress: true", concurrency)

        self.assertIn("github.event.action", authorization_job)
        self.assertIn('if [ "$EVENT_NAME" = "merge_group" ]', authorization_job)
        self.assertIn("github.event.label.name", authorization_job)
        self.assertIn("github.actor", authorization_job)
        self.assertIn("collaborators/${ACTOR}/permission", authorization_job)
        self.assertIn("admin|maintain|write|push", authorization_job)
        self.assertIn(
            'echo "run_validation=true" >> "$GITHUB_OUTPUT"', authorization_job
        )
        self.assertNotIn("actions/checkout", authorization_job)

        self.assertIn("needs: authorize-invocation", start_check_job)
        self.assertIn("checks: write", start_check_job)
        self.assertIn("github.event.pull_request.merge_commit_sha", start_check_job)
        self.assertIn("github.event.merge_group.head_sha", start_check_job)
        self.assertNotIn("github.event.pull_request.head.sha", start_check_job)
        self.assertIn('head_sha="$MERGE_SHA"', start_check_job)
        self.assertIn("CHECK_NAME: llama.cpp authorized validation v2", start_check_job)
        self.assertIn('name="$CHECK_NAME"', start_check_job)
        self.assertIn('external_id="$EXTERNAL_ID"', start_check_job)
        self.assertIn("github.run_id", start_check_job)
        self.assertIn("github.run_attempt", start_check_job)
        self.assertIn("status=in_progress", start_check_job)
        self.assertIn("check_run_id", start_check_job)

        self.assertIn("needs: [authorize-invocation, start-check]", validation_job)
        self.assertIn(
            "needs.authorize-invocation.outputs.run_validation == 'true'",
            validation_job,
        )
        self.assertIn(
            "uses: ./.github/workflows/validate_llamacpp_core.yml", validation_job
        )
        self.assertIn("permissions:\n      contents: read", validation_job)
        self.assertNotIn("secrets:", validation_job)
        self.assertNotIn("HUGGINGFACE_ACCESS_TOKEN", pull_request)

        self.assertIn(
            "needs: [authorize-invocation, start-check, validate]", report_job
        )
        self.assertIn("checks: write", report_job)
        self.assertIn("repos/${GITHUB_REPOSITORY}/check-runs", report_job)
        self.assertIn("github.event.pull_request.head.sha", report_job)
        self.assertIn("needs.start-check.outputs.check_run_id", report_job)
        self.assertIn("commits/${MERGE_SHA}/check-runs", report_step)
        self.assertIn(".external_id", report_step)
        self.assertIn("--arg external_id", report_step)
        self.assertIn('head_sha="$MERGE_SHA"', report_step)
        self.assertIn("CHECK_NAME: llama.cpp authorized validation v2", report_job)
        self.assertIn('name="$CHECK_NAME"', report_step)
        self.assertIn('external_id="$EXTERNAL_ID"', report_step)
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
        self.assertLess(
            report_step.index("repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}"),
            report_step.index("--method PATCH"),
        )
        self.assertIn("github.event.action == 'labeled'", report_job)
        self.assertIn("github.event.label.name == 'ci:upgrades'", report_job)
        self.assertIn("needs.validate.result", report_job)
        self.assertIn("github.event.action == 'labeled'", report_step)
        self.assertIn("github.event.label.name == 'ci:upgrades'", report_step)

        self.assertIn("  validate-invocation:\n", validation)
        invocation_job = validation.split("  validate-invocation:\n", 1)[1].split(
            "  plan:\n", 1
        )[0]
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
            "needs: [validate-invocation, build, validate, plan]", validation_gate
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
        self.assertEqual(validation.count("persist-credentials: false"), 4)

    def test_authorization_removal_revokes_the_merge_check(self) -> None:
        pull_request = PR_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("  revoke-check:\n", pull_request)
        revoke_job = pull_request.split("  revoke-check:\n", 1)[1].split(
            "  start-check:\n", 1
        )[0]

        self.assertIn("github.event.action == 'unlabeled'", revoke_job)
        self.assertIn("github.event.label.name == 'ci:upgrades'", revoke_job)
        self.assertIn("github.event.pull_request.state == 'open'", revoke_job)
        self.assertIn("checks: write", revoke_job)
        self.assertIn("pull-requests: read", revoke_job)
        self.assertIn("repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}", revoke_job)
        self.assertIn("current_state", revoke_job)
        self.assertIn("current_authorized", revoke_job)
        self.assertIn(
            'if [ "$current_state" != "open" ] || '
            '[ "$current_authorized" = "true" ]',
            revoke_job,
        )
        self.assertIn("github.event.pull_request.merge_commit_sha", revoke_job)
        self.assertIn("commits/${MERGE_SHA}/check-runs", revoke_job)
        self.assertIn("llama.cpp authorized validation v2", revoke_job)
        self.assertIn("conclusion=cancelled", revoke_job)
        self.assertIn("--method PATCH", revoke_job)
        self.assertIn("--method POST", revoke_job)
        self.assertNotIn("actions/checkout", revoke_job)

    def test_closed_pull_requests_cannot_start_protected_validation(self) -> None:
        pull_request = PR_WORKFLOW.read_text(encoding="utf-8")
        authorization_job = pull_request.split("  authorize-invocation:\n", 1)[1].split(
            "  revoke-check:\n", 1
        )[0]

        self.assertIn("current_state=$(jq -r '.state'", authorization_job)
        self.assertIn('if [ "$current_state" != "open" ]', authorization_job)
        state_guard = authorization_job.index('if [ "$current_state" != "open" ]')
        authorization_output = authorization_job.index(
            'echo "run_validation=true" >> "$GITHUB_OUTPUT"',
            state_guard,
        )
        self.assertLess(state_guard, authorization_output)

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
        self.assertEqual(trusted_step.count('capability_profile: "k2-horizon-v1"'), 3)
        self.assertEqual(
            trusted_step.count('capability_models: ["K2-Horizon-0.9B-GGUF"]'),
            3,
        )
        self.assertNotIn("python -m test.utils", trusted_step)
        self.assertIn("github.event_name != 'pull_request_target'", branch_step)
        self.assertIn("github.event_name != 'merge_group'", branch_step)
        self.assertIn("python -m test.utils.llamacpp_validation_plan", branch_step)
        self.assertIn("steps.protected-plan.outputs.matrix", plan_job)
        self.assertIn("steps.branch-plan.outputs.matrix", plan_job)

        validate_job = validation.split("  validate:\n", 1)[1].split(
            "  validation-gate:\n", 1
        )[0]
        self.assertIn('          $label = "${{ matrix.target }}"\n', validate_job)
        self.assertIn(
            "          CAPABILITY_PROFILE: ${{ matrix.capability_profile || '' }}\n",
            validate_job,
        )
        self.assertIn("          CAPABILITY_MODELS:", validate_job)
        self.assertIn("-CapabilityProfile $env:CAPABILITY_PROFILE", validate_job)
        self.assertIn("-CapabilityModels $env:CAPABILITY_MODELS", validate_job)
        self.assertIn("validation-results-${{ matrix.target }}", validate_job)
        self.assertIn("llamacpp_validation_${{ matrix.target }}.json", validate_job)

    def test_manual_validation_calls_the_read_only_core(self) -> None:
        manual = MANUAL_WORKFLOW.read_text(encoding="utf-8")
        manual_triggers = manual.split("permissions:", 1)[0]

        self.assertIn("  workflow_dispatch:\n", manual_triggers)
        self.assertNotIn("  schedule:\n", manual_triggers)
        self.assertIn("permissions:\n  contents: read\n", manual)
        self.assertNotIn("contents: write", manual)
        self.assertNotIn("checks: write", manual)
        self.assertNotIn("pull-requests: write", manual)
        self.assertNotIn("secrets: inherit", manual)
        self.assertIn(
            "HUGGINGFACE_ACCESS_TOKEN: ${{ secrets.HUGGINGFACE_ACCESS_TOKEN }}",
            manual,
        )
        self.assertIn(
            "uses: ./.github/workflows/validate_llamacpp_core.yml",
            manual,
        )

    def test_scheduled_publication_is_isolated_and_serialized(self) -> None:
        schedule = SCHEDULE_WORKFLOW.read_text(encoding="utf-8")
        schedule_triggers = schedule.split("permissions:", 1)[0]
        validation_job = schedule.split("  validate:\n", 1)[1].split("  publish:\n", 1)[
            0
        ]
        publication = schedule.split("  publish:\n", 1)[1]

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
        self.assertIn("permissions:\n      contents: read", validation_job)
        self.assertNotIn("contents: write", validation_job)
        self.assertNotIn("checks: write", validation_job)
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
        self.assertIn("contents: write", publication)
        self.assertIn("pull-requests: write", publication)
        self.assertIn("publish_llamacpp_update.sh", publication)
        self.assertIn(
            "      - uses: actions/checkout@v5\n        with:\n          fetch-depth: 0",
            publication,
        )
        self.assertIn("VALIDATED_BASE_SHA: ${{ github.sha }}", publication)
        self.assertIn(
            "BASE_BRANCH: ${{ github.event.repository.default_branch }}",
            publication,
        )
        evidence_step = publication.split(
            "      - name: Validate complete validation evidence\n", 1
        )[1].split(
            "      - name: Update backend_versions.json with verified releases\n", 1
        )[
            0
        ]
        manifest_step = publication.split(
            "      - name: Verify release asset manifest\n", 1
        )[1].split("      - name: Publish update pull request\n", 1)[0]
        self.assertIn("llamacpp_validation_evidence", evidence_step)
        self.assertIn("--expected-matrix-json", evidence_step)
        self.assertIn(
            "needs.validate.outputs.validation_matrix",
            evidence_step,
        )
        self.assertIn("llamacpp_release_manifest", manifest_step)
        self.assertIn(
            "needs.validate.outputs.release_asset_manifest",
            manifest_step,
        )
        self.assertIn("gh api", manifest_step)
        self.assertIn("repos/${repository}/commits/${tag}", manifest_step)
        self.assertIn("source_commit", manifest_step)
        self.assertNotIn("\n      - name:", manifest_step)
        self.assertLess(
            publication.index("      - name: Generate PR body\n"),
            publication.index("      - name: Verify release asset manifest\n"),
        )
        self.assertLess(
            publication.index("      - name: Verify release asset manifest\n"),
            publication.index("      - name: Publish update pull request\n"),
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
