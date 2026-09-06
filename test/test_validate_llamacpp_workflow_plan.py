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
MANUAL_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_manual.yml"
SCHEDULE_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_schedule.yml"


class LlamaCppValidationPlanTests(unittest.TestCase):
    def assert_active_lanes(self, rows):
        self.assertEqual(
            [(row["backend"], row["channel"]) for row in rows],
            [("vulkan", ""), ("rocm", "stable"), ("rocm", "nightly")],
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

    def test_schedule_captures_exact_hot_model_evidence_contract(self) -> None:
        rows = planning.create_validation_plan("schedule")["include"]

        self.assert_active_lanes(rows)
        expected = planning.load_hot_llamacpp_model_ids()
        self.assertIn(K2_SMALL, expected)
        self.assertTrue(all(row["models"] == [] for row in rows))
        self.assertTrue(all(row["expected_models"] == expected for row in rows))
        self.assertTrue(all(not row["lite"] for row in rows))
        self.assertTrue(all("128gb" in row["runner"] for row in rows))

    def test_default_dispatch_keeps_runtime_hot_selection(self) -> None:
        rows = planning.create_validation_plan("workflow_dispatch")["include"]

        self.assert_active_lanes(rows)
        self.assertTrue(all(row["models"] == [] for row in rows))
        self.assertTrue(all(not row["lite"] for row in rows))
        self.assertTrue(all("128gb" in row["runner"] for row in rows))

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
        self.assertIn("  pull_request:\n", validation_triggers)
        self.assertIn("  merge_group:\n", validation_triggers)
        self.assertNotIn("  workflow_dispatch:\n", validation_triggers)
        self.assertNotIn("  schedule:\n", validation_triggers)
        self.assertIn("permissions:\n  contents: read\n", validation)
        self.assertNotIn("contents: write", validation)
        self.assertNotIn("pull-requests: write", validation)
        self.assertNotIn("  create-pr:\n", validation)
        self.assertIn("      HUGGINGFACE_ACCESS_TOKEN:\n", validation_triggers)
        self.assertNotIn("HF_TOKEN:", workflow_scope)
        self.assertNotIn("secrets.HUGGINGFACE_ACCESS_TOKEN", workflow_scope)
        self.assertNotIn("HUGGINGFACE_ACCESS_TOKEN", plan_job)
        self.assertNotIn("HUGGINGFACE_ACCESS_TOKEN", build_job)
        self.assertEqual(validate_job.count("HF_TOKEN:"), 1)
        self.assertIn(
            "          HF_TOKEN: ${{ secrets.HUGGINGFACE_ACCESS_TOKEN }}\n",
            validation_step,
        )
        self.assertIn("          if-no-files-found: error\n", result_upload)

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
        self.assertNotIn("release_asset_manifest", release_job)

    def test_pr_label_revocation_cancels_authorized_validation(self) -> None:
        validation = VALIDATION_WORKFLOW.read_text(encoding="utf-8")
        validation_triggers = validation.split("permissions:", 1)[0]
        concurrency = validation.split("concurrency:\n", 1)[1].split("\nenv:\n", 1)[0]
        build_job_header = validation.split("  build:\n", 1)[1].split(
            "    runs-on:", 1
        )[0]

        self.assertIn(
            "    types: [opened, synchronize, reopened, labeled, unlabeled, closed]\n",
            validation_triggers,
        )
        self.assertIn("github.event.pull_request.number", concurrency)
        self.assertNotIn("github.ref", concurrency)
        self.assertIn(
            "cancel-in-progress: ${{ github.event_name == 'pull_request' }}",
            concurrency,
        )
        self.assertIn("github.event.action == 'labeled'", build_job_header)
        self.assertIn(
            "github.event.label.name == 'ci:upgrades'",
            build_job_header,
        )
        self.assertNotIn("github.event.pull_request.labels.*.name", build_job_header)

    def test_manual_validation_calls_the_read_only_core(self) -> None:
        manual = MANUAL_WORKFLOW.read_text(encoding="utf-8")
        manual_triggers = manual.split("permissions:", 1)[0]

        self.assertIn("  workflow_dispatch:\n", manual_triggers)
        self.assertNotIn("  schedule:\n", manual_triggers)
        self.assertIn("permissions:\n  contents: read\n", manual)
        self.assertNotIn("contents: write", manual)
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
