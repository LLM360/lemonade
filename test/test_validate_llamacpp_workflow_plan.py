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

    def test_schedule_and_default_dispatch_keep_full_hot_selection(self) -> None:
        for event_name in ("schedule", "workflow_dispatch"):
            with self.subTest(event_name=event_name):
                rows = planning.create_validation_plan(event_name)["include"]

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

    def test_github_output_contains_compact_matrix_json(self) -> None:
        plan = planning.create_validation_plan("merge_group")
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "github-output.txt"

            planning.write_github_output(output_path, plan)

            key, payload = (
                output_path.read_text(encoding="utf-8").rstrip().split("=", 1)
            )
            self.assertEqual(key, "matrix")
            self.assertEqual(json.loads(payload), plan)
            self.assertNotIn("\n", payload)


if __name__ == "__main__":
    unittest.main()
