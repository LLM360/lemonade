#!/usr/bin/env python3
"""Unit tests for the managed llama.cpp release-asset contract."""

from __future__ import annotations

import unittest

from test.utils import llamacpp_release_assets as assets


class LlamaCppReleaseAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requirements = assets.build_asset_requirements(
            ggml_release="b100",
            rocm_release="b200",
            lemonade_release="b300",
            backend_versions={
                "therock": {"version": "7.14.0"},
                "rocm_asset_families": {},
            },
        )

    def test_cuda_covers_every_published_compute_capability_and_platform(self) -> None:
        required = set(self.requirements["lemonade"]["cuda"])

        for compute_capability in (
            "sm_75",
            "sm_80",
            "sm_86",
            "sm_89",
            "sm_90",
            "sm_100",
            "sm_120",
            "sm_121",
        ):
            with self.subTest(compute_capability=compute_capability):
                self.assertIn(
                    f"llama-b300-windows-cuda-{compute_capability}-x64.7z",
                    required,
                )
                self.assertIn(
                    f"llama-b300-ubuntu-cuda-{compute_capability}-x64.tar.xz",
                    required,
                )
                self.assertIn(
                    f"llama-b300-ubuntu-cuda-{compute_capability}-arm64.tar.xz",
                    required,
                )

        self.assertEqual(len(required), 24)

    def test_rocm_nightly_covers_advertised_platform_targets(self) -> None:
        required = set(self.requirements["rocm"]["rocm-nightly"])
        both_platforms = (
            "gfx1151",
            "gfx1150",
            "gfx120X",
            "gfx110X",
            "gfx103X",
            "gfx90a",
            "gfx908",
            "gfx1152",
        )

        for target in both_platforms:
            with self.subTest(target=target):
                self.assertIn(f"llama-b200-windows-rocm-{target}-x64.zip", required)
                self.assertIn(f"llama-b200-ubuntu-rocm-{target}-x64.zip", required)

        self.assertIn("llama-b200-ubuntu-rocm-gfx942-x64.zip", required)
        self.assertNotIn("llama-b200-windows-rocm-gfx942-x64.zip", required)
        self.assertFalse(any("gfx950" in name for name in required))
        self.assertEqual(len(required), 17)

    def test_rocm_targets_use_resource_family_mapping_without_duplicates(self) -> None:
        requirements = assets.build_asset_requirements(
            ggml_release="b100",
            rocm_release="b200",
            lemonade_release="b300",
            backend_versions={
                "therock": {"version": "7.14.0"},
                "rocm_asset_families": {
                    "gfx1151": "gfx115X",
                    "gfx1152": "gfx115X",
                },
            },
        )
        required = requirements["rocm"]["rocm-nightly"]

        self.assertEqual(required.count("llama-b200-windows-rocm-gfx115X-x64.zip"), 1)
        self.assertEqual(required.count("llama-b200-ubuntu-rocm-gfx115X-x64.zip"), 1)

    def test_missing_new_target_keeps_rocm_nightly_pin_ineligible(self) -> None:
        required = self.requirements["rocm"]["rocm-nightly"]
        available = set(required)
        missing_asset = "llama-b200-ubuntu-rocm-gfx942-x64.zip"
        available.remove(missing_asset)

        eligible, missing = assets.evaluate_asset_group(
            available,
            self.requirements["rocm"],
        )

        self.assertEqual(eligible, [])
        self.assertEqual(missing, {"rocm-nightly": [missing_asset]})

    def test_invalid_release_tag_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid GGML_RELEASE"):
            assets.build_asset_requirements(
                ggml_release="main",
                rocm_release="b200",
                lemonade_release="b300",
                backend_versions={
                    "therock": {"version": "7.14.0"},
                    "rocm_asset_families": {},
                },
            )


if __name__ == "__main__":
    unittest.main()
