#!/usr/bin/env python3
"""Unit tests for the managed llama.cpp release-asset contract."""

from __future__ import annotations

import unittest

from test.utils import llamacpp_release_assets as assets

ROCM_ASSET_FAMILIES = {
    "gfx1200": "gfx120X",
    "gfx1201": "gfx120X",
    "gfx1100": "gfx110X",
    "gfx1101": "gfx110X",
    "gfx1102": "gfx110X",
    "gfx1103": "gfx110X",
    "gfx1030": "gfx103X",
    "gfx1031": "gfx103X",
    "gfx1032": "gfx103X",
    "gfx1033": "gfx103X",
    "gfx1034": "gfx103X",
    "gfx1035": "gfx103X",
    "gfx1036": "gfx103X",
}


class LlamaCppReleaseAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.requirements = assets.build_asset_requirements(
            ggml_release="b100",
            rocm_release="b200",
            lemonade_release="b300",
            backend_versions={
                "therock": {"version": "7.14.0"},
                "rocm_asset_families": ROCM_ASSET_FAMILIES,
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
            "gfx1152",
        )

        for target in both_platforms:
            with self.subTest(target=target):
                self.assertIn(f"llama-b200-windows-rocm-{target}-x64.zip", required)
                self.assertIn(f"llama-b200-ubuntu-rocm-{target}-x64.zip", required)

        for target in ("gfx908", "gfx90a", "gfx942"):
            with self.subTest(target=target):
                self.assertIn(f"llama-b200-ubuntu-rocm-{target}-x64.zip", required)
                self.assertNotIn(f"llama-b200-windows-rocm-{target}-x64.zip", required)
        self.assertFalse(any("gfx950" in name for name in required))
        self.assertEqual(len(required), 15)

    def test_rocm_requirements_bind_each_asset_to_concrete_build_targets(self) -> None:
        required = assets.build_rocm_asset_target_requirements(
            rocm_release="b200",
            backend_versions={"rocm_asset_families": ROCM_ASSET_FAMILIES},
        )

        expected_gfx103x = {
            "gfx1030",
            "gfx1031",
            "gfx1032",
            "gfx1033",
            "gfx1034",
            "gfx1035",
            "gfx1036",
        }
        self.assertEqual(
            set(required["llama-b200-windows-rocm-gfx103X-x64.zip"]),
            expected_gfx103x,
        )
        self.assertEqual(
            set(required["llama-b200-ubuntu-rocm-gfx103X-x64.zip"]),
            expected_gfx103x,
        )
        self.assertEqual(
            required["llama-b200-ubuntu-rocm-gfx90a-x64.zip"],
            ("gfx90a",),
        )
        self.assertNotIn("llama-b200-windows-rocm-gfx90a-x64.zip", required)
        self.assertFalse(
            any(
                "gfx950" in target
                for targets in required.values()
                for target in targets
            )
        )
        self.assertEqual(set(required), set(self.requirements["rocm"]["rocm-nightly"]))

    def test_rocm_target_grouping_tracks_asset_family_mapping_drift(self) -> None:
        changed_families = dict(ROCM_ASSET_FAMILIES)
        changed_families["gfx1033"] = "gfx1033"

        required = assets.build_rocm_asset_target_requirements(
            rocm_release="b200",
            backend_versions={"rocm_asset_families": changed_families},
        )

        self.assertEqual(
            required["llama-b200-windows-rocm-gfx1033-x64.zip"],
            ("gfx1033",),
        )
        self.assertNotIn(
            "gfx1033",
            required["llama-b200-windows-rocm-gfx103X-x64.zip"],
        )

    def test_rocm_targets_use_resource_family_mapping_without_duplicates(self) -> None:
        requirements = assets.build_asset_requirements(
            ggml_release="b100",
            rocm_release="b200",
            lemonade_release="b300",
            backend_versions={
                "therock": {"version": "7.14.0"},
                "rocm_asset_families": {
                    **ROCM_ASSET_FAMILIES,
                    "gfx1151": "gfx115X",
                    "gfx1152": "gfx115X",
                },
            },
        )
        required = requirements["rocm"]["rocm-nightly"]

        self.assertEqual(required.count("llama-b200-windows-rocm-gfx115X-x64.zip"), 1)
        self.assertEqual(required.count("llama-b200-ubuntu-rocm-gfx115X-x64.zip"), 1)

    def test_rocm_canonical_family_names_cannot_be_remapped(self) -> None:
        collapsing_mapping = {
            target: "gfx1151"
            for target in (
                "gfx1150",
                "gfx120X",
                "gfx110X",
                "gfx103X",
                "gfx90a",
                "gfx908",
                "gfx1152",
                "gfx942",
            )
        }

        with self.assertRaisesRegex(ValueError, "concrete ROCm ISA"):
            assets.build_asset_requirements(
                ggml_release="b100",
                rocm_release="b200",
                lemonade_release="b300",
                backend_versions={
                    "therock": {"version": "7.14.0"},
                    "rocm_asset_families": collapsing_mapping,
                },
            )

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

    def test_rocm_eligibility_requires_attested_targets_but_allows_extras(
        self,
    ) -> None:
        required_assets = self.requirements["rocm"]["rocm-nightly"]
        required_targets = assets.build_rocm_asset_target_requirements(
            rocm_release="b200",
            backend_versions={"rocm_asset_families": ROCM_ASSET_FAMILIES},
        )
        available = {name: 1 for name in required_assets}
        attested = {
            name: (*targets, "gfx999") for name, targets in required_targets.items()
        }

        eligible, missing = assets.evaluate_asset_group(
            available,
            self.requirements["rocm"],
            required_asset_targets=required_targets,
            attested_asset_targets=attested,
        )

        self.assertEqual(eligible, ["rocm-nightly"])
        self.assertEqual(missing, {"rocm-nightly": []})

        incomplete = dict(attested)
        target_asset = "llama-b200-windows-rocm-gfx103X-x64.zip"
        incomplete[target_asset] = tuple(
            target for target in incomplete[target_asset] if target != "gfx1035"
        )
        eligible, missing = assets.evaluate_asset_group(
            available,
            self.requirements["rocm"],
            required_asset_targets=required_targets,
            attested_asset_targets=incomplete,
        )
        self.assertEqual(eligible, [])
        self.assertEqual(missing, {"rocm-nightly": [target_asset]})

    def test_missing_rocm_attestation_does_not_affect_independent_groups(self) -> None:
        ggml_available = {
            name: 1
            for required in self.requirements["ggml"].values()
            for name in required
        }
        rocm_available = {name: 1 for name in self.requirements["rocm"]["rocm-nightly"]}
        required_targets = assets.build_rocm_asset_target_requirements(
            rocm_release="b200",
            backend_versions={"rocm_asset_families": ROCM_ASSET_FAMILIES},
        )

        ggml_eligible, _ = assets.evaluate_asset_group(
            ggml_available,
            self.requirements["ggml"],
        )
        rocm_eligible, missing = assets.evaluate_asset_group(
            rocm_available,
            self.requirements["rocm"],
            required_asset_targets=required_targets,
            attested_asset_targets={},
        )

        self.assertEqual(set(ggml_eligible), {"cpu", "metal", "vulkan"})
        self.assertEqual(rocm_eligible, [])
        self.assertEqual(
            missing["rocm-nightly"],
            self.requirements["rocm"]["rocm-nightly"],
        )

    def test_zero_sized_required_asset_keeps_backend_ineligible(self) -> None:
        required = self.requirements["ggml"]["cpu"]
        available = {name: 1 for name in required}
        zero_sized_asset = "llama-b100-bin-win-cpu-x64.zip"
        available[zero_sized_asset] = 0

        eligible, missing = assets.evaluate_asset_group(
            available,
            self.requirements["ggml"],
        )

        self.assertNotIn("cpu", eligible)
        self.assertEqual(missing["cpu"], [zero_sized_asset])

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
