import json
import tempfile
import unittest
from pathlib import Path

from test.utils.validation_model_catalog import (
    ModelCatalogOverlayError,
    merge_model_catalog_files,
    merge_model_catalogs,
)


class ValidationModelCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _write(self, name, payload):
        path = self.root / name
        path.write_text(payload, encoding="utf-8")
        return path

    def _merge(self, base, overlay):
        output = self.root / "merged.json"
        merge_model_catalog_files(base, overlay, output)
        return output.read_text(encoding="utf-8")

    def test_merges_disjoint_models_deterministically(self):
        base = self._write(
            "base.json",
            json.dumps({"Existing": {"recipe": "llamacpp", "size": 1.0}}),
        )
        overlay = self._write(
            "overlay.json",
            json.dumps({"Candidate": {"recipe": "llamacpp", "size": 2.0}}),
        )

        first = self._merge(base, overlay)
        second = self._merge(base, overlay)

        self.assertEqual(first, second)
        self.assertEqual(
            json.loads(first),
            {
                "Candidate": {"recipe": "llamacpp", "size": 2.0},
                "Existing": {"recipe": "llamacpp", "size": 1.0},
            },
        )
        self.assertTrue(first.endswith("\n"))

    def test_accepts_an_identical_promoted_model(self):
        model = {"recipe": "llamacpp", "size": 2.0}
        base = self._write("base.json", json.dumps({"Candidate": model}))
        overlay = self._write("overlay.json", json.dumps({"Candidate": model}))

        merged = json.loads(self._merge(base, overlay))

        self.assertEqual(merged, {"Candidate": model})

    def test_returns_the_merged_catalog_without_writing_a_file(self):
        base = self._write(
            "base.json",
            json.dumps({"Existing": {"recipe": "llamacpp"}}),
        )
        overlay = self._write(
            "overlay.json",
            json.dumps({"Candidate": {"recipe": "llamacpp"}}),
        )

        self.assertEqual(
            merge_model_catalogs(base, overlay),
            {
                "Existing": {"recipe": "llamacpp"},
                "Candidate": {"recipe": "llamacpp"},
            },
        )
        self.assertFalse((self.root / "merged.json").exists())

    def test_rejects_a_conflicting_promoted_model(self):
        base = self._write(
            "base.json", json.dumps({"Candidate": {"recipe": "llamacpp"}})
        )
        overlay = self._write(
            "overlay.json", json.dumps({"Candidate": {"recipe": "vllm"}})
        )

        with self.assertRaisesRegex(
            ModelCatalogOverlayError, "conflicting model definition: Candidate"
        ):
            self._merge(base, overlay)

    def test_rejects_duplicate_json_keys(self):
        base = self._write(
            "base.json",
            '{"Duplicate":{"recipe":"llamacpp"},' '"Duplicate":{"recipe":"llamacpp"}}',
        )
        overlay = self._write("overlay.json", "{}")

        with self.assertRaisesRegex(
            ModelCatalogOverlayError, "duplicate JSON member: Duplicate"
        ):
            self._merge(base, overlay)

    def test_rejects_non_finite_json_numbers(self):
        valid = self._write("valid.json", "{}")
        for literal in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(literal=literal):
                invalid = self._write(
                    "invalid.json",
                    '{"Candidate":{"recipe":"llamacpp","size":' + literal + "}}",
                )

                with self.assertRaisesRegex(
                    ModelCatalogOverlayError, "non-finite JSON number"
                ):
                    self._merge(valid, invalid)

    def test_rejects_floating_point_overflow_in_both_catalogs(self):
        for invalid_side in ("base", "overlay"):
            with self.subTest(invalid_side=invalid_side):
                invalid = self._write(
                    f"{invalid_side}-overflow.json",
                    '{"Candidate":{"recipe":"llamacpp","size":1e400}}',
                )
                valid = self._write(f"{invalid_side}-valid.json", "{}")
                base, overlay = (
                    (invalid, valid) if invalid_side == "base" else (valid, invalid)
                )

                with self.assertRaisesRegex(
                    ModelCatalogOverlayError, "non-finite JSON number"
                ):
                    self._merge(base, overlay)

    def test_promoted_model_comparison_is_type_sensitive(self):
        base = self._write(
            "base.json",
            '{"Candidate":{"recipe":"llamacpp","value":1}}',
        )
        overlay = self._write(
            "overlay.json",
            '{"Candidate":{"recipe":"llamacpp","value":true}}',
        )

        with self.assertRaisesRegex(
            ModelCatalogOverlayError, "conflicting model definition: Candidate"
        ):
            self._merge(base, overlay)

    def test_rejects_non_object_catalogs_and_models(self):
        valid = self._write("valid.json", "{}")
        catalog_list = self._write("list.json", "[]")
        invalid_model = self._write("model.json", '{"Candidate":[]}')

        with self.assertRaisesRegex(
            ModelCatalogOverlayError, "catalog must be a JSON object"
        ):
            self._merge(catalog_list, valid)
        with self.assertRaisesRegex(
            ModelCatalogOverlayError, "Candidate must be a JSON object"
        ):
            self._merge(valid, invalid_model)

    def test_does_not_replace_output_after_a_validation_failure(self):
        output = self.root / "merged.json"
        output.write_text('{"preserved":true}\n', encoding="utf-8")
        base = self._write("base.json", "{}")
        overlay = self._write("overlay.json", '{"Candidate":[]}')

        with self.assertRaises(ModelCatalogOverlayError):
            merge_model_catalog_files(base, overlay, output)

        self.assertEqual(output.read_text(encoding="utf-8"), '{"preserved":true}\n')


if __name__ == "__main__":
    unittest.main()
