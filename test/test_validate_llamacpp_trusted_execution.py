#!/usr/bin/env python3
"""Security-contract tests for protected llama.cpp validation execution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_core.yml"
TESTING_GUIDE = ROOT / "docs" / "dev" / "testing.md"
PROTECTED_EVENT = (
    "github.event_name == 'pull_request_target' || "
    "github.event_name == 'merge_group'"
)
UNPROTECTED_EVENT = (
    "github.event_name != 'pull_request_target' && "
    "github.event_name != 'merge_group'"
)
CHECKOUT_ACTION = "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09 # v5.1.0"


def job(workflow: str, name: str, following_name: str) -> str:
    return workflow.split(f"  {name}:\n", 1)[1].split(f"  {following_name}:\n", 1)[0]


def step(job_text: str, name: str, following_name: str) -> str:
    return job_text.split(f"      - name: {name}\n", 1)[1].split(
        f"      - name: {following_name}\n", 1
    )[0]


class TrustedExecutionTests(unittest.TestCase):
    def assert_trusted_validation_entrypoint(self, validation: str) -> None:
        run_script = validation.split("        run: >-\n", 1)[1]
        normalized = " ".join(run_script.split())
        trusted_entrypoint = (
            'python "$VALIDATION_ROOT/.github/scripts/' 'run_llamacpp_validation.py"'
        )

        self.assertEqual(normalized.count(trusted_entrypoint), 1)
        self.assertEqual(normalized.count("run_llamacpp_validation.py"), 1)
        self.assertNotIn("test/validate_llamacpp.py", normalized)
        self.assertNotIn("VALIDATION_CANDIDATE_ROOT", normalized)
        self.assertNotIn("candidate/", normalized)
        self.assertNotIn("cd candidate", normalized)

    def test_protected_validation_ignores_candidate_control_plane_files(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        build = job(workflow, "build", "validate")
        validate = job(workflow, "validate", "reverify-protected-release-manifest")

        for job_text in (build, validate):
            with self.subTest(job=job_text.splitlines()[0]):
                control_checkout = step(
                    job_text,
                    "Check out validation control plane",
                    "Check out the validation candidate",
                )
                candidate_checkout = step(
                    job_text,
                    "Check out the validation candidate",
                    "Verify authorized merge commit",
                )
                self.assertIn(CHECKOUT_ACTION, control_checkout)
                self.assertIn("github.event.pull_request.base.sha", control_checkout)
                self.assertIn("github.event.merge_group.base_sha", control_checkout)
                self.assertIn("github.sha", control_checkout)
                self.assertIn("persist-credentials: false", control_checkout)
                self.assertNotIn("allow-unsafe-pr-checkout", control_checkout)
                self.assertIn("path: candidate", candidate_checkout)
                self.assertIn("persist-credentials: false", candidate_checkout)
                self.assertIn(f"if: {PROTECTED_EVENT}", candidate_checkout)
                self.assertIn(
                    "github.event.pull_request.merge_commit_sha", candidate_checkout
                )
                self.assertIn("github.event.merge_group.head_sha", candidate_checkout)

        protected_catalog = step(
            build,
            "Stage protected validation-only model catalog",
            "Stage branch validation-only model catalog",
        )
        self.assertIn(f"if: {PROTECTED_EVENT}", protected_catalog)
        self.assertIn(
            'python "$GITHUB_WORKSPACE/test/utils/validation_model_catalog.py"',
            protected_catalog,
        )
        self.assertIn(
            'candidate_catalog="$GITHUB_WORKSPACE/candidate/src/cpp/resources/'
            'server_models.json"',
            protected_catalog,
        )
        self.assertIn(
            '--overlay "$GITHUB_WORKSPACE/test/fixtures/'
            'llamacpp_validation_models.json"',
            protected_catalog,
        )
        self.assertIn(
            "resolved_catalog.relative_to(candidate_root)",
            protected_catalog,
        )
        self.assertIn('--base "$candidate_catalog"', protected_catalog)
        self.assertIn('--output "$candidate_catalog"', protected_catalog)

        branch_catalog = step(
            build,
            "Stage branch validation-only model catalog",
            "Install Linux build dependencies",
        )
        self.assertIn(f"if: {UNPROTECTED_EVENT}", branch_catalog)
        self.assertIn(
            "working-directory: ${{ env.VALIDATION_CANDIDATE_ROOT }}",
            branch_catalog,
        )
        self.assertIn("python -m test.utils.validation_model_catalog", branch_catalog)

        protected_setup = step(
            validate,
            "Set up trusted or branch validation environment",
            "Run validation with lemond",
        )
        self.assertIn("uses: ./.github/actions/setup-venv", protected_setup)
        self.assertIn(
            "requirements-file: '${{ github.workspace }}/test/requirements.txt'",
            protected_setup,
        )

        before_trusted_runner = validate.split(
            "      - name: Run validation with lemond\n", 1
        )[0]
        self.assertNotIn("lemond --version", before_trusted_runner)
        self.assertNotIn("& $lemondExe --version", before_trusted_runner)
        self.assertNotIn('"$lemond" --version', before_trusted_runner)

        validation = step(validate, "Run validation with lemond", "Upload results")
        self.assertIn(
            "VALIDATION_ROOT: ${{ github.workspace }}",
            validation,
        )
        self.assertIn("PYTHONPATH: ${{ github.workspace }}", validation)
        self.assertIn("working-directory: ${{ github.workspace }}", validation)
        self.assertIn(
            'python "$VALIDATION_ROOT/.github/scripts/' 'run_llamacpp_validation.py"',
            validation,
        )
        self.assert_trusted_validation_entrypoint(validation)
        self.assertIn(
            "CANDIDATE_LEMOND: ${{ runner.os == 'Windows' && "
            "format('{0}/build/Release/lemond.exe', "
            "env.VALIDATION_CANDIDATE_ROOT) || "
            "format('{0}/build/lemond', env.VALIDATION_CANDIDATE_ROOT) }}",
            validation,
        )
        self.assertIn('--lemond "$CANDIDATE_LEMOND"', validation)
        self.assertNotIn("uses: ./candidate/", validate)
        self.assertNotIn("$GITHUB_WORKSPACE/candidate/.github", validation)
        self.assertNotIn("$GITHUB_WORKSPACE/candidate/test", validation)

        controlled_files = (
            ".github/actions/cleanup-processes-linux/action.yml",
            ".github/actions/cleanup-processes-windows/action.yml",
            ".github/actions/setup-python/action.yml",
            ".github/actions/setup-venv/action.yml",
            ".github/scripts/run_llamacpp_validation.py",
            "test/fixtures/llamacpp_validation_artifacts.json",
            "test/fixtures/llamacpp_validation_models.json",
            "test/requirements.txt",
            "test/validate_llamacpp.py",
            "test/utils/capabilities.py",
            "test/utils/llamacpp_capability_validation.py",
            "test/utils/llamacpp_validation_artifacts.py",
            "test/utils/llamacpp_validation_evidence.py",
            "test/utils/server_base.py",
            "test/utils/test_models.py",
            "test/utils/validation_model_catalog.py",
            "test/utils/validation_model_selection.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for relative_path in controlled_files:
                trusted_file = workspace / relative_path
                candidate_file = workspace / "candidate" / relative_path
                trusted_file.parent.mkdir(parents=True, exist_ok=True)
                candidate_file.parent.mkdir(parents=True, exist_ok=True)
                trusted_file.write_text("TRUSTED\n", encoding="utf-8")
                candidate_file.write_text("MALICIOUS\n", encoding="utf-8")

            execution_root = workspace
            for relative_path in controlled_files:
                with self.subTest(relative_path=relative_path):
                    selected = execution_root / relative_path
                    self.assertEqual(selected.read_text(encoding="utf-8"), "TRUSTED\n")

    def test_relative_candidate_validator_commands_fail_the_contract(self) -> None:
        trusted = (
            "        run: >-\n"
            '          python "$VALIDATION_ROOT/.github/scripts/'
            'run_llamacpp_validation.py"\n'
            '          --lemond "$CANDIDATE_LEMOND"\n'
        )
        substitutions = (
            trusted.replace(
                'python "$VALIDATION_ROOT/.github/scripts/'
                'run_llamacpp_validation.py"',
                "python candidate/test/validate_llamacpp.py",
            ),
            trusted + "          && python candidate/test/validate_llamacpp.py\n",
            trusted.replace(
                "$VALIDATION_ROOT/.github/scripts/run_llamacpp_validation.py",
                "$VALIDATION_CANDIDATE_ROOT/.github/scripts/"
                "run_llamacpp_validation.py",
            ),
            trusted.replace("python ", "cd candidate && python ", 1),
        )

        for substitution in substitutions:
            with self.subTest(substitution=substitution):
                with self.assertRaises(AssertionError):
                    self.assert_trusted_validation_entrypoint(substitution)

    def test_protected_ephemeral_mac_has_no_token_bearing_post_run_step(
        self,
    ) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        validate = job(workflow, "validate", "reverify-protected-release-manifest")
        validate_header = workflow.split("  validate:\n", 1)[1].split(
            "    steps:\n", 1
        )[0]
        runtime_and_later = validate.split(
            "      - name: Run validation with lemond\n", 1
        )[1]
        after_runtime = runtime_and_later.split("      - name: Upload results\n", 1)[1]
        protected_mac_exclusion = (
            "!((github.event_name == 'pull_request_target' || "
            "github.event_name == 'merge_group') && matrix.target == "
            "'macos-metal' && runner.environment == 'github-hosted' && "
            "join(matrix.runner, ',') == 'macos-latest')"
        )

        self.assertIn("permissions:\n      contents: read\n", validate_header)
        self.assertNotIn("GH_TOKEN:", validate)
        for upload_name in (
            "Upload results",
            "Upload restart results",
            "Upload server logs",
        ):
            upload = validate.split(f"      - name: {upload_name}\n", 1)[1]
            upload = upload.split("      - name:", 1)[0]
            with self.subTest(upload=upload_name):
                self.assertIn(protected_mac_exclusion, " ".join(upload.split()))
        self.assertNotIn("secrets.", after_runtime)
        self.assertNotIn("actions/cache", after_runtime)

    def test_trust_boundary_does_not_claim_hostile_native_isolation(self) -> None:
        guide = TESTING_GUIDE.read_text(encoding="utf-8")

        self.assertIn(
            "base-branch checkout supplies the setup and cleanup actions", guide
        )
        self.assertIn("every imported `test.utils` helper", guide)
        self.assertIn("built `lemond` and adjacent build resources", guide)
        self.assertIn(
            "does **not** make candidate native code an adversarial sandbox", guide
        )
        self.assertIn("execute as the runner's OS user", guide)
        self.assertIn("single-job ephemeral protected self-hosted workers", guide)
        self.assertIn("expose no repository secrets", guide)
        self.assertIn("grant only a read-only repository token", guide)
        self.assertIn("no artifact upload, cache upload", guide)


if __name__ == "__main__":
    unittest.main()
