#!/usr/bin/env python3
"""Supply-chain contract for protected llama.cpp validation workflows."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

ROOT = Path(__file__).resolve().parents[1]
CORE_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_core.yml"
DOCS_AND_STYLE_WORKFLOW = ROOT / ".github" / "workflows" / "docs_and_style.yml"
MANUAL_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_manual.yml"
PR_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_pr.yml"
SCHEDULE_WORKFLOW = ROOT / ".github" / "workflows" / "validate_llamacpp_schedule.yml"
SERVER_BASE = ROOT / "test" / "utils" / "server_base.py"
PROTECTED_WORKFLOWS = (
    PR_WORKFLOW,
    CORE_WORKFLOW,
    MANUAL_WORKFLOW,
    SCHEDULE_WORKFLOW,
)
PINNED_ACTION_PATTERN = re.compile(
    r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*@[0-9a-f]{40}$"
)
REVIEWED_ACTION_PINS = {
    "actions/checkout": "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09",
    "actions/download-artifact": "37930b1c2abaa49bbe596cd826c3c89aef350131",
    "actions/setup-python": "ece7cb06caefa5fff74198d8649806c4678c61a1",
    "actions/upload-artifact": "bbbca2ddaa5d8feaa63e36b76fdaad77386f024f",
}


def read_uses_references(path: Path) -> list[str]:
    references: list[str] = []
    visited_nodes: set[int] = set()

    def visit(node: Node | None) -> None:
        if node is None or id(node) in visited_nodes:
            return
        visited_nodes.add(id(node))

        if isinstance(node, MappingNode):
            for key, value in node.value:
                if isinstance(key, ScalarNode) and key.value == "uses":
                    if not isinstance(value, ScalarNode):
                        raise ValueError(f"uses value must be a scalar in {path}")
                    references.append(value.value)
                else:
                    visit(key)
                visit(value)
        elif isinstance(node, SequenceNode):
            for value in node.value:
                visit(value)

    try:
        for document in yaml.compose_all(
            path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader
        ):
            visit(document)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}: {error}") from error
    return references


def resolve_local_uses(repository_root: Path, reference: str) -> Path:
    target = (repository_root / reference.removeprefix("./")).resolve()
    try:
        target.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(
            f"local action escapes repository root: {reference}"
        ) from error
    if target.is_dir():
        manifests = [
            candidate
            for candidate in (target / "action.yml", target / "action.yaml")
            if candidate.is_file()
        ]
        if len(manifests) != 1:
            raise ValueError(f"local action has no unique manifest: {reference}")
        return manifests[0]
    if not target.is_file():
        raise ValueError(f"local uses target does not exist: {reference}")
    return target


def collect_external_actions(
    repository_root: Path, entrypoints: tuple[Path, ...]
) -> list[tuple[Path, str]]:
    active: set[Path] = set()
    visited: set[Path] = set()
    external: list[tuple[Path, str]] = []

    def visit(path: Path) -> None:
        path = path.resolve()
        try:
            path.relative_to(repository_root.resolve())
        except ValueError as error:
            raise ValueError(f"uses target escapes repository root: {path}") from error
        if path in active:
            raise ValueError(f"local uses cycle detected at {path}")
        if path in visited:
            return
        active.add(path)
        for reference in read_uses_references(path):
            if reference.startswith("./"):
                visit(resolve_local_uses(repository_root, reference))
            else:
                external.append((path, reference))
        active.remove(path)
        visited.add(path)

    for entrypoint in entrypoints:
        visit(entrypoint)
    return external


class LlamaCppProtectedActionPinTests(unittest.TestCase):
    def test_all_protected_entrypoints_are_scanned(self) -> None:
        self.assertEqual(
            set(PROTECTED_WORKFLOWS),
            {PR_WORKFLOW, CORE_WORKFLOW, MANUAL_WORKFLOW, SCHEDULE_WORKFLOW},
        )

    def test_uses_are_read_from_yaml_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository_root = Path(directory) / "repository"
            workflow = repository_root / ".github" / "workflows" / "entry.yml"
            workflow.parent.mkdir(parents=True)
            workflow.write_text(
                """\
metadata: !untrusted-tag ignored
jobs:
  quoted:
    "uses" : "actions/checkout@main"
  flow: {name: flow, 'uses': 'actions/setup-python@main'}
  spaced:
    uses : actions/upload-artifact@main
  script:
    run: |
      uses: actions/download-artifact@main
""",
                encoding="utf-8",
            )

            self.assertEqual(
                collect_external_actions(repository_root, (workflow,)),
                [
                    (workflow.resolve(), "actions/checkout@main"),
                    (workflow.resolve(), "actions/setup-python@main"),
                    (workflow.resolve(), "actions/upload-artifact@main"),
                ],
            )

    def test_manual_and_schedule_entrypoints_find_mutable_actions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository_root = Path(directory) / "repository"
            workflows = repository_root / ".github" / "workflows"
            action = repository_root / ".github" / "actions" / "nested"
            workflows.mkdir(parents=True)
            action.mkdir(parents=True)
            manual = workflows / "validate_llamacpp_manual.yml"
            schedule = workflows / "validate_llamacpp_schedule.yml"
            core = workflows / "validate_llamacpp_core.yml"
            manual.write_text(
                "jobs: {direct: {uses: actions/checkout@main}}\n",
                encoding="utf-8",
            )
            schedule.write_text(
                "jobs: {delegated: {'uses': './.github/workflows/validate_llamacpp_core.yml'}}\n",
                encoding="utf-8",
            )
            core.write_text(
                "jobs:\n  nested:\n    uses : ./.github/actions/nested\n",
                encoding="utf-8",
            )
            (action / "action.yml").write_text(
                "runs: {using: composite, steps: [{uses: actions/setup-python@v6}]}\n",
                encoding="utf-8",
            )

            external_actions = collect_external_actions(
                repository_root, (manual, schedule)
            )

            self.assertEqual(
                {reference for _, reference in external_actions},
                {"actions/checkout@main", "actions/setup-python@v6"},
            )
            for _, reference in external_actions:
                self.assertNotRegex(reference, PINNED_ACTION_PATTERN)

    def test_every_external_action_uses_an_immutable_commit(self) -> None:
        external_actions = collect_external_actions(ROOT, PROTECTED_WORKFLOWS)
        self.assertTrue(external_actions)
        for source, reference in external_actions:
            with self.subTest(source=source.relative_to(ROOT), reference=reference):
                self.assertRegex(reference, PINNED_ACTION_PATTERN)
                action, commit = reference.rsplit("@", 1)
                if action in REVIEWED_ACTION_PINS:
                    self.assertEqual(commit, REVIEWED_ACTION_PINS[action])

    def test_local_action_recursion_rejects_cycles_and_root_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository_root = Path(directory) / "repository"
            first = repository_root / ".github" / "actions" / "first"
            second = repository_root / ".github" / "actions" / "second"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            first_manifest = first / "action.yml"
            second_manifest = second / "action.yml"
            first_manifest.write_text(
                "runs:\n  using: composite\n  steps:\n    - uses: "
                "./.github/actions/second\n",
                encoding="utf-8",
            )
            second_manifest.write_text(
                "runs:\n  using: composite\n  steps:\n    - uses: "
                "./.github/actions/first\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "cycle"):
                collect_external_actions(repository_root, (first_manifest,))

            outside = Path(directory) / "outside.yml"
            outside.write_text(
                "runs:\n  using: composite\n  steps: []\n", encoding="utf-8"
            )
            first_manifest.write_text(
                "runs:\n  using: composite\n  steps:\n    - uses: ./../outside.yml\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "escapes repository root"):
                collect_external_actions(repository_root, (first_manifest,))

    def test_focused_test_dependencies_are_installed_explicitly(self) -> None:
        workflow = DOCS_AND_STYLE_WORKFLOW.read_text(encoding="utf-8")
        pre_commit_job = workflow.split("  pre-commit:\n", 1)[1].split(
            "  markdown-link-check:\n", 1
        )[0]
        install_step = pre_commit_job.split(
            "      - name: Run focused Python unit tests\n", 1
        )[0]
        self.assertIn(
            "pip install pre-commit psutil requests openai pyyaml", install_step
        )
        self.assertIn(
            'python -c "import test.utils.server_base"',
            pre_commit_job,
        )
        server_base = SERVER_BASE.read_text(encoding="utf-8")
        self.assertIn("from openai import OpenAI, AsyncOpenAI", server_base)
        self.assertIn("import requests", server_base)

    def test_cross_platform_validation_step_explicitly_uses_bash(self) -> None:
        workflow = CORE_WORKFLOW.read_text(encoding="utf-8")
        validate_job = workflow.split("  validate:\n", 1)[1].split(
            "  validation-gate:\n", 1
        )[0]
        step_marker = "      - name: Run validation with lemond\n"
        self.assertEqual(validate_job.count(step_marker), 1)
        step = validate_job.split(step_marker, 1)[1].split("      - name:", 1)[0]

        self.assertIn("    runs-on: ${{ matrix.runner }}\n", validate_job)
        self.assertIn("CANDIDATE_LEMOND:", step)
        self.assertIn("runner.os == 'Windows'", step)
        self.assertIn(
            "$VALIDATION_ROOT/.github/scripts/run_llamacpp_validation.py", step
        )
        self.assertIn('--lemond "$CANDIDATE_LEMOND"', step)
        self.assertIn("        shell: bash\n", step)
        self.assertNotIn("        if:", step)


if __name__ == "__main__":
    unittest.main()
