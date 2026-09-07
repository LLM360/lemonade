#!/usr/bin/env python3
"""Tests for the portable llama.cpp validation process runner."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from test.utils import llamacpp_capability_validation as capabilities
from test.utils.llamacpp_validation_plan import create_validation_plan

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / ".github" / "scripts" / "run_llamacpp_validation.py"
K2_SMALL = "K2-Horizon-0.9B-GGUF"
K2_MEDIUM = "K2-Horizon-3.7B-GGUF"
K2_SMALL_FILENAME = "K2-Horizon-1B-BF16.gguf"


def _passing_capability_matrix(model: str) -> dict:
    cases = []
    for case_id in capabilities.K2_HORIZON_CASE_IDS:
        protocol = "openai"
        if case_id == "ollama_thinking":
            protocol = "ollama"
        elif case_id == "anthropic_translation":
            protocol = "anthropic"
        stream = case_id in {
            "openai_plain_off_stream",
            "openai_reasoning_high_stream",
            "openai_tool_stream_xml",
        }
        tool_case = case_id in {
            "openai_tool_json",
            "openai_tool_xml",
            "openai_tool_xml_typed",
            "openai_tool_default_xml",
            "openai_tool_stream_xml",
        }
        reasoning_case = (
            case_id.startswith("openai_reasoning_")
            and case_id != "openai_reasoning_low"
        ) or case_id == "ollama_thinking"
        content_case = case_id in {
            "openai_plain_off_nonstream",
            "openai_plain_off_stream",
            "openai_tool_result_followup",
            "anthropic_translation",
            "ollama_thinking",
        } or case_id.startswith("openai_reasoning_")
        finish_reason = "stop"
        if tool_case:
            finish_reason = "tool_calls"
        elif case_id == "anthropic_translation":
            finish_reason = "end_turn"
        cases.append(
            {
                "id": case_id,
                "protocol": protocol,
                "stream": stream,
                "pass": True,
                "status_code": 200,
                "finish_reason": finish_reason,
                "content_chars": 2 if content_case else 0,
                "reasoning_chars": 8 if reasoning_case else 0,
                "tool_call_count": 1 if tool_case else 0,
                "terminal_frame": True,
                "tool_name": capabilities.TOOL_NAME if tool_case else None,
                "tool_arguments": (
                    capabilities.EXPECTED_TOOL_ARGUMENTS if tool_case else None
                ),
                "parity_with": (
                    "openai_tool_xml"
                    if case_id in {"openai_tool_default_xml", "openai_tool_stream_xml"}
                    else None
                ),
                "raw_ifm_marker": None,
                "error": None,
            }
        )
    return {
        "profile": capabilities.K2_HORIZON_PROFILE,
        "model": f"builtin.{model.removeprefix('builtin.')}",
        "pass": True,
        "cases": cases,
    }


def log_line(tag: str, message: str) -> str:
    if tag == "Process":
        message = f"[49972] 0.13.484.059 I {message}"
    return f"2026-09-06 12:00:00.000 [Info] ({tag}) {message}"


def backend_log_line(backend: str) -> str:
    return log_line("LlamaCpp", f"Using LlamaCpp Backend: {backend}")


def verified_artifacts(runner, root: Path | None = None):
    artifact_path = Path("/cache") / K2_SMALL_FILENAME
    if root is not None:
        artifact_path = root / "verified-artifacts" / K2_SMALL_FILENAME
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.touch()
        artifact_path = artifact_path.resolve()
    return {
        K2_SMALL: runner.VerifiedValidationArtifact(
            model_id=K2_SMALL,
            resolved_path=artifact_path,
            size_bytes=0,
            sha256="0" * 64,
        )
    }


def bind_verified_artifact_path(contents: str, expected_artifacts: dict) -> str:
    expected_path = expected_artifacts[K2_SMALL].resolved_path
    return contents.replace(
        f"/cache/{K2_SMALL_FILENAME}",
        str(expected_path),
    )


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_llamacpp_validation", RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load llama.cpp validation runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    next_pid = 4100

    def __init__(self):
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1

    @staticmethod
    def poll():
        return None


def _pid_is_active(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _pid_is_running(pid: int) -> bool:
    try:
        import psutil
    except ModuleNotFoundError:
        return _pid_is_active(pid)
    try:
        return psutil.Process(pid).status() not in {
            psutil.STATUS_DEAD,
            psutil.STATUS_ZOMBIE,
        }
    except psutil.Error:
        return False


def _process_tree_inspection_available() -> bool:
    if sys.platform != "darwin":
        return True
    try:
        import psutil

        psutil.Process(os.getpid()).children(recursive=True)
    except ModuleNotFoundError:
        return False
    except (OSError, psutil.Error):
        return False
    return True


DARWIN_PROCESS_TREE_INSPECTION_AVAILABLE = _process_tree_inspection_available()


class LlamaCppValidationRunnerTests(unittest.TestCase):
    def test_direct_help_invocation_imports_repository_modules(self):
        result = subprocess.run(
            [sys.executable, str(RUNNER_PATH), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_restart_command_is_bounded_cached_smoke_with_distinct_evidence(self):
        runner = load_runner()
        output_path = Path("isolated") / "restart.json"

        command = runner.build_restart_validation_command(
            python_executable=Path("python"),
            backend="vulkan",
            channel="",
            target="windows-vulkan",
            model=K2_SMALL,
            port=13305,
            output_path=output_path,
        )

        self.assertEqual(command[0], "python")
        self.assertEqual(
            Path(command[1]), (ROOT / "test/validate_llamacpp.py").resolve()
        )
        self.assertTrue(runner.VALIDATION_MODEL_CATALOG.is_absolute())
        self.assertTrue(runner.VALIDATION_ARTIFACT_LOCKS.is_absolute())
        runner.VALIDATION_MODEL_CATALOG.relative_to(ROOT.resolve())
        runner.VALIDATION_ARTIFACT_LOCKS.relative_to(ROOT.resolve())
        self.assertIn("--skip-install", command)
        self.assertEqual(command.count("--model"), 1)
        self.assertIn(K2_SMALL, command)
        self.assertNotIn("--capability-profile", command)
        self.assertNotIn("--capability-model", command)
        self.assertNotIn("--logs-dir", command)
        self.assertIn(str(output_path), command)
        self.assertNotIn("server-restart-logs-windows-vulkan", command)

    def test_protected_validation_scrubs_ambient_download_credentials(self):
        runner = load_runner()

        class FakeContainment:
            @staticmethod
            def popen_kwargs():
                return {} if os.name == "nt" else {"start_new_session": True}

            @staticmethod
            def assign(_process):
                return None

            @staticmethod
            def close():
                return True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lemond = root / "lemond"
            python = root / "python"
            runner_temp = root / "runner-temp"
            lemond.touch()
            python.touch()
            runner_temp.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "defaults.json").write_text("{}", encoding="utf-8")
            github_output = root / "github-output"
            args = argparse.Namespace(
                lemond=lemond,
                python=python,
                backend="metal",
                channel="",
                target="macos-metal",
                models=K2_SMALL,
                lite=False,
                capability_profile="k2-horizon-v1",
                capability_models=K2_SMALL,
                allow_huggingface_download_credentials=False,
                port=13305,
            )

            old_directory = Path.cwd()
            os.chdir(root)
            try:
                with (
                    mock.patch.object(runner, "_require_idle_runner"),
                    mock.patch.object(runner, "_require_unused_port"),
                    mock.patch.object(runner.time, "sleep"),
                    mock.patch.object(
                        runner,
                        "_create_process_containment",
                        side_effect=lambda **_kwargs: FakeContainment(),
                    ),
                    mock.patch.object(
                        runner.subprocess,
                        "Popen",
                        side_effect=(FakeProcess(), FakeProcess(), FakeProcess()),
                    ) as popen,
                    mock.patch.object(
                        runner,
                        "_run_contained_command",
                        side_effect=lambda command, **_kwargs: (
                            self._complete_command(command).returncode
                        ),
                    ) as run,
                    mock.patch.object(
                        runner,
                        "_shutdown_server",
                        side_effect=(True, True, True),
                    ) as shutdown,
                    mock.patch.object(
                        runner,
                        "verify_validation_artifacts",
                        return_value=verified_artifacts(runner),
                    ) as verify_artifacts,
                    mock.patch.object(
                        runner,
                        "require_accelerator_attestation",
                    ) as attest_accelerator,
                    mock.patch.object(
                        runner,
                        "revalidate_validation_cache",
                        wraps=runner.revalidate_validation_cache,
                    ) as revalidate_cache,
                    mock.patch.dict(
                        os.environ,
                        {
                            "RUNNER_TEMP": str(runner_temp),
                            "HF_ENDPOINT": "https://example.invalid",
                            "hF_ToKeN": "download-token",
                            "hUgGiNg_FaCe_HuB_tOkEn": "legacy-download-token",
                            "HuGgInGfAcE_ToKeN": "alternate-download-token",
                            "LEMONADE_API_KEY": "api-key",
                            "LEMONADE_DEFAULTS_PATH": "/host/defaults.json",
                            "LEMONADE_LLAMACPP_METAL_BIN": "/host/llama-server",
                            "lemonade_rocm_install_method": "wheel",
                            "LEMONADE_BACKEND_WATCHDOG_POLL_SECONDS": "1",
                            "LEMONADE_CACHE_DIR": "/host/lemonade-cache",
                            "LEMONADE_GGML_HIP_PATH": "/host/libggml-hip.so",
                            "HF_HOME": "/host/huggingface",
                            "LEMONADE_ALLOWED_ORIGINS": "https://example.invalid",
                            "ROCM_PATH": "/host/rocm",
                            "LLAMA_ARG_MODEL": "/host/model.gguf",
                            "LLAMA_ARG_CTX_SIZE": "1",
                            "HTTP_PROXY": "http://proxy.invalid:3128",
                            "all_proxy": "socks5://proxy.invalid:1080",
                            "NO_PROXY": "example.invalid",
                            "GITHUB_OUTPUT": str(github_output),
                            "GITHUB_ENV": "/runner/command/environment",
                            "ACTIONS_RUNTIME_TOKEN": "actions-token",
                            "LD_PRELOAD": "/attacker/inject.so",
                            "PYTHONPATH": "/attacker/python",
                            "UNKNOWN_DEPLOYMENT_SECRET": "secret",
                        },
                    ),
                ):
                    result = runner.run_validation(args)
            finally:
                os.chdir(old_directory)
            github_output_contents = github_output.read_text(encoding="utf-8")

        self.assertEqual(result, 0)
        self.assertEqual(popen.call_count, 3)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(shutdown.call_count, 3)
        self.assertEqual(verify_artifacts.call_count, 3)
        self.assertEqual(attest_accelerator.call_count, 2)
        self.assertEqual(revalidate_cache.call_count, 10)
        self.assertEqual(
            attest_accelerator.call_args_list[0].args[3],
            verified_artifacts(runner),
        )
        self.assertEqual(
            attest_accelerator.call_args_list[1].args[3],
            verified_artifacts(runner),
        )
        self.assertEqual(
            attest_accelerator.call_args_list[0].kwargs["selected_models"],
            [K2_SMALL],
        )
        self.assertEqual(
            attest_accelerator.call_args_list[1].kwargs["selected_models"],
            [K2_SMALL],
        )

        first_server = popen.call_args_list[0]
        second_server = popen.call_args_list[1]
        third_server = popen.call_args_list[2]
        self.assertEqual(first_server.args[0][1], second_server.args[0][1])
        self.assertEqual(first_server.args[0][1], third_server.args[0][1])
        self.assertFalse(first_server.args[0][2].startswith("--"))
        self.assertEqual(first_server.args[0][2], second_server.args[0][2])
        self.assertEqual(first_server.args[0][2], third_server.args[0][2])
        self.assertEqual(first_server.args[0][3:], second_server.args[0][3:])
        self.assertEqual(first_server.args[0][3:], third_server.args[0][3:])
        expected_hf_cache = first_server.kwargs["env"]["HF_HUB_CACHE"]
        hf_cache_path = Path(expected_hf_cache)
        hf_cache_path.relative_to(runner_temp.resolve())
        self.assertEqual(hf_cache_path.name, "hf-hub")
        self.assertEqual(hf_cache_path.parent.name, "lemonade-cache")
        self.assertTrue(
            hf_cache_path.parents[1].name.startswith("llamacpp-validation-macos-metal-")
        )
        private_temp = first_server.kwargs["env"]["TMPDIR"]
        self.assertEqual(first_server.kwargs["env"]["TMP"], private_temp)
        self.assertEqual(first_server.kwargs["env"]["TEMP"], private_temp)
        Path(private_temp).relative_to(runner_temp.resolve())
        self.assertEqual(Path(private_temp).name, "private-temp")
        download_credential_keys = {
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "HUGGINGFACE_TOKEN",
        }
        for call in popen.call_args_list:
            self.assertFalse(
                download_credential_keys & {key.upper() for key in call.kwargs["env"]}
            )
        for call in popen.call_args_list:
            if os.name != "nt":
                self.assertIs(call.kwargs.get("start_new_session"), True)
            self.assertEqual(call.kwargs["env"]["HF_HUB_CACHE"], expected_hf_cache)
            self.assertEqual(call.kwargs["env"]["TMPDIR"], private_temp)
            self.assertEqual(call.kwargs["env"]["TMP"], private_temp)
            self.assertEqual(call.kwargs["env"]["TEMP"], private_temp)
            self.assertEqual(call.kwargs["env"]["HOME"], private_temp)
            self.assertEqual(call.kwargs["env"]["USERPROFILE"], private_temp)
            self.assertEqual(call.kwargs["env"]["APPDATA"], private_temp)
            self.assertEqual(call.kwargs["env"]["LOCALAPPDATA"], private_temp)
            self.assertNotIn("HF_ENDPOINT", call.kwargs["env"])
            self.assertNotIn("LEMONADE_API_KEY", call.kwargs["env"])
            self.assertNotIn("LEMONADE_LLAMACPP_METAL_BIN", call.kwargs["env"])
            self.assertNotIn("lemonade_rocm_install_method", call.kwargs["env"])
            self.assertNotIn(
                "LEMONADE_BACKEND_WATCHDOG_POLL_SECONDS", call.kwargs["env"]
            )
            self.assertNotIn("LEMONADE_CACHE_DIR", call.kwargs["env"])
            self.assertNotIn("LEMONADE_GGML_HIP_PATH", call.kwargs["env"])
            self.assertNotIn("HF_HOME", call.kwargs["env"])
            self.assertNotIn("LEMONADE_ALLOWED_ORIGINS", call.kwargs["env"])
            self.assertNotIn("ROCM_PATH", call.kwargs["env"])
            self.assertEqual(
                {
                    key.upper(): value
                    for key, value in call.kwargs["env"].items()
                    if key.upper().startswith("LLAMA_ARG_")
                },
                {"LLAMA_ARG_LOG_VERBOSITY": "4"},
            )
            self.assertNotIn("HTTP_PROXY", call.kwargs["env"])
            self.assertNotIn("all_proxy", call.kwargs["env"])
            self.assertNotIn("GITHUB_ENV", call.kwargs["env"])
            self.assertNotIn("GITHUB_OUTPUT", call.kwargs["env"])
            self.assertNotIn("ACTIONS_RUNTIME_TOKEN", call.kwargs["env"])
            self.assertNotIn("LD_PRELOAD", call.kwargs["env"])
            self.assertNotIn("PYTHONPATH", call.kwargs["env"])
            self.assertNotIn("UNKNOWN_DEPLOYMENT_SECRET", call.kwargs["env"])
            self.assertEqual(call.kwargs["env"]["NO_PROXY"], "127.0.0.1")
            self.assertEqual(call.kwargs["env"]["no_proxy"], "127.0.0.1")
            self.assertEqual(call.kwargs["env"]["LEMONADE_CI_MODE"], "True")
            self.assertEqual(
                call.kwargs["env"]["LEMONADE_DEFAULTS_PATH"],
                str((resources / "defaults.json").resolve()),
            )

        prepare_command = run.call_args_list[0].args[0]
        primary_command = run.call_args_list[1].args[0]
        restart_command = run.call_args_list[2].args[0]
        self.assertIn("--prepare-only", prepare_command)
        self.assertNotIn("--skip-install", prepare_command)
        self.assertNotIn("--prepare-only", primary_command)
        self.assertIn("--skip-install", primary_command)
        self.assertIn("--skip-install", restart_command)
        for call in run.call_args_list:
            self.assertFalse(
                download_credential_keys & {key.upper() for key in call.kwargs["env"]}
            )
        self.assertEqual(
            Path(primary_command[primary_command.index("--output") + 1]).name,
            "llamacpp_validation_macos-metal.json",
        )
        self.assertEqual(
            Path(restart_command[restart_command.index("--output") + 1]).name,
            "llamacpp_restart_validation_macos-metal.json",
        )
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["env"]["HF_HUB_CACHE"], expected_hf_cache)
            self.assertEqual(call.kwargs["env"]["TMPDIR"], private_temp)
            self.assertEqual(call.kwargs["env"]["TMP"], private_temp)
            self.assertEqual(call.kwargs["env"]["TEMP"], private_temp)
            self.assertEqual(call.kwargs["env"]["HOME"], private_temp)
            self.assertEqual(call.kwargs["env"]["USERPROFILE"], private_temp)
            self.assertEqual(call.kwargs["env"]["APPDATA"], private_temp)
            self.assertEqual(call.kwargs["env"]["LOCALAPPDATA"], private_temp)
            self.assertNotIn("HF_ENDPOINT", call.kwargs["env"])
            self.assertNotIn("LEMONADE_API_KEY", call.kwargs["env"])
            self.assertNotIn("LEMONADE_LLAMACPP_METAL_BIN", call.kwargs["env"])
            self.assertEqual(
                {
                    key.upper(): value
                    for key, value in call.kwargs["env"].items()
                    if key.upper().startswith("LLAMA_ARG_")
                },
                {"LLAMA_ARG_LOG_VERBOSITY": "4"},
            )
            self.assertNotIn("GITHUB_ENV", call.kwargs["env"])
            self.assertNotIn("GITHUB_OUTPUT", call.kwargs["env"])
            self.assertNotIn("ACTIONS_RUNTIME_TOKEN", call.kwargs["env"])
            self.assertNotIn("LD_PRELOAD", call.kwargs["env"])
            self.assertNotIn("PYTHONPATH", call.kwargs["env"])
            self.assertNotIn("UNKNOWN_DEPLOYMENT_SECRET", call.kwargs["env"])

        server_log_names = {
            Path(call.kwargs[stream].name).name
            for call in popen.call_args_list
            for stream in ("stdout", "stderr")
        }
        self.assertEqual(
            server_log_names,
            {
                "lemond.prepare.stdout.log",
                "lemond.prepare.stderr.log",
                "lemond.stdout.log",
                "lemond.stderr.log",
                "lemond.restart.stdout.log",
                "lemond.restart.stderr.log",
            },
        )
        for call in popen.call_args_list:
            Path(call.kwargs["stdout"].name).relative_to(runner_temp.resolve())
            Path(call.kwargs["stderr"].name).relative_to(runner_temp.resolve())

        workflow_outputs = dict(
            line.split("=", 1) for line in github_output_contents.splitlines()
        )
        self.assertEqual(
            Path(workflow_outputs["validation_result"]).name,
            "llamacpp_validation_macos-metal.json",
        )
        self.assertEqual(
            Path(workflow_outputs["restart_result"]).name,
            "llamacpp_restart_validation_macos-metal.json",
        )
        self.assertEqual(len(workflow_outputs), 8)
        for output_path in workflow_outputs.values():
            Path(output_path).relative_to(runner_temp.resolve())

    @staticmethod
    def _complete_command(command):
        if "--prepare-only" not in command:
            output_path = Path(command[command.index("--output") + 1])
            models = [
                command[index + 1]
                for index, argument in enumerate(command)
                if argument == "--model"
            ]
            capability_models = {
                command[index + 1].removeprefix("builtin.")
                for index, argument in enumerate(command)
                if argument == "--capability-model"
            }
            records = []
            for model in models:
                record = {
                    "model": model,
                    "pass": True,
                    "response": "Four.",
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "time_to_first_token": 0.5,
                    "tokens_per_second": 4.0,
                }
                if model.removeprefix("builtin.") in capability_models:
                    record["capability_matrix"] = _passing_capability_matrix(model)
                records.append(record)
            output_path.write_text(json.dumps(records), encoding="utf-8")
        return argparse.Namespace(returncode=0)

    def _complete_process(self, **kwargs):
        kwargs["stdout_path"].touch()
        kwargs["stderr_path"].touch()
        return self._complete_command(kwargs["command"]).returncode, True

    def test_validation_attests_locked_model_with_selected_unlocked_schedule_model(
        self,
    ):
        runner = load_runner()
        schedule_models = create_validation_plan("schedule")["include"][0]["models"]
        artifact_locks = json.loads(
            (ROOT / "test/fixtures/llamacpp_validation_artifacts.json").read_text(
                encoding="utf-8"
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lemond = root / "lemond"
            python = root / "python"
            runner_temp = root / "runner-temp"
            lemond.touch()
            python.touch()
            runner_temp.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "defaults.json").write_text("{}", encoding="utf-8")
            expected_artifacts = {}
            artifact_root = root / "verified-artifacts"
            artifact_root.mkdir()
            for model, lock in artifact_locks.items():
                artifact_path = artifact_root / lock["filename"]
                artifact_path.touch()
                expected_artifacts[model] = runner.VerifiedValidationArtifact(
                    model_id=model,
                    resolved_path=artifact_path.resolve(),
                    size_bytes=0,
                    sha256="0" * 64,
                )
            args = argparse.Namespace(
                lemond=lemond,
                python=python,
                backend="vulkan",
                channel="",
                target="windows-vulkan",
                models=",".join(schedule_models),
                lite=False,
                capability_profile="k2-horizon-v1",
                capability_models=K2_SMALL,
                port=13305,
            )

            def completed_model_load(model: str, gguf_path: Path) -> str:
                return "\n".join(
                    (
                        log_line("LlamaCpp", f"Loading model: builtin.{model}"),
                        backend_log_line("vulkan"),
                        log_line("LlamaCpp", f"Using GGUF: {gguf_path}"),
                        log_line(
                            "Process",
                            "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                        ),
                        log_line(
                            "Process",
                            "load_tensors: offloaded 12/24 layers to GPU",
                        ),
                        log_line("Process", "srv  llama_server: model loaded"),
                    )
                )

            def complete_with_attestation_logs(**kwargs):
                result = self._complete_process(**kwargs)
                command = kwargs["command"]
                if "--prepare-only" in command:
                    return result
                models = [
                    command[index + 1].removeprefix("builtin.")
                    for index, argument in enumerate(command)
                    if argument == "--model"
                ]
                sessions = [
                    completed_model_load(
                        model,
                        (
                            expected_artifacts[model].resolved_path
                            if model in expected_artifacts
                            else Path("/cache") / f"{model}.gguf"
                        ),
                    )
                    for model in models
                ]
                kwargs["stderr_path"].write_text(
                    "\n".join(sessions) + "\n", encoding="utf-8"
                )
                return result

            with (
                mock.patch.object(runner, "_require_idle_runner"),
                mock.patch.object(runner, "_require_unused_port"),
                mock.patch.object(
                    runner,
                    "_run_validation_process",
                    side_effect=complete_with_attestation_logs,
                ),
                mock.patch.object(
                    runner,
                    "verify_validation_artifacts",
                    return_value=expected_artifacts,
                ),
                mock.patch.dict(os.environ, {"RUNNER_TEMP": str(runner_temp)}),
            ):
                try:
                    result = runner.run_validation(args)
                except RuntimeError as exc:
                    self.fail(f"selected unlocked schedule model was rejected: {exc}")

        self.assertEqual(result, 0)

    def test_validation_rejects_empty_primary_evidence_before_restart(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lemond = root / "lemond"
            python = root / "python"
            runner_temp = root / "runner-temp"
            lemond.touch()
            python.touch()
            runner_temp.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "defaults.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                lemond=lemond,
                python=python,
                backend="vulkan",
                channel="",
                target="linux-vulkan",
                models=K2_SMALL,
                lite=False,
                capability_profile="k2-horizon-v1",
                capability_models=K2_SMALL,
                port=13305,
            )
            phases = 0

            def write_empty_evidence(**kwargs):
                nonlocal phases
                phases += 1
                kwargs["stdout_path"].touch()
                kwargs["stderr_path"].touch()
                command = kwargs["command"]
                if "--prepare-only" not in command:
                    output_path = Path(command[command.index("--output") + 1])
                    output_path.write_text("[]", encoding="utf-8")
                return 0, True

            with (
                mock.patch.object(runner, "_require_idle_runner"),
                mock.patch.object(runner, "_require_unused_port"),
                mock.patch.object(
                    runner,
                    "_run_validation_process",
                    side_effect=write_empty_evidence,
                ),
                mock.patch.object(
                    runner,
                    "verify_validation_artifacts",
                    return_value={},
                ),
                mock.patch.dict(os.environ, {"RUNNER_TEMP": str(runner_temp)}),
                self.assertRaisesRegex(RuntimeError, "nonempty array"),
            ):
                runner.run_validation(args)

        self.assertEqual(phases, 2)

    def test_validation_rejects_restart_evidence_for_another_model(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lemond = root / "lemond"
            python = root / "python"
            runner_temp = root / "runner-temp"
            lemond.touch()
            python.touch()
            runner_temp.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "defaults.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                lemond=lemond,
                python=python,
                backend="vulkan",
                channel="",
                target="linux-vulkan",
                models=K2_SMALL,
                lite=False,
                capability_profile="k2-horizon-v1",
                capability_models=K2_SMALL,
                port=13305,
            )

            def write_wrong_restart_evidence(**kwargs):
                result = self._complete_process(**kwargs)
                command = kwargs["command"]
                output_path = Path(command[command.index("--output") + 1])
                if output_path.name.startswith("llamacpp_restart_validation_"):
                    output_path.write_text(
                        json.dumps(
                            [
                                {
                                    "model": "Other-Llama",
                                    "pass": True,
                                    "response": "Four.",
                                    "input_tokens": 10,
                                    "output_tokens": 2,
                                    "time_to_first_token": 0.5,
                                    "tokens_per_second": 4.0,
                                }
                            ]
                        ),
                        encoding="utf-8",
                    )
                return result

            with (
                mock.patch.object(runner, "_require_idle_runner"),
                mock.patch.object(runner, "_require_unused_port"),
                mock.patch.object(
                    runner,
                    "_run_validation_process",
                    side_effect=write_wrong_restart_evidence,
                ) as run_process,
                mock.patch.object(
                    runner,
                    "verify_validation_artifacts",
                    return_value={},
                ),
                mock.patch.dict(os.environ, {"RUNNER_TEMP": str(runner_temp)}),
                self.assertRaisesRegex(RuntimeError, "do not match planned models"),
            ):
                runner.run_validation(args)

        self.assertEqual(run_process.call_count, 3)

    def test_accelerator_attestation_requires_ordered_device_and_offload_markers(self):
        runner = load_runner()
        positive = "\n".join(
            (
                log_line("LlamaCpp", f"Loading model: builtin.{K2_SMALL}"),
                backend_log_line("vulkan"),
                log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}"),
                log_line(
                    "Process",
                    "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                ),
                log_line("Process", "load_tensors: offloaded 12/24 layers to GPU"),
                log_line("Process", "srv  llama_server: model loaded"),
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_artifacts = verified_artifacts(runner, root)
            log = root / "lemond.stderr.log"
            empty_companion = root / "lemond.stdout.log"
            empty_companion.touch()
            log.write_text(
                bind_verified_artifact_path(positive, expected_artifacts),
                encoding="utf-8",
            )
            identities = runner.require_accelerator_attestation(
                [empty_companion, log],
                root,
                "validation",
                expected_artifacts,
                "vulkan",
            )
            self.assertEqual(
                [identity.path for identity in identities], [empty_companion, log]
            )

            failures = {
                "missing backend": positive.replace(
                    backend_log_line("vulkan") + "\n", ""
                ),
                "wrong backend": positive.replace(
                    backend_log_line("vulkan"), backend_log_line("cuda")
                ),
                "duplicate backend": positive.replace(
                    backend_log_line("vulkan"),
                    backend_log_line("vulkan") + "\n" + backend_log_line("vulkan"),
                ),
                "malformed backend": positive.replace(
                    backend_log_line("vulkan"),
                    log_line("LlamaCpp", "Using LlamaCpp Backend: vulkan extra"),
                ),
                "backend after artifact identity": positive.replace(
                    backend_log_line("vulkan")
                    + "\n"
                    + log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}"),
                    log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}")
                    + "\n"
                    + backend_log_line("vulkan"),
                ),
                "backend after completed load": positive
                + "\n"
                + backend_log_line("vulkan"),
                "zero": positive.replace("12/24", "0/24"),
                "greater than total": positive.replace("12/24", "25/24"),
                "malformed offload": positive.replace("12/24", "many/all"),
                "missing offload": positive.replace(
                    log_line("Process", "load_tensors: offloaded 12/24 layers to GPU")
                    + "\n",
                    "",
                ),
                "missing child success": positive.replace(
                    log_line("Process", "srv  llama_server: model loaded"), ""
                ),
                "offload before device": positive.replace(
                    log_line(
                        "Process",
                        "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                    )
                    + "\n"
                    + log_line(
                        "Process", "load_tensors: offloaded 12/24 layers to GPU"
                    ),
                    log_line("Process", "load_tensors: offloaded 12/24 layers to GPU")
                    + "\n"
                    + log_line(
                        "Process",
                        "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                    ),
                ),
                "missing device": positive.replace(
                    log_line(
                        "Process",
                        "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                    )
                    + "\n",
                    "",
                ),
                "missing artifact identity": positive.replace(
                    log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}")
                    + "\n",
                    "",
                ),
                "artifact suffix is not exact": positive.replace(
                    K2_SMALL_FILENAME, f"{K2_SMALL_FILENAME}.evil"
                ),
                "device before artifact identity": positive.replace(
                    log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}")
                    + "\n"
                    + log_line(
                        "Process",
                        "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                    ),
                    log_line(
                        "Process",
                        "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                    )
                    + "\n"
                    + log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}"),
                ),
                "other model cannot mask target": positive.replace(
                    K2_SMALL, "Other-Llama"
                ),
                "later draft masks target": positive.replace(
                    log_line(
                        "Process",
                        "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                    )
                    + "\n"
                    + log_line(
                        "Process", "load_tensors: offloaded 12/24 layers to GPU"
                    ),
                    log_line(
                        "Process",
                        "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                    )
                    + "\n"
                    + log_line("Process", "load_tensors: offloaded 0/24 layers to GPU")
                    + "\n"
                    + log_line(
                        "Process",
                        "llama_prepare_model_devices: using device Vulkan1 (GPU)",
                    )
                    + "\n"
                    + log_line(
                        "Process", "load_tensors: offloaded 24/24 layers to GPU"
                    ),
                ),
                "deceptive child success text": positive.replace(
                    log_line("Process", "srv  llama_server: model loaded"),
                    log_line("Process", "did not see srv  llama_server: model loaded"),
                ),
                "second zero offload on same line": positive.replace(
                    log_line("Process", "load_tensors: offloaded 12/24 layers to GPU"),
                    log_line(
                        "Process",
                        "load_tensors: offloaded 12/24 layers to GPU "
                        "load_tensors: offloaded 0/24 layers to GPU",
                    ),
                ),
                "malformed then valid offload on same line": positive.replace(
                    log_line("Process", "load_tensors: offloaded 12/24 layers to GPU"),
                    log_line(
                        "Process",
                        "load_tensors: offloaded many/all layers to GPU "
                        "load_tensors: offloaded 12/24 layers to GPU",
                    ),
                ),
            }
            for name, contents in failures.items():
                with self.subTest(name=name):
                    log.write_text(
                        bind_verified_artifact_path(contents, expected_artifacts),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(RuntimeError, "accelerator"):
                        runner.require_accelerator_attestation(
                            [log],
                            root,
                            "validation",
                            expected_artifacts,
                            "vulkan",
                        )

            unexpected_model = (
                positive
                + "\n"
                + log_line("LlamaCpp", "Loading model: builtin.Unexpected-Llama")
            )
            log.write_text(
                bind_verified_artifact_path(unexpected_model, expected_artifacts),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "unexpected model"):
                runner.require_accelerator_attestation(
                    [log],
                    root,
                    "validation",
                    expected_artifacts,
                    "vulkan",
                )

    def test_selected_unlocked_load_cannot_replace_locked_model_attestation(self):
        runner = load_runner()
        schedule_models = create_validation_plan("schedule")["include"][0]["models"]
        unlocked_model = next(
            model
            for model in schedule_models
            if model not in {K2_SMALL, "K2-Horizon-3.7B-GGUF", "K2-Horizon-7B-GGUF"}
        )
        unlocked_load = "\n".join(
            (
                log_line("LlamaCpp", f"Loading model: builtin.{unlocked_model}"),
                log_line("Process", "srv  llama_server: model loaded"),
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "lemond.stderr.log"
            log.write_text(unlocked_load, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "missing models"):
                runner.require_accelerator_attestation(
                    [log],
                    root,
                    "validation",
                    verified_artifacts(runner, root),
                    "vulkan",
                    selected_models=[unlocked_model, K2_SMALL],
                )

    def test_accelerator_attestation_rejects_same_basename_from_another_path(self):
        runner = load_runner()
        positive = "\n".join(
            (
                log_line("LlamaCpp", f"Loading model: builtin.{K2_SMALL}"),
                backend_log_line("vulkan"),
                log_line("LlamaCpp", f"Using GGUF: /unverified/{K2_SMALL_FILENAME}"),
                log_line(
                    "Process",
                    "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                ),
                log_line("Process", "load_tensors: offloaded 12/24 layers to GPU"),
                log_line("Process", "srv  llama_server: model loaded"),
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_artifacts = verified_artifacts(runner, root)
            unverified_artifact = root / "unverified" / K2_SMALL_FILENAME
            unverified_artifact.parent.mkdir()
            unverified_artifact.touch()
            log = root / "lemond.stderr.log"
            log.write_text(
                positive.replace(
                    f"/unverified/{K2_SMALL_FILENAME}",
                    str(unverified_artifact.resolve()),
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "wrong artifact"):
                runner.require_accelerator_attestation(
                    [log],
                    root,
                    "validation",
                    expected_artifacts,
                    "vulkan",
                )

    def test_accelerator_attestation_binds_device_to_requested_backend(self):
        runner = load_runner()
        backend_devices = {
            "cuda": "CUDA0",
            "metal": "MTL0",
            "rocm-stable": "ROCm0",
            "vulkan": "Vulkan0",
        }

        def attestation(backend, device):
            return "\n".join(
                (
                    log_line("LlamaCpp", f"Loading model: builtin.{K2_SMALL}"),
                    backend_log_line(backend),
                    log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}"),
                    log_line(
                        "Process",
                        f"llama_prepare_model_devices: using device {device} (GPU)",
                    ),
                    log_line("Process", "load_tensors: offloaded 12/24 layers to GPU"),
                    log_line("Process", "srv  llama_server: model loaded"),
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_artifacts = verified_artifacts(runner, root)
            log = root / "lemond.stderr.log"
            for backend, device in backend_devices.items():
                with self.subTest(backend=backend, device=device):
                    log.write_text(
                        bind_verified_artifact_path(
                            attestation(backend, device), expected_artifacts
                        ),
                        encoding="utf-8",
                    )
                    runner.require_accelerator_attestation(
                        [log],
                        root,
                        "validation",
                        expected_artifacts,
                        expected_backend=backend,
                    )

                wrong_device = next(
                    candidate
                    for other_backend, candidate in backend_devices.items()
                    if other_backend != backend
                )
                with self.subTest(backend=backend, wrong_device=wrong_device):
                    log.write_text(
                        bind_verified_artifact_path(
                            attestation(backend, wrong_device), expected_artifacts
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(RuntimeError, "backend"):
                        runner.require_accelerator_attestation(
                            [log],
                            root,
                            "validation",
                            expected_artifacts,
                            expected_backend=backend,
                        )

    def test_rocm_attestation_binds_exact_release_channel(self):
        runner = load_runner()

        def attestation(channel):
            return "\n".join(
                (
                    log_line("LlamaCpp", f"Loading model: builtin.{K2_SMALL}"),
                    log_line("LlamaCpp", f"Using LlamaCpp Backend: rocm-{channel}"),
                    log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}"),
                    log_line(
                        "Process",
                        "llama_prepare_model_devices: using device ROCm0 (GPU)",
                    ),
                    log_line("Process", "load_tensors: offloaded 12/24 layers to GPU"),
                    log_line("Process", "srv  llama_server: model loaded"),
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_artifacts = verified_artifacts(runner, root)
            log = root / "lemond.stderr.log"
            for channel in ("stable", "nightly"):
                expected_backend = f"rocm-{channel}"
                log.write_text(
                    bind_verified_artifact_path(
                        attestation(channel), expected_artifacts
                    ),
                    encoding="utf-8",
                )
                runner.require_accelerator_attestation(
                    [log],
                    root,
                    "validation",
                    expected_artifacts,
                    expected_backend=expected_backend,
                )

                swapped_channel = "nightly" if channel == "stable" else "stable"
                with self.assertRaisesRegex(RuntimeError, "backend"):
                    runner.require_accelerator_attestation(
                        [log],
                        root,
                        "validation",
                        expected_artifacts,
                        expected_backend=f"rocm-{swapped_channel}",
                    )

    def test_attestation_backend_resolution_rejects_channel_ambiguity(self):
        runner = load_runner()
        self.assertEqual(
            runner.resolve_attestation_backend("rocm", "stable"), "rocm-stable"
        )
        self.assertEqual(
            runner.resolve_attestation_backend("rocm", "nightly"), "rocm-nightly"
        )
        self.assertEqual(runner.resolve_attestation_backend("metal", ""), "metal")

        for backend, channel in (
            ("rocm", ""),
            ("rocm", "candidate"),
            ("metal", "stable"),
        ):
            with self.subTest(backend=backend, channel=channel):
                with self.assertRaisesRegex(RuntimeError, "channel"):
                    runner.resolve_attestation_backend(backend, channel)

    def test_cpu_attestation_requires_cpu_only_build_markers(self):
        runner = load_runner()
        positive = "\n".join(
            (
                log_line("LlamaCpp", f"Loading model: builtin.{K2_SMALL}"),
                backend_log_line("cpu"),
                log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}"),
                log_line(
                    "Process",
                    "load_tensors: CPU_Mapped model buffer size = 2056.83 MiB",
                ),
                log_line("Process", "srv  llama_server: model loaded"),
            )
        )
        accelerator_markers = (
            log_line(
                "Process",
                "llama_prepare_model_devices: using device Vulkan0 (GPU)",
            ),
            log_line("Process", "load_tensors: offloaded 0/24 layers to GPU"),
            log_line("Process", "load_tensors: offloaded 12/24 layers to GPU"),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_artifacts = verified_artifacts(runner, root)
            log = root / "lemond.stderr.log"
            log.write_text(
                bind_verified_artifact_path(positive, expected_artifacts),
                encoding="utf-8",
            )
            runner.require_accelerator_attestation(
                [log],
                root,
                "validation",
                expected_artifacts,
                expected_backend="cpu",
            )

            missing_cpu_buffer = positive.replace(
                log_line(
                    "Process",
                    "load_tensors: CPU_Mapped model buffer size = 2056.83 MiB",
                )
                + "\n",
                "",
            )
            log.write_text(
                bind_verified_artifact_path(missing_cpu_buffer, expected_artifacts),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "CPU|cpu"):
                runner.require_accelerator_attestation(
                    [log],
                    root,
                    "validation",
                    expected_artifacts,
                    expected_backend="cpu",
                )

            for marker in accelerator_markers:
                with self.subTest(marker=marker):
                    invalid_cpu_log = positive.replace(
                        log_line("Process", "srv  llama_server: model loaded"),
                        marker
                        + "\n"
                        + log_line("Process", "srv  llama_server: model loaded"),
                    )
                    log.write_text(
                        bind_verified_artifact_path(
                            invalid_cpu_log, expected_artifacts
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(RuntimeError, "CPU|cpu"):
                        runner.require_accelerator_attestation(
                            [log],
                            root,
                            "validation",
                            expected_artifacts,
                            expected_backend="cpu",
                        )

    def test_accelerator_attestation_does_not_carry_state_across_log_streams(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_artifacts = verified_artifacts(runner, root)
            stdout_log = root / "stdout.log"
            stderr_log = root / "stderr.log"
            partial_load = "\n".join(
                (
                    log_line("LlamaCpp", f"Loading model: {K2_SMALL}"),
                    backend_log_line("vulkan"),
                    log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}"),
                    log_line(
                        "Process",
                        "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                    ),
                )
            )
            stdout_log.write_text(
                bind_verified_artifact_path(partial_load, expected_artifacts),
                encoding="utf-8",
            )
            stderr_log.write_text(
                "\n".join(
                    (
                        log_line(
                            "Process", "load_tensors: offloaded 12/24 layers to GPU"
                        ),
                        log_line("Process", "srv llama_server: model loaded"),
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                runner.require_accelerator_attestation(
                    [stdout_log, stderr_log],
                    root,
                    "validation",
                    expected_artifacts,
                    "vulkan",
                )

    def test_accelerator_attestation_rejects_every_bad_duplicate_or_late_offload(self):
        runner = load_runner()
        positive = "\n".join(
            (
                log_line("LlamaCpp", f"Loading model: {K2_SMALL}"),
                backend_log_line("vulkan"),
                log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}"),
                log_line(
                    "Process",
                    "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                ),
                log_line("Process", "load_tensors: offloaded 12/24 layers to GPU"),
                log_line("Process", "srv llama_server: model loaded"),
            )
        )
        bad_second_loads = {
            "zero": "\n".join(
                (
                    log_line("LlamaCpp", f"Loading model: {K2_SMALL}"),
                    backend_log_line("vulkan"),
                    log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}"),
                    log_line(
                        "Process",
                        "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                    ),
                    log_line("Process", "load_tensors: offloaded 0/24 layers to GPU"),
                    log_line("Process", "srv llama_server: model loaded"),
                )
            ),
            "missing": "\n".join(
                (
                    log_line("LlamaCpp", f"Loading model: {K2_SMALL}"),
                    backend_log_line("vulkan"),
                    log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}"),
                    log_line(
                        "Process",
                        "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                    ),
                    log_line("Process", "srv llama_server: model loaded"),
                )
            ),
            "incomplete": log_line("LlamaCpp", f"Loading model: {K2_SMALL}"),
            "late zero outside a load": log_line(
                "Process", "load_tensors: offloaded 0/24 layers to GPU"
            ),
            "late malformed outside a load": (
                log_line("Process", "load_tensors: offloaded many/all layers to GPU")
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_artifacts = verified_artifacts(runner, root)
            log = root / "lemond.stdout.log"
            for name, second_load in bad_second_loads.items():
                with self.subTest(name=name):
                    contents = bind_verified_artifact_path(
                        f"{positive}\n{second_load}\n", expected_artifacts
                    )
                    log.write_text(contents, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "accelerator|incomplete"):
                        runner.require_accelerator_attestation(
                            [log],
                            root,
                            "validation",
                            expected_artifacts,
                            "vulkan",
                        )

    def test_accelerator_attestation_rejects_oversized_log_lines(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_artifacts = verified_artifacts(runner, root)
            log = root / "lemond.stdout.log"
            log.write_bytes(b"x" * (runner.MAX_ATTESTATION_LINE_BYTES + 1))

            with self.assertRaisesRegex(RuntimeError, "line is too long"):
                runner.require_accelerator_attestation(
                    [log],
                    root,
                    "validation",
                    expected_artifacts,
                    "vulkan",
                )

    def test_accelerator_attestation_enforces_aggregate_log_budget(self):
        runner = load_runner()
        positive = "\n".join(
            (
                log_line("LlamaCpp", f"Loading model: {K2_SMALL}"),
                backend_log_line("vulkan"),
                log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}"),
                log_line(
                    "Process",
                    "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                ),
                log_line("Process", "load_tensors: offloaded 12/24 layers to GPU"),
                log_line("Process", "srv llama_server: model loaded"),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_artifacts = verified_artifacts(runner, root)
            positive = bind_verified_artifact_path(positive, expected_artifacts)
            first_log = root / "lemond.stdout.log"
            second_log = root / "lemond.stderr.log"
            first_log.write_text(positive, encoding="utf-8")
            second_log.write_text("12345", encoding="utf-8")
            budget = len(positive.encode("utf-8")) + 4

            with (
                mock.patch.object(runner, "MAX_ATTESTATION_LOG_BYTES", budget),
                self.assertRaisesRegex(RuntimeError, "byte budget"),
            ):
                runner.require_accelerator_attestation(
                    [first_log, second_log],
                    root,
                    "validation",
                    expected_artifacts,
                    "vulkan",
                )

    def test_attested_log_replacement_fails_final_identity_check(self):
        runner = load_runner()
        contents = "\n".join(
            (
                log_line("LlamaCpp", f"Loading model: {K2_SMALL}"),
                backend_log_line("vulkan"),
                log_line("LlamaCpp", f"Using GGUF: /cache/{K2_SMALL_FILENAME}"),
                log_line(
                    "Process",
                    "llama_prepare_model_devices: using device Vulkan0 (GPU)",
                ),
                log_line("Process", "load_tensors: offloaded 12/24 layers to GPU"),
                log_line("Process", "srv llama_server: model loaded"),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_artifacts = verified_artifacts(runner, root)
            contents = bind_verified_artifact_path(contents, expected_artifacts)
            log = root / "lemond.stdout.log"
            log.write_text(contents, encoding="utf-8")
            identity = runner.require_accelerator_attestation(
                [log],
                root,
                "validation",
                expected_artifacts,
                "vulkan",
            )[0]
            log.rename(root / "original.log")
            log.write_text(contents, encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "changed"):
                runner._revalidate_regular_file(
                    identity,
                    root,
                    require_nonempty=False,
                )

    def test_restart_is_not_attempted_when_primary_validation_fails(self):
        runner = load_runner()

        class FakeContainment:
            @staticmethod
            def popen_kwargs():
                return {}

            @staticmethod
            def assign(_process):
                return None

            @staticmethod
            def close():
                return True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lemond = root / "lemond"
            python = root / "python"
            runner_temp = root / "runner-temp"
            lemond.touch()
            python.touch()
            runner_temp.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "defaults.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                lemond=lemond,
                python=python,
                backend="vulkan",
                channel="",
                target="linux-vulkan",
                models=K2_SMALL,
                lite=False,
                capability_profile="k2-horizon-v1",
                capability_models=K2_SMALL,
                port=13305,
            )

            old_directory = Path.cwd()
            os.chdir(root)
            try:
                with (
                    mock.patch.object(runner, "_require_idle_runner"),
                    mock.patch.object(runner, "_require_unused_port"),
                    mock.patch.object(runner.time, "sleep"),
                    mock.patch.object(
                        runner,
                        "_create_process_containment",
                        side_effect=lambda **_kwargs: FakeContainment(),
                    ),
                    mock.patch.object(
                        runner.subprocess,
                        "Popen",
                        side_effect=(FakeProcess(), FakeProcess()),
                    ) as popen,
                    mock.patch.object(
                        runner,
                        "_run_contained_command",
                        side_effect=(0, 7),
                    ) as run,
                    mock.patch.object(
                        runner,
                        "_shutdown_server",
                        side_effect=(True, True),
                    ) as shutdown,
                    mock.patch.object(
                        runner,
                        "verify_validation_artifacts",
                        return_value=verified_artifacts(runner),
                    ) as verify,
                    mock.patch.dict(os.environ, {"RUNNER_TEMP": str(runner_temp)}),
                ):
                    result = runner.run_validation(args)
            finally:
                os.chdir(old_directory)

        self.assertEqual(result, 7)
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(shutdown.call_count, 2)
        verify.assert_called_once()

    def test_direct_locked_model_is_verified_without_a_capability_profile(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lemond = root / "lemond"
            python = root / "python"
            runner_temp = root / "runner-temp"
            lemond.touch()
            python.touch()
            runner_temp.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "defaults.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                lemond=lemond,
                python=python,
                backend="vulkan",
                channel="",
                target="linux-vulkan",
                models=f"builtin.{K2_SMALL}",
                lite=False,
                capability_profile="",
                capability_models="",
                port=13305,
            )

            old_directory = Path.cwd()
            os.chdir(root)
            try:
                with (
                    mock.patch.object(runner, "_require_idle_runner"),
                    mock.patch.object(runner, "_require_unused_port"),
                    mock.patch.object(
                        runner,
                        "_run_validation_process",
                        side_effect=self._complete_process,
                    ) as run_process,
                    mock.patch.object(
                        runner,
                        "verify_validation_artifacts",
                        return_value=verified_artifacts(runner),
                    ) as verify,
                    mock.patch.object(
                        runner,
                        "require_accelerator_attestation",
                        return_value=[],
                    ) as attest,
                    mock.patch.dict(os.environ, {"RUNNER_TEMP": str(runner_temp)}),
                ):
                    result = runner.run_validation(args)
            finally:
                os.chdir(old_directory)

        self.assertEqual(result, 0)
        self.assertEqual(run_process.call_count, 3)
        self.assertEqual(verify.call_count, 3)
        self.assertEqual(attest.call_count, 2)
        self.assertEqual(
            attest.call_args_list[0].args[3],
            verified_artifacts(runner),
        )
        self.assertEqual(
            attest.call_args_list[1].args[3],
            verified_artifacts(runner),
        )
        for call in verify.call_args_list:
            self.assertEqual(call.args[3], [f"builtin.{K2_SMALL}"])

    def test_direct_mixed_models_restart_the_selected_locked_candidate(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lemond = root / "lemond"
            python = root / "python"
            runner_temp = root / "runner-temp"
            lemond.touch()
            python.touch()
            runner_temp.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "defaults.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                lemond=lemond,
                python=python,
                backend="vulkan",
                channel="",
                target="linux-vulkan",
                models=f"Other-Model,builtin.{K2_SMALL}",
                lite=False,
                capability_profile="",
                capability_models="",
                port=13305,
            )

            with (
                mock.patch.object(runner, "_require_idle_runner"),
                mock.patch.object(runner, "_require_unused_port"),
                mock.patch.object(
                    runner,
                    "_run_validation_process",
                    side_effect=self._complete_process,
                ) as run_process,
                mock.patch.object(
                    runner,
                    "verify_validation_artifacts",
                    return_value=verified_artifacts(runner),
                ),
                mock.patch.object(
                    runner,
                    "require_accelerator_attestation",
                    return_value=[],
                ) as attest,
                mock.patch.dict(os.environ, {"RUNNER_TEMP": str(runner_temp)}),
            ):
                result = runner.run_validation(args)

        self.assertEqual(result, 0)
        restart_command = run_process.call_args_list[2].kwargs["command"]
        restart_model_index = restart_command.index("--model") + 1
        self.assertEqual(restart_command[restart_model_index], f"builtin.{K2_SMALL}")
        self.assertEqual(attest.call_count, 2)
        self.assertEqual(
            attest.call_args_list[0].kwargs["selected_models"],
            ["Other-Model", f"builtin.{K2_SMALL}"],
        )
        self.assertEqual(
            attest.call_args_list[1].kwargs["selected_models"],
            [f"builtin.{K2_SMALL}"],
        )

    def test_capability_model_is_the_restart_target_when_other_models_are_locked(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lemond = root / "lemond"
            python = root / "python"
            runner_temp = root / "runner-temp"
            lemond.touch()
            python.touch()
            runner_temp.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "defaults.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                lemond=lemond,
                python=python,
                backend="vulkan",
                channel="",
                target="windows-vulkan",
                models=f"{K2_MEDIUM},{K2_SMALL}",
                lite=False,
                capability_profile="k2-horizon-v1",
                capability_models=K2_SMALL,
                port=13305,
            )

            with (
                mock.patch.object(runner, "_require_idle_runner"),
                mock.patch.object(runner, "_require_unused_port"),
                mock.patch.object(
                    runner,
                    "_run_validation_process",
                    side_effect=self._complete_process,
                ) as run_process,
                mock.patch.object(
                    runner,
                    "verify_validation_artifacts",
                    return_value={},
                ),
                mock.patch.dict(os.environ, {"RUNNER_TEMP": str(runner_temp)}),
            ):
                result = runner.run_validation(args)

        self.assertEqual(result, 0)
        restart_command = run_process.call_args_list[2].kwargs["command"]
        restart_model_index = restart_command.index("--model") + 1
        self.assertEqual(restart_command[restart_model_index], K2_SMALL)

    def test_cpu_lane_verifies_locked_artifacts_with_cpu_attestation(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lemond = root / "lemond"
            python = root / "python"
            runner_temp = root / "runner-temp"
            lemond.touch()
            python.touch()
            runner_temp.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "defaults.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                lemond=lemond,
                python=python,
                backend="cpu",
                channel="",
                target="linux-cpu",
                models=K2_SMALL,
                lite=False,
                capability_profile="",
                capability_models="",
                port=13305,
            )

            with (
                mock.patch.object(runner, "_require_idle_runner"),
                mock.patch.object(runner, "_require_unused_port"),
                mock.patch.object(
                    runner,
                    "_run_validation_process",
                    side_effect=self._complete_process,
                ),
                mock.patch.object(
                    runner,
                    "verify_validation_artifacts",
                    return_value=verified_artifacts(runner),
                ) as verify,
                mock.patch.object(
                    runner,
                    "require_accelerator_attestation",
                    return_value=[],
                ) as attest,
                mock.patch.dict(os.environ, {"RUNNER_TEMP": str(runner_temp)}),
            ):
                result = runner.run_validation(args)

        self.assertEqual(result, 0)
        self.assertEqual(verify.call_count, 3)
        self.assertEqual(attest.call_count, 2)
        for call in attest.call_args_list:
            self.assertEqual(call.args[4], "cpu")

    def test_rocm_lane_attests_exact_channel_for_primary_and_restart(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lemond = root / "lemond"
            python = root / "python"
            runner_temp = root / "runner-temp"
            lemond.touch()
            python.touch()
            runner_temp.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "defaults.json").write_text("{}", encoding="utf-8")

            for channel in ("stable", "nightly"):
                with self.subTest(channel=channel):
                    args = argparse.Namespace(
                        lemond=lemond,
                        python=python,
                        backend="rocm",
                        channel=channel,
                        target=f"linux-rocm-{channel}",
                        models=K2_SMALL,
                        lite=False,
                        capability_profile="",
                        capability_models="",
                        port=13305,
                    )

                    with (
                        mock.patch.object(runner, "_require_idle_runner"),
                        mock.patch.object(runner, "_require_unused_port"),
                        mock.patch.object(
                            runner,
                            "_run_validation_process",
                            side_effect=self._complete_process,
                        ),
                        mock.patch.object(
                            runner,
                            "verify_validation_artifacts",
                            return_value=verified_artifacts(runner),
                        ),
                        mock.patch.object(
                            runner,
                            "require_accelerator_attestation",
                            return_value=[],
                        ) as attest,
                        mock.patch.dict(os.environ, {"RUNNER_TEMP": str(runner_temp)}),
                    ):
                        result = runner.run_validation(args)

                    self.assertEqual(result, 0)
                    self.assertEqual(attest.call_count, 2)
                    for call in attest.call_args_list:
                        self.assertEqual(call.args[4], f"rocm-{channel}")

    def test_artifact_change_during_full_validation_stops_before_restart(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lemond = root / "lemond"
            python = root / "python"
            runner_temp = root / "runner-temp"
            state = root / "artifact-state"
            lemond.touch()
            python.touch()
            runner_temp.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "defaults.json").write_text("{}", encoding="utf-8")
            state.write_text("trusted", encoding="ascii")
            args = argparse.Namespace(
                lemond=lemond,
                python=python,
                backend="metal",
                channel="",
                target="macos-metal",
                models=K2_SMALL,
                lite=False,
                capability_profile="",
                capability_models="",
                port=13305,
            )

            process_count = 0

            def run_process(**kwargs):
                nonlocal process_count
                process_count += 1
                self._complete_process(**kwargs)
                if process_count == 2:
                    state.write_text("swapped", encoding="ascii")
                return 0, True

            def verify(*_args):
                if state.read_text(encoding="ascii") != "trusted":
                    raise ValueError("artifact changed across inference")
                return verified_artifacts(runner)

            old_directory = Path.cwd()
            os.chdir(root)
            try:
                with (
                    mock.patch.object(runner, "_require_idle_runner"),
                    mock.patch.object(runner, "_require_unused_port"),
                    mock.patch.object(
                        runner,
                        "_run_validation_process",
                        side_effect=run_process,
                    ),
                    mock.patch.object(
                        runner,
                        "verify_validation_artifacts",
                        side_effect=verify,
                    ) as verify_artifacts,
                    mock.patch.dict(os.environ, {"RUNNER_TEMP": str(runner_temp)}),
                    self.assertRaisesRegex(ValueError, "across inference"),
                ):
                    runner.run_validation(args)
            finally:
                os.chdir(old_directory)

        self.assertEqual(process_count, 2)
        self.assertEqual(verify_artifacts.call_count, 2)

    def test_attested_log_replacement_stops_workflow_output_emission(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lemond = root / "lemond"
            python = root / "python"
            runner_temp = root / "runner-temp"
            github_output = root / "github-output"
            lemond.touch()
            python.touch()
            runner_temp.mkdir()
            resources = root / "resources"
            resources.mkdir()
            (resources / "defaults.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                lemond=lemond,
                python=python,
                backend="vulkan",
                channel="",
                target="linux-vulkan",
                models=K2_SMALL,
                lite=False,
                capability_profile="",
                capability_models="",
                port=13305,
            )
            validation_identity = None

            def attest(
                log_paths,
                allowed_root,
                _phase,
                _expected_artifacts,
                _expected_backend,
                **_kwargs,
            ):
                nonlocal validation_identity
                identities = [
                    runner._capture_regular_file(
                        path,
                        allowed_root,
                        require_nonempty=False,
                    )
                    for path in log_paths
                ]
                if validation_identity is None:
                    validation_identity = identities[0]
                else:
                    attacked_path = validation_identity.path
                    attacked_path.rename(attacked_path.with_suffix(".original"))
                    attacked_path.touch()
                return identities

            with (
                mock.patch.object(runner, "_require_idle_runner"),
                mock.patch.object(runner, "_require_unused_port"),
                mock.patch.object(
                    runner,
                    "_run_validation_process",
                    side_effect=self._complete_process,
                ),
                mock.patch.object(
                    runner,
                    "verify_validation_artifacts",
                    return_value=verified_artifacts(runner),
                ),
                mock.patch.object(
                    runner,
                    "require_accelerator_attestation",
                    side_effect=attest,
                ),
                mock.patch.dict(
                    os.environ,
                    {
                        "RUNNER_TEMP": str(runner_temp),
                        "GITHUB_OUTPUT": str(github_output),
                    },
                ),
                self.assertRaisesRegex(RuntimeError, "changed"),
            ):
                runner.run_validation(args)

            self.assertFalse(github_output.exists())

    def test_cache_root_rejects_a_preexisting_symlink(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_runner_temp = root / "real-runner-temp"
            runner_temp = root / "runner-temp"
            real_runner_temp.mkdir()
            runner_temp.symlink_to(real_runner_temp, target_is_directory=True)

            create_cache = getattr(runner, "create_validation_cache", None)
            self.assertIsNotNone(create_cache)
            with self.assertRaisesRegex(RuntimeError, "symlink or junction"):
                create_cache(runner_temp, "linux-vulkan")

    def test_validation_cache_is_unpredictable_and_has_a_revalidation_contract(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            runner_temp = Path(directory)
            first = runner.create_validation_cache(runner_temp, "linux-vulkan")
            runner.revalidate_validation_cache(first)
            for identity in first:
                self.assertTrue(identity.path.is_dir())
            try:
                second = runner.create_validation_cache(runner_temp, "linux-vulkan")
            except (OSError, RuntimeError) as exc:
                self.fail(f"cache path was predictable or reusable: {exc}")

        self.assertNotEqual(first, second)
        revalidate = getattr(runner, "revalidate_validation_cache", None)
        self.assertIsNotNone(revalidate)

    def test_server_log_files_are_created_exclusively(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            stdout_path = root / "stdout.log"
            stdout_path.write_text("stale", encoding="utf-8")
            with (
                mock.patch.object(runner.subprocess, "Popen") as popen,
                self.assertRaises(FileExistsError),
            ):
                runner._run_validation_process(
                    lemond=root / "lemond",
                    cache_directory=cache,
                    config_directory=cache,
                    command=["validate"],
                    environment={},
                    port=13305,
                    pid_path=root / "lemond.pid",
                    stdout_path=stdout_path,
                    stderr_path=root / "stderr.log",
                )

        popen.assert_not_called()

    def test_containment_assignment_precedes_pid_and_close_failure_keeps_pid(self):
        runner = load_runner()
        events = []

        class FakeContainment:
            @staticmethod
            def popen_kwargs():
                return {}

            @staticmethod
            def assign(_process):
                events.append("assign")

            @staticmethod
            def close():
                events.append("close")
                return False

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            pid_path = root / "lemond.pid"

            def run_client(
                selected_command,
                *,
                env,
                containment,
                timeout_seconds,
            ):
                self.assertTrue(pid_path.is_file())
                self.assertEqual(selected_command, ["validate"])
                self.assertEqual(env, {})
                self.assertIsInstance(containment, FakeContainment)
                self.assertEqual(
                    timeout_seconds,
                    runner.VALIDATION_PHASE_TIMEOUT_SECONDS,
                )
                events.append("client")
                return 0

            def shutdown(*_args):
                events.append("shutdown")
                return True

            with (
                mock.patch.object(
                    runner,
                    "_create_process_containment",
                    return_value=FakeContainment(),
                ),
                mock.patch.object(
                    runner.subprocess,
                    "Popen",
                    side_effect=lambda *_args, **_kwargs: (
                        events.append("popen") or FakeProcess()
                    ),
                ),
                mock.patch.object(runner.time, "sleep"),
                mock.patch.object(
                    runner,
                    "_run_contained_command",
                    side_effect=run_client,
                ),
                mock.patch.object(
                    runner,
                    "_shutdown_server",
                    side_effect=shutdown,
                ),
            ):
                exit_code, cleanup_succeeded = runner._run_validation_process(
                    lemond=root / "lemond",
                    cache_directory=cache,
                    config_directory=cache,
                    command=["validate"],
                    environment={},
                    port=13305,
                    pid_path=pid_path,
                    stdout_path=root / "stdout.log",
                    stderr_path=root / "stderr.log",
                )
            pid_evidence_retained = pid_path.is_file()

        self.assertEqual(exit_code, 0)
        self.assertFalse(cleanup_succeeded)
        self.assertEqual(events, ["popen", "assign", "client", "shutdown", "close"])
        self.assertTrue(pid_evidence_retained)

    def test_validation_cache_rejects_directory_replacement_and_symlink_swap(self):
        runner = load_runner()

        for cache_name in (
            "hf_cache_directory",
            "private_temp_directory",
            "results_directory",
            "logs_directory",
        ):
            for attack in ("directory", "symlink"):
                self._assert_cache_directory_attack_is_rejected(
                    runner, cache_name, attack
                )

    def _assert_cache_directory_attack_is_rejected(
        self, runner, cache_name: str, attack: str
    ) -> None:
        with (
            self.subTest(cache_name=cache_name, attack=attack),
            tempfile.TemporaryDirectory() as directory,
        ):
            root = Path(directory)
            cache = runner.create_validation_cache(root, "linux-vulkan")
            attacked_path = getattr(cache, cache_name).path
            original = attacked_path.with_name(f"{attacked_path.name}-original")
            attacked_path.rename(original)
            if attack == "directory":
                attacked_path.mkdir()
            else:
                outside = root / "outside"
                outside.mkdir()
                try:
                    attacked_path.symlink_to(outside, target_is_directory=True)
                except OSError as exc:
                    self.skipTest(f"directory symlinks are unavailable: {exc}")

            with self.assertRaisesRegex(RuntimeError, "changed|symlink or junction"):
                runner.revalidate_validation_cache(cache)

    def test_pid_file_failure_still_stops_the_started_server(self):
        runner = load_runner()

        class FakeContainment:
            @staticmethod
            def popen_kwargs():
                return {}

            @staticmethod
            def assign(_process):
                return None

            @staticmethod
            def close():
                return True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            pid_path = root / "pid-as-directory"
            pid_path.mkdir()
            with (
                mock.patch.object(
                    runner,
                    "_create_process_containment",
                    return_value=FakeContainment(),
                ),
                mock.patch.object(
                    runner.subprocess, "Popen", return_value=FakeProcess()
                ),
                mock.patch.object(runner, "_run_contained_command") as run,
                mock.patch.object(
                    runner, "_shutdown_server", return_value=True
                ) as shutdown,
                self.assertRaises(OSError),
            ):
                runner._run_validation_process(
                    lemond=root / "lemond",
                    cache_directory=cache,
                    config_directory=cache,
                    command=["validate"],
                    environment={},
                    port=13305,
                    pid_path=pid_path,
                    stdout_path=root / "stdout.log",
                    stderr_path=root / "stderr.log",
                )

        run.assert_not_called()
        shutdown.assert_called_once()

    def test_already_exited_server_is_an_unclean_validation_failure(self):
        runner = load_runner()

        class ExitedProcess:
            pid = 999

            @staticmethod
            def poll():
                return 17

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            cache.mkdir()
            with mock.patch.object(
                runner,
                "_cache_processes",
                return_value=runner._CacheProcessScan([], True),
            ):
                clean = runner._shutdown_server(ExitedProcess(), 13305, cache)

        self.assertFalse(clean)

    def test_nonzero_clean_shutdown_exit_is_a_validation_failure(self):
        runner = load_runner()

        class NonzeroShutdownProcess:
            pid = 1001

            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                del timeout
                self.returncode = 139
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            cache.mkdir()
            with (
                mock.patch.object(runner, "_open_local_url"),
                mock.patch.object(
                    runner,
                    "_cache_processes",
                    return_value=runner._CacheProcessScan([], True),
                ),
            ):
                clean = runner._shutdown_server(NonzeroShutdownProcess(), 13305, cache)

        self.assertFalse(clean)

    def test_clean_shutdown_uses_http_and_returns_true(self):
        runner = load_runner()
        wait_timeouts = []

        class CleanShutdownProcess:
            pid = 1003

            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                wait_timeouts.append(timeout)
                self.returncode = 0
                return self.returncode

        process = CleanShutdownProcess()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            cache.mkdir()
            with (
                mock.patch.object(runner, "_open_local_url") as open_url,
                mock.patch.object(
                    runner,
                    "_cache_processes",
                    return_value=runner._CacheProcessScan([], True),
                ),
            ):
                clean = runner._shutdown_server(process, 13305, cache)

        self.assertTrue(clean)
        self.assertTrue(wait_timeouts)
        self.assertTrue(all(timeout > 0 for timeout in wait_timeouts))
        open_url.assert_called_once()
        self.assertEqual(open_url.call_args.kwargs, {"timeout": 10})

    def test_shutdown_cancellation_rescans_cache_before_bounded_abort(self):
        runner = load_runner()
        cancellation_deadline = time.monotonic() + 4
        cancellation = runner._ValidationCancelled(
            signal.SIGINT,
            cancellation_deadline,
        )
        orphan = argparse.Namespace(pid=2002)
        process = argparse.Namespace(pid=2001)
        process.poll = mock.Mock(return_value=None)
        containment = object()

        with (
            mock.patch.object(
                runner,
                "_open_local_url_with_cancellation",
                side_effect=cancellation,
            ),
            mock.patch.object(
                runner,
                "_cache_processes",
                return_value=runner._CacheProcessScan([orphan], True),
            ) as scan,
            mock.patch.object(runner, "_force_kill_processes") as force_kill,
            mock.patch.object(runner, "_abort_process_containment") as abort,
            mock.patch.object(runner, "_reap_direct_process") as reap,
            self.assertRaises(runner._ValidationCancelled) as raised,
        ):
            runner._shutdown_server(
                process,
                13305,
                Path("/validation-cache"),
                containment,
            )

        self.assertIs(raised.exception, cancellation)
        scan.assert_called_once_with(Path("/validation-cache"), process.pid)
        force_kill.assert_called_once_with([orphan], cancellation_deadline)
        abort.assert_called_once_with(containment, cancellation_deadline)
        reap.assert_called_once_with(process, cancellation_deadline)

    def test_cleaned_orphan_process_still_invalidates_the_lane(self):
        runner = load_runner()

        class CleanShutdownProcess:
            pid = 1002

            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                del timeout
                self.returncode = 0
                return self.returncode

        orphan = object()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            cache.mkdir()
            with (
                mock.patch.object(runner, "_open_local_url"),
                mock.patch.object(
                    runner,
                    "_cache_processes",
                    return_value=runner._CacheProcessScan([orphan], True),
                ),
                mock.patch.object(runner, "_stop_processes", return_value=True) as stop,
            ):
                clean = runner._shutdown_server(CleanShutdownProcess(), 13305, cache)

        self.assertFalse(clean)
        stop.assert_called_once_with([orphan])

    def test_orphan_cleanup_waits_for_term_exit_before_using_kill(self):
        runner = load_runner()
        events = []

        class FakePsutilError(Exception):
            pass

        class ExitsOnTermProcess:
            @staticmethod
            def terminate():
                events.append("terminate")

            @staticmethod
            def kill():
                events.append("kill")

        def wait_procs(processes, timeout):
            events.append(("wait", timeout))
            return processes, []

        fake_psutil = argparse.Namespace(
            Error=FakePsutilError,
            wait_procs=wait_procs,
        )
        with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
            clean = runner._stop_processes([ExitsOnTermProcess()])

        self.assertTrue(clean)
        self.assertEqual(events[0], "terminate")
        self.assertEqual(events[1][0], "wait")
        self.assertGreater(events[1][1], 0)
        self.assertLessEqual(events[1][1], runner.PROCESS_WAIT_POLL_SECONDS)
        self.assertNotIn("kill", events)

    @unittest.skipIf(os.name == "nt", "POSIX process-group test")
    def test_process_group_probe_falls_back_when_psutil_enumeration_fails(self):
        runner = load_runner()

        for enumeration_error in (
            PermissionError("denied"),
            OSError("enumeration failed"),
        ):
            fake_psutil = mock.Mock()
            fake_psutil.STATUS_DEAD = "dead"
            fake_psutil.STATUS_ZOMBIE = "zombie"
            fake_psutil.Error = RuntimeError
            fake_psutil.process_iter.side_effect = enumeration_error
            with (
                self.subTest(error=type(enumeration_error).__name__),
                mock.patch.dict(sys.modules, {"psutil": fake_psutil}),
                mock.patch.object(runner.os, "killpg", return_value=None) as killpg,
            ):
                self.assertTrue(
                    runner._PosixProcessGroupContainment._has_live_members(4321)
                )
                killpg.assert_called_once_with(4321, 0)

        fake_psutil = mock.Mock()
        fake_psutil.STATUS_DEAD = "dead"
        fake_psutil.STATUS_ZOMBIE = "zombie"
        fake_psutil.Error = RuntimeError
        fake_psutil.process_iter.side_effect = PermissionError("denied")
        with (
            mock.patch.dict(sys.modules, {"psutil": fake_psutil}),
            mock.patch.object(
                runner.os,
                "killpg",
                side_effect=ProcessLookupError,
            ),
        ):
            self.assertFalse(
                runner._PosixProcessGroupContainment._has_live_members(4321)
            )

        dead_process = mock.Mock()
        dead_process.pid = 9875
        dead_process.info = {"status": "dead"}

        def partial_enumeration(_attributes):
            yield dead_process
            raise OSError("enumeration failed")

        fake_psutil = mock.Mock()
        fake_psutil.STATUS_DEAD = "dead"
        fake_psutil.STATUS_ZOMBIE = "zombie"
        fake_psutil.Error = RuntimeError
        fake_psutil.process_iter.side_effect = partial_enumeration
        with (
            mock.patch.dict(sys.modules, {"psutil": fake_psutil}),
            mock.patch.object(runner.os, "killpg", return_value=None) as killpg,
        ):
            self.assertTrue(
                runner._PosixProcessGroupContainment._has_live_members(4321)
            )
            killpg.assert_called_once_with(4321, 0)

    @unittest.skipIf(os.name == "nt", "POSIX process-group test")
    def test_normal_process_group_close_force_kills_when_cancellation_is_pending(self):
        runner = load_runner()
        containment = runner._PosixProcessGroupContainment(
            allow_untracked_descendants=sys.platform == "darwin"
        )
        containment._process_groups = [4324]
        cancellation = argparse.Namespace(signum=signal.SIGINT)

        with (
            mock.patch.object(
                runner,
                "_ACTIVE_CANCELLATION_STATE",
                cancellation,
            ),
            mock.patch.object(
                runner._PosixProcessGroupContainment,
                "_has_live_members",
                return_value=True,
            ),
            mock.patch.object(runner.os, "killpg") as killpg,
        ):
            clean = containment.close()

        self.assertFalse(clean)
        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(4324, signal.SIGTERM),
                mock.call(4324, signal.SIGKILL),
            ],
        )

    def test_cache_process_scan_fails_closed_when_psutil_enumeration_fails(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            for enumeration_error in (
                PermissionError("denied"),
                OSError("enumeration failed"),
            ):
                fake_psutil = mock.Mock()
                fake_psutil.Error = RuntimeError
                fake_psutil.process_iter.side_effect = enumeration_error
                with (
                    self.subTest(error=type(enumeration_error).__name__),
                    mock.patch.dict(sys.modules, {"psutil": fake_psutil}),
                ):
                    scan = runner._cache_processes(Path(directory), os.getpid())

                self.assertFalse(scan.complete)
                self.assertEqual(scan.processes, [])

    def test_partial_cache_process_scan_retains_known_matches(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory).resolve()
            known_process = mock.Mock()
            known_process.pid = 9876
            known_process.info = {"exe": str(cache / "llama-server")}

            def partial_enumeration(_attributes):
                yield known_process
                raise PermissionError("enumeration failed")

            fake_psutil = mock.Mock()
            fake_psutil.Error = RuntimeError
            fake_psutil.process_iter.side_effect = partial_enumeration
            with mock.patch.dict(sys.modules, {"psutil": fake_psutil}):
                scan = runner._cache_processes(cache, os.getpid())

        self.assertFalse(scan.complete)
        self.assertEqual(scan.processes, [known_process])

    def test_incomplete_cache_process_scan_invalidates_cleanup_and_stops_matches(self):
        runner = load_runner()

        class ExitedProcess:
            pid = 9877

            @staticmethod
            def poll():
                return 0

        known_process = object()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with (
                mock.patch.object(
                    runner,
                    "_cache_processes",
                    return_value=runner._CacheProcessScan([known_process], False),
                ),
                mock.patch.object(runner, "_stop_processes", return_value=True) as stop,
            ):
                clean = runner._shutdown_server(ExitedProcess(), 13305, cache)

        self.assertFalse(clean)
        stop.assert_called_once_with([known_process])

    def test_empty_incomplete_cache_process_scan_invalidates_cleanup(self):
        runner = load_runner()

        class ExitedProcess:
            pid = 9878

            @staticmethod
            def poll():
                return 0

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                runner,
                "_cache_processes",
                return_value=runner._CacheProcessScan([], False),
            ):
                clean = runner._shutdown_server(
                    ExitedProcess(),
                    13305,
                    Path(directory),
                )

        self.assertFalse(clean)

    def test_contained_command_timeout_closes_containment(self):
        runner = load_runner()
        events = []

        class TimedOutProcess:
            pid = 4322

            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                events.append(("wait", timeout))
                if self.returncode is None:
                    time.sleep(timeout)
                    raise subprocess.TimeoutExpired(["validate"], timeout)
                return self.returncode

            def terminate(self):
                events.append("terminate")
                self.returncode = -signal.SIGTERM

            def kill(self):
                events.append("kill")
                self.returncode = -signal.SIGKILL

        class FakeContainment:
            @staticmethod
            def popen_kwargs():
                return {}

            @staticmethod
            def assign(_process):
                events.append("assign")

            @staticmethod
            def abort(_deadline):
                events.append("abort")
                return False

            @staticmethod
            def close():
                events.append("close")
                return False

        process = TimedOutProcess()
        with (
            mock.patch.object(runner.subprocess, "Popen", return_value=process),
            self.assertRaisesRegex(RuntimeError, "timed out after 0.01 seconds"),
        ):
            runner._run_contained_command(
                ["validate"],
                env={},
                containment=FakeContainment(),
                timeout_seconds=0.01,
            )

        self.assertIn("abort", events)
        self.assertLess(events.index("abort"), events.index("kill"))
        self.assertNotIn("terminate", events)
        self.assertNotIn(("wait", 5), events)

    def test_contained_command_preserves_nonzero_exit_code(self):
        runner = load_runner()

        class ExitedProcess:
            pid = 4323

            @staticmethod
            def wait(timeout):
                del timeout
                return 23

        class FakeContainment:
            @staticmethod
            def popen_kwargs():
                return {}

            @staticmethod
            def assign(_process):
                return None

            @staticmethod
            def close():
                raise AssertionError("normal process exit must not close containment")

        with mock.patch.object(
            runner.subprocess,
            "Popen",
            return_value=ExitedProcess(),
        ):
            exit_code = runner._run_contained_command(
                ["validate"],
                env={},
                containment=FakeContainment(),
                timeout_seconds=30,
            )

        self.assertEqual(exit_code, 23)

    def test_darwin_containment_requires_the_trusted_ephemeral_runner_contract(self):
        runner = load_runner()
        trusted_environment = {
            "GITHUB_ACTIONS": "true",
            runner.DARWIN_EPHEMERAL_RUNNER_CONTEXT_KEY: (
                runner.DARWIN_EPHEMERAL_RUNNER_CONTEXT
            ),
        }

        with mock.patch.object(runner.sys, "platform", "darwin"):
            with self.assertRaisesRegex(
                RuntimeError, "Darwin cannot contain session-escaping descendants"
            ):
                runner._create_process_containment()
            with (
                mock.patch.dict(
                    os.environ,
                    {"GITHUB_ACTIONS": "true"},
                    clear=True,
                ),
                self.assertRaisesRegex(RuntimeError, "requires the trusted"),
            ):
                runner._create_process_containment(
                    allow_darwin_github_hosted_ephemeral_runner=True,
                    backend="metal",
                    target="macos-metal",
                )
            with (
                mock.patch.dict(os.environ, trusted_environment, clear=True),
                self.assertRaisesRegex(RuntimeError, "restricted to macos-metal"),
            ):
                runner._create_process_containment(
                    allow_darwin_github_hosted_ephemeral_runner=True,
                    backend="vulkan",
                    target="macos-metal",
                )
            with mock.patch.dict(os.environ, trusted_environment, clear=True):
                containment = runner._create_process_containment(
                    allow_darwin_github_hosted_ephemeral_runner=True,
                    backend="metal",
                    target="macos-metal",
                )

        self.assertTrue(containment.close())

    def test_linux_subreaper_lease_restores_only_the_original_disabled_state(self):
        runner = load_runner()

        class RunnerProcess:
            @staticmethod
            def children(recursive):
                del recursive
                return []

        fake_psutil = argparse.Namespace(
            DEAD_STATUS="dead",
            Error=OSError,
            NoSuchProcess=ProcessLookupError,
            Process=lambda _pid: RunnerProcess(),
            STATUS_DEAD="dead",
            STATUS_ZOMBIE="zombie",
            ZombieProcess=ProcessLookupError,
        )
        with (
            mock.patch.object(runner.sys, "platform", "linux"),
            mock.patch.dict(sys.modules, {"psutil": fake_psutil}),
            mock.patch.object(
                runner, "_linux_child_subreaper_state", return_value=False
            ) as get_state,
            mock.patch.object(runner, "_set_linux_child_subreaper") as set_state,
        ):
            first = runner._PosixProcessGroupContainment()
            second = runner._PosixProcessGroupContainment()
            self.assertTrue(first.close())
            self.assertTrue(second.close())

        get_state.assert_called_once_with()
        self.assertEqual(set_state.call_args_list, [mock.call(True), mock.call(False)])

    def test_linux_subreaper_initialization_fails_before_process_launch(self):
        runner = load_runner()

        class RunnerProcess:
            @staticmethod
            def children(recursive):
                del recursive
                return []

        fake_psutil = argparse.Namespace(
            Error=OSError,
            Process=lambda _pid: RunnerProcess(),
        )
        with (
            mock.patch.object(runner.sys, "platform", "linux"),
            mock.patch.dict(sys.modules, {"psutil": fake_psutil}),
            mock.patch.object(
                runner,
                "_acquire_linux_child_subreaper",
                side_effect=OSError("prctl denied"),
            ),
            mock.patch.object(runner.subprocess, "Popen") as popen,
            self.assertRaisesRegex(OSError, "prctl denied"),
        ):
            runner._PosixProcessGroupContainment()

        popen.assert_not_called()

    def test_linux_subreaper_rejects_unsafe_preexisting_process_state(self):
        runner = load_runner()

        class RunnerProcess:
            descendants = []

            @classmethod
            def children(cls, recursive):
                del recursive
                return cls.descendants

        fake_psutil = argparse.Namespace(
            Error=OSError,
            Process=lambda _pid: RunnerProcess(),
        )
        with (
            mock.patch.object(runner.sys, "platform", "linux"),
            mock.patch.dict(sys.modules, {"psutil": fake_psutil}),
            mock.patch.object(
                runner.signal,
                "getsignal",
                return_value=signal.SIG_IGN,
            ),
            mock.patch.object(runner, "_acquire_linux_child_subreaper") as acquire,
            self.assertRaisesRegex(RuntimeError, "default SIGCHLD"),
        ):
            runner._PosixProcessGroupContainment()
        acquire.assert_not_called()

        RunnerProcess.descendants = [object()]
        with (
            mock.patch.object(runner.sys, "platform", "linux"),
            mock.patch.dict(sys.modules, {"psutil": fake_psutil}),
            mock.patch.object(
                runner.signal,
                "getsignal",
                return_value=signal.SIG_DFL,
            ),
            mock.patch.object(runner, "_acquire_linux_child_subreaper") as acquire,
            self.assertRaisesRegex(RuntimeError, "pre-existing runner descendants"),
        ):
            runner._PosixProcessGroupContainment()
        acquire.assert_not_called()

    @unittest.skipIf(os.name == "nt", "POSIX descendant tracking test")
    @unittest.skipIf(
        not DARWIN_PROCESS_TREE_INSPECTION_AVAILABLE,
        "Darwin process-tree inspection is unavailable",
    )
    def test_clean_contained_process_exit_remains_clean_with_tracking(self):
        runner = load_runner()
        containment = runner._PosixProcessGroupContainment(
            allow_untracked_descendants=sys.platform == "darwin"
        )

        exit_code = runner._run_contained_command(
            [sys.executable, "-c", "pass"],
            env=dict(os.environ),
            containment=containment,
            timeout_seconds=30,
        )

        self.assertEqual(exit_code, 0)
        self.assertTrue(containment.close())

    @unittest.skipIf(os.name == "nt", "POSIX gated process launch test")
    @unittest.skipIf(
        not DARWIN_PROCESS_TREE_INSPECTION_AVAILABLE,
        "Darwin process-tree inspection is unavailable",
    )
    def test_posix_gate_assigns_containment_before_candidate_code_runs(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "candidate-ran.txt"
            containment = runner._PosixProcessGroupContainment(
                allow_untracked_descendants=sys.platform == "darwin"
            )
            original_assign = containment.assign

            def assert_candidate_is_blocked(process):
                self.assertFalse(marker.exists())
                original_assign(process)

            try:
                with mock.patch.object(
                    containment,
                    "assign",
                    side_effect=assert_candidate_is_blocked,
                ):
                    process = containment.start_process(
                        [
                            sys.executable,
                            "-c",
                            (
                                "import pathlib, sys; "
                                "pathlib.Path(sys.argv[1]).write_text("
                                "'ran', encoding='ascii')"
                            ),
                            str(marker),
                        ],
                        env=dict(os.environ),
                    )
                self.assertEqual(process.wait(timeout=10), 0)
                self.assertEqual(marker.read_text(encoding="ascii"), "ran")
            finally:
                self.assertTrue(containment.close())

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "Linux child-subreaper containment test"
    )
    def test_linux_subreaper_reaps_adopted_child_without_stealing_root_status(self):
        runner = load_runner()
        daemon_program = """
import os
import pathlib
import sys

first_child = os.fork()
if first_child:
    os._exit(0)
os.setsid()
second_child = os.fork()
if second_child:
    os._exit(0)
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding="ascii")
"""
        root_program = f"""
import pathlib
import subprocess
import sys
import time

subprocess.Popen(
    [sys.executable, "-c", {daemon_program!r}, sys.argv[1]],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
while not pathlib.Path(sys.argv[1]).is_file():
    time.sleep(0.01)
time.sleep(0.3)
raise SystemExit(23)
"""

        with tempfile.TemporaryDirectory() as directory:
            daemon_pid_path = Path(directory) / "daemon.pid"
            containment = runner._PosixProcessGroupContainment()
            try:
                exit_code = runner._run_contained_command(
                    [
                        sys.executable,
                        "-c",
                        root_program,
                        str(daemon_pid_path),
                    ],
                    env=dict(os.environ),
                    containment=containment,
                    timeout_seconds=30,
                )
                daemon_pid = int(daemon_pid_path.read_text(encoding="ascii"))
                self.assertEqual(exit_code, 23)
                self.assertFalse(_pid_is_active(daemon_pid))
            finally:
                self.assertTrue(containment.close())

    def test_cancellation_reuses_one_cleanup_deadline(self):
        runner = load_runner()
        cancellation = argparse.Namespace(signum=signal.SIGINT)

        with (
            mock.patch.object(
                runner,
                "_ACTIVE_CANCELLATION_STATE",
                cancellation,
            ),
            mock.patch.object(runner.time, "monotonic", return_value=100.0) as clock,
        ):
            with self.assertRaises(runner._ValidationCancelled) as first:
                runner._raise_if_cancelled()
            with self.assertRaises(runner._ValidationCancelled) as second:
                runner._raise_if_cancelled()

        self.assertEqual(first.exception.cleanup_deadline, 104.0)
        self.assertEqual(
            second.exception.cleanup_deadline,
            first.exception.cleanup_deadline,
        )
        clock.assert_called_once_with()

    @unittest.skipIf(os.name == "nt", "POSIX signal containment test")
    def test_sigint_and_sigterm_kill_contained_command_descendants(self):
        for selected_signal in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(selected_signal=selected_signal):
                self._assert_runner_signal_kills_descendants(selected_signal)

    @unittest.skipIf(os.name == "nt", "POSIX signal containment test")
    def test_sigint_force_kills_term_ignoring_tree_before_escalation(self):
        elapsed = self._assert_runner_signal_kills_descendants(
            signal.SIGINT,
            ignore_sigterm=True,
        )

        self.assertLess(elapsed, 6.0)

    @unittest.skipIf(os.name == "nt", "POSIX signal containment test")
    @unittest.skipIf(
        not DARWIN_PROCESS_TREE_INSPECTION_AVAILABLE,
        "Darwin process-tree inspection is unavailable",
    )
    def test_sigint_and_sigterm_kill_session_escaping_descendants(self):
        for selected_signal in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(selected_signal=selected_signal):
                elapsed = self._assert_runner_signal_kills_descendants(
                    selected_signal,
                    ignore_sigterm=True,
                    escape_session=True,
                )
                self.assertLess(elapsed, 6.0)

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "Linux child-subreaper containment test"
    )
    def test_sigint_and_sigterm_kill_fast_double_fork_session_escape(self):
        for selected_signal in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(selected_signal=selected_signal):
                elapsed = self._assert_runner_signal_kills_descendants(
                    selected_signal,
                    ignore_sigterm=True,
                    double_fork=True,
                )
                self.assertLess(elapsed, 6.0)

    def _assert_runner_signal_kills_descendants(
        self,
        selected_signal: int,
        *,
        ignore_sigterm: bool = False,
        escape_session: bool = False,
        double_fork: bool = False,
    ) -> float:
        descendant_setup = []
        if double_fork:
            descendant_setup.extend(
                [
                    "first_child = os.fork()",
                    "if first_child:",
                    "    os._exit(0)",
                    "os.setsid()",
                    "second_child = os.fork()",
                    "if second_child:",
                    "    os._exit(0)",
                ]
            )
        elif escape_session:
            descendant_setup.append("os.setsid()")
        if ignore_sigterm:
            descendant_setup.append("signal.signal(signal.SIGTERM, signal.SIG_IGN)")
        descendant_program = f"""
import os
import pathlib
import signal
import sys
import time

{chr(10).join(descendant_setup)}
pathlib.Path(sys.argv[1]).write_text(str(os.getpid()), encoding="ascii")
time.sleep(30)
"""
        parent_signal_setup = ""
        if ignore_sigterm:
            parent_signal_setup = "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        child_program = f"""
import os
import pathlib
import signal
import subprocess
import sys
import time

{parent_signal_setup}
child = subprocess.Popen(
    [sys.executable, "-c", {descendant_program!r}, sys.argv[2]],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
while not pathlib.Path(sys.argv[2]).is_file():
    time.sleep(0.01)
descendant_pid = pathlib.Path(sys.argv[2]).read_text(encoding="ascii")
pathlib.Path(sys.argv[1]).write_text(
    f"{{os.getpid()}},{{descendant_pid}}",
    encoding="ascii",
)
time.sleep(30)
"""
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "contained-pids.txt"
            descendant_ready_path = Path(directory) / "descendant-ready.txt"
            helper_program = f"""
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location("signal_runner", {str(RUNNER_PATH)!r})
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
command = [
    sys.executable,
    "-c",
    {child_program!r},
    {str(pid_path)!r},
    {str(descendant_ready_path)!r},
]

def run_validation(_args):
    containment = runner._PosixProcessGroupContainment(
        allow_untracked_descendants=sys.platform == "darwin"
    )
    try:
        return runner._run_contained_command(
            command,
            env=dict(os.environ),
            containment=containment,
            timeout_seconds=30,
        )
    finally:
        runner._close_process_containment(containment)

runner.run_validation = run_validation
sys.argv = [
    "run_llamacpp_validation.py",
    "--lemond",
    sys.executable,
    "--python",
    sys.executable,
    "--backend",
    "cpu",
    "--target",
    "linux-cpu",
]
runner.main()
"""
            helper = subprocess.Popen(
                [sys.executable, "-c", helper_program],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            command_pid = None
            descendant_pid = None
            try:
                deadline = time.monotonic() + 10
                pid_values = None
                while pid_values is None and time.monotonic() < deadline:
                    if helper.poll() is not None:
                        break
                    try:
                        values = pid_path.read_text(encoding="ascii").split(",")
                        if len(values) == 2:
                            pid_values = tuple(int(value) for value in values)
                    except (FileNotFoundError, ValueError):
                        pass
                    time.sleep(0.05)
                if pid_values is None:
                    stdout, stderr = helper.communicate(timeout=5)
                    self.fail(
                        "signal test helper did not start its descendants: "
                        f"stdout={stdout!r}, stderr={stderr!r}"
                    )
                command_pid, descendant_pid = pid_values
                signal_started = time.monotonic()
                helper.send_signal(selected_signal)
                stdout, stderr = helper.communicate(timeout=15)
                signal_elapsed = time.monotonic() - signal_started
                deadline = time.monotonic() + 10
                while (
                    _pid_is_running(command_pid) or _pid_is_running(descendant_pid)
                ) and time.monotonic() < deadline:
                    time.sleep(0.05)

                self.assertEqual(
                    helper.returncode,
                    128 + selected_signal,
                    f"stdout={stdout!r}, stderr={stderr!r}",
                )
                self.assertFalse(_pid_is_running(command_pid))
                self.assertFalse(_pid_is_running(descendant_pid))
            finally:
                if helper.poll() is None:
                    helper.kill()
                    helper.wait(timeout=5)
                if command_pid is not None:
                    try:
                        if (
                            os.getpgid(command_pid) == command_pid
                            and command_pid != os.getpgrp()
                        ):
                            os.killpg(command_pid, signal.SIGKILL)
                    except OSError:
                        pass
                if descendant_pid is not None and _pid_is_active(descendant_pid):
                    os.kill(descendant_pid, signal.SIGKILL)
        return signal_elapsed

    @unittest.skipIf(os.name == "nt", "POSIX signal containment test")
    def test_sigint_during_blocked_shutdown_aborts_term_ignoring_server_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_path = root / "server-pids.txt"
            descendant_ready_path = root / "server-descendant-ready.txt"
            shutdown_blocked_path = root / "shutdown-blocked.txt"
            descendant_program = (
                "import pathlib, signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "pathlib.Path(sys.argv[1]).write_text('ready', encoding='ascii'); "
                "time.sleep(30)"
            )
            server_program = f"""
import os
import pathlib
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c", {descendant_program!r}, sys.argv[2]])
while not pathlib.Path(sys.argv[2]).is_file():
    time.sleep(0.01)
pathlib.Path(sys.argv[1]).write_text(
    f"{{os.getpid()}},{{child.pid}}",
    encoding="ascii",
)
time.sleep(30)
"""
            helper_program = f"""
import contextlib
import importlib.util
import os
import pathlib
import sys
import time

spec = importlib.util.spec_from_file_location("shutdown_runner", {str(RUNNER_PATH)!r})
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
containment = runner._PosixProcessGroupContainment(
    allow_untracked_descendants=sys.platform == "darwin"
)
server = runner.subprocess.Popen(
    [
        sys.executable,
        "-c",
        {server_program!r},
        {str(pid_path)!r},
        {str(descendant_ready_path)!r},
    ],
    **containment.popen_kwargs(),
)
containment.assign(server)

def blocked_shutdown(_request, timeout):
    del timeout
    pathlib.Path({str(shutdown_blocked_path)!r}).write_text(
        "blocked",
        encoding="ascii",
    )
    time.sleep(30)
    return contextlib.nullcontext()

runner._open_local_url = blocked_shutdown
try:
    with runner._CancellationSignalGuard():
        try:
            runner._shutdown_server(
                server,
                13305,
                pathlib.Path({str(root)!r}),
                containment,
            )
            runner._raise_if_cancelled()
        finally:
            runner._close_process_containment(containment)
except runner._ValidationCancelled as exc:
    raise SystemExit(128 + exc.signum) from None
"""
            helper = subprocess.Popen(
                [sys.executable, "-c", helper_program],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            server_pid = None
            descendant_pid = None
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        pid_values = tuple(
                            int(value)
                            for value in pid_path.read_text(encoding="ascii").split(",")
                        )
                    except (FileNotFoundError, ValueError):
                        pid_values = ()
                    if len(pid_values) == 2 and shutdown_blocked_path.is_file():
                        server_pid, descendant_pid = pid_values
                        break
                    if helper.poll() is not None:
                        break
                    time.sleep(0.05)
                if server_pid is None or descendant_pid is None:
                    stdout, stderr = helper.communicate(timeout=5)
                    self.fail(
                        "shutdown signal helper did not reach the blocked request: "
                        f"stdout={stdout!r}, stderr={stderr!r}"
                    )

                signal_started = time.monotonic()
                helper.send_signal(signal.SIGINT)
                try:
                    stdout, stderr = helper.communicate(timeout=6)
                except subprocess.TimeoutExpired:
                    self.fail("blocked shutdown ignored cancellation for 6 seconds")
                signal_elapsed = time.monotonic() - signal_started

                self.assertEqual(
                    helper.returncode,
                    128 + signal.SIGINT,
                    f"stdout={stdout!r}, stderr={stderr!r}",
                )
                self.assertLess(signal_elapsed, 6.0)
                self.assertFalse(_pid_is_running(server_pid))
                self.assertFalse(_pid_is_running(descendant_pid))
            finally:
                if helper.poll() is None:
                    helper.kill()
                if server_pid is not None:
                    try:
                        if (
                            os.getpgid(server_pid) == server_pid
                            and server_pid != os.getpgrp()
                        ):
                            os.killpg(server_pid, signal.SIGKILL)
                    except OSError:
                        pass
                try:
                    helper.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    helper.kill()
                    helper.wait(timeout=5)

    @unittest.skipIf(os.name == "nt", "POSIX signal cleanup test")
    @unittest.skipIf(
        not DARWIN_PROCESS_TREE_INSPECTION_AVAILABLE,
        "Darwin process-tree inspection is unavailable",
    )
    def test_sigint_during_orphan_cleanup_aborts_containment_and_orphan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            cache.mkdir()
            contained_pids_path = root / "contained-pids.txt"
            contained_descendant_ready_path = root / "contained-ready.txt"
            orphan_pid_path = root / "orphan.pid"
            orphan_ready_path = root / "orphan-ready.txt"
            orphan_term_path = root / "orphan-term.txt"
            contained_descendant_program = (
                "import pathlib, signal, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "pathlib.Path(sys.argv[1]).write_text('ready', encoding='ascii'); "
                "time.sleep(30)"
            )
            contained_program = f"""
import os
import pathlib
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen(
    [sys.executable, "-c", {contained_descendant_program!r}, sys.argv[2]],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
while not pathlib.Path(sys.argv[2]).is_file():
    time.sleep(0.01)
pathlib.Path(sys.argv[1]).write_text(
    f"{{os.getpid()}},{{child.pid}}",
    encoding="ascii",
)
time.sleep(30)
"""
            orphan_program = """
import pathlib
import signal
import sys
import time

term_path = pathlib.Path(sys.argv[2])

def ignore_term(_signum, _frame):
    term_path.write_text("term", encoding="ascii")

signal.signal(signal.SIGTERM, ignore_term)
pathlib.Path(sys.argv[1]).write_text("ready", encoding="ascii")
while True:
    time.sleep(1)
"""
            helper_program = f"""
import importlib.util
import os
import pathlib
import signal
import subprocess
import sys
import time
import psutil as real_psutil

spec = importlib.util.spec_from_file_location("orphan_runner", {str(RUNNER_PATH)!r})
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
actual_containment = runner._PosixProcessGroupContainment(
    allow_untracked_descendants=sys.platform == "darwin"
)
contained = subprocess.Popen(
    [
        sys.executable,
        "-c",
        {contained_program!r},
        {str(contained_pids_path)!r},
        {str(contained_descendant_ready_path)!r},
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    **actual_containment.popen_kwargs(),
)
actual_containment.assign(contained)
orphan = subprocess.Popen(
    [
        sys.executable,
        "-c",
        {orphan_program!r},
        {str(orphan_ready_path)!r},
        {str(orphan_term_path)!r},
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
while not pathlib.Path({str(orphan_ready_path)!r}).is_file():
    time.sleep(0.01)
pathlib.Path({str(orphan_pid_path)!r}).write_text(str(orphan.pid), encoding="ascii")

class OrphanProcess:
    pid = orphan.pid

    @staticmethod
    def terminate():
        os.kill(orphan.pid, signal.SIGTERM)

    @staticmethod
    def kill():
        os.kill(orphan.pid, signal.SIGKILL)

def wait_procs(processes, timeout):
    deadline = time.monotonic() + timeout
    while True:
        alive = []
        for process in processes:
            try:
                waited_pid, _status = os.waitpid(process.pid, os.WNOHANG)
            except ChildProcessError:
                waited_pid = process.pid
            if waited_pid == 0:
                alive.append(process)
        if not alive:
            return processes, []
        if time.monotonic() >= deadline:
            return [], alive
        time.sleep(0.01)

real_psutil.wait_procs = wait_procs
sys.modules["psutil"] = real_psutil
runner._cache_processes = lambda _cache, _excluded: runner._CacheProcessScan(
    [OrphanProcess()],
    True,
)

class DeferredContainment:
    @staticmethod
    def close():
        return False

    @staticmethod
    def abort(deadline):
        return actual_containment.abort(deadline)

class ExitedProcess:
    pid = os.getpid()

    @staticmethod
    def poll():
        return 0

    @staticmethod
    def wait(timeout):
        del timeout
        return 0

try:
    with runner._CancellationSignalGuard():
        runner._shutdown_server(
            ExitedProcess(),
            13305,
            pathlib.Path({str(cache)!r}),
            DeferredContainment(),
        )
        runner._raise_if_cancelled()
except runner._ValidationCancelled as exc:
    raise SystemExit(128 + exc.signum) from None
"""
            helper = subprocess.Popen(
                [sys.executable, "-c", helper_program],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            orphan_pid = None
            contained_pid = None
            contained_descendant_pid = None
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    try:
                        orphan_pid = int(orphan_pid_path.read_text(encoding="ascii"))
                    except (FileNotFoundError, ValueError):
                        orphan_pid = None
                    try:
                        contained_pids = tuple(
                            int(value)
                            for value in contained_pids_path.read_text(
                                encoding="ascii"
                            ).split(",")
                        )
                    except (FileNotFoundError, ValueError):
                        contained_pids = ()
                    if len(contained_pids) == 2:
                        contained_pid, contained_descendant_pid = contained_pids
                    if (
                        orphan_pid is not None
                        and contained_pid is not None
                        and contained_descendant_pid is not None
                        and orphan_term_path.is_file()
                    ):
                        break
                    if helper.poll() is not None:
                        break
                    time.sleep(0.05)
                if (
                    orphan_pid is None
                    or contained_pid is None
                    or contained_descendant_pid is None
                    or not orphan_term_path.is_file()
                ):
                    stdout, stderr = helper.communicate(timeout=5)
                    self.fail(
                        "shutdown helper did not reach orphan cleanup: "
                        f"stdout={stdout!r}, stderr={stderr!r}"
                    )

                signal_started = time.monotonic()
                helper.send_signal(signal.SIGINT)
                try:
                    stdout, stderr = helper.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    self.fail("orphan cleanup ignored cancellation for 3 seconds")
                signal_elapsed = time.monotonic() - signal_started

                self.assertEqual(
                    helper.returncode,
                    128 + signal.SIGINT,
                    f"stdout={stdout!r}, stderr={stderr!r}",
                )
                self.assertLess(signal_elapsed, 3.0)
                self.assertFalse(
                    _pid_is_running(orphan_pid),
                    f"stdout={stdout!r}, stderr={stderr!r}",
                )
                self.assertFalse(
                    _pid_is_running(contained_pid),
                    f"stdout={stdout!r}, stderr={stderr!r}",
                )
                self.assertFalse(
                    _pid_is_running(contained_descendant_pid),
                    f"stdout={stdout!r}, stderr={stderr!r}",
                )
            finally:
                if helper.poll() is None:
                    helper.kill()
                if orphan_pid is not None and _pid_is_active(orphan_pid):
                    os.kill(orphan_pid, signal.SIGKILL)
                if contained_pid is not None:
                    try:
                        if (
                            os.getpgid(contained_pid) == contained_pid
                            and contained_pid != os.getpgrp()
                        ):
                            os.killpg(contained_pid, signal.SIGKILL)
                    except OSError:
                        pass
                try:
                    helper.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    helper.kill()
                    helper.wait(timeout=5)

    @unittest.skipIf(os.name == "nt", "POSIX process-group test")
    def test_shutdown_terminates_an_external_descendant_in_the_server_session(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_pid_path = root / "child.pid"
            containment = runner._PosixProcessGroupContainment(
                allow_untracked_descendants=sys.platform == "darwin"
            )
            parent = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, subprocess, sys; "
                        "child = subprocess.Popen(['/bin/sleep', '30']); "
                        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), "
                        "encoding='ascii')"
                    ),
                    str(child_pid_path),
                ],
                **containment.popen_kwargs(),
            )
            containment.assign(parent)
            parent.wait(timeout=5)
            child_pid = int(child_pid_path.read_text(encoding="ascii"))
            try:
                self.assertTrue(_pid_is_active(child_pid))
                with mock.patch.object(
                    runner,
                    "_cache_processes",
                    return_value=runner._CacheProcessScan([], True),
                ):
                    clean = runner._shutdown_server(
                        parent,
                        13305,
                        root,
                        containment,
                    )

                self.assertFalse(clean)
                deadline = time.monotonic() + 5
                while _pid_is_active(child_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(_pid_is_active(child_pid))
                self.assertFalse(containment.close())
            finally:
                if _pid_is_active(child_pid):
                    os.kill(child_pid, signal.SIGKILL)

    @unittest.skipIf(os.name == "nt", "POSIX process-group test")
    def test_validation_command_descendants_are_in_the_contained_session(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            child_pid_path = Path(directory) / "child.pid"
            containment = runner._PosixProcessGroupContainment(
                allow_untracked_descendants=sys.platform == "darwin"
            )
            exit_code = runner._run_contained_command(
                [
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, subprocess, sys; "
                        "child = subprocess.Popen(['/bin/sleep', '30']); "
                        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), "
                        "encoding='ascii')"
                    ),
                    str(child_pid_path),
                ],
                env=dict(os.environ),
                containment=containment,
                timeout_seconds=30,
            )
            child_pid = int(child_pid_path.read_text(encoding="ascii"))
            try:
                self.assertEqual(exit_code, 0)
                self.assertTrue(_pid_is_active(child_pid))
                self.assertFalse(containment.close())
                deadline = time.monotonic() + 5
                while _pid_is_active(child_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(_pid_is_active(child_pid))
            finally:
                if _pid_is_active(child_pid):
                    os.kill(child_pid, signal.SIGKILL)

    @unittest.skipIf(os.name == "nt", "POSIX descendant tracking test")
    @unittest.skipIf(
        not DARWIN_PROCESS_TREE_INSPECTION_AVAILABLE,
        "Darwin process-tree inspection is unavailable",
    )
    def test_close_kills_retained_session_escape_after_parent_exit(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child_pid_path = root / "child.pid"
            child_ready_path = root / "child-ready.txt"
            descendant_program = (
                "import os, pathlib, sys, time; "
                "os.setsid(); "
                "pathlib.Path(sys.argv[1]).write_text('ready', encoding='ascii'); "
                "time.sleep(30)"
            )
            parent_program = f"""
import pathlib
import subprocess
import sys
import time

child = subprocess.Popen(
    [sys.executable, "-c", {descendant_program!r}, sys.argv[2]],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
ready = pathlib.Path(sys.argv[2])
deadline = time.monotonic() + 5
while not ready.is_file() and time.monotonic() < deadline:
    time.sleep(0.01)
pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding="ascii")
time.sleep(0.5)
"""
            containment = runner._PosixProcessGroupContainment(
                allow_untracked_descendants=sys.platform == "darwin"
            )
            exit_code = runner._run_contained_command(
                [
                    sys.executable,
                    "-c",
                    parent_program,
                    str(child_pid_path),
                    str(child_ready_path),
                ],
                env=dict(os.environ),
                containment=containment,
                timeout_seconds=30,
            )
            child_pid = int(child_pid_path.read_text(encoding="ascii"))
            try:
                self.assertEqual(exit_code, 0)
                self.assertTrue(_pid_is_running(child_pid))
                self.assertFalse(containment.close())
                deadline = time.monotonic() + 5
                while _pid_is_running(child_pid) and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertFalse(_pid_is_running(child_pid))
            finally:
                if _pid_is_active(child_pid):
                    os.kill(child_pid, signal.SIGKILL)

    def test_windows_job_object_sets_kill_on_close_and_closes_once(self):
        runner = load_runner()
        events = []

        class FakeKernel32:
            @staticmethod
            def CreateJobObjectW(_attributes, _name):
                events.append("create")
                return 123

            @staticmethod
            def SetInformationJobObject(handle, info_class, pointer, size):
                information = ctypes.cast(
                    pointer,
                    ctypes.POINTER(runner._JobObjectExtendedLimitInformation),
                ).contents
                events.append(
                    (
                        "configure",
                        handle,
                        info_class,
                        information.basic_limit_information.limit_flags,
                        size,
                    )
                )
                return 1

            @staticmethod
            def AssignProcessToJobObject(handle, process_handle):
                events.append(("assign", handle, process_handle))
                return 1

            @staticmethod
            def QueryInformationJobObject(
                handle, info_class, pointer, size, _return_size
            ):
                information = ctypes.cast(
                    pointer,
                    ctypes.POINTER(runner._JobObjectBasicAccountingInformation),
                ).contents
                information.active_processes = 0
                events.append(("query", handle, info_class, size))
                return 1

            @staticmethod
            def CloseHandle(handle):
                events.append(("close", handle))
                return 1

        job = runner._WindowsKillOnCloseJob.create(FakeKernel32())
        process = argparse.Namespace(_handle=456)
        job.assign(process)

        self.assertTrue(job.abort(time.monotonic() + 4))
        self.assertTrue(job.close())
        self.assertEqual(events[0], "create")
        self.assertEqual(
            events[1][0:4],
            (
                "configure",
                123,
                9,
                0x00002000,
            ),
        )
        self.assertEqual(
            events[1][4], ctypes.sizeof(runner._JobObjectExtendedLimitInformation)
        )
        self.assertEqual(events[2], ("assign", 123, 456))
        self.assertEqual(
            events[3:],
            [
                (
                    "query",
                    123,
                    1,
                    ctypes.sizeof(runner._JobObjectBasicAccountingInformation),
                ),
                ("close", 123),
            ],
        )

    def test_windows_gate_assigns_before_fast_descendants_and_job_close_kills_tree(
        self,
    ):
        runner = load_runner()
        events = []
        assigned = False
        descendants = {"child": False, "grandchild": False}

        class FakeStartupInfo:
            pass

        class FakeProcess:
            _handle = 456
            pid = 654

        class FakeKernel32:
            @staticmethod
            def CreateEventW(_attributes, _manual_reset, _initial_state, _name):
                events.append("create-gate")
                return 789

            @staticmethod
            def SetHandleInformation(handle, mask, flags):
                events.append(("inherit", handle, mask, flags))
                return 1

            @staticmethod
            def AssignProcessToJobObject(handle, process_handle):
                nonlocal assigned
                events.append(("assign", handle, process_handle))
                assigned = True
                return 1

            @staticmethod
            def SetEvent(handle):
                events.append(("release", handle))
                if not assigned:
                    raise AssertionError(
                        "candidate gate released before Job assignment"
                    )
                descendants["child"] = True
                descendants["grandchild"] = True
                return 1

            @staticmethod
            def QueryInformationJobObject(
                _handle, _info_class, pointer, _size, _return_size
            ):
                information = ctypes.cast(
                    pointer,
                    ctypes.POINTER(runner._JobObjectBasicAccountingInformation),
                ).contents
                information.active_processes = sum(descendants.values()) + 1
                return 1

            @staticmethod
            def CloseHandle(handle):
                events.append(("close", handle))
                if handle == 123:
                    descendants["child"] = False
                    descendants["grandchild"] = False
                return 1

        def popen(command, **kwargs):
            events.append("popen-gate")
            self.assertEqual(
                command[0:5],
                [sys.executable, "-I", "-S", "-c", runner._WINDOWS_GATE_PROGRAM],
            )
            self.assertEqual(command[5:], ["789", "candidate", "spawn-now"])
            self.assertTrue(kwargs["close_fds"])
            self.assertEqual(
                kwargs["startupinfo"].lpAttributeList,
                {"handle_list": [789]},
            )
            self.assertFalse(any(descendants.values()))
            return FakeProcess()

        job = runner._WindowsKillOnCloseJob(FakeKernel32(), 123)
        with (
            mock.patch.object(
                runner.subprocess,
                "STARTUPINFO",
                FakeStartupInfo,
                create=True,
            ),
            mock.patch.object(runner.subprocess, "Popen", side_effect=popen),
        ):
            process = runner._start_contained_process(
                job,
                ["candidate", "spawn-now"],
                env={},
            )

        self.assertEqual(process.pid, 654)
        self.assertTrue(all(descendants.values()))
        self.assertLess(
            events.index(("assign", 123, 456)), events.index(("release", 789))
        )
        self.assertLess(
            events.index(("inherit", 789, 1, 0)), events.index(("release", 789))
        )
        self.assertFalse(job.close())
        self.assertFalse(any(descendants.values()))

    def test_windows_gate_assignment_failure_kills_wrapper_and_closes_raw_handles(
        self,
    ):
        runner = load_runner()
        events = []

        class FakeStartupInfo:
            pass

        class FakeProcess:
            _handle = 456
            pid = 654

            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def kill(self):
                events.append("kill-wrapper")
                self.returncode = 1

            def wait(self, timeout):
                events.append(("wait-wrapper", timeout))
                return self.returncode

        class FakeKernel32:
            @staticmethod
            def CreateEventW(_attributes, _manual_reset, _initial_state, _name):
                return 789

            @staticmethod
            def SetHandleInformation(_handle, _mask, _flags):
                return 1

            @staticmethod
            def AssignProcessToJobObject(_handle, _process_handle):
                events.append("assign-failed")
                return 0

            @staticmethod
            def SetEvent(_handle):
                events.append("released")
                return 1

            @staticmethod
            def QueryInformationJobObject(
                _handle, _info_class, pointer, _size, _return_size
            ):
                information = ctypes.cast(
                    pointer,
                    ctypes.POINTER(runner._JobObjectBasicAccountingInformation),
                ).contents
                information.active_processes = 0
                return 1

            @staticmethod
            def CloseHandle(handle):
                events.append(("close", handle))
                return 1

        process = FakeProcess()
        job = runner._WindowsKillOnCloseJob(FakeKernel32(), 123)
        with (
            mock.patch.object(
                runner.subprocess,
                "STARTUPINFO",
                FakeStartupInfo,
                create=True,
            ),
            mock.patch.object(runner.subprocess, "Popen", return_value=process),
            mock.patch.object(runner, "_windows_last_error", return_value=5),
            self.assertRaisesRegex(OSError, "AssignProcessToJobObject"),
        ):
            job.start_process(["candidate"], env={})

        self.assertNotIn("released", events)
        self.assertIn("kill-wrapper", events)
        self.assertIn(("close", 123), events)
        self.assertIn(("close", 789), events)

    @unittest.skipUnless(os.name == "nt", "Windows Job Object integration test")
    def test_windows_job_close_kills_real_gated_child_and_grandchild(self):
        runner = load_runner()
        grandchild_program = "import time; time.sleep(60)"
        child_program = f"""
import os
import pathlib
import subprocess
import sys
import time

grandchild = subprocess.Popen([sys.executable, "-c", {grandchild_program!r}])
pathlib.Path(sys.argv[1]).write_text(
    f"{{os.getpid()}},{{grandchild.pid}}", encoding="ascii"
)
time.sleep(60)
"""
        candidate_program = f"""
import subprocess
import sys
import time

subprocess.Popen([sys.executable, "-c", {child_program!r}, sys.argv[1]])
time.sleep(60)
"""

        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "descendants.txt"
            job = runner._WindowsKillOnCloseJob.create()
            process = None
            descendant_pids = []
            try:
                process = job.start_process(
                    [sys.executable, "-c", candidate_program, str(pid_path)],
                    env=dict(os.environ),
                )
                deadline = time.monotonic() + 10
                while not pid_path.is_file() and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(pid_path.is_file())
                descendant_pids = [
                    int(value)
                    for value in pid_path.read_text(encoding="ascii").split(",")
                ]
                self.assertTrue(all(_pid_is_running(pid) for pid in descendant_pids))
                self.assertFalse(job.close())
                deadline = time.monotonic() + 5
                while (
                    any(_pid_is_running(pid) for pid in descendant_pids)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
                self.assertFalse(any(_pid_is_running(pid) for pid in descendant_pids))
            finally:
                job.close()
                if process is not None:
                    runner._reap_direct_process(process, time.monotonic() + 5)
                try:
                    import psutil
                except ModuleNotFoundError:
                    psutil = None
                if psutil is not None:
                    for pid in descendant_pids:
                        try:
                            psutil.Process(pid).kill()
                        except psutil.Error:
                            pass

    def test_windows_job_query_failure_closes_the_job_and_fails_closed(self):
        runner = load_runner()
        closed_handles = []

        class FakeKernel32:
            @staticmethod
            def QueryInformationJobObject(
                _handle, _info_class, _pointer, _size, _return_size
            ):
                return 0

            @staticmethod
            def CloseHandle(handle):
                closed_handles.append(handle)
                return 1

        job = runner._WindowsKillOnCloseJob(FakeKernel32(), 123)

        self.assertFalse(job.close())
        self.assertFalse(job.close())
        self.assertEqual(closed_handles, [123])

    def test_windows_job_configuration_failure_closes_the_raw_handle(self):
        runner = load_runner()
        closed_handles = []

        class FakeKernel32:
            @staticmethod
            def CreateJobObjectW(_attributes, _name):
                return 123

            @staticmethod
            def SetInformationJobObject(_handle, _info_class, _pointer, _size):
                return 0

            @staticmethod
            def CloseHandle(handle):
                closed_handles.append(handle)
                return 1

        with self.assertRaisesRegex(OSError, "SetInformationJobObject"):
            runner._WindowsKillOnCloseJob.create(FakeKernel32())

        self.assertEqual(closed_handles, [123])

    def test_server_environment_inherits_only_minimal_runtime_values(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            defaults = Path(directory) / "defaults.json"
            defaults.write_text("{}", encoding="utf-8")
            environment = runner.build_server_environment(
                Path(directory),
                inherited={
                    "LEMONADE_API_KEY": "user-key",
                    "LEMONADE_ADMIN_API_KEY": "admin-key",
                    "LEMONADE_DEFAULTS_PATH": "/host/defaults.json",
                    "LEMONADE_LLAMACPP_VULKAN_BIN": "/host/llama-server",
                    "lemonade_rocm_install_method": "wheel",
                    "LEMONADE_BACKEND_WATCHDOG": "0",
                    "lemonade_backend_watchdog_poll_seconds": "999",
                    "LEMONADE_CACHE_DIR": "/host/lemonade-cache",
                    "LEMONADE_GGML_HIP_PATH": "/host/libggml-hip.so",
                    "HF_HOME": "/host/huggingface",
                    "HF_TOKEN": "download-token",
                    "hugging_face_hub_token": "legacy-download-token",
                    "HUGGINGFACE_TOKEN": "alternate-download-token",
                    "LEMONADE_ALLOWED_ORIGINS": "https://example.invalid",
                    "rocm_path": "/host/rocm",
                    "HF_ENDPOINT": "https://example.invalid",
                    "HTTP_PROXY": "http://proxy.invalid:3128",
                    "https_proxy": "http://proxy.invalid:3128",
                    "ALL_PROXY": "socks5://proxy.invalid:1080",
                    "NO_PROXY": "example.invalid",
                    "LLAMA_ARG_MODEL": "/host/model.gguf",
                    "LLAMA_ARG_N_GPU_LAYERS": "999",
                    "llama_arg_ctx_size": "1",
                    "lemonade_ci_mode": "false",
                    "PATH": "/trusted/bin",
                    "LANG": "en_US.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "LC_CTYPE": "UTF-8",
                    "TZ": "UTC",
                    "SSL_CERT_FILE": "/trusted/ca.pem",
                    "SSL_CERT_DIR": "/trusted/ca",
                    "CURL_CA_BUNDLE": "/trusted/curl-ca.pem",
                    "REQUESTS_CA_BUNDLE": "/trusted/requests-ca.pem",
                    "SystemRoot": "C:\\Windows",
                    "windir": "C:\\Windows",
                    "COMSPEC": "C:\\Windows\\System32\\cmd.exe",
                    "PATHEXT": ".COM;.EXE;.BAT;.CMD",
                    "SYSTEMDRIVE": "C:",
                    "PROCESSOR_ARCHITECTURE": "AMD64",
                    "NUMBER_OF_PROCESSORS": "8",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "NVIDIA_VISIBLE_DEVICES": "GPU-123",
                    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
                    "__NV_PRIME_RENDER_OFFLOAD": "1",
                    "HIP_VISIBLE_DEVICES": "0",
                    "ROCR_VISIBLE_DEVICES": "0",
                    "GPU_DEVICE_ORDINAL": "0",
                    "HSA_OVERRIDE_GFX_VERSION": "11.5.1",
                    "VK_ICD_FILENAMES": "/trusted/vulkan/icd.json",
                    "VK_DRIVER_FILES": "/trusted/vulkan/driver.json",
                    "VK_ADD_DRIVER_FILES": "/trusted/vulkan/additional.json",
                    "GGML_METAL_NO_RESIDENCY": "1",
                    "HOME": "/runner/profile",
                    "USERPROFILE": "C:\\Users\\runner",
                    "APPDATA": "C:\\Users\\runner\\AppData\\Roaming",
                    "LOCALAPPDATA": "C:\\Users\\runner\\AppData\\Local",
                    "GITHUB_OUTPUT": "/runner/command/output",
                    "GITHUB_ENV": "/runner/command/environment",
                    "GITHUB_PATH": "/runner/command/path",
                    "GITHUB_STEP_SUMMARY": "/runner/command/summary",
                    "ACTIONS_RUNTIME_TOKEN": "actions-token",
                    "LD_PRELOAD": "/attacker/inject.so",
                    "DYLD_INSERT_LIBRARIES": "/attacker/inject.dylib",
                    "DYLD_LIBRARY_PATH": "/attacker/dylibs",
                    "PYTHONPATH": "/attacker/python",
                    "PYTHONSTARTUP": "/attacker/startup.py",
                    "BASH_ENV": "/attacker/bashrc",
                    "UNKNOWN_DEPLOYMENT_SECRET": "secret",
                    "PRESERVED": "value",
                },
                private_temp_directory=Path(directory) / "private-temp",
                candidate_defaults_path=defaults,
            )

        expected_private_temp = str((Path(directory) / "private-temp").resolve())
        self.assertNotIn("LEMONADE_API_KEY", environment)
        self.assertNotIn("LEMONADE_ADMIN_API_KEY", environment)
        self.assertNotIn("LEMONADE_LLAMACPP_VULKAN_BIN", environment)
        self.assertNotIn("lemonade_rocm_install_method", environment)
        self.assertNotIn("LEMONADE_BACKEND_WATCHDOG", environment)
        self.assertNotIn("lemonade_backend_watchdog_poll_seconds", environment)
        self.assertNotIn("LEMONADE_CACHE_DIR", environment)
        self.assertNotIn("LEMONADE_GGML_HIP_PATH", environment)
        self.assertNotIn("HF_HOME", environment)
        self.assertFalse(
            {
                "HF_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
                "HUGGINGFACE_TOKEN",
            }
            & {key.upper() for key in environment}
        )
        self.assertNotIn("LEMONADE_ALLOWED_ORIGINS", environment)
        self.assertNotIn("rocm_path", environment)
        self.assertNotIn("HF_ENDPOINT", environment)
        self.assertEqual(
            {
                key.upper(): value
                for key, value in environment.items()
                if key.upper().startswith("LLAMA_ARG_")
            },
            {"LLAMA_ARG_LOG_VERBOSITY": "4"},
        )
        self.assertNotIn("HTTP_PROXY", environment)
        self.assertNotIn("https_proxy", environment)
        self.assertNotIn("ALL_PROXY", environment)
        self.assertEqual(environment["NO_PROXY"], "127.0.0.1")
        self.assertEqual(environment["no_proxy"], "127.0.0.1")
        self.assertEqual(environment["TMPDIR"], expected_private_temp)
        self.assertEqual(environment["TMP"], expected_private_temp)
        self.assertEqual(environment["TEMP"], expected_private_temp)
        self.assertEqual(environment["LEMONADE_DEFAULTS_PATH"], str(defaults.resolve()))
        self.assertEqual(environment["LEMONADE_CI_MODE"], "True")
        self.assertNotIn("lemonade_ci_mode", environment)
        self.assertEqual(environment["PATH"], "/trusted/bin")
        for key, value in {
            "LANG": "en_US.UTF-8",
            "LC_ALL": "C.UTF-8",
            "LC_CTYPE": "UTF-8",
            "TZ": "UTC",
            "SSL_CERT_FILE": "/trusted/ca.pem",
            "SSL_CERT_DIR": "/trusted/ca",
            "CURL_CA_BUNDLE": "/trusted/curl-ca.pem",
            "REQUESTS_CA_BUNDLE": "/trusted/requests-ca.pem",
            "COMSPEC": "C:\\Windows\\System32\\cmd.exe",
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
            "SYSTEMDRIVE": "C:",
            "PROCESSOR_ARCHITECTURE": "AMD64",
            "NUMBER_OF_PROCESSORS": "8",
            "CUDA_VISIBLE_DEVICES": "0",
            "NVIDIA_VISIBLE_DEVICES": "GPU-123",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "__NV_PRIME_RENDER_OFFLOAD": "1",
            "HIP_VISIBLE_DEVICES": "0",
            "ROCR_VISIBLE_DEVICES": "0",
            "GPU_DEVICE_ORDINAL": "0",
            "HSA_OVERRIDE_GFX_VERSION": "11.5.1",
            "VK_ICD_FILENAMES": "/trusted/vulkan/icd.json",
            "VK_DRIVER_FILES": "/trusted/vulkan/driver.json",
            "VK_ADD_DRIVER_FILES": "/trusted/vulkan/additional.json",
            "GGML_METAL_NO_RESIDENCY": "1",
        }.items():
            self.assertEqual(environment.get(key), value)
        self.assertEqual(environment.get("SYSTEMROOT"), "C:\\Windows")
        self.assertEqual(environment.get("WINDIR"), "C:\\Windows")
        for key in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
            self.assertEqual(environment[key], expected_private_temp)
        for key in (
            "ACTIONS_RUNTIME_TOKEN",
            "BASH_ENV",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
            "GITHUB_ENV",
            "GITHUB_OUTPUT",
            "GITHUB_PATH",
            "GITHUB_STEP_SUMMARY",
            "LD_PRELOAD",
            "PRESERVED",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "UNKNOWN_DEPLOYMENT_SECRET",
        ):
            self.assertNotIn(key, environment)

    def test_server_environment_maps_only_explicit_canonical_download_token(self):
        runner = load_runner()
        inherited = {
            "HF_TOKEN": "download-token",
            "hf_token": "ambient-case-variant",
            "hugging_face_hub_token": "legacy-download-token",
            "HuGgInGfAcE_ToKeN": "alternate-download-token",
        }

        with tempfile.TemporaryDirectory() as directory:
            environment = runner.build_server_environment(
                Path(directory),
                inherited=inherited,
                private_temp_directory=Path(directory) / "private-temp",
                allow_download_credentials=True,
            )

        self.assertEqual(
            {
                key.upper(): value
                for key, value in environment.items()
                if key.upper()
                in {
                    "HF_TOKEN",
                    "HUGGING_FACE_HUB_TOKEN",
                    "HUGGINGFACE_TOKEN",
                }
            },
            {"HF_TOKEN": "download-token"},
        )

    def test_candidate_defaults_must_be_a_real_adjacent_build_resource(self):
        runner = load_runner()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build"
            outside = root / "outside-resources"
            build.mkdir()
            outside.mkdir()
            (outside / "defaults.json").write_text("{}", encoding="utf-8")
            (build / "lemond").touch()
            try:
                (build / "resources").symlink_to(
                    outside,
                    target_is_directory=True,
                )
            except OSError as exc:
                self.skipTest(f"directory symlinks are unavailable: {exc}")

            with self.assertRaisesRegex(RuntimeError, "build resource"):
                runner.resolve_candidate_defaults(build / "lemond")


if __name__ == "__main__":
    unittest.main()
