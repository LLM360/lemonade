#!/usr/bin/env python3
"""Source contracts for subprocess handle inheritance."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_PROCESS_SOURCE = (
    ROOT / "src" / "cpp" / "server" / "utils" / "platform" / "process_windows.cpp"
)
WINDOWS_PROCESS_TEST_SOURCE = ROOT / "test" / "cpp" / "test_process_manager_windows.cpp"
CMAKE_SOURCE = ROOT / "CMakeLists.txt"
CPP_WORKFLOW_SOURCE = (
    ROOT / ".github" / "workflows" / "cpp_server_build_test_release.yml"
)
LINUX_PROCESS_SOURCE = (
    ROOT / "src" / "cpp" / "server" / "utils" / "platform" / "process_linux.cpp"
)
MACOS_PROCESS_SOURCE = (
    ROOT / "src" / "cpp" / "server" / "utils" / "platform" / "process_macos.cpp"
)
MCP_CLIENT_SOURCE = ROOT / "src" / "cpp" / "server" / "mcp_client.cpp"
SYSTEM_INFO_SOURCE = ROOT / "src" / "cpp" / "server" / "system_info.cpp"


class ProcessInheritanceContractTests(unittest.TestCase):
    def test_windows_process_creation_uses_one_restricted_handle_gateway(self):
        source = WINDOWS_PROCESS_SOURCE.read_text(encoding="utf-8")

        self.assertEqual(source.count("CreateProcessA("), 1)
        self.assertEqual(
            source.count("create_process_with_restricted_handles("),
            4,
        )
        for required in (
            "STARTUPINFOEXA",
            "PROC_THREAD_ATTRIBUTE_HANDLE_LIST",
            "EXTENDED_STARTUPINFO_PRESENT",
            "UpdateProcThreadAttribute",
            "DuplicateHandle",
            "DUPLICATE_SAME_ACCESS",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_windows_gateway_normalizes_missing_standard_handles(self):
        source = WINDOWS_PROCESS_SOURCE.read_text(encoding="utf-8")
        helper_start = source.index(
            "static BOOL create_process_with_restricted_handles("
        )
        helper_end = source.index("\n}\n\n// Helper function", helper_start) + 2
        helper = source[helper_start:helper_end]

        for required in (
            "STARTF_USESTDHANDLES",
            "fallback_handles",
            '"NUL", access',
            "GENERIC_READ",
            "GENERIC_WRITE",
            "GetFileType(source_handles[i])",
            "source_handles[i] = fallback",
            "CloseHandle(fallback_handles[i])",
        ):
            with self.subTest(required=required):
                self.assertIn(required, helper)

        self.assertLess(helper.index('"NUL", access'), helper.index("DuplicateHandle("))

    def test_windows_missing_standard_handle_runtime_test_is_wired_into_ci(self):
        runtime_test = WINDOWS_PROCESS_TEST_SOURCE.read_text(encoding="utf-8")
        cmake_source = CMAKE_SOURCE.read_text(encoding="utf-8")
        workflow = CPP_WORKFLOW_SOURCE.read_text(encoding="utf-8")

        for required in (
            "InvalidStandardHandles",
            "INVALID_HANDLE_VALUE",
            "stale_value",
            "GetFileType(handle)",
            "--probe-stdio",
            "quiet child receives valid standard handles",
            "captured child receives valid standard handles",
        ):
            with self.subTest(required=required):
                self.assertIn(required, runtime_test)

        self.assertIn("BUILD_TESTING AND WIN32", cmake_source)
        self.assertIn("add_cpp_ci_test(ProcessManagerWindowsTest CI ON", cmake_source)
        self.assertIn(
            "--target test_mcp_client_config test_process_manager_windows", workflow
        )
        self.assertIn("mcp_client_config|ProcessManagerWindowsTest", workflow)

    def test_other_windows_process_creation_cannot_inherit_unlisted_handles(self):
        mcp_source = MCP_CLIENT_SOURCE.read_text(encoding="utf-8")
        system_info_source = SYSTEM_INFO_SOURCE.read_text(encoding="utf-8")

        create_process_sites = {}
        for source_path in (ROOT / "src" / "cpp").rglob("*.cpp"):
            source = source_path.read_text(encoding="utf-8")
            call_count = source.count("CreateProcessA(") + source.count(
                "CreateProcessW("
            )
            if call_count:
                create_process_sites[source_path.relative_to(ROOT).as_posix()] = (
                    call_count
                )
        self.assertEqual(
            create_process_sites,
            {
                "src/cpp/server/mcp_client.cpp": 1,
                "src/cpp/server/system_info.cpp": 1,
                "src/cpp/server/utils/platform/process_windows.cpp": 1,
            },
        )

        self.assertEqual(mcp_source.count("CreateProcessW("), 1)
        for required in (
            "STARTUPINFOEXW",
            "PROC_THREAD_ATTRIBUTE_HANDLE_LIST",
            "EXTENDED_STARTUPINFO_PRESENT",
        ):
            with self.subTest(required=required):
                self.assertIn(required, mcp_source)

        self.assertEqual(system_info_source.count("CreateProcessA("), 1)
        self.assertIn(
            "nullptr, nullptr, FALSE, CREATE_NO_WINDOW",
            system_info_source,
        )

    def test_linux_capture_pipes_are_atomically_close_on_exec(self):
        source = LINUX_PROCESS_SOURCE.read_text(encoding="utf-8")
        system_info_source = SYSTEM_INFO_SOURCE.read_text(encoding="utf-8")
        helper_start = source.index("static bool create_pipe_above_standard_streams")
        helper_end = source.index("\n}\n", helper_start) + 2
        helper = source[helper_start:helper_end]

        self.assertIn("pipe2(pipe_fds, O_CLOEXEC)", helper)
        self.assertNotRegex(helper, r"(?<![A-Za-z0-9_])pipe\(pipe_fds\)")

        linux_section_start = system_info_source.index(
            "#ifdef __linux__", system_info_source.index("// Linux implementation")
        )
        linux_section_end = system_info_source.index(
            "#endif // __linux__", linux_section_start
        )
        linux_system_info = system_info_source[linux_section_start:linux_section_end]
        self.assertEqual(source.count('popen(command.c_str(), "re")'), 1)
        self.assertEqual(linux_system_info.count(', "re")'), 5)
        self.assertNotRegex(source, r'popen\([^;\n]+,\s*"r"\)')
        self.assertNotRegex(linux_system_info, r'popen\([^;\n]+,\s*"r"\)')

    def test_macos_server_launches_cannot_use_plain_fork(self):
        process_source = MACOS_PROCESS_SOURCE.read_text(encoding="utf-8")
        mcp_source = MCP_CLIENT_SOURCE.read_text(encoding="utf-8")
        system_info_source = SYSTEM_INFO_SOURCE.read_text(encoding="utf-8")

        fork_sites = {}
        fork_call = re.compile(r"(?<![A-Za-z0-9_])(?:::)?fork\(\);")
        for source_path in (ROOT / "src" / "cpp" / "server").rglob("*.cpp"):
            call_count = len(fork_call.findall(source_path.read_text(encoding="utf-8")))
            if call_count:
                fork_sites[source_path.relative_to(ROOT).as_posix()] = call_count
        self.assertEqual(
            fork_sites,
            {
                "src/cpp/server/mcp_client.cpp": 1,
                "src/cpp/server/utils/platform/process_linux.cpp": 2,
            },
        )

        popen_sites = {}
        popen_call = re.compile(r"(?<![A-Za-z0-9_])popen\(")
        for source_path in (ROOT / "src" / "cpp" / "server").rglob("*.cpp"):
            call_count = len(
                popen_call.findall(source_path.read_text(encoding="utf-8"))
            )
            if call_count:
                popen_sites[source_path.relative_to(ROOT).as_posix()] = call_count
        self.assertEqual(
            popen_sites,
            {
                "src/cpp/server/system_info.cpp": 5,
                "src/cpp/server/utils/platform/process_linux.cpp": 1,
            },
        )
        linux_section_start = system_info_source.index(
            "#ifdef __linux__", system_info_source.index("// Linux implementation")
        )
        linux_section_end = system_info_source.index(
            "#endif // __linux__", linux_section_start
        )
        mac_compiled_system_info = (
            system_info_source[:linux_section_start]
            + system_info_source[linux_section_end:]
        )
        self.assertNotIn("popen(", mac_compiled_system_info)

        pipe_helper_start = process_source.index(
            "static bool create_pipe_above_standard_streams"
        )
        pipe_helper_end = process_source.index("\n}\n", pipe_helper_start) + 2
        pipe_helper = process_source[pipe_helper_start:pipe_helper_end]
        self.assertLess(
            pipe_helper.index("pipe(pipe_fds)"),
            pipe_helper.index("LEMONADE_PROCESS_TEST_HOOK"),
        )
        self.assertLess(
            pipe_helper.index("LEMONADE_PROCESS_TEST_HOOK"),
            pipe_helper.index("fcntl(pipe_fds[i], F_DUPFD_CLOEXEC"),
        )
        self.assertIn("set_close_on_exec(pipe_fds[i])", pipe_helper)
        self.assertNotIn("popen(", process_source)
        self.assertIn("posix_spawn_file_actions_addinherit_np", process_source)

        run_with_output_start = process_source.index(
            "int MacOSProcessPlatform::run_with_output("
        )
        run_with_output_end = process_source.index(
            "int MacOSProcessPlatform::find_free_port", run_with_output_start
        )
        run_with_output = process_source[run_with_output_start:run_with_output_end]
        capture_spawn_start = process_source.index(
            "static pid_t spawn_capturing_output"
        )
        capture_spawn_end = process_source.index(
            "static void read_process_output", capture_spawn_start
        )
        capture_spawn = process_source[capture_spawn_start:capture_spawn_end]
        for required in (
            "file_actions.add_inherit_if_open(STDIN_FILENO)",
            "file_actions.add_inherit_if_open(STDERR_FILENO)",
            "file_actions.add_dup2(output_pipe[1], STDOUT_FILENO)",
            "file_actions.add_dup2(output_pipe[1], STDERR_FILENO)",
            "file_actions.add_close(output_pipe[0])",
            "file_actions.add_close(output_pipe[1])",
            "posix_spawnp(",
            "POSIX_SPAWN_CLOEXEC_DEFAULT",
        ):
            with self.subTest(required=required):
                self.assertIn(required, capture_spawn)
        self.assertIn("spawn_capturing_output(", run_with_output)
        self.assertNotRegex(run_with_output, fork_call)

        run_command_start = process_source.index(
            "int MacOSProcessPlatform::run_command("
        )
        run_command_end = process_source.index(
            "std::unique_ptr<ProcessPlatform>", run_command_start
        )
        run_command = process_source[run_command_start:run_command_end]
        self.assertIn("spawn_capturing_output(", run_command)
        self.assertNotIn("popen(", run_command)
        reserved_options = run_command.index("int wait_options = WEXITED | WNOWAIT;")
        reserved_exit = run_command.index("waitid(", reserved_options)
        group_cleanup = run_command.index(
            "kill_process_group(pid, false);", reserved_exit
        )
        final_reap = run_command.index("waitpid(pid, &status, 0)", group_cleanup)
        self.assertNotIn("waitpid(pid, &status, WNOHANG)", run_command)
        self.assertLess(reserved_options, reserved_exit)
        self.assertLess(reserved_exit, group_cleanup)
        self.assertLess(group_cleanup, final_reap)

        spawn_start = process_source.index("ProcessHandle MacOSProcessPlatform::spawn(")
        spawn_end = process_source.index(
            "void MacOSProcessPlatform::terminate(", spawn_start
        )
        spawn = process_source[spawn_start:spawn_end]
        for standard_stream in ("STDIN_FILENO", "STDOUT_FILENO", "STDERR_FILENO"):
            with self.subTest(standard_stream=standard_stream):
                self.assertIn(
                    f"file_actions.add_inherit_if_open({standard_stream})", spawn
                )

        apple_branch_start = mcp_source.index(
            "#ifdef __APPLE__", mcp_source.index("if (pipe_creation_failed)")
        )
        apple_branch_end = mcp_source.index("#else", apple_branch_start)
        apple_branch = mcp_source[apple_branch_start:apple_branch_end]
        self.assertNotIn("::fork()", apple_branch)
        for required in (
            "posix_spawn(",
            "posix_spawn_file_actions_adddup2",
            "posix_spawn_file_actions_addclose",
            "posix_spawn_file_actions_addchdir_np",
            "posix_spawnattr_setpgroup(&attributes, 0)",
            "POSIX_SPAWN_CLOEXEC_DEFAULT",
            "POSIX_SPAWN_SETPGROUP",
        ):
            with self.subTest(required=required):
                self.assertIn(required, apple_branch)


if __name__ == "__main__":
    unittest.main()
