#!/usr/bin/env python3
"""Run one llama.cpp validation lane on Windows, Linux, or macOS."""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test.utils.llamacpp_validation_artifacts import (
    VerifiedValidationArtifact,
    verify_validation_artifacts,
)
from test.utils.llamacpp_validation_evidence import (
    ValidationEvidenceError,
    load_and_validate_result_file,
)
from test.utils.validation_model_catalog import load_model_catalog

K2_HORIZON_PROFILE = "k2-horizon-v1"
TARGET_PATTERN = re.compile(r"^[a-z0-9-]+$")
INFERENCE_PROCESS_NAMES = {
    "flm",
    "lemond",
    "lemonade",
    "lemonadeserver",
    "llama-server",
    "moonshine-server",
    "ort-server",
}
VALIDATION_MODEL_CATALOG = ROOT / "test/fixtures/llamacpp_validation_models.json"
VALIDATION_ARTIFACT_LOCKS = ROOT / "test/fixtures/llamacpp_validation_artifacts.json"
INHERITED_RUNTIME_ENVIRONMENT_KEYS = {
    "COMSPEC",
    "CUDA_VISIBLE_DEVICES",
    "CURL_CA_BUNDLE",
    "GGML_METAL_NO_RESIDENCY",
    "GPU_DEVICE_ORDINAL",
    "HIP_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "NUMBER_OF_PROCESSORS",
    "PATH",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "REQUESTS_CA_BUNDLE",
    "ROCR_VISIBLE_DEVICES",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TZ",
    "VK_ADD_DRIVER_FILES",
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "WINDIR",
    "__NV_PRIME_RENDER_OFFLOAD",
}
LOG_PREFIX = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \[[A-Za-z]+\]"
CHILD_LOG_PREFIX = r"(?:\[\d{1,10}\]\s+)?\d{1,6}\.\d{2}\.\d{3}\.\d{3}\s+[A-Z]\s+"
LOADING_MODEL_MARKER = re.compile(
    LOG_PREFIX + r" \(LlamaCpp\) Loading model:\s+(?P<model>\S+)\s*$"
)
BACKEND_MARKER = re.compile(
    LOG_PREFIX + r" \(LlamaCpp\) Using LlamaCpp Backend:\s+(?P<backend>\S+)\s*$"
)
GGUF_MARKER = re.compile(
    LOG_PREFIX + r" \(LlamaCpp\) Using GGUF:\s+(?P<path>.+?\S)\s*$"
)
LOADED_MODEL_MARKER = re.compile(
    LOG_PREFIX
    + r" \(Process\) "
    + CHILD_LOG_PREFIX
    + r"srv\s+llama_server:\s+model loaded\s*$"
)
DEVICE_MARKER = re.compile(
    LOG_PREFIX
    + r" \(Process\) "
    + CHILD_LOG_PREFIX
    + r"llama_prepare_model_devices:\s+using device\s+(?P<device>\S+).*$"
)
BACKEND_PREFIX = "Using LlamaCpp Backend:"
DEVICE_PREFIX = "llama_prepare_model_devices: using device"
OFFLOAD_MARKER = re.compile(
    LOG_PREFIX
    + r" \(Process\) "
    + CHILD_LOG_PREFIX
    + r"load_tensors:\s+offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU\s*$"
)
CPU_BUFFER_MARKER = re.compile(
    LOG_PREFIX
    + r" \(Process\) "
    + CHILD_LOG_PREFIX
    + r"load_tensors:\s+CPU(?:_Mapped)?\s+model buffer size\s*=\s*"
    + r"(?P<size>[0-9]+(?:\.[0-9]+)?)\s+MiB\s*$"
)
OFFLOAD_PREFIX = "load_tensors: offloaded"
MAX_ATTESTATION_LOG_BYTES = 512 * 1024 * 1024
MAX_ATTESTATION_LINE_BYTES = 1024 * 1024
VALIDATION_PHASE_TIMEOUT_SECONDS = 4 * 60 * 60
PROCESS_WAIT_POLL_SECONDS = 0.2
EXCEPTIONAL_CLEANUP_TIMEOUT_SECONDS = 4.0
EXCEPTIONAL_TERM_GRACE_SECONDS = 2.0
DARWIN_EPHEMERAL_RUNNER_CONTEXT_KEY = "LEMONADE_PROTECTED_RUNNER_CONTEXT"
DARWIN_EPHEMERAL_RUNNER_CONTEXT = "github-hosted:macos-latest:macos-metal"
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37
_POSIX_GATE_PROGRAM = """
import os
import sys

gate = int(sys.argv[1])
command = sys.argv[2:]
released = os.read(gate, 1)
os.close(gate)
if released != b"1" or not command:
    raise SystemExit(125)
os.execvpe(command[0], command, os.environ)
"""
_WINDOWS_GATE_PROGRAM = """
import ctypes
import subprocess
import sys

gate = int(sys.argv[1])
command = sys.argv[2:]
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
kernel32.WaitForSingleObject.restype = ctypes.c_uint32
kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
kernel32.CloseHandle.restype = ctypes.c_int
wait_result = kernel32.WaitForSingleObject(gate, 0xFFFFFFFF)
closed = kernel32.CloseHandle(gate)
if wait_result != 0 or not closed or not command:
    raise SystemExit(125)
raise SystemExit(subprocess.call(command))
"""
ACCELERATOR_DEVICE_PREFIXES = {
    "cuda": "CUDA",
    "metal": "MTL",
    "rocm": "ROCm",
    "vulkan": "Vulkan",
}
ATTESTED_BACKENDS = {
    "cpu",
    "cuda",
    "metal",
    "rocm-nightly",
    "rocm-stable",
    "vulkan",
}
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_HANDLE_FLAG_INHERIT = 0x00000001
_LINUX_SUBREAPER_LOCK = threading.Lock()
_LINUX_SUBREAPER_LEASES = 0
_LINUX_SUBREAPER_ORIGINAL_STATE: bool | None = None


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = (
        ("per_process_user_time_limit", ctypes.c_longlong),
        ("per_job_user_time_limit", ctypes.c_longlong),
        ("limit_flags", ctypes.c_uint32),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority_class", ctypes.c_uint32),
        ("scheduling_class", ctypes.c_uint32),
    )


class _JobObjectBasicAccountingInformation(ctypes.Structure):
    _fields_ = (
        ("total_user_time", ctypes.c_longlong),
        ("total_kernel_time", ctypes.c_longlong),
        ("this_period_total_user_time", ctypes.c_longlong),
        ("this_period_total_kernel_time", ctypes.c_longlong),
        ("total_page_fault_count", ctypes.c_uint32),
        ("total_processes", ctypes.c_uint32),
        ("active_processes", ctypes.c_uint32),
        ("total_terminated_processes", ctypes.c_uint32),
    )


class _IoCounters(ctypes.Structure):
    _fields_ = (
        ("read_operation_count", ctypes.c_ulonglong),
        ("write_operation_count", ctypes.c_ulonglong),
        ("other_operation_count", ctypes.c_ulonglong),
        ("read_transfer_count", ctypes.c_ulonglong),
        ("write_transfer_count", ctypes.c_ulonglong),
        ("other_transfer_count", ctypes.c_ulonglong),
    )


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = (
        ("basic_limit_information", _JobObjectBasicLimitInformation),
        ("io_info", _IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    )


class _DirectoryIdentity(NamedTuple):
    path: Path
    resolved_path: Path
    device: int
    inode: int


class _FileIdentity(NamedTuple):
    path: Path
    resolved_path: Path
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


class ValidationCache(NamedTuple):
    temporary_root: _DirectoryIdentity
    validation_root: _DirectoryIdentity
    cache_directory: _DirectoryIdentity
    config_directory: _DirectoryIdentity
    hf_cache_directory: _DirectoryIdentity
    private_temp_directory: _DirectoryIdentity
    results_directory: _DirectoryIdentity
    logs_directory: _DirectoryIdentity


class _CacheProcessScan(NamedTuple):
    processes: list[object]
    complete: bool


class _ValidationCancelled(RuntimeError):
    def __init__(self, signum: int, cleanup_deadline: float):
        self.signum = signum
        self.cleanup_deadline = cleanup_deadline
        super().__init__(signum)


_ACTIVE_CANCELLATION_STATE = None


class _CancellationSignalGuard:
    def __init__(self):
        self.signum: int | None = None
        self.cleanup_deadline: float | None = None
        self._previous_handlers: dict[int, object] = {}

    def _handle_signal(self, signum: int, _frame) -> None:
        if self.signum is None:
            self.signum = signum

    def _restore_handlers(self) -> None:
        for selected_signal, previous_handler in self._previous_handlers.items():
            signal.signal(selected_signal, previous_handler)
        self._previous_handlers.clear()

    def __enter__(self):
        global _ACTIVE_CANCELLATION_STATE
        if _ACTIVE_CANCELLATION_STATE is not None:
            raise RuntimeError("Cancellation signal guard is already active")
        _ACTIVE_CANCELLATION_STATE = self
        if os.name == "nt":
            return self
        try:
            for selected_signal in (signal.SIGINT, signal.SIGTERM):
                self._previous_handlers[selected_signal] = signal.getsignal(
                    selected_signal
                )
                signal.signal(selected_signal, self._handle_signal)
        except BaseException:
            self._restore_handlers()
            _ACTIVE_CANCELLATION_STATE = None
            raise
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        global _ACTIVE_CANCELLATION_STATE
        try:
            if os.name != "nt":
                self._restore_handlers()
        finally:
            _ACTIVE_CANCELLATION_STATE = None


def _raise_if_cancelled() -> None:
    cancellation = _ACTIVE_CANCELLATION_STATE
    if cancellation is not None and cancellation.signum is not None:
        cleanup_deadline = getattr(cancellation, "cleanup_deadline", None)
        if cleanup_deadline is None:
            cleanup_deadline = time.monotonic() + EXCEPTIONAL_CLEANUP_TIMEOUT_SECONDS
            cancellation.cleanup_deadline = cleanup_deadline
        raise _ValidationCancelled(
            cancellation.signum,
            cleanup_deadline,
        )


def _cancellation_pending() -> bool:
    cancellation = _ACTIVE_CANCELLATION_STATE
    return cancellation is not None and cancellation.signum is not None


def _windows_last_error() -> int:
    getter = getattr(ctypes, "get_last_error", None)
    return getter() if getter is not None else 0


def _load_windows_kernel32():
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise RuntimeError("Windows Job Objects are unavailable")
    kernel32 = loader("kernel32", use_last_error=True)
    handle_type = ctypes.c_void_p
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
    kernel32.CreateJobObjectW.restype = handle_type
    kernel32.SetInformationJobObject.argtypes = (
        handle_type,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    )
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.AssignProcessToJobObject.argtypes = (handle_type, handle_type)
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.QueryInformationJobObject.argtypes = (
        handle_type,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    kernel32.QueryInformationJobObject.restype = ctypes.c_int
    kernel32.CreateEventW.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_wchar_p,
    )
    kernel32.CreateEventW.restype = handle_type
    kernel32.SetEvent.argtypes = (handle_type,)
    kernel32.SetEvent.restype = ctypes.c_int
    kernel32.SetHandleInformation.argtypes = (
        handle_type,
        ctypes.c_uint32,
        ctypes.c_uint32,
    )
    kernel32.SetHandleInformation.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = (handle_type,)
    kernel32.CloseHandle.restype = ctypes.c_int
    return kernel32


class _WindowsKillOnCloseJob:
    def __init__(self, api, handle: int):
        self._api = api
        self._handle = handle
        self._close_result: bool | None = None

    @classmethod
    def create(cls, api=None):
        selected_api = api or _load_windows_kernel32()
        handle = selected_api.CreateJobObjectW(None, None)
        if not handle:
            error = _windows_last_error()
            raise OSError(error, "CreateJobObjectW failed")
        information = _JobObjectExtendedLimitInformation()
        information.basic_limit_information.limit_flags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        configured = selected_api.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        )
        if not configured:
            error = _windows_last_error()
            selected_api.CloseHandle(handle)
            raise OSError(error, "SetInformationJobObject failed")
        return cls(selected_api, handle)

    @staticmethod
    def popen_kwargs() -> dict[str, object]:
        return {}

    def start_process(self, command: list[str], **kwargs) -> subprocess.Popen:
        if not command:
            raise ValueError("Contained command must not be empty")
        if self._handle is None:
            raise RuntimeError("Windows Job Object is already closed")
        if kwargs.get("startupinfo") is not None:
            raise ValueError("Contained Windows commands cannot override startupinfo")
        if kwargs.get("close_fds") is False:
            raise ValueError("Contained Windows commands require close_fds=True")
        kwargs.pop("startupinfo", None)
        kwargs.pop("close_fds", None)

        gate_handle = self._api.CreateEventW(None, True, False, None)
        if not gate_handle:
            error = _windows_last_error()
            raise OSError(error, "CreateEventW failed")
        gate_inheritable = False
        process = None
        try:
            if not self._api.SetHandleInformation(
                gate_handle,
                _HANDLE_FLAG_INHERIT,
                _HANDLE_FLAG_INHERIT,
            ):
                error = _windows_last_error()
                raise OSError(error, "SetHandleInformation failed")
            gate_inheritable = True
            startup_info = subprocess.STARTUPINFO()
            startup_info.lpAttributeList = {"handle_list": [int(gate_handle)]}
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    _WINDOWS_GATE_PROGRAM,
                    str(int(gate_handle)),
                    *command,
                ],
                startupinfo=startup_info,
                close_fds=True,
                **kwargs,
            )
            self.assign(process)
            if not self._api.SetHandleInformation(
                gate_handle,
                _HANDLE_FLAG_INHERIT,
                0,
            ):
                error = _windows_last_error()
                raise OSError(error, "SetHandleInformation failed")
            gate_inheritable = False
            _raise_if_cancelled()
            if not self._api.SetEvent(gate_handle):
                error = _windows_last_error()
                raise OSError(error, "SetEvent failed")
            if not self._api.CloseHandle(gate_handle):
                error = _windows_last_error()
                raise OSError(error, "CloseHandle failed for Windows launch gate")
            gate_handle = None
            return process
        except BaseException as exc:
            cleanup_deadline = (
                exc.cleanup_deadline
                if isinstance(exc, _ValidationCancelled)
                else time.monotonic() + EXCEPTIONAL_CLEANUP_TIMEOUT_SECONDS
            )
            if process is not None:
                self.abort(cleanup_deadline)
                _reap_direct_process(process, cleanup_deadline)
            raise
        finally:
            if gate_handle is not None:
                if gate_inheritable:
                    self._api.SetHandleInformation(
                        gate_handle,
                        _HANDLE_FLAG_INHERIT,
                        0,
                    )
                self._api.CloseHandle(gate_handle)

    def assign(self, process: subprocess.Popen) -> None:
        process_handle = getattr(process, "_handle", None)
        if process_handle is None:
            raise RuntimeError("Windows child process handle is unavailable")
        if not self._api.AssignProcessToJobObject(self._handle, int(process_handle)):
            error = _windows_last_error()
            raise OSError(error, "AssignProcessToJobObject failed")

    def close(self) -> bool:
        if self._close_result is not None:
            return self._close_result
        handle = self._handle
        self._handle = None
        accounting = _JobObjectBasicAccountingInformation()
        queried = bool(
            self._api.QueryInformationJobObject(
                handle,
                _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                ctypes.byref(accounting),
                ctypes.sizeof(accounting),
                None,
            )
        )
        closed = bool(self._api.CloseHandle(handle))
        self._close_result = queried and accounting.active_processes == 0 and closed
        return self._close_result

    def abort(self, deadline: float) -> bool:
        del deadline
        return self.close()


def _linux_child_subreaper_state() -> bool:
    libc = ctypes.CDLL(None, use_errno=True)
    state = ctypes.c_int()
    if libc.prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(state), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "PR_GET_CHILD_SUBREAPER failed")
    return bool(state.value)


def _set_linux_child_subreaper(enabled: bool) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, int(enabled), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, "PR_SET_CHILD_SUBREAPER failed")


def _acquire_linux_child_subreaper() -> None:
    global _LINUX_SUBREAPER_LEASES
    global _LINUX_SUBREAPER_ORIGINAL_STATE
    with _LINUX_SUBREAPER_LOCK:
        if _LINUX_SUBREAPER_LEASES == 0:
            original_state = _linux_child_subreaper_state()
            if not original_state:
                _set_linux_child_subreaper(True)
            _LINUX_SUBREAPER_ORIGINAL_STATE = original_state
        _LINUX_SUBREAPER_LEASES += 1


def _release_linux_child_subreaper() -> None:
    global _LINUX_SUBREAPER_LEASES
    global _LINUX_SUBREAPER_ORIGINAL_STATE
    with _LINUX_SUBREAPER_LOCK:
        if _LINUX_SUBREAPER_LEASES <= 0:
            raise RuntimeError("Linux child-subreaper lease was not acquired")
        if _LINUX_SUBREAPER_LEASES == 1 and _LINUX_SUBREAPER_ORIGINAL_STATE is False:
            _set_linux_child_subreaper(False)
        _LINUX_SUBREAPER_LEASES -= 1
        if _LINUX_SUBREAPER_LEASES == 0:
            _LINUX_SUBREAPER_ORIGINAL_STATE = None


class _PosixProcessGroupContainment:
    def __init__(self, *, allow_untracked_descendants: bool = False):
        self._process_groups: list[int] = []
        self._root_processes: dict[tuple[int, float], object] = {}
        self._retained_descendants: dict[tuple[int, float], object] = {}
        self._tracking_complete = True
        self._cleanup_finished = False
        self._close_result: bool | None = None
        self._linux_runner_process = None
        self._linux_subreaper_lease = False
        if sys.platform.startswith("linux"):
            self._initialize_linux_subreaper()
        elif sys.platform == "darwin":
            if not allow_untracked_descendants:
                raise RuntimeError(
                    "Darwin cannot contain session-escaping descendants; protected "
                    "validation requires --allow-darwin-github-hosted-ephemeral-runner "
                    "on the trusted macos-latest job"
                )
        elif not allow_untracked_descendants:
            raise RuntimeError(
                "Kernel-backed descendant containment is unavailable on this platform"
            )

    @staticmethod
    def _process_identity(process) -> tuple[int, float]:
        return process.pid, process.create_time()

    def _initialize_linux_subreaper(self) -> None:
        if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            raise RuntimeError(
                "Linux child-subreaper containment requires the default SIGCHLD handler"
            )
        try:
            import psutil
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Linux child-subreaper containment requires psutil"
            ) from exc
        try:
            runner_process = psutil.Process(os.getpid())
            existing_descendants = runner_process.children(recursive=True)
        except (OSError, psutil.Error) as exc:
            raise RuntimeError(
                "Could not verify an idle process tree before enabling containment"
            ) from exc
        if existing_descendants:
            raise RuntimeError(
                "Cannot enable Linux child-subreaper containment with pre-existing "
                "runner descendants"
            )
        _acquire_linux_child_subreaper()
        self._linux_runner_process = runner_process
        self._linux_subreaper_lease = True

    @staticmethod
    def popen_kwargs() -> dict[str, object]:
        return {"start_new_session": True}

    def start_process(self, command: list[str], **kwargs) -> subprocess.Popen:
        if not command:
            raise ValueError("Contained command must not be empty")
        read_gate, write_gate = os.pipe()
        process = None
        assigned = False
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    _POSIX_GATE_PROGRAM,
                    str(read_gate),
                    *command,
                ],
                pass_fds=(read_gate,),
                **kwargs,
                **self.popen_kwargs(),
            )
            os.close(read_gate)
            read_gate = -1
            self.assign(process)
            assigned = True
            _raise_if_cancelled()
            os.write(write_gate, b"1")
            return process
        except BaseException as exc:
            cleanup_deadline = (
                exc.cleanup_deadline
                if isinstance(exc, _ValidationCancelled)
                else time.monotonic() + EXCEPTIONAL_CLEANUP_TIMEOUT_SECONDS
            )
            if assigned:
                self.abort(cleanup_deadline)
            elif process is not None:
                _reap_direct_process(process, cleanup_deadline)
            raise
        finally:
            if read_gate >= 0:
                os.close(read_gate)
            os.close(write_gate)

    def assign(self, process: subprocess.Popen) -> None:
        process_group = process.pid
        if process_group == os.getpgrp():
            raise RuntimeError(
                "Refusing to contain the validation runner process group"
            )
        if process_group in self._process_groups:
            raise RuntimeError("Process group was assigned more than once")
        self._process_groups.append(process_group)
        self._remember_root_process(process.pid)
        try:
            actual_process_group = os.getpgid(process.pid)
        except ProcessLookupError:
            self.refresh_descendants()
            return
        if actual_process_group != process_group:
            raise RuntimeError("Child process did not start in an isolated session")
        self.refresh_descendants()

    def _remember_root_process(self, process_id: int) -> None:
        try:
            import psutil
        except ModuleNotFoundError:
            self._tracking_complete = False
            return
        try:
            process = psutil.Process(process_id)
            identity = self._process_identity(process)
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return
        except (OSError, psutil.Error):
            self._tracking_complete = False
            return
        self._root_processes.setdefault(identity, process)

    def refresh_descendants(self) -> bool:
        if (
            not self._root_processes
            and not self._retained_descendants
            and self._linux_runner_process is None
        ):
            return self._tracking_complete
        try:
            import psutil
        except ModuleNotFoundError:
            self._tracking_complete = False
            return False

        complete = True
        if self._linux_runner_process is not None:
            try:
                adopted_processes = self._linux_runner_process.children(recursive=False)
            except (OSError, psutil.Error):
                adopted_processes = []
                complete = False
            root_identities = set(self._root_processes)
            for adopted_process in adopted_processes:
                try:
                    identity = self._process_identity(adopted_process)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                except (OSError, psutil.Error):
                    complete = False
                    continue
                if identity not in root_identities:
                    self._retained_descendants.setdefault(identity, adopted_process)
        discovery_roots = list(self._root_processes.values()) + list(
            self._retained_descendants.values()
        )
        for root_process in discovery_roots:
            try:
                descendants = root_process.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (OSError, psutil.Error):
                complete = False
                continue
            for descendant in descendants:
                if descendant.pid == os.getpid():
                    complete = False
                    continue
                try:
                    identity = self._process_identity(descendant)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                except (OSError, psutil.Error):
                    complete = False
                    continue
                self._retained_descendants.setdefault(identity, descendant)
        if self._linux_runner_process is not None:
            for identity, descendant in self._retained_descendants.items():
                try:
                    if descendant.ppid() != os.getpid():
                        continue
                    waited_pid, _status = os.waitpid(descendant.pid, os.WNOHANG)
                    if waited_pid not in (0, descendant.pid):
                        complete = False
                except ChildProcessError:
                    try:
                        if (
                            self._process_identity(descendant) == identity
                            and descendant.is_running()
                            and descendant.status()
                            not in {psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE}
                        ):
                            complete = False
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        pass
                    except (OSError, psutil.Error):
                        complete = False
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                except (OSError, psutil.Error):
                    complete = False
        if not complete:
            self._tracking_complete = False
        return complete

    def _release_kernel_tracking(self) -> bool:
        if not self._linux_subreaper_lease:
            return True
        try:
            _release_linux_child_subreaper()
        except (OSError, RuntimeError):
            self._tracking_complete = False
            return False
        self._linux_subreaper_lease = False
        return True

    def _live_retained_descendants(self) -> list[object]:
        try:
            import psutil
        except ModuleNotFoundError:
            self._tracking_complete = False
            return list(self._retained_descendants.values())

        dead_statuses = {psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE}
        live_processes = []
        for identity, process in self._retained_descendants.items():
            try:
                if (
                    self._process_identity(process) == identity
                    and process.is_running()
                    and process.status() not in dead_statuses
                ):
                    live_processes.append(process)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (OSError, psutil.Error):
                self._tracking_complete = False
                live_processes.append(process)
        return live_processes

    def _signal_retained_descendants(
        self,
        processes: list[object],
        selected_signal: int,
    ) -> None:
        try:
            import psutil
        except ModuleNotFoundError:
            self._tracking_complete = False
            return
        for process in processes:
            try:
                process.send_signal(selected_signal)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (OSError, psutil.Error):
                self._tracking_complete = False
                continue

    @staticmethod
    def _has_live_members(process_group: int) -> bool:
        def probe_process_group() -> bool:
            try:
                os.killpg(process_group, 0)
            except ProcessLookupError:
                return False
            except OSError:
                return True
            return True

        try:
            import psutil
        except ModuleNotFoundError:
            return probe_process_group()

        dead_statuses = {psutil.STATUS_DEAD, psutil.STATUS_ZOMBIE}
        try:
            for process in psutil.process_iter(("pid", "status")):
                try:
                    if (
                        process.info["status"] not in dead_statuses
                        and os.getpgid(process.pid) == process_group
                    ):
                        return True
                except (OSError, psutil.Error):
                    continue
        except (OSError, psutil.Error):
            return probe_process_group()
        return False

    def _wait_for_exit(
        self,
        process_groups: list[int],
        timeout: float,
        *,
        stop_on_cancellation: bool = False,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if stop_on_cancellation and _cancellation_pending():
                return False
            self.refresh_descendants()
            if (
                not any(self._has_live_members(group) for group in process_groups)
                and not self._live_retained_descendants()
            ):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    @classmethod
    def _signal_groups(cls, process_groups: list[int], selected_signal: int) -> None:
        for process_group in process_groups:
            if not cls._has_live_members(process_group):
                continue
            try:
                os.killpg(process_group, selected_signal)
            except OSError:
                continue

    def close(self) -> bool:
        if self._cleanup_finished:
            return self._close_result
        self.refresh_descendants()
        live_groups = [
            group for group in self._process_groups if self._has_live_members(group)
        ]
        live_descendants = self._live_retained_descendants()
        had_survivors = bool(live_groups or live_descendants)
        self._signal_groups(live_groups, signal.SIGTERM)
        self._signal_retained_descendants(live_descendants, signal.SIGTERM)
        if not self._wait_for_exit(
            live_groups,
            5,
            stop_on_cancellation=True,
        ):
            remaining_groups = [
                group for group in live_groups if self._has_live_members(group)
            ]
            remaining_descendants = self._live_retained_descendants()
            self._signal_groups(remaining_groups, signal.SIGKILL)
            self._signal_retained_descendants(
                remaining_descendants,
                signal.SIGKILL,
            )
            self._wait_for_exit(
                remaining_groups,
                5,
                stop_on_cancellation=True,
            )
        self.refresh_descendants()
        cleanup_finished = (
            self._tracking_complete
            and not any(self._has_live_members(group) for group in self._process_groups)
            and not self._live_retained_descendants()
        )
        if cleanup_finished and not self._release_kernel_tracking():
            cleanup_finished = False
        self._cleanup_finished = cleanup_finished
        clean = not had_survivors and cleanup_finished and self._tracking_complete
        self._close_result = clean if self._close_result is None else False
        return self._close_result

    def abort(self, deadline: float) -> bool:
        if self._cleanup_finished:
            return self._close_result
        self.refresh_descendants()
        live_groups = [
            group for group in self._process_groups if self._has_live_members(group)
        ]
        live_descendants = self._live_retained_descendants()
        had_survivors = bool(live_groups or live_descendants)
        self._signal_groups(live_groups, signal.SIGTERM)
        self._signal_retained_descendants(live_descendants, signal.SIGTERM)
        term_timeout = min(
            EXCEPTIONAL_TERM_GRACE_SECONDS,
            max(0.0, deadline - time.monotonic()),
        )
        if not self._wait_for_exit(live_groups, term_timeout):
            remaining_groups = [
                group for group in live_groups if self._has_live_members(group)
            ]
            remaining_descendants = self._live_retained_descendants()
            self._signal_groups(remaining_groups, signal.SIGKILL)
            self._signal_retained_descendants(
                remaining_descendants,
                signal.SIGKILL,
            )
            kill_timeout = max(0.0, deadline - time.monotonic())
            self._wait_for_exit(remaining_groups, kill_timeout)
        self.refresh_descendants()
        cleanup_finished = (
            self._tracking_complete
            and not any(self._has_live_members(group) for group in self._process_groups)
            and not self._live_retained_descendants()
        )
        if cleanup_finished and not self._release_kernel_tracking():
            cleanup_finished = False
        self._cleanup_finished = cleanup_finished
        clean = not had_survivors and cleanup_finished and self._tracking_complete
        self._close_result = clean if self._close_result is None else False
        return self._close_result


def _verified_darwin_ephemeral_runner(
    *,
    allow_darwin_github_hosted_ephemeral_runner: bool,
    backend: str | None,
    target: str | None,
    environment: dict[str, str] | None = None,
) -> bool:
    if sys.platform != "darwin":
        return False
    selected_environment = os.environ if environment is None else environment
    if not allow_darwin_github_hosted_ephemeral_runner:
        return False
    if backend != "metal" or target != "macos-metal":
        raise RuntimeError(
            "The Darwin ephemeral-runner exception is restricted to macos-metal"
        )
    if (
        selected_environment.get(DARWIN_EPHEMERAL_RUNNER_CONTEXT_KEY)
        != DARWIN_EPHEMERAL_RUNNER_CONTEXT
        or selected_environment.get("GITHUB_ACTIONS") != "true"
    ):
        raise RuntimeError(
            "The Darwin ephemeral-runner exception requires the trusted "
            "github-hosted macos-latest workflow context"
        )
    return True


def _create_process_containment(
    *,
    allow_darwin_github_hosted_ephemeral_runner: bool = False,
    backend: str | None = None,
    target: str | None = None,
):
    if os.name == "nt":
        return _WindowsKillOnCloseJob.create()
    allow_untracked_descendants = _verified_darwin_ephemeral_runner(
        allow_darwin_github_hosted_ephemeral_runner=(
            allow_darwin_github_hosted_ephemeral_runner
        ),
        backend=backend,
        target=target,
    )
    return _PosixProcessGroupContainment(
        allow_untracked_descendants=allow_untracked_descendants
    )


def _start_contained_process(containment, command: list[str], **kwargs):
    start_process = getattr(containment, "start_process", None)
    if start_process is not None:
        return start_process(command, **kwargs)
    process = subprocess.Popen(
        command,
        **kwargs,
        **containment.popen_kwargs(),
    )
    containment.assign(process)
    return process


def _close_process_containment(containment) -> bool:
    try:
        return containment.close()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Process containment cleanup failed: {exc}", file=sys.stderr)
        return False


def _abort_process_containment(containment, deadline: float) -> bool:
    try:
        return containment.abort(deadline)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Process containment abort failed: {exc}", file=sys.stderr)
        return False


def _refresh_process_containment(containment) -> bool:
    refresh = getattr(containment, "refresh_descendants", None)
    return True if refresh is None else bool(refresh())


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_validation_command(
    *,
    python_executable: Path,
    backend: str,
    channel: str,
    target: str,
    models_csv: str,
    lite: bool,
    capability_profile: str,
    capability_models_csv: str,
    port: int,
    output_path: Path | None = None,
) -> list[str]:
    if not TARGET_PATTERN.fullmatch(target):
        raise ValueError(
            "target must contain only lowercase letters, digits, and hyphens"
        )
    models = _csv_values(models_csv)
    capability_models = _csv_values(capability_models_csv)
    if models and lite:
        raise ValueError("Explicit models and lite mode are mutually exclusive")
    if bool(capability_profile) != bool(capability_models):
        raise ValueError("Capability profile and capability models must be paired")
    if capability_profile and capability_profile != K2_HORIZON_PROFILE:
        raise ValueError(f"Unsupported capability profile: {capability_profile}")
    if models:
        selected_model_ids = {model.removeprefix("builtin.") for model in models}
        missing = [
            model
            for model in capability_models
            if model.removeprefix("builtin.") not in selected_model_ids
        ]
        if missing:
            raise ValueError(
                "Capability models must be present in explicit models: "
                + ", ".join(missing)
            )

    command = [
        str(python_executable),
        str((ROOT / "test" / "validate_llamacpp.py").resolve()),
        "--backend",
        backend,
        "--port",
        str(port),
        "--output",
        str(output_path or f"llamacpp_validation_{target}.json"),
    ]
    if channel:
        command.extend(("--channel", channel))
    for model in models:
        command.extend(("--model", model))
    if lite:
        command.append("--lite")
    if capability_profile:
        command.extend(("--capability-profile", capability_profile))
        for model in capability_models:
            command.extend(("--capability-model", model))
    return command


def build_restart_validation_command(
    *,
    python_executable: Path,
    backend: str,
    channel: str,
    target: str,
    model: str,
    port: int,
    output_path: Path | None = None,
) -> list[str]:
    if not model.strip() or "," in model:
        raise ValueError("Restart validation requires exactly one model")
    command = build_validation_command(
        python_executable=python_executable,
        backend=backend,
        channel=channel,
        target=target,
        models_csv=model,
        lite=False,
        capability_profile="",
        capability_models_csv="",
        port=port,
        output_path=output_path,
    )
    if output_path is None:
        command[command.index("--output") + 1] = (
            f"llamacpp_restart_validation_{target}.json"
        )
    command.append("--skip-install")
    return command


def build_prepare_validation_command(
    *,
    python_executable: Path,
    backend: str,
    channel: str,
    target: str,
    models_csv: str,
    lite: bool,
    capability_profile: str,
    capability_models_csv: str,
    port: int,
    output_path: Path | None = None,
) -> list[str]:
    command = build_validation_command(
        python_executable=python_executable,
        backend=backend,
        channel=channel,
        target=target,
        models_csv=models_csv,
        lite=lite,
        capability_profile=capability_profile,
        capability_models_csv=capability_models_csv,
        port=port,
        output_path=output_path,
    )
    command.append("--prepare-only")
    return command


def _require_unused_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is already in use") from exc


def _require_idle_runner() -> None:
    import psutil

    conflicts = []
    for process in psutil.process_iter(("pid", "name")):
        try:
            name = (process.info.get("name") or "").lower().removesuffix(".exe")
            if name in INFERENCE_PROCESS_NAMES:
                conflicts.append(f"{name}:{process.pid}")
        except psutil.Error:
            continue
    if conflicts:
        raise RuntimeError("Runner is not idle: " + ", ".join(sorted(conflicts)))


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(os.path, "isjunction", None)
    if is_junction is not None and is_junction(path):
        return True
    try:
        path_status = path.lstat()
    except OSError:
        return False
    reparse_attribute = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(path_status, "st_file_attributes", 0)
    return bool(reparse_attribute and file_attributes & reparse_attribute)


def _capture_directory(path: Path, allowed_root: Path | None) -> _DirectoryIdentity:
    if _is_link_or_junction(path):
        raise RuntimeError(f"Directory must not be a symlink or junction: {path}")
    try:
        resolved_path = path.resolve(strict=True)
        path_status = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"Directory does not exist: {path}") from exc
    if not stat.S_ISDIR(path_status.st_mode):
        raise RuntimeError(f"Path is not a directory: {path}")
    if allowed_root is not None:
        try:
            resolved_path.relative_to(allowed_root)
        except ValueError as exc:
            raise RuntimeError(f"Directory escapes its allowed root: {path}") from exc
    return _DirectoryIdentity(
        path=path,
        resolved_path=resolved_path,
        device=path_status.st_dev,
        inode=path_status.st_ino,
    )


def create_validation_cache(temporary_root: Path, target: str) -> ValidationCache:
    if not TARGET_PATTERN.fullmatch(target):
        raise ValueError(
            "target must contain only lowercase letters, digits, and hyphens"
        )
    temporary_identity = _capture_directory(temporary_root, None)
    validation_root = Path(
        tempfile.mkdtemp(
            prefix=f"llamacpp-validation-{target}-",
            dir=temporary_identity.resolved_path,
        )
    )
    validation_identity = _capture_directory(
        validation_root, temporary_identity.resolved_path
    )
    cache_directory = validation_identity.resolved_path / "lemonade-cache"
    config_directory = validation_identity.resolved_path / "lemonade-config"
    cache_directory.mkdir(mode=0o700)
    hf_cache_directory = cache_directory / "hf-hub"
    hf_cache_directory.mkdir(mode=0o700)
    config_directory.mkdir(mode=0o700)
    private_temp_directory = validation_identity.resolved_path / "private-temp"
    private_temp_directory.mkdir(mode=0o700)
    results_directory = validation_identity.resolved_path / "results"
    results_directory.mkdir(mode=0o700)
    logs_directory = validation_identity.resolved_path / "logs"
    logs_directory.mkdir(mode=0o700)
    return ValidationCache(
        temporary_root=temporary_identity,
        validation_root=validation_identity,
        cache_directory=_capture_directory(
            cache_directory, validation_identity.resolved_path
        ),
        config_directory=_capture_directory(
            config_directory, validation_identity.resolved_path
        ),
        hf_cache_directory=_capture_directory(
            hf_cache_directory, validation_identity.resolved_path
        ),
        private_temp_directory=_capture_directory(
            private_temp_directory, validation_identity.resolved_path
        ),
        results_directory=_capture_directory(
            results_directory, validation_identity.resolved_path
        ),
        logs_directory=_capture_directory(
            logs_directory, validation_identity.resolved_path
        ),
    )


def revalidate_validation_cache(cache: ValidationCache) -> None:
    current_temporary_root = _capture_directory(cache.temporary_root.path, None)
    current_validation_root = _capture_directory(
        cache.validation_root.path,
        current_temporary_root.resolved_path,
    )
    expected_and_current = (
        (cache.temporary_root, current_temporary_root),
        (cache.validation_root, current_validation_root),
        (
            cache.cache_directory,
            _capture_directory(
                cache.cache_directory.path,
                current_validation_root.resolved_path,
            ),
        ),
        (
            cache.config_directory,
            _capture_directory(
                cache.config_directory.path,
                current_validation_root.resolved_path,
            ),
        ),
        (
            cache.hf_cache_directory,
            _capture_directory(
                cache.hf_cache_directory.path,
                current_validation_root.resolved_path,
            ),
        ),
        (
            cache.private_temp_directory,
            _capture_directory(
                cache.private_temp_directory.path,
                current_validation_root.resolved_path,
            ),
        ),
        (
            cache.results_directory,
            _capture_directory(
                cache.results_directory.path,
                current_validation_root.resolved_path,
            ),
        ),
        (
            cache.logs_directory,
            _capture_directory(
                cache.logs_directory.path,
                current_validation_root.resolved_path,
            ),
        ),
    )
    for expected, current in expected_and_current:
        if expected != current:
            raise RuntimeError(
                f"Validation cache directory changed after creation: {expected.path}"
            )


def _file_state(file_status: os.stat_result) -> tuple[int, ...]:
    return (
        file_status.st_dev,
        file_status.st_ino,
        file_status.st_mode,
        file_status.st_size,
        file_status.st_mtime_ns,
        file_status.st_ctime_ns,
    )


def _open_binary_no_follow(path: Path):
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags | no_follow)
    return os.fdopen(descriptor, "rb")


def _capture_regular_file(
    path: Path, allowed_root: Path, *, require_nonempty: bool
) -> _FileIdentity:
    try:
        allowed_root = allowed_root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"Validation output root does not exist: {allowed_root}"
        ) from exc
    if _is_link_or_junction(path):
        raise RuntimeError(f"Validation output must not be a link: {path}")
    try:
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(allowed_root)
        with _open_binary_no_follow(path) as output_file:
            file_status = os.fstat(output_file.fileno())
            path_after = path.resolve(strict=True)
            path_after.relative_to(allowed_root)
            path_status = path_after.stat()
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Validation output is absent or escapes its root: {path}"
        ) from exc
    if _is_link_or_junction(path):
        raise RuntimeError(f"Validation output must not be a link: {path}")
    if not stat.S_ISREG(file_status.st_mode) or (
        require_nonempty and file_status.st_size == 0
    ):
        qualifier = "nonempty " if require_nonempty else ""
        raise RuntimeError(
            f"Validation output must be a {qualifier}regular file: {path}"
        )
    if path_after != resolved_path or _file_state(file_status) != _file_state(
        path_status
    ):
        raise RuntimeError(f"Validation output changed while inspected: {path}")
    return _FileIdentity(
        path=path,
        resolved_path=resolved_path,
        device=file_status.st_dev,
        inode=file_status.st_ino,
        size=file_status.st_size,
        modified_ns=file_status.st_mtime_ns,
        changed_ns=file_status.st_ctime_ns,
    )


def _revalidate_regular_file(
    identity: _FileIdentity, allowed_root: Path, *, require_nonempty: bool
) -> None:
    if (
        _capture_regular_file(
            identity.path,
            allowed_root,
            require_nonempty=require_nonempty,
        )
        != identity
    ):
        raise RuntimeError(f"Validation output changed after creation: {identity.path}")


def _scan_stable_log(
    path: Path, allowed_root: Path, handle_line, byte_budget: int
) -> tuple[_FileIdentity, int]:
    try:
        allowed_root = allowed_root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"Validation log root does not exist: {allowed_root}"
        ) from exc
    if _is_link_or_junction(path):
        raise RuntimeError(f"Validation log must not be a link: {path}")
    try:
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(allowed_root)
        with _open_binary_no_follow(path) as log_file:
            status_before = os.fstat(log_file.fileno())
            if not stat.S_ISREG(status_before.st_mode):
                raise RuntimeError(f"Validation log must be a regular file: {path}")
            consumed = 0
            while True:
                raw_line = log_file.readline(MAX_ATTESTATION_LINE_BYTES + 1)
                if not raw_line:
                    break
                consumed += len(raw_line)
                if consumed > byte_budget:
                    raise RuntimeError(
                        "Accelerator attestation log exceeds byte budget"
                    )
                if len(raw_line) > MAX_ATTESTATION_LINE_BYTES or (
                    len(raw_line) == MAX_ATTESTATION_LINE_BYTES
                    and not raw_line.endswith(b"\n")
                ):
                    raise RuntimeError("Accelerator attestation log line is too long")
                handle_line(raw_line.decode("utf-8", errors="replace"))
            status_after = os.fstat(log_file.fileno())
        path_after = path.resolve(strict=True)
        path_after.relative_to(allowed_root)
        path_status = path_after.stat()
    except RuntimeError:
        raise
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not safely scan validation log: {path}") from exc
    if _is_link_or_junction(path):
        raise RuntimeError(f"Validation log must not be a link: {path}")
    if (
        path_after != resolved_path
        or _file_state(status_before) != _file_state(status_after)
        or _file_state(status_after) != _file_state(path_status)
    ):
        raise RuntimeError(f"Validation log changed while scanned: {path}")
    identity = _FileIdentity(
        path=path,
        resolved_path=resolved_path,
        device=status_after.st_dev,
        inode=status_after.st_ino,
        size=status_after.st_size,
        modified_ns=status_after.st_mtime_ns,
        changed_ns=status_after.st_ctime_ns,
    )
    return identity, consumed


def require_accelerator_attestation(
    log_paths: list[Path],
    allowed_root: Path,
    phase: str,
    expected_artifacts: dict[str, VerifiedValidationArtifact],
    expected_backend: str,
    *,
    selected_models: list[str] | None = None,
) -> list[_FileIdentity]:
    cpu_backend = expected_backend == "cpu"
    backend_family = expected_backend.split("-", maxsplit=1)[0]
    expected_device_prefix = ACCELERATOR_DEVICE_PREFIXES.get(backend_family)
    if expected_backend not in ATTESTED_BACKENDS:
        raise RuntimeError(
            f"{phase} accelerator attestation has unsupported backend: "
            f"{expected_backend}"
        )
    expected_models = {}
    for model_id, artifact in expected_artifacts.items():
        canonical_model_id = model_id.removeprefix("builtin.")
        if (
            not isinstance(artifact, VerifiedValidationArtifact)
            or artifact.model_id.removeprefix("builtin.") != canonical_model_id
        ):
            raise RuntimeError(
                f"{phase} accelerator attestation has an invalid artifact binding"
            )
        expected_models[canonical_model_id] = artifact
    canonical_selected_models = [
        model.removeprefix("builtin.")
        for model in (
            selected_models if selected_models is not None else list(expected_models)
        )
        if isinstance(model, str) and model
    ]
    if (
        not canonical_selected_models
        or len(canonical_selected_models)
        != len(selected_models if selected_models is not None else expected_models)
        or len(canonical_selected_models) != len(set(canonical_selected_models))
        or not set(expected_models).issubset(canonical_selected_models)
    ):
        raise RuntimeError(
            f"{phase} accelerator attestation has an invalid selected-model binding"
        )
    selected_model_ids = set(canonical_selected_models)
    attested_models = set()
    scanned_identities = []
    total_bytes = 0
    for log_path in log_paths:
        current_model = None
        current_backend_seen = False
        current_identity_seen = False
        current_device_seen = False
        current_cpu_buffer_seen = False
        current_offload = None

        def handle_line(line: str) -> None:
            nonlocal current_model, current_identity_seen
            nonlocal current_backend_seen, current_device_seen
            nonlocal current_cpu_buffer_seen, current_offload
            backend_marker = BACKEND_MARKER.search(line)
            offload = OFFLOAD_MARKER.search(line)
            device = DEVICE_MARKER.search(line)
            cpu_buffer = CPU_BUFFER_MARKER.search(line)
            if cpu_backend and (DEVICE_PREFIX in line or OFFLOAD_PREFIX in line):
                raise RuntimeError(
                    f"{phase} CPU attestation contains accelerator markers"
                )
            if OFFLOAD_PREFIX in line:
                if line.count(OFFLOAD_PREFIX) != 1 or offload is None:
                    raise RuntimeError(
                        f"{phase} accelerator attestation has malformed offload proof"
                    )
                offloaded_layers, total_layers = map(int, offload.groups())
                if not 0 < offloaded_layers <= total_layers:
                    raise RuntimeError(
                        f"{phase} accelerator attestation has invalid layer offload proof"
                    )

            loading = LOADING_MODEL_MARKER.search(line)
            if loading is not None:
                if current_model is not None:
                    raise RuntimeError(
                        f"{phase} accelerator attestation has an incomplete model load"
                    )
                current_model = loading.group("model").removeprefix("builtin.")
                if current_model not in selected_model_ids:
                    raise RuntimeError(
                        f"{phase} accelerator attestation has an unexpected model "
                        f"load: {current_model}"
                    )
                current_backend_seen = False
                current_identity_seen = False
                current_device_seen = False
                current_cpu_buffer_seen = False
                current_offload = None
                return
            if current_model not in expected_models:
                if current_model is None and BACKEND_PREFIX in line:
                    raise RuntimeError(
                        f"{phase} accelerator attestation has an unbound backend proof"
                    )
                if current_model is not None and LOADED_MODEL_MARKER.search(line):
                    current_model = None
                return
            if BACKEND_PREFIX in line:
                if line.count(BACKEND_PREFIX) != 1 or backend_marker is None:
                    raise RuntimeError(
                        f"{phase} accelerator attestation has malformed backend proof"
                    )
                if current_backend_seen:
                    raise RuntimeError(
                        f"{phase} accelerator attestation has duplicate backend proof"
                    )
                if (
                    current_identity_seen
                    or current_device_seen
                    or current_cpu_buffer_seen
                    or current_offload is not None
                ):
                    raise RuntimeError(
                        f"{phase} accelerator attestation has late backend proof"
                    )
                actual_backend = backend_marker.group("backend")
                if actual_backend != expected_backend:
                    raise RuntimeError(
                        f"{phase} accelerator attestation has backend "
                        f"'{actual_backend}', expected '{expected_backend}'"
                    )
                current_backend_seen = True
                return
            gguf = GGUF_MARKER.search(line)
            if gguf is not None:
                if not current_backend_seen:
                    raise RuntimeError(
                        f"{phase} accelerator attestation is missing backend proof"
                    )
                loaded_path = Path(gguf.group("path"))
                try:
                    if not loaded_path.is_absolute():
                        raise ValueError("artifact path is not absolute")
                    resolved_loaded_path = loaded_path.resolve(strict=True)
                except (OSError, ValueError) as exc:
                    raise RuntimeError(
                        f"{phase} accelerator attestation has the wrong artifact"
                    ) from exc
                expected_path = expected_models[current_model].resolved_path
                if os.path.normcase(os.path.normpath(str(resolved_loaded_path))) != (
                    os.path.normcase(os.path.normpath(str(expected_path)))
                ):
                    raise RuntimeError(
                        f"{phase} accelerator attestation has the wrong artifact"
                    )
                current_identity_seen = True
            if (
                cpu_backend
                and cpu_buffer is not None
                and current_identity_seen
                and float(cpu_buffer.group("size")) > 0
            ):
                current_cpu_buffer_seen = True
            if current_offload is None:
                if device is not None and current_identity_seen:
                    device_name = device.group("device")
                    if (
                        re.fullmatch(
                            rf"{re.escape(expected_device_prefix)}[0-9]+", device_name
                        )
                        is None
                    ):
                        raise RuntimeError(
                            f"{phase} accelerator attestation has device "
                            f"'{device_name}' for the wrong backend"
                        )
                    current_device_seen = True
                if offload is not None:
                    if not current_identity_seen or not current_device_seen:
                        raise RuntimeError(
                            f"{phase} accelerator attestation has unbound device/offload proof"
                        )
                    current_offload = tuple(map(int, offload.groups()))
            if LOADED_MODEL_MARKER.search(line) is None:
                return
            if not current_identity_seen:
                raise RuntimeError(
                    f"{phase} accelerator attestation is missing artifact identity"
                )
            if not current_backend_seen:
                raise RuntimeError(
                    f"{phase} accelerator attestation is missing backend proof"
                )
            if cpu_backend and not current_cpu_buffer_seen:
                raise RuntimeError(
                    f"{phase} CPU attestation is missing a CPU model buffer"
                )
            if not cpu_backend and current_offload is None:
                raise RuntimeError(
                    f"{phase} accelerator attestation is missing layer offload proof"
                )
            attested_models.add(current_model)
            current_model = None
            current_backend_seen = False
            current_identity_seen = False
            current_device_seen = False
            current_cpu_buffer_seen = False
            current_offload = None

        identity, consumed = _scan_stable_log(
            log_path,
            allowed_root,
            handle_line,
            MAX_ATTESTATION_LOG_BYTES - total_bytes,
        )
        scanned_identities.append(identity)
        total_bytes += consumed
        if current_model is not None:
            raise RuntimeError(
                f"{phase} accelerator attestation has an incomplete model load"
            )
    missing_models = sorted(set(expected_models) - attested_models)
    if missing_models:
        raise RuntimeError(
            f"{phase} accelerator attestation is missing models: "
            + ", ".join(missing_models)
        )
    return scanned_identities


def resolve_attestation_backend(backend: str, channel: str) -> str:
    if backend == "rocm":
        if channel not in {"stable", "nightly"}:
            raise RuntimeError(
                "ROCm validation requires an exact stable or nightly channel"
            )
        return f"rocm-{channel}"
    if channel:
        raise RuntimeError(
            f"{backend} validation must not specify a ROCm release channel"
        )
    if backend not in ATTESTED_BACKENDS:
        raise RuntimeError(f"Unsupported validation backend: {backend}")
    return backend


def _emit_workflow_outputs(outputs: dict[str, Path]) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    with Path(github_output).open("a", encoding="utf-8") as output_file:
        for name, path in outputs.items():
            output_file.write(f"{name}={path}\n")


def _cache_processes(cache_directory: Path, excluded_pid: int) -> _CacheProcessScan:
    try:
        import psutil
    except ModuleNotFoundError:
        return _CacheProcessScan([], False)

    cache_prefix = os.path.normcase(str(cache_directory.resolve()) + os.sep)
    matches = []
    try:
        for process in psutil.process_iter(("pid", "exe")):
            if process.pid == excluded_pid:
                continue
            try:
                executable = process.info.get("exe")
                if executable and os.path.normcase(
                    str(Path(executable).resolve())
                ).startswith(cache_prefix):
                    matches.append(process)
            except (OSError, psutil.Error):
                continue
    except (OSError, psutil.Error):
        return _CacheProcessScan(matches, False)
    return _CacheProcessScan(matches, True)


def _stop_processes(processes: list[object]) -> bool:
    import psutil

    alive = list(processes)
    for process in processes:
        try:
            process.terminate()
        except psutil.Error:
            pass
    alive = _wait_for_psutil_processes(alive, timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass
    alive = _wait_for_psutil_processes(alive, timeout=5)
    return not alive


def _force_kill_processes(processes: list[object], deadline: float) -> bool:
    import psutil

    alive = list(processes)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass
    reap_timeout = min(
        PROCESS_WAIT_POLL_SECONDS,
        max(0.0, deadline - time.monotonic()),
    )
    try:
        _, alive = psutil.wait_procs(alive, timeout=reap_timeout)
    except (OSError, psutil.Error):
        return False
    return not alive


def _wait_for_psutil_processes(
    processes: list[object],
    timeout: float,
) -> list[object]:
    import psutil

    deadline = time.monotonic() + timeout
    alive = list(processes)
    while alive:
        _raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        _, alive = psutil.wait_procs(
            alive,
            timeout=min(PROCESS_WAIT_POLL_SECONDS, remaining),
        )
    _raise_if_cancelled()
    return alive


def _open_local_url(request: urllib.request.Request, timeout: int):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def _open_local_url_with_cancellation(
    request: urllib.request.Request,
    timeout: int,
    containment=None,
) -> None:
    outcome: list[BaseException | None] = []

    def open_url() -> None:
        try:
            with _open_local_url(request, timeout=timeout):
                pass
        except BaseException as exc:  # pylint: disable=broad-exception-caught
            outcome.append(exc)
        else:
            outcome.append(None)

    worker = threading.Thread(target=open_url, daemon=True)
    worker.start()
    deadline = time.monotonic() + timeout
    while worker.is_alive():
        if containment is not None:
            _refresh_process_containment(containment)
        _raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Timed out requesting graceful server shutdown")
        worker.join(min(PROCESS_WAIT_POLL_SECONDS, remaining))
    _raise_if_cancelled()
    if not outcome:
        raise RuntimeError("Graceful server shutdown request returned no result")
    if outcome[0] is not None:
        raise outcome[0]


def _wait_for_process_with_cancellation(
    process: subprocess.Popen,
    timeout: float,
    containment=None,
) -> int:
    deadline = time.monotonic() + timeout
    while True:
        if containment is not None:
            _refresh_process_containment(containment)
        _raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(
                getattr(process, "args", "process"), timeout
            )
        try:
            exit_code = process.wait(timeout=min(PROCESS_WAIT_POLL_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            continue
        _raise_if_cancelled()
        return exit_code


def _shutdown_server(
    process: subprocess.Popen,
    port: int,
    cache_directory: Path,
    containment=None,
) -> bool:
    orphan_processes = []
    try:
        running_on_entry = process.poll() is None
        clean = running_on_entry
        if running_on_entry:
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/internal/shutdown",
                data=b"",
                method="POST",
            )
            try:
                _open_local_url_with_cancellation(
                    request,
                    timeout=10,
                    containment=containment,
                )
                if (
                    _wait_for_process_with_cancellation(
                        process,
                        timeout=10,
                        containment=containment,
                    )
                    != 0
                ):
                    clean = False
            except (OSError, subprocess.TimeoutExpired):
                clean = False
        if process.poll() is None:
            process.terminate()
            try:
                _wait_for_process_with_cancellation(
                    process,
                    timeout=5,
                    containment=containment,
                )
            except subprocess.TimeoutExpired:
                process.kill()
                _wait_for_process_with_cancellation(
                    process,
                    timeout=5,
                    containment=containment,
                )
                clean = False

        _raise_if_cancelled()
        if containment is not None and not _close_process_containment(containment):
            clean = False
        _raise_if_cancelled()
        orphan_scan = _cache_processes(cache_directory, process.pid)
        orphan_processes = list(orphan_scan.processes)
        if not orphan_scan.complete:
            clean = False
        if orphan_processes:
            clean = False
            _stop_processes(orphan_processes)
        _raise_if_cancelled()
        return clean
    except _ValidationCancelled as exc:
        cleanup_deadline = exc.cleanup_deadline
        cancellation_orphans = list(orphan_processes)
        cancellation_scan = _cache_processes(cache_directory, process.pid)
        retained_orphan_pids = {
            getattr(orphan, "pid", None) for orphan in cancellation_orphans
        }
        for orphan in cancellation_scan.processes:
            if getattr(orphan, "pid", None) not in retained_orphan_pids:
                cancellation_orphans.append(orphan)
        if cancellation_orphans:
            _force_kill_processes(cancellation_orphans, cleanup_deadline)
        if containment is not None:
            _abort_process_containment(containment, cleanup_deadline)
        _reap_direct_process(process, cleanup_deadline)
        raise


def _reap_direct_process(process: subprocess.Popen, deadline: float) -> None:
    try:
        if process.poll() is None:
            process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except (OSError, subprocess.TimeoutExpired):
        pass


def _wait_for_contained_process(
    process: subprocess.Popen,
    timeout_seconds: float,
    containment,
) -> int:
    if timeout_seconds <= 0:
        raise ValueError("Validation phase timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    while True:
        _refresh_process_containment(containment)
        _raise_if_cancelled()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"Validation command timed out after {timeout_seconds:g} seconds"
            )
        try:
            exit_code = process.wait(timeout=min(PROCESS_WAIT_POLL_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            continue
        _raise_if_cancelled()
        return exit_code


def _run_contained_command(
    command: list[str],
    *,
    env: dict[str, str],
    containment,
    timeout_seconds: float,
) -> int:
    process = None
    try:
        process = _start_contained_process(
            containment,
            command,
            env=env,
        )
        return _wait_for_contained_process(process, timeout_seconds, containment)
    except BaseException as exc:
        cleanup_deadline = (
            exc.cleanup_deadline
            if isinstance(exc, _ValidationCancelled)
            else time.monotonic() + EXCEPTIONAL_CLEANUP_TIMEOUT_SECONDS
        )
        _abort_process_containment(containment, cleanup_deadline)
        if process is not None:
            _reap_direct_process(process, cleanup_deadline)
        if isinstance(exc, _ValidationCancelled):
            raise
        _raise_if_cancelled()
        raise


def build_server_environment(
    hf_cache_directory: Path,
    inherited: dict[str, str] | None = None,
    *,
    private_temp_directory: Path,
    candidate_defaults_path: Path | None = None,
    allow_download_credentials: bool = False,
) -> dict[str, str]:
    inherited_environment = os.environ if inherited is None else inherited
    environment = {}
    for key, value in inherited_environment.items():
        normalized_key = key.upper()
        if normalized_key in INHERITED_RUNTIME_ENVIRONMENT_KEYS:
            environment[normalized_key] = value
    if allow_download_credentials and inherited_environment.get("HF_TOKEN"):
        environment["HF_TOKEN"] = inherited_environment["HF_TOKEN"]
    environment["HF_HUB_CACHE"] = str(hf_cache_directory.resolve())
    environment["LEMONADE_CI_MODE"] = "True"
    environment["LLAMA_ARG_LOG_VERBOSITY"] = "4"
    environment["NO_PROXY"] = "127.0.0.1"
    environment["no_proxy"] = "127.0.0.1"
    resolved_private_temp = str(private_temp_directory.resolve())
    environment["TMPDIR"] = resolved_private_temp
    environment["TMP"] = resolved_private_temp
    environment["TEMP"] = resolved_private_temp
    environment["HOME"] = resolved_private_temp
    environment["USERPROFILE"] = resolved_private_temp
    environment["APPDATA"] = resolved_private_temp
    environment["LOCALAPPDATA"] = resolved_private_temp
    if candidate_defaults_path is not None:
        environment["LEMONADE_DEFAULTS_PATH"] = str(candidate_defaults_path.resolve())
    return environment


def resolve_candidate_defaults(lemond: Path) -> Path:
    resource_directory = (
        lemond.parent.parent / "resources"
        if lemond.suffix.lower() == ".exe"
        else lemond.parent / "resources"
    )
    defaults_path = resource_directory / "defaults.json"
    if _is_link_or_junction(resource_directory):
        raise RuntimeError(
            f"Candidate defaults directory is not a regular build resource: "
            f"{resource_directory}"
        )
    try:
        resolved_resource_directory = resource_directory.resolve(strict=True)
        resolved_defaults = defaults_path.resolve(strict=True)
        resolved_defaults.relative_to(resolved_resource_directory)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Candidate defaults file was not found in build resources: {defaults_path}"
        ) from exc
    if _is_link_or_junction(defaults_path) or not resolved_defaults.is_file():
        raise RuntimeError(
            f"Candidate defaults file is not a regular build resource: {defaults_path}"
        )
    return resolved_defaults


def _run_validation_process(
    *,
    lemond: Path,
    cache_directory: Path,
    config_directory: Path,
    command: list[str],
    environment: dict[str, str],
    port: int,
    pid_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    backend: str | None = None,
    target: str | None = None,
    allow_darwin_github_hosted_ephemeral_runner: bool = False,
) -> tuple[int, bool]:
    containment = None
    cleanup_succeeded = False
    try:
        with stdout_path.open("x", encoding="utf-8") as stdout_log, stderr_path.open(
            "x", encoding="utf-8"
        ) as stderr_log:
            containment = _create_process_containment(
                allow_darwin_github_hosted_ephemeral_runner=(
                    allow_darwin_github_hosted_ephemeral_runner
                ),
                backend=backend,
                target=target,
            )
            process = _start_contained_process(
                containment,
                [
                    str(lemond),
                    str(cache_directory.resolve()),
                    str(config_directory.resolve()),
                    "--port",
                    str(port),
                    "--host",
                    "127.0.0.1",
                ],
                stdout=stdout_log,
                stderr=stderr_log,
                env=environment,
            )
            try:
                with pid_path.open("x", encoding="ascii") as pid_file:
                    pid_file.write(f"{process.pid}\n")
                print(f"Started lemond PID {process.pid}", flush=True)
                time.sleep(0.5)
                if process.poll() is not None:
                    raise RuntimeError(
                        f"lemond exited before validation started; see {stderr_path}"
                    )
                exit_code = _run_contained_command(
                    command,
                    env=environment,
                    containment=containment,
                    timeout_seconds=VALIDATION_PHASE_TIMEOUT_SECONDS,
                )
            finally:
                cleanup_succeeded = _shutdown_server(
                    process,
                    port,
                    cache_directory,
                    containment,
                )
    finally:
        if containment is not None and not _close_process_containment(containment):
            cleanup_succeeded = False
        _raise_if_cancelled()
    if cleanup_succeeded:
        pid_path.unlink(missing_ok=True)
    return exit_code, cleanup_succeeded


def require_validation_result_evidence(
    identity: _FileIdentity,
    allowed_root: Path,
    expected_models: list[str],
    *,
    capability_profile: str = "",
    capability_models: list[str] | None = None,
) -> None:
    try:
        load_and_validate_result_file(
            identity.resolved_path,
            expected_models,
            capability_profile=capability_profile,
            capability_models=capability_models,
        )
    except ValidationEvidenceError as exc:
        raise RuntimeError(str(exc)) from exc
    _revalidate_regular_file(identity, allowed_root, require_nonempty=True)


def run_validation(args: argparse.Namespace) -> int:
    lemond = args.lemond.resolve()
    python_executable = args.python.resolve()
    allow_huggingface_download_credentials = bool(
        getattr(args, "allow_huggingface_download_credentials", False)
    )
    allow_darwin_github_hosted_ephemeral_runner = bool(
        getattr(args, "allow_darwin_github_hosted_ephemeral_runner", False)
    )
    if not lemond.is_file():
        raise RuntimeError(f"lemond executable not found: {lemond}")
    if not python_executable.is_file():
        raise RuntimeError(f"Python executable not found: {python_executable}")
    attestation_backend = resolve_attestation_backend(args.backend, args.channel)

    command = build_validation_command(
        python_executable=python_executable,
        backend=args.backend,
        channel=args.channel,
        target=args.target,
        models_csv=args.models,
        lite=args.lite,
        capability_profile=args.capability_profile,
        capability_models_csv=args.capability_models,
        port=args.port,
    )
    command.append("--skip-install")
    prepare_command = build_prepare_validation_command(
        python_executable=python_executable,
        backend=args.backend,
        channel=args.channel,
        target=args.target,
        models_csv=args.models,
        lite=args.lite,
        capability_profile=args.capability_profile,
        capability_models_csv=args.capability_models,
        port=args.port,
    )
    selected_models = _csv_values(args.models)
    capability_models = _csv_values(args.capability_models)
    restart_models = capability_models or selected_models
    if not selected_models:
        raise ValueError("Validation requires explicit selected models")
    if not restart_models:
        raise ValueError(
            "A capability model or explicit model is required for restart validation"
        )
    artifact_locks = load_model_catalog(VALIDATION_ARTIFACT_LOCKS)
    locked_restart_models = [
        model
        for model in restart_models
        if model.removeprefix("builtin.") in artifact_locks
    ]
    restart_model_value = (locked_restart_models or restart_models)[0]
    restart_command = build_restart_validation_command(
        python_executable=python_executable,
        backend=args.backend,
        channel=args.channel,
        target=args.target,
        model=restart_model_value,
        port=args.port,
    )
    _require_idle_runner()
    _require_unused_port(args.port)
    candidate_defaults_path = resolve_candidate_defaults(lemond)

    temporary_root_value = os.environ.get("RUNNER_TEMP")
    if not temporary_root_value:
        raise RuntimeError("RUNNER_TEMP is required for validation cache isolation")
    validation_cache = create_validation_cache(Path(temporary_root_value), args.target)
    cache_directory = validation_cache.cache_directory.resolved_path
    config_directory = validation_cache.config_directory.resolved_path
    hf_cache_directory = validation_cache.hf_cache_directory.resolved_path
    private_temp_directory = validation_cache.private_temp_directory.resolved_path
    results_directory = validation_cache.results_directory.resolved_path
    logs_directory = validation_cache.logs_directory.resolved_path
    validation_result_path = (
        results_directory / f"llamacpp_validation_{args.target}.json"
    )
    restart_result_path = (
        results_directory / f"llamacpp_restart_validation_{args.target}.json"
    )
    command[command.index("--output") + 1] = str(validation_result_path)
    prepare_command[prepare_command.index("--output") + 1] = str(validation_result_path)
    restart_command[restart_command.index("--output") + 1] = str(restart_result_path)
    prepare_environment = build_server_environment(
        hf_cache_directory,
        private_temp_directory=private_temp_directory,
        candidate_defaults_path=candidate_defaults_path,
        allow_download_credentials=allow_huggingface_download_credentials,
    )
    environment = build_server_environment(
        hf_cache_directory,
        private_temp_directory=private_temp_directory,
        candidate_defaults_path=candidate_defaults_path,
    )
    pid_path = validation_cache.validation_root.resolved_path / "lemond.pid"
    prepare_stdout_path = logs_directory / "lemond.prepare.stdout.log"
    prepare_stderr_path = logs_directory / "lemond.prepare.stderr.log"
    validation_stdout_path = logs_directory / "lemond.stdout.log"
    validation_stderr_path = logs_directory / "lemond.stderr.log"
    restart_stdout_path = logs_directory / "lemond.restart.stdout.log"
    restart_stderr_path = logs_directory / "lemond.restart.stderr.log"

    revalidate_validation_cache(validation_cache)
    validation_exit_code, cleanup_succeeded = _run_validation_process(
        lemond=lemond,
        cache_directory=cache_directory,
        config_directory=config_directory,
        command=prepare_command,
        environment=prepare_environment,
        port=args.port,
        pid_path=pid_path,
        stdout_path=prepare_stdout_path,
        stderr_path=prepare_stderr_path,
        backend=args.backend,
        target=args.target,
        allow_darwin_github_hosted_ephemeral_runner=(
            allow_darwin_github_hosted_ephemeral_runner
        ),
    )
    revalidate_validation_cache(validation_cache)
    if validation_exit_code:
        return validation_exit_code
    if not cleanup_succeeded:
        print("Prepare process cleanup did not complete", file=sys.stderr)
        return 1

    def verify_selected_artifacts() -> dict[str, VerifiedValidationArtifact]:
        revalidate_validation_cache(validation_cache)
        verified = verify_validation_artifacts(
            hf_cache_directory,
            VALIDATION_MODEL_CATALOG,
            VALIDATION_ARTIFACT_LOCKS,
            selected_models,
        )
        if verified:
            print(
                "Verified immutable validation artifacts: " + ", ".join(verified),
                flush=True,
            )
        return verified

    verified_artifacts = verify_selected_artifacts()
    require_backend_proof = bool(verified_artifacts)
    attested_log_identities = []
    _require_unused_port(args.port)
    revalidate_validation_cache(validation_cache)
    validation_exit_code, cleanup_succeeded = _run_validation_process(
        lemond=lemond,
        cache_directory=cache_directory,
        config_directory=config_directory,
        command=command,
        environment=environment,
        port=args.port,
        pid_path=pid_path,
        stdout_path=validation_stdout_path,
        stderr_path=validation_stderr_path,
        backend=args.backend,
        target=args.target,
        allow_darwin_github_hosted_ephemeral_runner=(
            allow_darwin_github_hosted_ephemeral_runner
        ),
    )
    revalidate_validation_cache(validation_cache)
    if validation_exit_code:
        return validation_exit_code
    if not cleanup_succeeded:
        print("Validation process cleanup did not complete", file=sys.stderr)
        return 1
    validation_result = _capture_regular_file(
        validation_result_path,
        results_directory,
        require_nonempty=True,
    )
    require_validation_result_evidence(
        validation_result,
        results_directory,
        selected_models,
        capability_profile=args.capability_profile,
        capability_models=capability_models,
    )
    verified_artifacts = verify_selected_artifacts()
    if require_backend_proof:
        attested_log_identities.extend(
            require_accelerator_attestation(
                [validation_stdout_path, validation_stderr_path],
                logs_directory,
                "validation",
                verified_artifacts,
                attestation_backend,
                selected_models=selected_models,
            )
        )
    _require_unused_port(args.port)
    revalidate_validation_cache(validation_cache)
    restart_exit_code, restart_cleanup_succeeded = _run_validation_process(
        lemond=lemond,
        cache_directory=cache_directory,
        config_directory=config_directory,
        command=restart_command,
        environment=environment,
        port=args.port,
        pid_path=pid_path,
        stdout_path=restart_stdout_path,
        stderr_path=restart_stderr_path,
        backend=args.backend,
        target=args.target,
        allow_darwin_github_hosted_ephemeral_runner=(
            allow_darwin_github_hosted_ephemeral_runner
        ),
    )
    revalidate_validation_cache(validation_cache)
    if restart_exit_code:
        return restart_exit_code
    if not restart_cleanup_succeeded:
        print("Restart validation process cleanup did not complete", file=sys.stderr)
        return 1
    restart_result = _capture_regular_file(
        restart_result_path,
        results_directory,
        require_nonempty=True,
    )
    require_validation_result_evidence(
        restart_result,
        results_directory,
        [restart_model_value],
    )
    verified_artifacts = verify_selected_artifacts()
    restart_model = restart_model_value.removeprefix("builtin.")
    if require_backend_proof and restart_model in verified_artifacts:
        attested_log_identities.extend(
            require_accelerator_attestation(
                [restart_stdout_path, restart_stderr_path],
                logs_directory,
                "restart validation",
                {restart_model: verified_artifacts[restart_model]},
                attestation_backend,
                selected_models=[restart_model_value],
            )
        )
    revalidate_validation_cache(validation_cache)
    _revalidate_regular_file(
        validation_result,
        results_directory,
        require_nonempty=True,
    )
    _revalidate_regular_file(
        restart_result,
        results_directory,
        require_nonempty=True,
    )
    log_paths = (
        prepare_stdout_path,
        prepare_stderr_path,
        validation_stdout_path,
        validation_stderr_path,
        restart_stdout_path,
        restart_stderr_path,
    )
    log_identities_by_path = {
        identity.path: identity for identity in attested_log_identities
    }
    for path in log_paths:
        if path not in log_identities_by_path:
            log_identities_by_path[path] = _capture_regular_file(
                path,
                logs_directory,
                require_nonempty=False,
            )
    log_identities = list(log_identities_by_path.values())
    for identity in log_identities:
        _revalidate_regular_file(identity, logs_directory, require_nonempty=False)
    output_log_paths = {
        identity.path: identity.resolved_path for identity in log_identities
    }
    _emit_workflow_outputs(
        {
            "validation_result": validation_result.resolved_path,
            "restart_result": restart_result.resolved_path,
            "prepare_stdout_log": output_log_paths[prepare_stdout_path],
            "prepare_stderr_log": output_log_paths[prepare_stderr_path],
            "validation_stdout_log": output_log_paths[validation_stdout_path],
            "validation_stderr_log": output_log_paths[validation_stderr_path],
            "restart_stdout_log": output_log_paths[restart_stdout_path],
            "restart_stderr_log": output_log_paths[restart_stderr_path],
        }
    )
    return 0


def _bool_argument(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized in {"", "false"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lemond", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument(
        "--backend",
        default=os.environ.get("LLAMACPP_BACKEND"),
        choices=("cpu", "cuda", "metal", "rocm", "vulkan"),
    )
    parser.add_argument(
        "--channel",
        default=os.environ.get("LLAMACPP_CHANNEL", ""),
        choices=("", "stable", "nightly"),
    )
    parser.add_argument("--target", default=os.environ.get("TARGET"))
    parser.add_argument("--models", default=os.environ.get("LLAMACPP_TEST_MODELS", ""))
    parser.add_argument(
        "--lite",
        default=_bool_argument(os.environ.get("LITE_MODE", "false")),
        type=_bool_argument,
    )
    parser.add_argument(
        "--capability-profile", default=os.environ.get("CAPABILITY_PROFILE", "")
    )
    parser.add_argument(
        "--capability-models", default=os.environ.get("CAPABILITY_MODELS", "")
    )
    parser.add_argument(
        "--allow-huggingface-download-credentials",
        action="store_true",
        help="allow the prepare phase to inherit the canonical HF_TOKEN value",
    )
    parser.add_argument(
        "--allow-darwin-github-hosted-ephemeral-runner",
        action="store_true",
        help=(
            "accept github-hosted macos-latest VM disposal as the Darwin "
            "descendant-containment boundary"
        ),
    )
    parser.add_argument("--port", default=13305, type=int)
    args = parser.parse_args()
    if not args.backend:
        parser.error("--backend or LLAMACPP_BACKEND is required")
    if not args.target:
        parser.error("--target or TARGET is required")
    try:
        with _CancellationSignalGuard():
            exit_code = run_validation(args)
            _raise_if_cancelled()
    except _ValidationCancelled as exc:
        signal_name = signal.Signals(exc.signum).name
        parser.exit(
            128 + exc.signum,
            f"ERROR: validation cancelled by {signal_name}\n",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
