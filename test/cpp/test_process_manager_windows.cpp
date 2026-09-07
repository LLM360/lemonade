#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <windows.h>

#include <lemon/utils/process_manager.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <future>
#include <string>
#include <thread>
#include <vector>

using lemon::utils::ProcessHandle;
using lemon::utils::ProcessManager;

namespace {

int failures = 0;

void check(const char* name, bool condition) {
    std::printf("[%s] %s\n", condition ? "PASS" : "FAIL", name);
    if (!condition) {
        ++failures;
    }
}

bool standard_handle_is_valid(DWORD stream) {
    const HANDLE handle = GetStdHandle(stream);
    if (handle == nullptr || handle == INVALID_HANDLE_VALUE) {
        return false;
    }

    SetLastError(ERROR_SUCCESS);
    const DWORD type = GetFileType(handle);
    return type != FILE_TYPE_UNKNOWN || GetLastError() == ERROR_SUCCESS;
}

int probe_standard_handles() {
    int result = 0;
    if (!standard_handle_is_valid(STD_INPUT_HANDLE)) {
        result |= 1;
    }
    if (!standard_handle_is_valid(STD_OUTPUT_HANDLE)) {
        result |= 2;
    }
    if (!standard_handle_is_valid(STD_ERROR_HANDLE)) {
        result |= 4;
    }

    if ((result & 2) == 0) {
        constexpr char message[] = "stdout-ready\n";
        DWORD written = 0;
        WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), message,
                  static_cast<DWORD>(sizeof(message) - 1), &written, nullptr);
    }
    if ((result & 4) == 0) {
        constexpr char message[] = "stderr-ready\n";
        DWORD written = 0;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), message,
                  static_cast<DWORD>(sizeof(message) - 1), &written, nullptr);
    }
    return result;
}

std::string current_executable() {
    std::vector<char> path(32768);
    const DWORD length = GetModuleFileNameA(
        nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0 ||
        static_cast<std::size_t>(length) >= path.size()) {
        return {};
    }
    return std::string(path.data(), length);
}

std::string unique_sentinel_path() {
    char temp_directory[MAX_PATH + 1] = {};
    const DWORD directory_length = GetTempPathA(
        static_cast<DWORD>(sizeof(temp_directory)), temp_directory);
    if (directory_length == 0 || directory_length >= sizeof(temp_directory)) {
        return {};
    }

    char path[MAX_PATH + 1] = {};
    if (GetTempFileNameA(temp_directory, "lpm", 0, path) == 0) {
        return {};
    }
    if (!DeleteFileA(path)) {
        return {};
    }
    return path;
}

bool create_sentinel(const std::string& path) {
    const HANDLE file = CreateFileA(
        path.c_str(), GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }
    return CloseHandle(file) != FALSE;
}

bool sentinel_exists(const std::string& path) {
    const DWORD attributes = GetFileAttributesA(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES &&
           (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

bool wait_for_sentinel(
    const std::string& path,
    std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        if (sentinel_exists(path)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    return sentinel_exists(path);
}

bool wait_for_flag(
    const std::atomic<bool>& flag,
    std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        if (flag.load(std::memory_order_acquire)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    return flag.load(std::memory_order_acquire);
}

bool pipe_closed_without_marker(HANDLE pipe) {
    char marker = '\0';
    DWORD bytes_read = 0;
    if (ReadFile(pipe, &marker, 1, &bytes_read, nullptr)) {
        return bytes_read == 0;
    }
    return GetLastError() == ERROR_BROKEN_PIPE;
}

class TemporarySentinel {
public:
    TemporarySentinel() : path_(unique_sentinel_path()) {
    }

    ~TemporarySentinel() {
        if (!path_.empty()) {
            DeleteFileA(path_.c_str());
        }
    }

    TemporarySentinel(const TemporarySentinel&) = delete;
    TemporarySentinel& operator=(const TemporarySentinel&) = delete;

    const std::string& path() const {
        return path_;
    }

    bool valid() const {
        return !path_.empty();
    }

    bool signal() const {
        return valid() && create_sentinel(path_);
    }

private:
    std::string path_;
};

class ScopedHandle {
public:
    ScopedHandle() = default;

    ~ScopedHandle() {
        reset();
    }

    ScopedHandle(const ScopedHandle&) = delete;
    ScopedHandle& operator=(const ScopedHandle&) = delete;

    HANDLE get() const {
        return handle_;
    }

    HANDLE* receive() {
        reset();
        return &handle_;
    }

    void reset(HANDLE handle = nullptr) {
        if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
        }
        handle_ = handle;
    }

private:
    HANDLE handle_ = nullptr;
};

int capture_wait_child(const std::string& release_path) {
    constexpr char message[] = "capture-ready\n";
    DWORD written = 0;
    if (!WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), message,
                   static_cast<DWORD>(sizeof(message) - 1), &written,
                   nullptr) ||
        written != sizeof(message) - 1) {
        return 10;
    }
    return wait_for_sentinel(release_path, std::chrono::seconds(30)) ? 0 : 11;
}

int long_wait_child(
    const std::string& ready_path,
    const std::string& release_path,
    const std::string& canary_handle_value) {
    std::uintptr_t canary_value = 0;
    try {
        std::size_t parsed = 0;
        canary_value = static_cast<std::uintptr_t>(
            std::stoull(canary_handle_value, &parsed));
        if (parsed != canary_handle_value.size()) {
            return 20;
        }
    } catch (const std::exception&) {
        return 20;
    }

    constexpr char marker = 'I';
    DWORD written = 0;
    WriteFile(reinterpret_cast<HANDLE>(canary_value), &marker, 1, &written,
              nullptr);
    if (!create_sentinel(ready_path)) {
        return 21;
    }
    return wait_for_sentinel(release_path, std::chrono::seconds(30)) ? 0 : 22;
}

class InvalidStandardHandles {
public:
    explicit InvalidStandardHandles(
        HANDLE replacement = INVALID_HANDLE_VALUE)
        : input_(GetStdHandle(STD_INPUT_HANDLE)),
          output_(GetStdHandle(STD_OUTPUT_HANDLE)),
          error_(GetStdHandle(STD_ERROR_HANDLE)) {
        const BOOL input_set =
            SetStdHandle(STD_INPUT_HANDLE, replacement);
        const BOOL output_set =
            SetStdHandle(STD_OUTPUT_HANDLE, replacement);
        const BOOL error_set =
            SetStdHandle(STD_ERROR_HANDLE, replacement);
        installed_ = input_set && output_set && error_set;
    }

    ~InvalidStandardHandles() {
        SetStdHandle(STD_INPUT_HANDLE, input_);
        SetStdHandle(STD_OUTPUT_HANDLE, output_);
        SetStdHandle(STD_ERROR_HANDLE, error_);
    }

    bool installed() const {
        return installed_;
    }

private:
    HANDLE input_;
    HANDLE output_;
    HANDLE error_;
    bool installed_ = false;
};

void test_concurrent_capture_handle_isolation(const std::string& executable) {
    TemporarySentinel capture_release;
    TemporarySentinel long_ready;
    TemporarySentinel long_release;
    const bool sentinels_ready = capture_release.valid() &&
                                 long_ready.valid() && long_release.valid();
    check("create process-overlap sentinels", sentinels_ready);
    if (!sentinels_ready) {
        return;
    }

    std::atomic<bool> capture_ready{false};
    std::string captured_output;
    std::future<int> capture_future;
    bool capture_launched = false;
    try {
        capture_future = std::async(
            std::launch::async,
            [&]() {
                return ProcessManager::run_process_with_output(
                    executable, {"--capture-wait", capture_release.path()},
                    [&](const std::string& line) {
                        captured_output += line;
                        captured_output.push_back('\n');
                        if (line == "capture-ready") {
                            capture_ready.store(true, std::memory_order_release);
                        }
                        return true;
                    },
                    "", 20);
            });
        capture_launched = true;
    } catch (const std::exception&) {
    }
    check("launch capture child A", capture_launched);
    if (!capture_launched) {
        return;
    }

    const bool capture_announced =
        wait_for_flag(capture_ready, std::chrono::seconds(5));
    check("capture child A announces readiness", capture_announced);

    ScopedHandle canary_read;
    ScopedHandle canary_write;
    bool canary_created = false;
    if (capture_announced) {
        SECURITY_ATTRIBUTES attributes{};
        attributes.nLength = sizeof(attributes);
        attributes.bInheritHandle = TRUE;
        if (CreatePipe(
                canary_read.receive(), canary_write.receive(), &attributes,
                0) &&
            SetHandleInformation(
                canary_read.get(), HANDLE_FLAG_INHERIT, 0)) {
            canary_created = true;
        }
    }
    check("create inheritable overlap canary", canary_created);

    ProcessHandle long_handle{};
    bool long_started = false;
    if (canary_created) {
        try {
            long_handle = ProcessManager::start_process(
                executable,
                {"--long-wait", long_ready.path(), long_release.path(),
                 std::to_string(reinterpret_cast<std::uintptr_t>(
                     canary_write.get()))}, "",
                false, false);
            long_started = long_handle.handle != nullptr;
        } catch (const std::exception&) {
        }
    }
    check("start long-lived child B during capture", long_started);

    bool long_announced = false;
    if (long_started) {
        long_announced =
            wait_for_sentinel(long_ready.path(), std::chrono::seconds(5));
    }
    check("long-lived child B announces readiness", long_announced);

    const bool long_running_before_release =
        long_announced && ProcessManager::is_running(long_handle);
    check("long-lived child B is running", long_running_before_release);

    canary_write.reset();
    const bool canary_excluded =
        canary_created && long_running_before_release &&
        pipe_closed_without_marker(canary_read.get());
    check("child B excludes unrelated inheritable pipe writers",
          canary_excluded);

    const bool capture_released = capture_release.signal();
    check("release capture child A", capture_released);

    bool capture_completed_promptly = false;
    int capture_exit = -1;
    bool capture_result_collected = false;
    if (capture_released && long_running_before_release) {
        capture_completed_promptly =
            capture_future.wait_for(std::chrono::seconds(3)) ==
            std::future_status::ready;
        if (capture_completed_promptly) {
            try {
                capture_exit = capture_future.get();
                capture_result_collected = true;
            } catch (const std::exception&) {
            }
        }
    }

    const bool long_running_after_capture =
        capture_result_collected && ProcessManager::is_running(long_handle);
    check("capture returns promptly while child B remains running",
          capture_completed_promptly && long_running_after_capture);

    capture_release.signal();
    const bool long_released = long_release.signal();
    check("release long-lived child B", long_released);

    int long_exit = -1;
    if (long_started) {
        long_exit = ProcessManager::wait_for_exit(long_handle, 10);
        if (long_exit >= 0) {
            ProcessManager::reap_process(long_handle);
        } else {
            ProcessManager::kill_process(long_handle);
        }
    }
    check("long-lived child B exits after release", long_exit == 0);

    if (!capture_result_collected) {
        try {
            capture_exit = capture_future.get();
            capture_result_collected = true;
        } catch (const std::exception&) {
        }
    }
    check("capture child A exits after release",
          capture_result_collected && capture_exit == 0);
    check("capture child A output is complete",
          captured_output.find("capture-ready\n") != std::string::npos);
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--probe-stdio") {
        return probe_standard_handles();
    }
    if (argc == 3 && std::string(argv[1]) == "--capture-wait") {
        return capture_wait_child(argv[2]);
    }
    if (argc == 5 && std::string(argv[1]) == "--long-wait") {
        return long_wait_child(argv[2], argv[3], argv[4]);
    }

    const std::string executable = current_executable();
    check("resolve test executable", !executable.empty());
    if (executable.empty()) {
        return failures;
    }

    ProcessHandle quiet_handle{};
    bool quiet_override_installed = false;
    bool quiet_started = false;
    {
        InvalidStandardHandles invalid_stdio;
        quiet_override_installed = invalid_stdio.installed();
        if (quiet_override_installed) {
            try {
                quiet_handle = ProcessManager::start_process(
                    executable, {"--probe-stdio"}, "", false, false);
                quiet_started = quiet_handle.handle != nullptr;
            } catch (const std::exception&) {
            }
        }
    }
    check("replace parent standard handles", quiet_override_installed);
    check("start quiet child without parent standard handles", quiet_started);

    int quiet_exit = -1;
    if (quiet_started) {
        quiet_exit = ProcessManager::wait_for_exit(quiet_handle, 10);
        if (quiet_exit >= 0) {
            ProcessManager::reap_process(quiet_handle);
        } else {
            ProcessManager::kill_process(quiet_handle);
        }
    }
    check("quiet child receives valid standard handles", quiet_exit == 0);

    ProcessHandle stale_handle{};
    bool stale_override_installed = false;
    bool stale_started = false;
    const HANDLE stale_value = reinterpret_cast<HANDLE>(
        static_cast<std::uintptr_t>(0x1234567));
    {
        InvalidStandardHandles stale_stdio(stale_value);
        stale_override_installed = stale_stdio.installed();
        if (stale_override_installed) {
            try {
                stale_handle = ProcessManager::start_process(
                    executable, {"--probe-stdio"}, "", false, false);
                stale_started = stale_handle.handle != nullptr;
            } catch (const std::exception&) {
            }
        }
    }
    check("replace parent standard handles with stale values",
          stale_override_installed);
    check("start child with stale parent standard handles", stale_started);

    int stale_exit = -1;
    if (stale_started) {
        stale_exit = ProcessManager::wait_for_exit(stale_handle, 10);
        if (stale_exit >= 0) {
            ProcessManager::reap_process(stale_handle);
        } else {
            ProcessManager::kill_process(stale_handle);
        }
    }
    check("stale standard handles are replaced with valid handles",
          stale_exit == 0);

    bool capture_override_installed = false;
    int capture_exit = -1;
    std::string captured_output;
    {
        InvalidStandardHandles invalid_stdio;
        capture_override_installed = invalid_stdio.installed();
        if (capture_override_installed) {
            try {
                capture_exit = ProcessManager::run_process_with_output(
                    executable, {"--probe-stdio"},
                    [&captured_output](const std::string& line) {
                        captured_output += line;
                        captured_output.push_back('\n');
                        return true;
                    },
                    "", 10);
            } catch (const std::exception&) {
            }
        }
    }
    check("replace standard handles for captured child",
          capture_override_installed);
    check("captured child receives valid standard handles", capture_exit == 0);
    check("captured stdout remains connected",
          captured_output.find("stdout-ready\n") != std::string::npos);
    check("captured stderr remains connected",
          captured_output.find("stderr-ready\n") != std::string::npos);

    test_concurrent_capture_handle_isolation(executable);

    if (failures == 0) {
        std::printf("\nAll Windows process_manager tests passed\n");
        return 0;
    }
    std::printf("\n%d Windows process_manager test(s) failed\n", failures);
    return 1;
}
