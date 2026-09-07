// Windows header discipline — must precede all other includes.
// Mirrors the setup that was previously in process_manager.cpp before
// the platform files were made self-contained.
#ifdef _WIN32
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <windows.h>
#include <processenv.h>
#pragma comment(lib, "ws2_32.lib")
// Un-define ERROR to avoid conflict with LOG(ERROR, ...) / SEVERITY::ERROR
#ifdef ERROR
#undef ERROR
#endif
#endif

#include <lemon/utils/process_platform.h>
#include <lemon/utils/aixlog.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <new>
#include <stdexcept>
#include <thread>

namespace lemon {
namespace utils {

// Helper function: escape Windows command-line arguments
static std::string escape_windows_arg(const std::string& arg) {
    std::string result = "\"";
    for (size_t i = 0; i < arg.size(); ++i) {
        if (arg[i] == '"') {
            // Escape the quote with a backslash
            result += "\\\"";
        } else if (arg[i] == '\\') {
            // Check if this backslash is followed by a quote
            // If so, we need to escape the backslash too
            if (i + 1 < arg.size() && arg[i + 1] == '"') {
                result += "\\\\";
            } else {
                result += '\\';
            }
        } else {
            result += arg[i];
        }
    }
    result += "\"";
    return result;
}

static BOOL create_process_with_restricted_handles(
    LPSTR command_line,
    DWORD creation_flags,
    LPVOID environment,
    LPCSTR working_directory,
    const STARTUPINFOA& startup_info,
    PROCESS_INFORMATION* process_information) {
    STARTUPINFOEXA extended_startup_info{};
    extended_startup_info.StartupInfo = startup_info;

    HANDLE source_handles[] = {
        startup_info.hStdInput,
        startup_info.hStdOutput,
        startup_info.hStdError,
    };
    HANDLE fallback_handles[3] = {};
    HANDLE unique_source_handles[3] = {};
    HANDLE inherited_handles[3] = {};
    DWORD inherited_handle_count = 0;
    SIZE_T attribute_list_size = 0;
    void* attribute_storage = nullptr;
    bool attribute_list_initialized = false;
    auto cleanup = [&]() {
        if (attribute_list_initialized) {
            DeleteProcThreadAttributeList(
                extended_startup_info.lpAttributeList);
        }
        if (attribute_storage != nullptr) {
            HeapFree(GetProcessHeap(), 0, attribute_storage);
        }
        for (DWORD i = 0; i < inherited_handle_count; ++i) {
            CloseHandle(inherited_handles[i]);
        }
        for (DWORD i = 0; i < 3; ++i) {
            if (fallback_handles[i] != nullptr &&
                fallback_handles[i] != INVALID_HANDLE_VALUE) {
                CloseHandle(fallback_handles[i]);
            }
        }
    };

    if ((startup_info.dwFlags & STARTF_USESTDHANDLES) != 0) {
        for (DWORD i = 0; i < 3; ++i) {
            bool usable = source_handles[i] != nullptr &&
                          source_handles[i] != INVALID_HANDLE_VALUE;
            if (usable) {
                SetLastError(ERROR_SUCCESS);
                const DWORD type = GetFileType(source_handles[i]);
                usable = type != FILE_TYPE_UNKNOWN ||
                         GetLastError() == ERROR_SUCCESS;
            }
            if (usable) {
                continue;
            }

            const DWORD access = i == 0 ? GENERIC_READ : GENERIC_WRITE;
            const HANDLE fallback = CreateFileA(
                "NUL", access, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (fallback == INVALID_HANDLE_VALUE) {
                const DWORD error = GetLastError();
                cleanup();
                SetLastError(error);
                return FALSE;
            }
            fallback_handles[i] = fallback;
            source_handles[i] = fallback;
        }

        for (DWORD i = 0; i < 3; ++i) {
            const HANDLE source = source_handles[i];
            DWORD existing = 0;
            while (existing < inherited_handle_count &&
                   unique_source_handles[existing] != source) {
                ++existing;
            }

            HANDLE inherited = nullptr;
            if (existing < inherited_handle_count) {
                inherited = inherited_handles[existing];
            } else {
                if (!DuplicateHandle(
                        GetCurrentProcess(), source, GetCurrentProcess(),
                        &inherited, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
                    const DWORD error = GetLastError();
                    cleanup();
                    SetLastError(error);
                    return FALSE;
                }
                unique_source_handles[inherited_handle_count] = source;
                inherited_handles[inherited_handle_count] = inherited;
                ++inherited_handle_count;
            }

            if (i == 0) {
                extended_startup_info.StartupInfo.hStdInput = inherited;
            } else if (i == 1) {
                extended_startup_info.StartupInfo.hStdOutput = inherited;
            } else {
                extended_startup_info.StartupInfo.hStdError = inherited;
            }
        }
    }

    BOOL inherit_handles = FALSE;
    DWORD effective_creation_flags = creation_flags;
    if (inherited_handle_count > 0) {
        InitializeProcThreadAttributeList(
            nullptr, 1, 0, &attribute_list_size);
        if (attribute_list_size == 0) {
            const DWORD error = GetLastError();
            cleanup();
            SetLastError(error);
            return FALSE;
        }

        attribute_storage =
            HeapAlloc(GetProcessHeap(), 0, attribute_list_size);
        if (attribute_storage == nullptr) {
            cleanup();
            SetLastError(ERROR_NOT_ENOUGH_MEMORY);
            return FALSE;
        }
        extended_startup_info.lpAttributeList =
            static_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attribute_storage);
        if (!InitializeProcThreadAttributeList(
                extended_startup_info.lpAttributeList, 1, 0,
                &attribute_list_size)) {
            const DWORD error = GetLastError();
            cleanup();
            SetLastError(error);
            return FALSE;
        }
        attribute_list_initialized = true;
        if (!UpdateProcThreadAttribute(
                extended_startup_info.lpAttributeList, 0,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited_handles,
                inherited_handle_count * sizeof(HANDLE), nullptr, nullptr)) {
            const DWORD error = GetLastError();
            cleanup();
            SetLastError(error);
            return FALSE;
        }
        extended_startup_info.StartupInfo.cb = sizeof(STARTUPINFOEXA);
        effective_creation_flags |= EXTENDED_STARTUPINFO_PRESENT;
        inherit_handles = TRUE;
    }

    const BOOL success = CreateProcessA(
        nullptr, command_line, nullptr, nullptr, inherit_handles,
        effective_creation_flags, environment, working_directory,
        &extended_startup_info.StartupInfo, process_information);
    const DWORD error = success ? ERROR_SUCCESS : GetLastError();
    cleanup();
    SetLastError(error);
    return success;
}

// Helper function to check if a line should be filtered
static bool should_filter_line(const std::string& line) {
    // Filter out health check requests (both /health and /v1/health)
    // Also filter FLM's interactive prompt spam
    return (line.find("GET /health") != std::string::npos ||
            line.find("GET /v1/health") != std::string::npos ||
            // idle heartbeat returned by llamma cpp when its /metrics is scrapped. supressed to decrease visual clutering
            line.find("srv  update_slots: all slots are idle") != std::string::npos ||
            line.find("Enter 'exit' to stop the server") != std::string::npos);
}

static bool is_error_line(const std::string& line) {
    std::string lowered = line;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered.find("error") != std::string::npos;
}

// Helper function: filter and log process output
static void log_process_line(const std::string& line) {
    if (should_filter_line(line)) {
        return;
    }

    if (is_error_line(line)) {
        LOG(ERROR, "Process") << line << std::endl;
    } else {
        LOG(INFO, "Process") << line << std::endl;
    }
}

struct OutputReaderContext {
    HANDLE pipe;
    bool log_output;
    std::shared_ptr<ProcessOutputCapture> output_capture;
};

static DWORD WINAPI output_filter_thread(LPVOID param) {
    std::unique_ptr<OutputReaderContext> context(
        static_cast<OutputReaderContext*>(param));
    char buffer[4096];
    DWORD bytes_read;
    std::string line_buffer;

    while (ReadFile(context->pipe, buffer, sizeof(buffer), &bytes_read, nullptr) &&
           bytes_read > 0) {
        if (context->output_capture) {
            context->output_capture->append(
                buffer, static_cast<std::size_t>(bytes_read));
        }
        if (!context->log_output) {
            continue;
        }

        line_buffer.append(buffer, static_cast<std::size_t>(bytes_read));

        size_t pos;
        while ((pos = line_buffer.find('\n')) != std::string::npos) {
            std::string line = line_buffer.substr(0, pos);
            line_buffer.erase(0, pos + 1);
            log_process_line(line);
        }
    }

    if (context->log_output && !line_buffer.empty()) {
        log_process_line(line_buffer);
    }
    CloseHandle(context->pipe);
    if (context->output_capture) {
        context->output_capture->finish_reader();
    }
    return 0;
}

static void start_output_reader(
    HANDLE pipe,
    bool log_output,
    const std::shared_ptr<ProcessOutputCapture>& output_capture) {
    auto* context =
        new (std::nothrow) OutputReaderContext{pipe, log_output, output_capture};
    if (context == nullptr) {
        CloseHandle(pipe);
        if (output_capture) {
            output_capture->finish_reader();
        }
        LOG(ERROR, "ProcessManager")
            << "Failed to allocate process output reader context" << std::endl;
        return;
    }

    HANDLE thread = CreateThread(
        nullptr, 0, output_filter_thread, context, 0, nullptr);
    if (thread != nullptr) {
        CloseHandle(thread);
        return;
    }

    CloseHandle(pipe);
    if (output_capture) {
        output_capture->finish_reader();
    }
    delete context;
}

// Helper function: lowercase ASCII string for case-insensitive comparison
static std::string lowercase_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Helper function: build Windows environment block
static std::vector<char> build_windows_environment_block(
    const std::vector<std::pair<std::string, std::string>>& env_vars) {
    std::vector<std::string> merged_entries;

    LPWCH environment = GetEnvironmentStringsW();
    if (environment) {
        for (const wchar_t* entry = environment; *entry != L'\0';
             entry += std::wcslen(entry) + 1) {
            int size_needed = WideCharToMultiByte(CP_UTF8, 0, entry, -1, nullptr, 0, nullptr, nullptr);
            if (size_needed > 0) {
                std::string narrow(size_needed - 1, '\0');
                WideCharToMultiByte(CP_UTF8, 0, entry, -1, &narrow[0], size_needed, nullptr, nullptr);
                merged_entries.emplace_back(std::move(narrow));
            }
        }
        FreeEnvironmentStringsW(environment);
    }

    for (const auto& env : env_vars) {
        const std::string key_lower = lowercase_ascii(env.first);
        const std::string new_entry = env.first + "=" + env.second;

        bool replaced = false;
        for (auto& existing : merged_entries) {
            size_t equals = existing.find('=');
            if (equals == std::string::npos) {
                continue;
            }

            std::string existing_key = lowercase_ascii(existing.substr(0, equals));
            if (existing_key == key_lower) {
                existing = new_entry;
                replaced = true;
                break;
            }
        }

        if (!replaced) {
            merged_entries.push_back(new_entry);
        }
    }

    std::vector<char> block;
    for (const auto& entry : merged_entries) {
        block.insert(block.end(), entry.begin(), entry.end());
        block.push_back('\0');
    }
    block.push_back('\0');
    return block;
}

// Windows ProcessPlatform implementation
class WindowsProcessPlatform : public ProcessPlatform {
public:
    ProcessHandle spawn(
        const std::string& executable,
        const std::vector<std::string>& args,
        const std::string& working_dir,
        bool inherit_output,
        bool filter_health_logs,
        const std::vector<std::pair<std::string, std::string>>& env_vars,
        std::shared_ptr<ProcessOutputCapture> output_capture) override {

        ProcessHandle handle;
        handle.handle = nullptr;
        handle.pid = 0;
        handle.output_capture = output_capture;

        std::string cmdline = escape_windows_arg(executable);
        for (const auto& arg : args) {
            cmdline += " " + escape_windows_arg(arg);
        }

        STARTUPINFOA si;
        PROCESS_INFORMATION pi;
        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        ZeroMemory(&pi, sizeof(pi));

        HANDLE stdout_read = nullptr;
        HANDLE stdout_write = nullptr;
        HANDLE stderr_read = nullptr;
        HANDLE stderr_write = nullptr;
        HANDLE nul_input = nullptr;

        bool use_filtered_output =
            (inherit_output && filter_health_logs) || output_capture != nullptr;

        if (inherit_output && !filter_health_logs) {
            const HANDLE std_in = GetStdHandle(STD_INPUT_HANDLE);
            const HANDLE std_out = GetStdHandle(STD_OUTPUT_HANDLE);
            const HANDLE std_err = GetStdHandle(STD_ERROR_HANDLE);

            const bool invalid_stdio =
                (std_in == nullptr || std_in == INVALID_HANDLE_VALUE) ||
                (std_out == nullptr || std_out == INVALID_HANDLE_VALUE) ||
                (std_err == nullptr || std_err == INVALID_HANDLE_VALUE);

            if (invalid_stdio) {
                use_filtered_output = true;
                if (!output_capture) {
                    output_capture =
                        std::make_shared<ProcessOutputCapture>(0);
                    handle.output_capture = output_capture;
                }
                LOG(WARNING, "ProcessManager")
                    << "Parent std handles are unavailable; enabling filtered output capture"
                    << std::endl;
            }
        }

        if (use_filtered_output) {
            if (!CreatePipe(&stdout_read, &stdout_write, nullptr, 0)) {
                throw std::runtime_error("Failed to create stdout pipe");
            }
            if (!CreatePipe(&stderr_read, &stderr_write, nullptr, 0)) {
                CloseHandle(stdout_read);
                CloseHandle(stdout_write);
                throw std::runtime_error("Failed to create stderr pipe");
            }

            si.dwFlags = STARTF_USESTDHANDLES;
            si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
            if (si.hStdInput == nullptr || si.hStdInput == INVALID_HANDLE_VALUE) {
                nul_input = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                if (nul_input != INVALID_HANDLE_VALUE) {
                    si.hStdInput = nul_input;
                } else {
                    si.hStdInput = nullptr;
                }
            }
            si.hStdOutput = stdout_write;
            si.hStdError = stderr_write;

            if (inherit_output) {
                LOG(DEBUG, "ProcessManager") << "Starting process with filtered output: " << cmdline << std::endl;
            }
        } else if (inherit_output) {
            // Direct inheritance without filtering
            si.dwFlags |= STARTF_USESTDHANDLES;
            si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
            si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
            si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
            LOG(DEBUG, "ProcessManager") << "Starting process with inherited output: " << cmdline << std::endl;
        } else {
            // Redirect to NUL to suppress output when not in debug mode
            si.dwFlags |= STARTF_USESTDHANDLES;
            si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);

            HANDLE hNul = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (hNul != INVALID_HANDLE_VALUE) {
                si.hStdOutput = hNul;
                si.hStdError = hNul;
            }
        }

        std::vector<char> environment_block;
        if (!env_vars.empty()) {
            environment_block = build_windows_environment_block(env_vars);
        }

        BOOL success = create_process_with_restricted_handles(
            const_cast<char*>(cmdline.c_str()),
            (inherit_output && !use_filtered_output) ? 0 : CREATE_NO_WINDOW,
            environment_block.empty() ? nullptr : environment_block.data(),
            working_dir.empty() ? nullptr : working_dir.c_str(),
            si,
            &pi
        );

        // If we opened a NUL handle, we can close it now (the child process has its own inherited handle)
        if (!inherit_output && !use_filtered_output &&
            si.hStdOutput != nullptr && si.hStdOutput != INVALID_HANDLE_VALUE) {
            CloseHandle(si.hStdOutput);
        }

        if (!success) {
            DWORD error = GetLastError();
            char error_msg[256];
            FormatMessageA(
                FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                nullptr,
                error,
                0,
                error_msg,
                sizeof(error_msg),
                nullptr
            );

            if (stdout_write) CloseHandle(stdout_write);
            if (stderr_write) CloseHandle(stderr_write);
            if (stdout_read) CloseHandle(stdout_read);
            if (stderr_read) CloseHandle(stderr_read);
            if (nul_input && nul_input != INVALID_HANDLE_VALUE) CloseHandle(nul_input);

            std::string full_error = "Failed to start process '" + executable +
                                    "': " + error_msg + " (Error code: " + std::to_string(error) + ")";
            LOG(ERROR, "ProcessManager") << full_error << std::endl;
            throw std::runtime_error(full_error);
        }

        if (nul_input && nul_input != INVALID_HANDLE_VALUE) {
            CloseHandle(nul_input);
        }

        // Close write ends of pipes in parent process
        if (stdout_write) CloseHandle(stdout_write);
        if (stderr_write) CloseHandle(stderr_write);

        if (use_filtered_output) {
            start_output_reader(stdout_read, inherit_output, output_capture);
            start_output_reader(stderr_read, inherit_output, output_capture);
        }

        if (inherit_output) {
            LOG(INFO, "ProcessManager") << "Process started successfully, PID: " << pi.dwProcessId << std::endl;
        }

        handle.handle = pi.hProcess;
        handle.pid = pi.dwProcessId;
        CloseHandle(pi.hThread);

        return handle;
    }

    void terminate(ProcessHandle handle) override {
        if (handle.handle) {
            TerminateProcess(handle.handle, 0);
            WaitForSingleObject(handle.handle, 5000);  // Wait up to 5 seconds
            CloseHandle(handle.handle);
        }
    }

    bool is_running(ProcessHandle handle) override {
        if (!handle.handle) {
            return false;
        }

        DWORD wait_result = WaitForSingleObject(handle.handle, 0);
        if (wait_result == WAIT_OBJECT_0) {
            return false;
        }
        if (wait_result == WAIT_TIMEOUT) {
            return true;
        }

        DWORD exit_code;
        if (!GetExitCodeProcess(handle.handle, &exit_code)) {
            return false;
        }

        return exit_code == STILL_ACTIVE;
    }

    int get_exit_code(ProcessHandle handle) override {
        if (!handle.handle) {
            return -1;
        }

        DWORD exit_code;
        if (!GetExitCodeProcess(handle.handle, &exit_code)) {
            return -1;
        }

        if (exit_code == STILL_ACTIVE) {
            return -1;
        }

        return static_cast<int>(exit_code);
    }

    int wait_for_exit(ProcessHandle handle, int timeout_seconds) override {
        if (!handle.handle) {
            return -1;
        }

        DWORD timeout_ms = (timeout_seconds < 0) ? INFINITE : (timeout_seconds * 1000);
        DWORD result = WaitForSingleObject(handle.handle, timeout_ms);

        if (result == WAIT_TIMEOUT) {
            return -1;
        }

        DWORD exit_code;
        GetExitCodeProcess(handle.handle, &exit_code);
        return exit_code;
    }

    int reap(ProcessHandle handle) override {
        if (!handle.handle) {
            return -1;
        }

        DWORD wait_result = WaitForSingleObject(handle.handle, 0);
        if (wait_result != WAIT_OBJECT_0) {
            return -1;
        }

        DWORD exit_code = STILL_ACTIVE;
        if (!GetExitCodeProcess(handle.handle, &exit_code)) {
            CloseHandle(handle.handle);
            return -1;
        }

        CloseHandle(handle.handle);
        return exit_code == STILL_ACTIVE ? -1 : static_cast<int>(exit_code);
    }

    void kill(ProcessHandle handle) override {
        if (handle.handle) {
            TerminateProcess(handle.handle, 1);
            CloseHandle(handle.handle);
        }
    }

    void terminate_without_cleanup(ProcessHandle handle) override {
        if (handle.handle) {
            TerminateProcess(handle.handle, 1);
        }
    }

    int run_with_output(
        const std::string& executable,
        const std::vector<std::string>& args,
        OutputLineCallback on_line,
        const std::string& working_dir,
        int timeout_seconds,
        bool capture_stderr = true) override {

        std::string cmdline = escape_windows_arg(executable);
        for (const auto& arg : args) {
            cmdline += " " + escape_windows_arg(arg);
        }

        // Create pipes for stdout
        HANDLE stdout_read = nullptr;
        HANDLE stdout_write = nullptr;

        if (!CreatePipe(&stdout_read, &stdout_write, nullptr, 0)) {
            throw std::runtime_error("Failed to create stdout pipe");
        }

        STARTUPINFOA si;
        PROCESS_INFORMATION pi;
        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        si.hStdOutput = stdout_write;
        si.hStdError = capture_stderr ? stdout_write : GetStdHandle(STD_ERROR_HANDLE);
        ZeroMemory(&pi, sizeof(pi));

        BOOL success = create_process_with_restricted_handles(
            const_cast<char*>(cmdline.c_str()),
            CREATE_NO_WINDOW,
            nullptr,
            working_dir.empty() ? nullptr : working_dir.c_str(),
            si,
            &pi
        );

        // Close write end in parent
        CloseHandle(stdout_write);

        if (!success) {
            CloseHandle(stdout_read);
            DWORD error = GetLastError();
            throw std::runtime_error("Failed to start process: error " + std::to_string(error));
        }

        // Read output line by line
        std::string line_buffer;
        char buffer[4096];
        DWORD bytes_read;
        bool killed_by_callback = false;

        auto start_time = std::chrono::steady_clock::now();

        while (true) {
            // Check timeout
            if (timeout_seconds > 0) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - start_time).count();
                if (elapsed > timeout_seconds) {
                    TerminateProcess(pi.hProcess, 1);
                    killed_by_callback = true;
                    break;
                }
            }

            // Check if there's data to read (non-blocking peek)
            DWORD available = 0;
            if (!PeekNamedPipe(stdout_read, nullptr, 0, nullptr, &available, nullptr)) {
                break;  // Pipe closed or error
            }

            if (available > 0) {
                DWORD to_read = (std::min)(available, (DWORD)(sizeof(buffer) - 1));
                if (ReadFile(stdout_read, buffer, to_read, &bytes_read, nullptr) && bytes_read > 0) {
                    buffer[bytes_read] = '\0';
                    line_buffer += buffer;

                    // Process complete lines (split on \n or \r for in-place progress updates)
                    size_t pos;
                    while (true) {
                        // Find the first line terminator (\n or \r)
                        size_t newline_pos = line_buffer.find('\n');
                        size_t cr_pos = line_buffer.find('\r');

                        if (newline_pos == std::string::npos && cr_pos == std::string::npos) {
                            break;  // No complete line yet
                        }

                        // Use whichever comes first
                        if (newline_pos == std::string::npos) {
                            pos = cr_pos;
                        } else if (cr_pos == std::string::npos) {
                            pos = newline_pos;
                        } else {
                            pos = (std::min)(newline_pos, cr_pos);
                        }

                        std::string line = line_buffer.substr(0, pos);

                        // Skip \r\n as a single delimiter
                        size_t skip = 1;
                        if (pos + 1 < line_buffer.size() &&
                            line_buffer[pos] == '\r' && line_buffer[pos + 1] == '\n') {
                            skip = 2;
                        }
                        line_buffer = line_buffer.substr(pos + skip);

                        // Skip empty lines
                        if (line.empty()) {
                            continue;
                        }

                        // Call the callback
                        if (on_line && !on_line(line)) {
                            TerminateProcess(pi.hProcess, 1);
                            killed_by_callback = true;
                            break;
                        }
                    }

                    if (killed_by_callback) break;
                }
            } else {
                // No data available, check if process is still running
                DWORD exit_code;
                if (GetExitCodeProcess(pi.hProcess, &exit_code) && exit_code != STILL_ACTIVE) {
                    // Process exited, drain any remaining output
                    while (ReadFile(stdout_read, buffer, sizeof(buffer) - 1, &bytes_read, nullptr) && bytes_read > 0) {
                        buffer[bytes_read] = '\0';
                        line_buffer += buffer;
                    }
                    break;
                }

                // Sleep briefly to avoid busy-waiting
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        // Process any remaining partial line
        if (!line_buffer.empty() && on_line && !killed_by_callback) {
            // Remove trailing \r if present
            if (!line_buffer.empty() && line_buffer.back() == '\r') {
                line_buffer.pop_back();
            }
            if (!line_buffer.empty()) {
                on_line(line_buffer);
            }
        }

        CloseHandle(stdout_read);

        // Get exit code
        DWORD exit_code = 0;
        WaitForSingleObject(pi.hProcess, 5000);
        GetExitCodeProcess(pi.hProcess, &exit_code);

        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);

        return killed_by_callback ? -1 : static_cast<int>(exit_code);
    }

    int find_free_port(int start_port) override {
        WSADATA wsa_data;
        WSAStartup(MAKEWORD(2, 2), &wsa_data);

        for (int port = start_port; port < start_port + 1000; ++port) {
            // Test if port is free by attempting to bind to localhost
            int sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock < 0) {
                continue;
            }

            sockaddr_in addr;
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            addr.sin_addr.s_addr = inet_addr("127.0.0.1");

            int result = bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
            closesocket(sock);

            if (result == 0) {
                WSACleanup();
                return port;
            }
        }

        WSACleanup();
        return -1;
    }

    int run_command(const std::string& command, std::string& output, int timeout_seconds) override {
        output.clear();

        // Windows: use CreateProcess + pipe to avoid console window flash.
        // This is a drop-in replacement for _popen() that works in SUBSYSTEM:WINDOWS apps.
        HANDLE stdout_read = nullptr;
        HANDLE stdout_write = nullptr;

        if (!CreatePipe(&stdout_read, &stdout_write, nullptr, 0)) {
            return -1;
        }

        STARTUPINFOA si = {};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = INVALID_HANDLE_VALUE;
        si.hStdOutput = stdout_write;
        si.hStdError = stdout_write;

        PROCESS_INFORMATION pi = {};
        // Wrap in cmd /c so shell features (redirection, pipes) work
        std::string cmdline = "cmd /c " + command;
        BOOL success = create_process_with_restricted_handles(
            const_cast<char*>(cmdline.c_str()), CREATE_NO_WINDOW,
            nullptr, nullptr, si, &pi);

        CloseHandle(stdout_write);

        if (!success) {
            CloseHandle(stdout_read);
            return -1;
        }

        // Read all output
        char buf[4096];
        DWORD bytes_read;
        while (ReadFile(stdout_read, buf, sizeof(buf), &bytes_read, nullptr) && bytes_read > 0) {
            output.append(buf, bytes_read);
        }
        CloseHandle(stdout_read);

        WaitForSingleObject(pi.hProcess, timeout_seconds > 0 ? timeout_seconds * 1000 : INFINITE);
        DWORD exit_code = 1;
        GetExitCodeProcess(pi.hProcess, &exit_code);
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);

        return static_cast<int>(exit_code);
    }
};

// Factory function
std::unique_ptr<ProcessPlatform> create_process_platform() {
    return std::make_unique<WindowsProcessPlatform>();
}

} // namespace utils
} // namespace lemon
