#include <lemon/utils/process_platform.h>
#include <lemon/utils/aixlog.hpp>

#include <stdexcept>

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <errno.h>
#include <spawn.h>
#include <cstring>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cctype>

extern char** environ;
#ifdef LEMONADE_PROCESS_TEST_HOOK
extern "C" void lemonade_test_run_with_output_pipe_created();
extern "C" void lemonade_test_run_command_exit_observed(pid_t pid);
extern "C" void lemonade_test_run_command_group_cleanup_complete(pid_t pid);
extern "C" void lemonade_test_run_command_final_reap_complete(pid_t pid);
#endif

namespace lemon::utils {

// Reuse helper functions from Unix implementation
static bool should_filter_line(const std::string& line) {
    return (line.find("GET /health") != std::string::npos ||
            line.find("GET /v1/health") != std::string::npos ||
            line.find("srv  update_slots: all slots are idle") != std::string::npos ||
            line.find("Enter 'exit' to stop the server") != std::string::npos);
}

static bool is_error_line(const std::string& line) {
    std::string lowered = line;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered.find("error") != std::string::npos;
}

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

static void close_pipe(int pipe_fds[2]) {
    for (int i = 0; i < 2; ++i) {
        if (pipe_fds[i] >= 0) {
            close(pipe_fds[i]);
            pipe_fds[i] = -1;
        }
    }
}

static bool set_close_on_exec(int fd) {
    int flags;
    do {
        flags = fcntl(fd, F_GETFD);
    } while (flags < 0 && errno == EINTR);
    if (flags < 0) {
        return false;
    }

    int result;
    do {
        result = fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
    } while (result < 0 && errno == EINTR);
    return result == 0;
}

static bool create_pipe_above_standard_streams(int pipe_fds[2]) {
    if (pipe(pipe_fds) < 0) {
        return false;
    }
#ifdef LEMONADE_PROCESS_TEST_HOOK
    lemonade_test_run_with_output_pipe_created();
#endif

    for (int i = 0; i < 2; ++i) {
        if (pipe_fds[i] > STDERR_FILENO) {
            if (!set_close_on_exec(pipe_fds[i])) {
                close_pipe(pipe_fds);
                return false;
            }
            continue;
        }

        int replacement;
        do {
            replacement =
                fcntl(pipe_fds[i], F_DUPFD_CLOEXEC, STDERR_FILENO + 1);
        } while (replacement < 0 && errno == EINTR);
        if (replacement < 0) {
            close_pipe(pipe_fds);
            return false;
        }
        close(pipe_fds[i]);
        pipe_fds[i] = replacement;
    }
    return true;
}

static void kill_process_group(pid_t pid, bool fallback_to_process = true) {
    int result;
    do {
        result = ::kill(-pid, SIGKILL);
    } while (result < 0 && errno == EINTR);
    if (fallback_to_process && result < 0 && errno == ESRCH) {
        do {
            result = ::kill(pid, SIGKILL);
        } while (result < 0 && errno == EINTR);
    }
}

class SpawnFileActions {
public:
    SpawnFileActions() {
        error_ = posix_spawn_file_actions_init(&actions_);
        initialized_ = error_ == 0;
    }

    ~SpawnFileActions() {
        if (initialized_) {
            posix_spawn_file_actions_destroy(&actions_);
        }
    }

    SpawnFileActions(const SpawnFileActions&) = delete;
    SpawnFileActions& operator=(const SpawnFileActions&) = delete;

    void add_close(int fd) {
        if (error_ == 0) {
            error_ = posix_spawn_file_actions_addclose(&actions_, fd);
        }
    }

    void add_dup2(int source, int destination) {
        if (error_ == 0) {
            error_ =
                posix_spawn_file_actions_adddup2(&actions_, source, destination);
        }
    }

    void add_open(int fd, const char* path, int flags, mode_t mode) {
        if (error_ == 0) {
            error_ = posix_spawn_file_actions_addopen(&actions_, fd, path,
                                                      flags, mode);
        }
    }

    void add_inherit_if_open(int fd) {
        if (error_ != 0) {
            return;
        }

        int flags;
        do {
            flags = fcntl(fd, F_GETFD);
        } while (flags < 0 && errno == EINTR);
        if (flags < 0) {
            if (errno != EBADF) {
                error_ = errno;
            }
            return;
        }
        if ((flags & FD_CLOEXEC) == 0) {
            error_ = posix_spawn_file_actions_addinherit_np(&actions_, fd);
        }
    }

    void add_chdir(const std::string& working_dir) {
        if (error_ != 0 || working_dir.empty()) {
            return;
        }
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        error_ = posix_spawn_file_actions_addchdir_np(&actions_,
                                                      working_dir.c_str());
#pragma clang diagnostic pop
    }

    int error() const { return error_; }
    posix_spawn_file_actions_t* get() { return &actions_; }

private:
    posix_spawn_file_actions_t actions_;
    bool initialized_ = false;
    int error_ = 0;
};

class SpawnAttributes {
public:
    SpawnAttributes() {
        error_ = posix_spawnattr_init(&attributes_);
        initialized_ = error_ == 0;
    }

    ~SpawnAttributes() {
        if (initialized_) {
            posix_spawnattr_destroy(&attributes_);
        }
    }

    SpawnAttributes(const SpawnAttributes&) = delete;
    SpawnAttributes& operator=(const SpawnAttributes&) = delete;

    void set_default_signals() {
        if (error_ != 0) {
            return;
        }

        sigset_t default_signals;
        if (sigfillset(&default_signals) != 0) {
            error_ = errno;
            return;
        }
        error_ = posix_spawnattr_setsigdefault(&attributes_, &default_signals);
    }

    void set_process_group(pid_t process_group) {
        if (error_ == 0) {
            error_ = posix_spawnattr_setpgroup(&attributes_, process_group);
        }
    }

    void set_flags(short flags) {
        if (error_ == 0) {
            error_ = posix_spawnattr_setflags(&attributes_, flags);
        }
    }

    int error() const { return error_; }
    posix_spawnattr_t* get() { return &attributes_; }

private:
    posix_spawnattr_t attributes_;
    bool initialized_ = false;
    int error_ = 0;
};

static pid_t spawn_capturing_output(const std::string& executable,
                                    const std::vector<std::string>& args,
                                    const std::string& working_dir,
                                    bool capture_stderr,
                                    bool create_process_group,
                                    int output_pipe[2]) {
    SpawnFileActions file_actions;
    file_actions.add_inherit_if_open(STDIN_FILENO);
    file_actions.add_close(output_pipe[0]);
    file_actions.add_dup2(output_pipe[1], STDOUT_FILENO);
    if (capture_stderr) {
        file_actions.add_dup2(output_pipe[1], STDERR_FILENO);
    } else {
        file_actions.add_inherit_if_open(STDERR_FILENO);
    }
    file_actions.add_close(output_pipe[1]);
    file_actions.add_chdir(working_dir);

    SpawnAttributes attributes;
    short spawn_flags = POSIX_SPAWN_CLOEXEC_DEFAULT;
    if (create_process_group) {
        attributes.set_process_group(0);
        spawn_flags |= POSIX_SPAWN_SETPGROUP;
    }
    attributes.set_flags(spawn_flags);

    std::vector<char*> argv_ptrs;
    argv_ptrs.reserve(args.size() + 2);
    argv_ptrs.push_back(const_cast<char*>(executable.c_str()));
    for (const auto& arg : args) {
        argv_ptrs.push_back(const_cast<char*>(arg.c_str()));
    }
    argv_ptrs.push_back(nullptr);

    int spawn_result = file_actions.error();
    if (spawn_result == 0) {
        spawn_result = attributes.error();
    }

    pid_t pid = 0;
    if (spawn_result == 0) {
        spawn_result = posix_spawnp(&pid, executable.c_str(),
                                    file_actions.get(), attributes.get(),
                                    argv_ptrs.data(), environ);
    }
    if (spawn_result != 0) {
        close_pipe(output_pipe);
        throw std::runtime_error(std::string("posix_spawn failed: ") +
                                 strerror(spawn_result));
    }

    close(output_pipe[1]);
    output_pipe[1] = -1;
    return pid;
}

static void read_process_output(
    int fd,
    bool log_output,
    const std::shared_ptr<ProcessOutputCapture>& output_capture) {
    char buffer[4096];
    std::string line_buffer;
    ssize_t bytes_read;

    while ((bytes_read = read(fd, buffer, sizeof(buffer))) > 0) {
        if (output_capture) {
            output_capture->append(buffer, static_cast<std::size_t>(bytes_read));
        }
        if (!log_output) {
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

    if (log_output && !line_buffer.empty()) {
        log_process_line(line_buffer);
    }
    close(fd);
    if (output_capture) {
        output_capture->finish_reader();
    }
}

static void start_process_output_reader(
    int fd,
    bool log_output,
    const std::shared_ptr<ProcessOutputCapture>& output_capture) {
    try {
        std::thread(read_process_output, fd, log_output, output_capture).detach();
    } catch (const std::exception& error) {
        close(fd);
        if (output_capture) {
            output_capture->finish_reader();
        }
        LOG(ERROR, "ProcessManager")
            << "Failed to start process output reader: " << error.what()
            << std::endl;
    }
}

// Forward declare UnixProcessPlatform base class methods
class MacOSProcessPlatform : public ProcessPlatform {
public:
    ProcessHandle spawn(
        const std::string& executable,
        const std::vector<std::string>& args,
        const std::string& working_dir,
        bool inherit_output,
        bool filter_health_logs,
        const std::vector<std::pair<std::string, std::string>>& env_vars,
        std::shared_ptr<ProcessOutputCapture> output_capture) override;

    void terminate(ProcessHandle handle) override;
    bool is_running(ProcessHandle handle) override;
    int get_exit_code(ProcessHandle handle) override;
    int wait_for_exit(ProcessHandle handle, int timeout_seconds) override;
    int reap(ProcessHandle handle) override;
    void kill(ProcessHandle handle) override;
    void terminate_without_cleanup(ProcessHandle handle) override;

    int run_with_output(
        const std::string& executable,
        const std::vector<std::string>& args,
        OutputLineCallback on_line,
        const std::string& working_dir,
        int timeout_seconds,
        bool capture_stderr = true) override;

    int find_free_port(int start_port) override;
    int run_command(const std::string& command, std::string& output, int timeout_seconds) override;
};

ProcessHandle MacOSProcessPlatform::spawn(
    const std::string& executable,
    const std::vector<std::string>& args,
    const std::string& working_dir,
    bool inherit_output,
    bool filter_health_logs,
    const std::vector<std::pair<std::string, std::string>>& env_vars,
    std::shared_ptr<ProcessOutputCapture> output_capture) {

    ProcessHandle handle;
    handle.handle = nullptr;
    handle.pid = 0;
    handle.output_capture = output_capture;

    int stdout_pipe[2] = {-1, -1};
    int stderr_pipe[2] = {-1, -1};
    const bool redirect_output =
        (inherit_output && filter_health_logs) || output_capture != nullptr;

    if (redirect_output) {
        if (!create_pipe_above_standard_streams(stdout_pipe)) {
            throw std::runtime_error("Failed to create pipes for output filtering");
        }
        if (!create_pipe_above_standard_streams(stderr_pipe)) {
            close_pipe(stdout_pipe);
            throw std::runtime_error("Failed to create pipes for output filtering");
        }
    }

    if (inherit_output) {
        std::string cmdline = executable;
        for (const auto& arg : args) {
            cmdline += " " + arg;
        }
        if (filter_health_logs) {
            LOG(DEBUG, "ProcessManager") << "Starting process with filtered output: " << cmdline << std::endl;
        } else {
            LOG(DEBUG, "ProcessManager") << "Starting process with inherited output: " << cmdline << std::endl;
        }
    }

    // macOS: use posix_spawn instead of fork+exec
    //
    // Problem: lemond spawns llama-server via fork()+execvp(). On macOS, fork()
    // leaves the child with corrupted Mach-port and XPC-bootstrap state that
    // execvp() does not reset. llama.cpp b8884+ now runs a ggml-metal probe at
    // startup that calls [MTLDevice newLibraryWithSource:] — which routes
    // through MTLCompilerService XPC — and dies on the broken channel before
    // the model is opened. Direct terminal runs work; only lemond-spawned
    // children fail (~130ms, exit code -1).
    //
    // Fix: on macOS, use posix_spawn instead of fork+exec. Preserves
    // pipe/working-dir semantics. Adds POSIX_SPAWN_CLOEXEC_DEFAULT to avoid
    // leaking lemond FDs into the child, and POSIX_SPAWN_SETSIGDEF to reset
    // inherited SIG_IGN dispositions.
    SpawnFileActions file_actions;
    file_actions.add_inherit_if_open(STDIN_FILENO);

    if (redirect_output) {
        file_actions.add_close(stdout_pipe[0]);
        file_actions.add_close(stderr_pipe[0]);
        file_actions.add_dup2(stdout_pipe[1], STDOUT_FILENO);
        file_actions.add_dup2(stderr_pipe[1], STDERR_FILENO);
        file_actions.add_close(stdout_pipe[1]);
        file_actions.add_close(stderr_pipe[1]);
    } else if (!inherit_output) {
        file_actions.add_open(STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
        file_actions.add_dup2(STDOUT_FILENO, STDERR_FILENO);
    } else {
        file_actions.add_inherit_if_open(STDOUT_FILENO);
        file_actions.add_inherit_if_open(STDERR_FILENO);
    }

    file_actions.add_chdir(working_dir);

    SpawnAttributes attributes;
    attributes.set_default_signals();
    attributes.set_flags(POSIX_SPAWN_CLOEXEC_DEFAULT |
                         POSIX_SPAWN_SETSIGDEF);

    // Build envp
    std::vector<std::string> env_strings;
    for (char** e = environ; e && *e; ++e) {
        bool override_existing = false;
        for (const auto& env_pair : env_vars) {
            std::string prefix = env_pair.first + "=";
            if (std::strncmp(*e, prefix.c_str(), prefix.size()) == 0) {
                override_existing = true;
                break;
            }
        }
        if (!override_existing) {
            env_strings.emplace_back(*e);
        }
    }
    for (const auto& env_pair : env_vars) {
        env_strings.emplace_back(env_pair.first + "=" + env_pair.second);
    }
    std::vector<char*> envp;
    envp.reserve(env_strings.size() + 1);
    for (auto& s : env_strings) {
        envp.push_back(&s[0]);
    }
    envp.push_back(nullptr);

    std::vector<char*> argv_ptrs;
    argv_ptrs.reserve(args.size() + 2);
    argv_ptrs.push_back(const_cast<char*>(executable.c_str()));
    for (const auto& arg : args) {
        argv_ptrs.push_back(const_cast<char*>(arg.c_str()));
    }
    argv_ptrs.push_back(nullptr);

    int spawn_rc = file_actions.error();
    if (spawn_rc == 0) {
        spawn_rc = attributes.error();
    }

    pid_t pid = 0;
    if (spawn_rc == 0) {
        spawn_rc = posix_spawnp(&pid, executable.c_str(), file_actions.get(),
                                attributes.get(), argv_ptrs.data(),
                                envp.data());
    }

    if (spawn_rc != 0) {
        if (redirect_output) {
            close_pipe(stdout_pipe);
            close_pipe(stderr_pipe);
        }
        throw std::runtime_error(std::string("posix_spawn failed: ") + strerror(spawn_rc));
    }

    handle.pid = pid;

    if (inherit_output) {
        LOG(INFO, "ProcessManager") << "Process started successfully, PID: " << pid << std::endl;
    }

    if (redirect_output) {
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        start_process_output_reader(stdout_pipe[0], inherit_output,
                                    output_capture);
        start_process_output_reader(stderr_pipe[0], inherit_output,
                                    output_capture);
    }

    return handle;
}

// Reuse Unix implementations for other methods
void MacOSProcessPlatform::terminate(ProcessHandle handle) {
    if (handle.pid <= 0) {
        return;
    }

#ifdef WNOWAIT
    siginfo_t info;
    std::memset(&info, 0, sizeof(info));
    if (waitid(P_PID, static_cast<id_t>(handle.pid), &info, WEXITED | WNOHANG | WNOWAIT) == 0) {
        if (info.si_pid != 0) {
            reap(handle);
            LOG(INFO, "ProcessManager") << "Process already exited; reaped PID "
                                        << handle.pid << std::endl;
            LOG(INFO, "ProcessManager") << "Process terminated, waiting for GPU driver cleanup..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            return;
        }
    } else if (errno == ECHILD) {
        LOG(WARNING, "ProcessManager") << "Process PID " << handle.pid
                                       << " is no longer an owned child; skipping termination"
                                       << std::endl;
        return;
    }
#endif

    errno = 0;
    if (::kill(handle.pid, SIGTERM) != 0 && errno == ESRCH) {
        LOG(INFO, "ProcessManager") << "Process PID " << handle.pid
                                    << " was already gone before SIGTERM" << std::endl;
        return;
    }

    int status = 0;
    bool exited_gracefully = false;
    for (int i = 0; i < 50; ++i) {
        pid_t result = waitpid(handle.pid, &status, WNOHANG);
        if (result > 0) {
            exited_gracefully = true;
            break;
        }
        if (result < 0 && errno == ECHILD) {
            exited_gracefully = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!exited_gracefully) {
        LOG(WARNING, "ProcessManager") << "Process did not respond to SIGTERM, using SIGKILL" << std::endl;
        errno = 0;
        if (::kill(handle.pid, SIGKILL) == 0 || errno != ESRCH) {
            waitpid(handle.pid, &status, 0);
        }
    }

    LOG(INFO, "ProcessManager") << "Process terminated, waiting for GPU driver cleanup..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
}

bool MacOSProcessPlatform::is_running(ProcessHandle handle) {
    if (handle.pid <= 0) {
        return false;
    }

#ifdef WNOWAIT
    siginfo_t info;
    std::memset(&info, 0, sizeof(info));
    if (waitid(P_PID, static_cast<id_t>(handle.pid), &info, WEXITED | WNOHANG | WNOWAIT) == 0) {
        return info.si_pid == 0;
    }

    if (errno == ECHILD) {
        return false;
    }
#endif

    errno = 0;
    return ::kill(handle.pid, 0) == 0 || errno == EPERM;
}

int MacOSProcessPlatform::get_exit_code(ProcessHandle handle) {
    if (handle.pid <= 0) {
        return -1;
    }

    int status = 0;
    pid_t result = waitpid(handle.pid, &status, WNOHANG);

    if (result == 0 || result < 0) {
        return -1;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }

    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }

    return -1;
}

int MacOSProcessPlatform::wait_for_exit(ProcessHandle handle, int timeout_seconds) {
    if (handle.pid <= 0) {
        return -1;
    }

    int status = 0;
    if (timeout_seconds < 0) {
        if (waitpid(handle.pid, &status, 0) <= 0) {
            return -1;
        }
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        }
        if (WIFSIGNALED(status)) {
            return 128 + WTERMSIG(status);
        }
        return -1;
    }

    for (int i = 0; i < timeout_seconds * 10; ++i) {
        pid_t result = waitpid(handle.pid, &status, WNOHANG);
        if (result > 0) {
            if (WIFEXITED(status)) {
                return WEXITSTATUS(status);
            }
            if (WIFSIGNALED(status)) {
                return 128 + WTERMSIG(status);
            }
            return -1;
        }
        if (result < 0) {
            return -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return -1;
}

int MacOSProcessPlatform::reap(ProcessHandle handle) {
    if (handle.pid <= 0) {
        return -1;
    }

    int status = 0;
    pid_t result = waitpid(handle.pid, &status, WNOHANG);
    if (result <= 0) {
        return -1;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }

    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }

    return -1;
}

void MacOSProcessPlatform::kill(ProcessHandle handle) {
    if (handle.pid > 0) {
        errno = 0;
        if (::kill(handle.pid, SIGKILL) == 0 || errno != ESRCH) {
            int status = 0;
            waitpid(handle.pid, &status, 0);
        }
    }
}

void MacOSProcessPlatform::terminate_without_cleanup(ProcessHandle handle) {
    if (handle.pid > 0) {
        ::kill(handle.pid, SIGKILL);
    }
}

int MacOSProcessPlatform::run_with_output(
    const std::string& executable,
    const std::vector<std::string>& args,
    OutputLineCallback on_line,
    const std::string& working_dir,
    int timeout_seconds,
    bool capture_stderr) {

    int stdout_pipe[2] = {-1, -1};

    if (!create_pipe_above_standard_streams(stdout_pipe)) {
        throw std::runtime_error("Failed to create pipe");
    }
    const pid_t pid = spawn_capturing_output(
        executable, args, working_dir, capture_stderr, false, stdout_pipe);

    std::string line_buffer;
    char buffer[4096];
    ssize_t bytes_read;
    bool killed_by_callback = false;

    auto start_time = std::chrono::steady_clock::now();

    int flags = fcntl(stdout_pipe[0], F_GETFL, 0);
    fcntl(stdout_pipe[0], F_SETFL, flags | O_NONBLOCK);

    while (true) {
        if (timeout_seconds > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();
            if (elapsed > timeout_seconds) {
                ::kill(pid, SIGKILL);
                killed_by_callback = true;
                break;
            }
        }

        bytes_read = read(stdout_pipe[0], buffer, sizeof(buffer) - 1);

        if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            line_buffer += buffer;

            size_t pos;
            while (true) {
                size_t newline_pos = line_buffer.find('\n');
                size_t cr_pos = line_buffer.find('\r');

                if (newline_pos == std::string::npos && cr_pos == std::string::npos) {
                    break;
                }

                if (newline_pos == std::string::npos) {
                    pos = cr_pos;
                } else if (cr_pos == std::string::npos) {
                    pos = newline_pos;
                } else {
                    pos = std::min(newline_pos, cr_pos);
                }

                std::string line = line_buffer.substr(0, pos);

                size_t skip = 1;
                if (pos + 1 < line_buffer.size() &&
                    line_buffer[pos] == '\r' && line_buffer[pos + 1] == '\n') {
                    skip = 2;
                }
                line_buffer = line_buffer.substr(pos + skip);

                if (line.empty()) {
                    continue;
                }

                if (on_line && !on_line(line)) {
                    ::kill(pid, SIGKILL);
                    killed_by_callback = true;
                    break;
                }
            }

            if (killed_by_callback) break;
        } else if (bytes_read == 0) {
            break;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                int status = 0;
                pid_t result = waitpid(pid, &status, WNOHANG);
                if (result > 0) {
                    fcntl(stdout_pipe[0], F_SETFL, flags);
                    while ((bytes_read = read(stdout_pipe[0], buffer, sizeof(buffer) - 1)) > 0) {
                        buffer[bytes_read] = '\0';
                        line_buffer += buffer;
                    }
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } else {
                break;
            }
        }
    }

    if (!line_buffer.empty() && on_line && !killed_by_callback) {
        on_line(line_buffer);
    }

    close(stdout_pipe[0]);

    int status = 0;
    waitpid(pid, &status, 0);

    if (killed_by_callback) {
        return -1;
    }

    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

int MacOSProcessPlatform::find_free_port(int start_port) {
    for (int port = start_port; port < start_port + 1000; ++port) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            continue;
        }

        sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = inet_addr("127.0.0.1");

        int result = bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        close(sock);

        if (result == 0) {
            return port;
        }
    }

    return -1;
}

int MacOSProcessPlatform::run_command(const std::string& command, std::string& output, int timeout_seconds) {
    output.clear();
    int output_pipe[2] = {-1, -1};
    if (!create_pipe_above_standard_streams(output_pipe)) {
        return -1;
    }

    pid_t pid;
    try {
        pid = spawn_capturing_output("/bin/sh", {"-c", command}, "", false,
                                     true, output_pipe);
    } catch (...) {
        return -1;
    }

    const int flags = fcntl(output_pipe[0], F_GETFL, 0);
    if (flags < 0 ||
        fcntl(output_pipe[0], F_SETFL, flags | O_NONBLOCK) != 0) {
        kill_process_group(pid);
        close(output_pipe[0]);
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
        return -1;
    }

    const auto start_time = std::chrono::steady_clock::now();
    const auto deadline =
        start_time + std::chrono::seconds(timeout_seconds > 0
                                              ? timeout_seconds
                                              : 0);
    bool timed_out = false;
    bool read_failed = false;
    char buffer[4096];
    while (true) {
        if (timeout_seconds > 0) {
            if (std::chrono::steady_clock::now() >= deadline) {
                kill_process_group(pid);
                timed_out = true;
                break;
            }
        }

        const ssize_t bytes_read = read(output_pipe[0], buffer, sizeof(buffer));
        if (bytes_read > 0) {
            output.append(buffer, static_cast<std::size_t>(bytes_read));
            continue;
        }
        if (bytes_read == 0) {
            break;
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            kill_process_group(pid);
            read_failed = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    close(output_pipe[0]);
    int status = 0;
    pid_t wait_result = -1;
    bool child_identity_lost = false;
    if (!timed_out && !read_failed) {
        while (true) {
            siginfo_t child_info{};
            int wait_options = WEXITED | WNOWAIT;
            if (timeout_seconds > 0) {
                wait_options |= WNOHANG;
            }

            int observe_result;
            do {
                observe_result =
                    waitid(P_PID, static_cast<id_t>(pid), &child_info,
                           wait_options);
            } while (observe_result < 0 && errno == EINTR);

            if (observe_result == 0 && child_info.si_pid == pid) {
                break;
            }
            if (observe_result < 0) {
                child_identity_lost = errno == ECHILD;
                if (!child_identity_lost) {
                    kill_process_group(pid);
                }
                read_failed = true;
                break;
            }
            if (timeout_seconds > 0 &&
                std::chrono::steady_clock::now() >= deadline) {
                kill_process_group(pid);
                timed_out = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    if (timed_out || read_failed) {
        if (child_identity_lost) {
            return -1;
        }
        do {
            wait_result = waitpid(pid, &status, 0);
        } while (wait_result < 0 && errno == EINTR);
        return -1;
    }

#ifdef LEMONADE_PROCESS_TEST_HOOK
    lemonade_test_run_command_exit_observed(pid);
#endif
    kill_process_group(pid, false);
#ifdef LEMONADE_PROCESS_TEST_HOOK
    lemonade_test_run_command_group_cleanup_complete(pid);
#endif
    do {
        wait_result = waitpid(pid, &status, 0);
    } while (wait_result < 0 && errno == EINTR);
    if (wait_result != pid) {
        return -1;
    }
#ifdef LEMONADE_PROCESS_TEST_HOOK
    lemonade_test_run_command_final_reap_complete(pid);
#endif
    return status;
}

std::unique_ptr<ProcessPlatform> create_process_platform() {
    return std::make_unique<MacOSProcessPlatform>();
}

} // namespace lemon::utils
