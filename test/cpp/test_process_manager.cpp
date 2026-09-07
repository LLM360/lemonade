#include <lemon/utils/process_manager.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <exception>
#include <fstream>
#include <future>
#include <limits>
#include <string>
#include <thread>

#include <fcntl.h>
#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>

using lemon::utils::ProcessHandle;
using lemon::utils::ProcessManager;

namespace {

int failures = 0;
std::atomic<bool> coordinate_forks{false};
std::atomic<bool> first_fork_ready{false};
std::atomic<bool> release_first_fork{false};
std::atomic<bool> second_fork_completed{false};
std::atomic<int> fork_prepare_count{0};
std::atomic<int> fork_parent_count{0};
#ifdef __linux__
std::atomic<bool> run_command_pipe_ready{false};
std::atomic<bool> release_run_command_pipe{false};
std::atomic<int> run_command_pipe_fd{-1};
std::atomic<int> run_command_pipe_flags{-1};
#endif
#ifdef __APPLE__
std::atomic<bool> first_output_pipe_ready{false};
std::atomic<bool> release_first_output_command{false};
std::atomic<int> output_pipe_hook_count{0};
std::atomic<bool> inspect_run_command_reap_order{false};
std::atomic<int> run_command_reap_sequence{0};
std::atomic<int> run_command_exit_observed_order{-1};
std::atomic<int> run_command_group_cleanup_order{-1};
std::atomic<int> run_command_final_reap_order{-1};
std::atomic<bool> run_command_leader_reserved{false};
std::atomic<bool> run_command_leader_reserved_after_cleanup{false};
std::atomic<bool> run_command_leader_absent_after_reap{false};
#endif

void before_coordinated_fork() {
    if (!coordinate_forks.load()) {
        return;
    }

#ifdef __linux__
    if (fork_prepare_count.fetch_add(1) == 0) {
        first_fork_ready.store(true);
        while (coordinate_forks.load() && !release_first_fork.load()) {
            std::this_thread::yield();
        }
    }
#else
    fork_prepare_count.fetch_add(1);
#endif
}

void after_coordinated_fork_in_parent() {
    fork_parent_count.fetch_add(1);
    if (coordinate_forks.load() && fork_prepare_count.load() >= 2) {
        second_fork_completed.store(true);
    }
}

void check(const char* name, bool condition) {
    std::printf("[%s] %s\n", condition ? "PASS" : "FAIL", name);
    if (!condition) {
        ++failures;
    }
}

#ifdef __linux__
ProcessHandle make_handle(pid_t pid) {
    return {nullptr, static_cast<int>(pid)};
}

pid_t spawn_exiting_child(int exit_code) {
    const pid_t pid = fork();
    if (pid == 0) {
        _exit(exit_code);
    }
    return pid;
}

pid_t spawn_running_child() {
    const pid_t pid = fork();
    if (pid == 0) {
        for (;;) {
            pause();
        }
    }
    return pid;
}

bool is_zombie(pid_t pid) {
    std::ifstream stat_file("/proc/" + std::to_string(pid) + "/stat");
    std::string stat_line;
    if (!std::getline(stat_file, stat_line)) {
        return false;
    }

    const auto close_paren = stat_line.rfind(')');
    return close_paren != std::string::npos &&
           close_paren + 2 < stat_line.size() &&
           stat_line[close_paren + 2] == 'Z';
}

bool wait_for_zombie(pid_t pid, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        if (is_zombie(pid)) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);

    return is_zombie(pid);
}

void kill_and_reap(pid_t pid) {
    if (pid <= 0) {
        return;
    }

    ::kill(pid, SIGKILL);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
}

std::string read_fd_target(pid_t pid, int fd) {
    const std::string path = "/proc/" + std::to_string(pid) + "/fd/" +
                             std::to_string(fd);
    char target[256];
    const ssize_t length = readlink(path.c_str(), target, sizeof(target) - 1);
    if (length < 0) {
        return {};
    }
    target[length] = '\0';
    return target;
}

bool wait_for_process_name(pid_t pid, const std::string& expected,
                           std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        std::ifstream comm_file("/proc/" + std::to_string(pid) + "/comm");
        std::string process_name;
        if (std::getline(comm_file, process_name) && process_name == expected) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}
#endif

int capture_with_closed_standard_output() {
    const pid_t probe = fork();
    if (probe == 0) {
        close(STDOUT_FILENO);
        close(STDERR_FILENO);
        try {
            const ProcessHandle handle = ProcessManager::start_process(
                "/bin/sh",
                {"-c", "printf 'closed-stdio-error' >&2"},
                "",
                false,
                false,
                {},
                64);
            if (handle.pid <= 0 ||
                ProcessManager::wait_for_exit(handle, 5) != 0) {
                _exit(2);
            }
            if (ProcessManager::read_output(handle, 64) !=
                "closed-stdio-error") {
                _exit(3);
            }

            std::string command_output;
            const int result = ProcessManager::run_process_with_output(
                "/bin/sh",
                {"-c", "printf 'closed-stdio-version' >&2"},
                [&command_output](const std::string& line) {
                    command_output += line;
                    return true;
                },
                "",
                5);
            _exit(result == 0 && command_output == "closed-stdio-version" ? 0
                                                                           : 5);
        } catch (...) {
            _exit(4);
        }
    }
    if (probe <= 0) {
        return -1;
    }

    int status = 0;
    if (waitpid(probe, &status, 0) != probe || !WIFEXITED(status)) {
        return -1;
    }
    return WEXITSTATUS(status);
}

#ifdef __linux__
struct ConcurrentCaptureResult {
    bool setup_succeeded = false;
    int short_exit_code = -1;
    long short_read_ms = -1;
    std::string short_output;
};

struct ConcurrentPopenResult {
    bool setup_succeeded = false;
    bool close_on_exec = false;
    bool descriptor_inherited = false;
    int command_status = -1;
    std::string command_output;
};

ConcurrentPopenResult run_overlapping_popen_and_exec() {
    ConcurrentPopenResult result;
    char sentinel_path[] = "/tmp/lemonade-popen-overlap-XXXXXX";
    const int sentinel_fd = mkstemp(sentinel_path);
    if (sentinel_fd < 0) {
        return result;
    }
    close(sentinel_fd);
    unlink(sentinel_path);

    run_command_pipe_ready.store(false);
    release_run_command_pipe.store(false);
    run_command_pipe_fd.store(-1);
    run_command_pipe_flags.store(-1);

    auto command = std::async(std::launch::async, [&]() {
        return ProcessManager::run_command("printf popen-output",
                                           result.command_output, 5);
    });

    const auto hook_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!run_command_pipe_ready.load() &&
           std::chrono::steady_clock::now() < hook_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    ProcessHandle long_handle{nullptr, 0, nullptr};
    bool child_exec_observed = false;
    std::string parent_pipe_target;
    std::string child_pipe_target;
    const int pipe_fd = run_command_pipe_fd.load();
    if (run_command_pipe_ready.load() && pipe_fd >= 0) {
        parent_pipe_target = read_fd_target(getpid(), pipe_fd);
        try {
            long_handle = ProcessManager::start_process(
                "/bin/sh",
                {"-c", "printf ready > " + std::string(sentinel_path) +
                           "; exec sleep 3"},
                "", false, false, {}, 64);
            child_exec_observed =
                wait_for_process_name(long_handle.pid, "sleep",
                                      std::chrono::milliseconds(1000)) &&
                access(sentinel_path, F_OK) == 0;
            if (child_exec_observed) {
                child_pipe_target = read_fd_target(long_handle.pid, pipe_fd);
            }
        } catch (...) {
        }
    }

    result.close_on_exec =
        (run_command_pipe_flags.load() & FD_CLOEXEC) != 0;
    result.descriptor_inherited = !parent_pipe_target.empty() &&
                                  child_pipe_target == parent_pipe_target;
    result.setup_succeeded = run_command_pipe_ready.load() &&
                             parent_pipe_target.rfind("pipe:[", 0) == 0 &&
                             long_handle.pid > 0 && child_exec_observed;

    release_run_command_pipe.store(true);
    if (long_handle.pid > 0) {
        ProcessManager::kill_process(long_handle);
    }
    result.command_status = command.get();
    unlink(sentinel_path);
    return result;
}

ConcurrentCaptureResult run_overlapping_captured_spawns() {
    ConcurrentCaptureResult result;
    if (pthread_atfork(before_coordinated_fork,
                       after_coordinated_fork_in_parent, nullptr) != 0) {
        return result;
    }

    coordinate_forks.store(true);
    first_fork_ready.store(false);
    release_first_fork.store(false);
    second_fork_completed.store(false);
    fork_prepare_count.store(0);
    fork_parent_count.store(0);

    ProcessHandle short_handle{nullptr, 0, nullptr};
    ProcessHandle long_handle{nullptr, 0, nullptr};
    std::exception_ptr short_error;
    std::exception_ptr long_error;
    std::atomic<bool> short_spawn_finished{false};
    std::atomic<bool> long_spawn_finished{false};
    std::atomic<bool> workers_can_exit{false};

    std::thread short_thread([&]() {
        try {
            short_handle = ProcessManager::start_process(
                "/bin/sh", {"-c", "printf 'short-capture' >&2"}, "", false,
                false, {}, 64);
        } catch (...) {
            short_error = std::current_exception();
            second_fork_completed.store(true);
        }
        short_spawn_finished.store(true);
        while (!workers_can_exit.load()) {
            std::this_thread::yield();
        }
    });

    const auto coordination_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!first_fork_ready.load() &&
           std::chrono::steady_clock::now() < coordination_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::thread long_thread;
    if (first_fork_ready.load()) {
        long_thread = std::thread([&]() {
            try {
                long_handle = ProcessManager::start_process(
                    "/bin/sh", {"-c", "exec sleep 3"}, "", false, false, {},
                    64);
            } catch (...) {
                long_error = std::current_exception();
                second_fork_completed.store(true);
            }
            long_spawn_finished.store(true);
            while (!workers_can_exit.load()) {
                std::this_thread::yield();
            }
        });
    } else {
        second_fork_completed.store(true);
        long_spawn_finished.store(true);
    }

    while (first_fork_ready.load() && !second_fork_completed.load() &&
           std::chrono::steady_clock::now() < coordination_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool forks_overlapped = second_fork_completed.load();
    release_first_fork.store(true);
    if (!forks_overlapped) {
        second_fork_completed.store(true);
    }

    while ((!short_spawn_finished.load() || !long_spawn_finished.load()) &&
           std::chrono::steady_clock::now() < coordination_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    coordinate_forks.store(false);

    if (!forks_overlapped || !short_spawn_finished.load() ||
        !long_spawn_finished.load() || short_error || long_error ||
        short_handle.pid <= 0 || long_handle.pid <= 0) {
        ProcessManager::kill_process(short_handle);
        ProcessManager::kill_process(long_handle);
        workers_can_exit.store(true);
        short_thread.join();
        if (long_thread.joinable()) {
            long_thread.join();
        }
        return result;
    }

    const auto exit_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (result.short_exit_code < 0 &&
           std::chrono::steady_clock::now() < exit_deadline) {
        result.short_exit_code = ProcessManager::get_exit_code(short_handle);
        if (result.short_exit_code < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    const auto read_started = std::chrono::steady_clock::now();
    result.short_output = ProcessManager::read_output(short_handle, 64, 1000);
    result.short_read_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - read_started)
                               .count();
    if (result.short_exit_code < 0) {
        ProcessManager::kill_process(short_handle);
    }
    ProcessManager::kill_process(long_handle);
    workers_can_exit.store(true);
    short_thread.join();
    long_thread.join();
    result.setup_succeeded = true;
    return result;
}
#endif

#ifdef __APPLE__
struct ConcurrentRunWithOutputResult {
    bool setup_succeeded = false;
    bool short_completed_promptly = false;
    int short_exit_code = -1;
    int long_exit_code = -1;
    long short_wait_ms = -1;
    std::string short_output;
};

ConcurrentRunWithOutputResult run_overlapping_output_commands() {
    ConcurrentRunWithOutputResult result;
    char sentinel_path[] = "/tmp/lemonade-process-overlap-XXXXXX";
    const int sentinel_fd = mkstemp(sentinel_path);
    if (sentinel_fd < 0) {
        return result;
    }
    close(sentinel_fd);
    unlink(sentinel_path);

    coordinate_forks.store(true);
    first_output_pipe_ready.store(false);
    release_first_output_command.store(false);
    output_pipe_hook_count.store(0);

    std::string short_output;
    auto short_command = std::async(std::launch::async, [&]() {
        return ProcessManager::run_process_with_output(
            "/bin/sh", {"-c", "printf 'short-capture\\n'"},
            [&short_output](const std::string& line) {
                short_output += line;
                return true;
            },
            "", 5);
    });

    const auto first_pipe_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!first_output_pipe_ready.load() &&
           std::chrono::steady_clock::now() < first_pipe_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::string long_output;
    const std::string long_command_text =
        "printf ready > " + std::string(sentinel_path) + "; sleep 3";
    auto long_command = std::async(std::launch::async, [&]() {
        return ProcessManager::run_command(long_command_text, long_output, 5);
    });

    const auto sentinel_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (access(sentinel_path, F_OK) != 0 &&
           std::chrono::steady_clock::now() < sentinel_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const bool setup_succeeded = first_output_pipe_ready.load() &&
                                 output_pipe_hook_count.load() == 2 &&
                                 access(sentinel_path, F_OK) == 0;
    release_first_output_command.store(true);
    coordinate_forks.store(false);

    const auto wait_started = std::chrono::steady_clock::now();
    result.short_completed_promptly =
        short_command.wait_for(std::chrono::milliseconds(1000)) ==
        std::future_status::ready;
    result.short_wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - wait_started)
                               .count();
    result.long_exit_code = long_command.get();
    result.short_exit_code = short_command.get();
    result.short_output = std::move(short_output);
    result.setup_succeeded = setup_succeeded;
    unlink(sentinel_path);
    return result;
}

bool wait_for_process_gone(pid_t pid, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        errno = 0;
        if (::kill(pid, 0) != 0 && errno == ESRCH) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);

    errno = 0;
    return ::kill(pid, 0) != 0 && errno == ESRCH;
}

struct RunCommandTimeoutResult {
    bool setup_succeeded = false;
    bool descendant_gone = false;
    int command_status = 0;
};

RunCommandTimeoutResult run_command_with_descendant_timeout() {
    RunCommandTimeoutResult result;
    char sentinel_path[] = "/tmp/lemonade-command-descendant-XXXXXX";
    const int sentinel_fd = mkstemp(sentinel_path);
    if (sentinel_fd < 0) {
        return result;
    }
    close(sentinel_fd);
    unlink(sentinel_path);

    std::string output;
    result.command_status = ProcessManager::run_command(
        "sleep 30 & echo $! > " + std::string(sentinel_path) + "; wait",
        output, 1);

    std::ifstream sentinel(sentinel_path);
    pid_t descendant = -1;
    sentinel >> descendant;
    result.setup_succeeded = descendant > 0;
    if (result.setup_succeeded) {
        result.descendant_gone =
            wait_for_process_gone(descendant, std::chrono::seconds(2));
        if (!result.descendant_gone) {
            ::kill(descendant, SIGKILL);
        }
    }
    unlink(sentinel_path);
    return result;
}

struct RunCommandBackgroundResult {
    bool setup_succeeded = false;
    bool descendant_gone = false;
    int command_status = -1;
};

RunCommandBackgroundResult run_command_with_background_descendant() {
    RunCommandBackgroundResult result;
    char sentinel_path[] = "/tmp/lemonade-command-background-XXXXXX";
    const int sentinel_fd = mkstemp(sentinel_path);
    if (sentinel_fd < 0) {
        return result;
    }
    close(sentinel_fd);
    unlink(sentinel_path);

    std::string output;
    result.command_status = ProcessManager::run_command(
        "sleep 30 </dev/null >/dev/null 2>&1 & echo $! > " +
            std::string(sentinel_path),
        output, 5);

    std::ifstream sentinel(sentinel_path);
    pid_t descendant = -1;
    sentinel >> descendant;
    result.setup_succeeded = descendant > 0;
    if (result.setup_succeeded) {
        result.descendant_gone =
            wait_for_process_gone(descendant, std::chrono::seconds(2));
        if (!result.descendant_gone) {
            ::kill(descendant, SIGKILL);
        }
    }
    unlink(sentinel_path);
    return result;
}

struct RunCommandReapOrderResult {
    bool leader_reserved = false;
    bool leader_reserved_after_cleanup = false;
    bool leader_absent_after_reap = false;
    int exit_observed_order = -1;
    int group_cleanup_order = -1;
    int final_reap_order = -1;
    int command_status = -1;
    std::string command_output;
};

RunCommandReapOrderResult run_command_without_descendants() {
    inspect_run_command_reap_order.store(true);
    run_command_reap_sequence.store(0);
    run_command_exit_observed_order.store(-1);
    run_command_group_cleanup_order.store(-1);
    run_command_final_reap_order.store(-1);
    run_command_leader_reserved.store(false);
    run_command_leader_reserved_after_cleanup.store(false);
    run_command_leader_absent_after_reap.store(false);

    RunCommandReapOrderResult result;
    result.command_status =
        ProcessManager::run_command("printf no-descendant; exit 9",
                                    result.command_output, 5);
    inspect_run_command_reap_order.store(false);
    result.leader_reserved = run_command_leader_reserved.load();
    result.leader_reserved_after_cleanup =
        run_command_leader_reserved_after_cleanup.load();
    result.leader_absent_after_reap =
        run_command_leader_absent_after_reap.load();
    result.exit_observed_order = run_command_exit_observed_order.load();
    result.group_cleanup_order = run_command_group_cleanup_order.load();
    result.final_reap_order = run_command_final_reap_order.load();
    return result;
}
#endif

} // namespace

#ifdef __linux__
extern "C" void lemonade_test_run_command_pipe_created(int fd) {
    int flags;
    do {
        flags = fcntl(fd, F_GETFD);
    } while (flags < 0 && errno == EINTR);
    run_command_pipe_fd.store(fd);
    run_command_pipe_flags.store(flags);
    run_command_pipe_ready.store(true);
    while (!release_run_command_pipe.load()) {
        std::this_thread::yield();
    }
}
#endif

#ifdef __APPLE__
extern "C" void lemonade_test_run_with_output_pipe_created() {
    if (!coordinate_forks.load()) {
        return;
    }

    const int index = output_pipe_hook_count.fetch_add(1);
    if (index == 0) {
        first_output_pipe_ready.store(true);
        while (coordinate_forks.load() &&
               !release_first_output_command.load()) {
            std::this_thread::yield();
        }
    } else if (index == 1) {
        return;
    }
}

bool exited_child_is_reserved(pid_t pid) {
    siginfo_t child_info{};
    int result;
    do {
        result = waitid(P_PID, static_cast<id_t>(pid), &child_info,
                        WEXITED | WNOHANG | WNOWAIT);
    } while (result < 0 && errno == EINTR);
    return result == 0 && child_info.si_pid == pid;
}

extern "C" void lemonade_test_run_command_exit_observed(pid_t pid) {
    if (!inspect_run_command_reap_order.load()) {
        return;
    }
    run_command_leader_reserved.store(exited_child_is_reserved(pid));
    run_command_exit_observed_order.store(
        run_command_reap_sequence.fetch_add(1));
}

extern "C" void lemonade_test_run_command_group_cleanup_complete(pid_t pid) {
    if (!inspect_run_command_reap_order.load()) {
        return;
    }
    run_command_leader_reserved_after_cleanup.store(
        exited_child_is_reserved(pid));
    run_command_group_cleanup_order.store(
        run_command_reap_sequence.fetch_add(1));
}

extern "C" void lemonade_test_run_command_final_reap_complete(pid_t pid) {
    if (!inspect_run_command_reap_order.load()) {
        return;
    }
    siginfo_t child_info{};
    errno = 0;
    int result;
    do {
        result = waitid(P_PID, static_cast<id_t>(pid), &child_info,
                        WEXITED | WNOHANG | WNOWAIT);
    } while (result < 0 && errno == EINTR);
    run_command_leader_absent_after_reap.store(result < 0 && errno == ECHILD);
    run_command_final_reap_order.store(run_command_reap_sequence.fetch_add(1));
}
#endif

int main() {
    {
        const ProcessHandle handle = ProcessManager::start_process(
            "/bin/sh",
            {"-c", "printf 'discarded-prefix:original-startup-error' >&2"},
            "",
            false,
            false,
            {},
            22);
        check("start_process returns a child handle for captured output",
              handle.pid > 0);

        if (handle.pid > 0) {
            check("captured-output child exits successfully",
                  ProcessManager::wait_for_exit(handle, 5) == 0);
            const std::string output = ProcessManager::read_output(handle, 22);
            check("read_output returns a bounded tail from the original child",
                  output == "original-startup-error");
        }
    }

    check("process output capture survives closed stdout and stderr",
          capture_with_closed_standard_output() == 0);

#ifdef __linux__
    {
        const ConcurrentPopenResult result = run_overlapping_popen_and_exec();
        check("overlapping popen and exec complete setup",
              result.setup_succeeded);
        check("popen stream is close-on-exec", result.close_on_exec);
        check("exec child cannot inherit the popen stream",
              !result.descriptor_inherited);
        check("popen command preserves output and wait status",
              WIFEXITED(result.command_status) &&
                  WEXITSTATUS(result.command_status) == 0 &&
                  result.command_output == "popen-output");
    }

    {
        const ConcurrentCaptureResult result = run_overlapping_captured_spawns();
        check("overlapping captured spawns complete setup",
              result.setup_succeeded);
        if (result.setup_succeeded) {
            std::printf("[INFO] short exit=%d read=%ldms output='%s'\n",
                        result.short_exit_code, result.short_read_ms,
                        result.short_output.c_str());
            check("short captured process exits successfully",
                  result.short_exit_code == 0);
            check("long process cannot retain the short process capture pipes",
                  result.short_read_ms < 750);
            check("short process output remains isolated",
                  result.short_output == "short-capture");
        }
    }
#endif

#ifdef __APPLE__
    {
        const RunCommandReapOrderResult result =
            run_command_without_descendants();
        check("run_command no-descendant probe preserves output and status",
              result.command_output == "no-descendant" &&
                  WIFEXITED(result.command_status) &&
                  WEXITSTATUS(result.command_status) == 9);
        check("run_command reserves the exited leader through group cleanup",
              result.leader_reserved &&
                  result.leader_reserved_after_cleanup);
        check("run_command reaps only after process-group cleanup",
              result.exit_observed_order == 0 &&
                  result.group_cleanup_order == 1 &&
                  result.final_reap_order == 2 &&
                  result.leader_absent_after_reap);
    }

    {
        const ConcurrentRunWithOutputResult result =
            run_overlapping_output_commands();
        check("overlapping output commands complete setup",
              result.setup_succeeded);
        std::printf("[INFO] short exit=%d wait=%ldms output='%s'\n",
                    result.short_exit_code, result.short_wait_ms,
                    result.short_output.c_str());
        check("short output command exits successfully",
              result.short_exit_code == 0);
        check("long output command exits successfully",
              result.long_exit_code == 0);
        check("long command cannot retain the short command capture pipe",
              result.short_completed_promptly);
        check("short command output remains isolated",
              result.short_output == "short-capture");
    }

    {
        std::string output;
        const int status = ProcessManager::run_command(
            "printf 'line-one\\nunterminated'; exit 7", output, 5);
        check("run_command preserves raw output bytes",
              output == "line-one\nunterminated");
        check("run_command preserves wait-status semantics",
              WIFEXITED(status) && WEXITSTATUS(status) == 7);
    }

    {
        const RunCommandTimeoutResult result =
            run_command_with_descendant_timeout();
        check("run_command timeout descendant test completes setup",
              result.setup_succeeded);
        check("run_command reports timeout", result.command_status == -1);
        check("run_command timeout terminates descendant processes",
              result.descendant_gone);
    }

    {
        std::string output;
        const auto started = std::chrono::steady_clock::now();
        const int status = ProcessManager::run_command(
            "exec 1>&-; sleep 5", output, 1);
        const auto elapsed = std::chrono::steady_clock::now() - started;
        check("run_command enforces timeout after output closes",
              status == -1);
        check("run_command returns promptly after output closes",
              elapsed < std::chrono::seconds(3));
    }

    {
        const RunCommandBackgroundResult result =
            run_command_with_background_descendant();
        check("run_command background descendant test completes setup",
              result.setup_succeeded);
        check("run_command preserves successful shell status",
              WIFEXITED(result.command_status) &&
                  WEXITSTATUS(result.command_status) == 0);
        check("run_command success terminates residual process-group members",
              result.descendant_gone);
    }

    {
        std::string output;
        const int status = ProcessManager::run_command(
            "sleep 1; printf 'no-deadline'", output, 0);
        check("run_command zero timeout waits without a deadline",
              WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
                  output == "no-deadline");
    }

    {
        std::string output;
        const int exit_code = ProcessManager::run_process_with_output(
            "/bin/pwd", {},
            [&output](const std::string& line) {
                output += line;
                return true;
            },
            "/private/tmp", 5);
        check("output command preserves working-directory semantics",
              exit_code == 0 && output == "/private/tmp");
    }

    {
        const auto started = std::chrono::steady_clock::now();
        const int exit_code = ProcessManager::run_process_with_output(
            "/bin/sh", {"-c", "printf 'cancel\\n'; sleep 3"},
            [](const std::string&) { return false; }, "", 5);
        const auto elapsed = std::chrono::steady_clock::now() - started;
        check("output callback cancellation kills and reaps promptly",
              exit_code == -1 && elapsed < std::chrono::seconds(2));
    }
#endif

    {
        const ProcessHandle handle = ProcessManager::start_process(
            "/bin/sh", {"-c", "printf 'GET /health\\n'"}, "", true, true);
        check("filtered output creates a reader-completion tracker",
              handle.output_capture != nullptr);
        check("filtered-output child exits successfully",
              ProcessManager::wait_for_exit(handle, 5) == 0);
    }

#ifdef __linux__
    {
        const pid_t child = spawn_exiting_child(42);
        check("fork exited child", child > 0);

        if (child > 0) {
            const bool zombie = wait_for_zombie(child, std::chrono::seconds(5));
            check("child reaches zombie state without reaping", zombie);

            if (zombie) {
                const ProcessHandle handle = make_handle(child);
                check("is_running() returns false for zombie",
                      !ProcessManager::is_running(handle));
                check("reap_process() preserves exit code 42",
                      ProcessManager::reap_process(handle) == 42);
            } else {
                kill_and_reap(child);
            }
        }
    }

    {
        const pid_t child = spawn_running_child();
        check("fork running child", child > 0);

        if (child > 0) {
            const ProcessHandle handle = make_handle(child);
            check("is_running() returns true for running child",
                  ProcessManager::is_running(handle));
            check("reap_process() does not reap running child",
                  ProcessManager::reap_process(handle) == -1);
            kill_and_reap(child);
        }
    }

    check("is_running() rejects PID 0",
          !ProcessManager::is_running(make_handle(0)));
    check("is_running() rejects negative PID",
          !ProcessManager::is_running(make_handle(-1)));
    check("is_running() rejects non-existent PID",
          !ProcessManager::is_running(
              make_handle(std::numeric_limits<int>::max())));
#endif

    if (failures == 0) {
        std::printf("\nAll process_manager tests passed\n");
        return 0;
    }

    std::printf("\n%d process_manager test(s) failed\n", failures);
    return 1;
}
