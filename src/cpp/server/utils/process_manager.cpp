#include <lemon/utils/process_manager.h>
#include <lemon/utils/process_platform.h>

#include <algorithm>
#include <chrono>

namespace lemon {
namespace utils {

namespace {

constexpr int output_reader_drain_timeout_ms = 1000;

void wait_for_output_readers(const ProcessHandle& handle) {
    if (handle.output_capture) {
        handle.output_capture->read_tail(0, output_reader_drain_timeout_ms);
    }
}

}  // namespace

ProcessOutputCapture::ProcessOutputCapture(std::size_t max_bytes)
    : max_bytes_(max_bytes) {
}

void ProcessOutputCapture::append(const char* data, std::size_t size) {
    if (data == nullptr || size == 0 || max_bytes_ == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (size >= max_bytes_) {
        tail_.assign(data + size - max_bytes_, max_bytes_);
        return;
    }

    const std::size_t overflow =
        tail_.size() + size > max_bytes_ ? tail_.size() + size - max_bytes_ : 0;
    if (overflow > 0) {
        tail_.erase(0, overflow);
    }
    tail_.append(data, size);
}

void ProcessOutputCapture::finish_reader() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_readers_ > 0) {
        --active_readers_;
    }
    if (active_readers_ == 0) {
        completion_cv_.notify_all();
    }
}

std::string ProcessOutputCapture::read_tail(std::size_t max_bytes,
                                            int wait_timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (wait_timeout_ms > 0 && active_readers_ > 0) {
        completion_cv_.wait_for(
            lock,
            std::chrono::milliseconds(wait_timeout_ms),
            [this]() { return active_readers_ == 0; });
    }

    const std::size_t size = std::min(max_bytes, tail_.size());
    return tail_.substr(tail_.size() - size, size);
}

ProcessHandle ProcessManager::start_process(
    const std::string& executable,
    const std::vector<std::string>& args,
    const std::string& working_dir,
    bool inherit_output,
    bool filter_health_logs,
    const std::vector<std::pair<std::string, std::string>>& env_vars,
    std::size_t capture_output_bytes) {

    std::shared_ptr<ProcessOutputCapture> output_capture;
    if (capture_output_bytes > 0 ||
        (inherit_output && filter_health_logs)) {
        output_capture =
            std::make_shared<ProcessOutputCapture>(capture_output_bytes);
    }

    auto platform = create_process_platform();
    return platform->spawn(executable, args, working_dir, inherit_output,
                           filter_health_logs, env_vars,
                           std::move(output_capture));
}

void ProcessManager::stop_process(ProcessHandle handle) {
    auto platform = create_process_platform();
    platform->terminate(handle);
    wait_for_output_readers(handle);
}

bool ProcessManager::is_running(ProcessHandle handle) {
    auto platform = create_process_platform();
    return platform->is_running(handle);
}

int ProcessManager::get_exit_code(ProcessHandle handle) {
    auto platform = create_process_platform();
    return platform->get_exit_code(handle);
}

int ProcessManager::wait_for_exit(ProcessHandle handle, int timeout_seconds) {
    auto platform = create_process_platform();
    const int exit_code = platform->wait_for_exit(handle, timeout_seconds);
    if (exit_code >= 0) {
        wait_for_output_readers(handle);
    }
    return exit_code;
}

int ProcessManager::reap_process(ProcessHandle handle) {
    auto platform = create_process_platform();
    const int exit_code = platform->reap(handle);
    if (exit_code >= 0) {
        wait_for_output_readers(handle);
    }
    return exit_code;
}

std::string ProcessManager::read_output(ProcessHandle handle,
                                        int max_bytes,
                                        int wait_timeout_ms) {
    if (!handle.output_capture || max_bytes <= 0) {
        return "";
    }
    return handle.output_capture->read_tail(
        static_cast<std::size_t>(max_bytes), wait_timeout_ms);
}

int ProcessManager::run_process_with_output(
    const std::string& executable,
    const std::vector<std::string>& args,
    OutputLineCallback on_line,
    const std::string& working_dir,
    int timeout_seconds,
    bool capture_stderr) {

    auto platform = create_process_platform();
    return platform->run_with_output(executable, args, on_line, working_dir, timeout_seconds, capture_stderr);
}

void ProcessManager::kill_process(ProcessHandle handle) {
    auto platform = create_process_platform();
    platform->kill(handle);
    wait_for_output_readers(handle);
}

void ProcessManager::terminate_process(ProcessHandle handle) {
    auto platform = create_process_platform();
    platform->terminate_without_cleanup(handle);
}

int ProcessManager::find_free_port(int start_port) {
    auto platform = create_process_platform();
    return platform->find_free_port(start_port);
}

int ProcessManager::run_command(const std::string& command, std::string& output, int timeout_seconds) {
    auto platform = create_process_platform();
    return platform->run_command(command, output, timeout_seconds);
}

} // namespace utils
} // namespace lemon
