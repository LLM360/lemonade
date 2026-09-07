#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

#ifndef LEMONADE_MCP_CLIENT_SOURCE
    #error "LEMONADE_MCP_CLIENT_SOURCE must name mcp_client.cpp"
#endif

namespace {

int failures = 0;

void check(const std::string& name, bool condition) {
    if (condition) {
        std::cout << "[PASS] " << name << '\n';
        return;
    }
    std::cerr << "[FAIL] " << name << '\n';
    ++failures;
}

std::string read_source() {
    std::ifstream input(LEMONADE_MCP_CLIENT_SOURCE);
    if (!input) {
        throw std::runtime_error("failed to open mcp_client.cpp");
    }
    return {std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()};
}

std::string stop_body(const std::string& source) {
    const std::size_t start = source.find("void stop_locked()");
    const std::size_t end = source.find(
        "if (stdout_thread_.joinable())", start);
    if (start == std::string::npos || end == std::string::npos) {
        throw std::runtime_error("failed to locate MCP stop body");
    }
    return source.substr(start, end - start);
}

}  // namespace

int main() {
    try {
        const std::string body = stop_body(read_source());
        const std::size_t windows_process =
            body.find("if (process_info_.hProcess)");
        const std::size_t posix_start =
            body.find("#else", windows_process);
        if (windows_process == std::string::npos ||
            posix_start == std::string::npos) {
            throw std::runtime_error("failed to locate MCP POSIX stop path");
        }
        const std::string posix = body.substr(posix_start);
        check("MCP stop uses one POSIX lifecycle path",
              posix.find("#ifdef __APPLE__") == std::string::npos);
        if (posix.find("#ifdef __APPLE__") != std::string::npos) {
            return 1;
        }

        const std::size_t observe = posix.find("::waitid(");
        const std::size_t terminate =
            posix.find("::kill(-pid_, SIGTERM)");
        const std::size_t force_kill =
            posix.rfind("::kill(-pid_, SIGKILL)");
        const std::size_t final_reap = posix.find("::waitpid(pid_");

        check("MCP stop observes the POSIX child without reaping",
              observe != std::string::npos &&
                  posix.find("WNOWAIT", observe) != std::string::npos);
        check("MCP stop cleans the POSIX process group before final reap",
              observe < terminate && terminate < force_kill &&
                  force_kill < final_reap);
        check("MCP stop does not use a reaping POSIX status probe",
              posix.find("::waitpid(pid_, &status, WNOHANG)") ==
                      std::string::npos &&
                  posix.find("::waitpid(pid_, &status, 0)") == final_reap);
    } catch (const std::exception& error) {
        std::cerr << "mcp client process contract failed: " << error.what()
                  << '\n';
        return 1;
    }

    if (failures != 0) {
        std::cerr << failures << " MCP client process contract check(s) failed\n";
        return 1;
    }
    std::cout << "MCP client process contract checks passed\n";
    return 0;
}
