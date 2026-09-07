#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <regex>
#include <string>
#include <system_error>

#ifdef __linux__
#include <unistd.h>
#endif

namespace lemon {
namespace backends {
namespace llamacpp {
namespace detail {

inline std::string lowercase_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline bool identifies_k2_horizon_model(const std::string& model_name,
                                         const std::string& checkpoint,
                                         const std::string& architecture) {
    const std::string normalized_architecture = lowercase_ascii(architecture);
    if (normalized_architecture == "k2-horizon" ||
        normalized_architecture == "k2_horizon") {
        return true;
    }
    if (!normalized_architecture.empty()) {
        return false;
    }

    const auto contains_k2_horizon = [](const std::string& value) {
        const std::string normalized = lowercase_ascii(value);
        return normalized.find("k2-horizon") != std::string::npos ||
               normalized.find("k2_horizon") != std::string::npos;
    };
    return contains_k2_horizon(model_name) || contains_k2_horizon(checkpoint);
}

inline std::string k2_horizon_unsupported_architecture_error(
    const std::string& startup_output) {
    size_t start = 0;
    while (start <= startup_output.size()) {
        const size_t end = startup_output.find_first_of("\r\n", start);
        std::string line = startup_output.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        const std::string normalized = lowercase_ascii(line);
        const bool names_k2 =
            normalized.find("k2-horizon") != std::string::npos ||
            normalized.find("k2_horizon") != std::string::npos;
        if (names_k2 &&
            normalized.find("unknown model architecture") != std::string::npos) {
            const size_t first = line.find_first_not_of(" \t");
            const size_t last = line.find_last_not_of(" \t");
            return first == std::string::npos
                       ? std::string()
                       : line.substr(first, last - first + 1);
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
        if (startup_output[end] == '\r' && start < startup_output.size() &&
            startup_output[start] == '\n') {
            ++start;
        }
    }
    return "";
}

inline bool should_report_k2_system_startup_diagnostic(
    const std::string& backend,
    bool load_cancelled,
    const std::string& model_name,
    const std::string& checkpoint,
    const std::string& architecture,
    const std::string& startup_output = "") {
    return backend == "system" && !load_cancelled &&
           identifies_k2_horizon_model(model_name, checkpoint, architecture) &&
           !k2_horizon_unsupported_architecture_error(startup_output).empty();
}

inline std::string parse_system_llamacpp_version(const std::string& output,
                                                 int exit_code = 0) {
    if (exit_code != 0) {
        return "unknown";
    }

    std::smatch match;
    if (std::regex_search(output, match, std::regex(R"(build\s+(\d+))"))) {
        return "b" + match[1].str();
    }

    if (std::regex_search(
            output,
            match,
            std::regex(R"(version:\s*(\d+)|version\s+b?(\d+))"))) {
        for (size_t i = 1; i < match.size(); ++i) {
            if (match[i].matched) {
                return "b" + match[i].str();
            }
        }
    }
    return output.empty() ? "unknown" : "detected";
}

inline std::string k2_horizon_system_startup_error(
    const std::string& model_name,
    const std::string& executable_path,
    const std::string& version,
    const std::string& original_error) {
    const std::string displayed_path = executable_path.empty() ? "unknown" : executable_path;
    const std::string displayed_version =
        version.empty() || version == "detected" ? "unknown" : version;
    return "System llama-server failed to start while loading K2-Horizon model '" +
           model_name + "' using executable '" + displayed_path + "' (version " +
           displayed_version + "). Original error: " + original_error +
           ". Install a K2-Horizon-capable llama.cpp build from "
           "https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon and "
           "ensure that executable is first on PATH, then retry.";
}

// Private, deterministic pieces of the system llama.cpp HIP lookup. The full
// production availability check stays in llamacpp_server.cpp so this header is
// only a narrow test seam, not a second implementation of the policy.
inline bool is_valid_ggml_hip_plugin_path(const std::filesystem::path& path) {
#ifdef __linux__
    std::string name = path.filename().string();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    const bool name_matches = name.rfind("libggml-hip", 0) == 0 &&
                              name.find(".so") != std::string::npos;

    std::error_code ec;
    return name_matches && std::filesystem::is_regular_file(path, ec);
#else
    (void)path;
    return false;
#endif
}

inline bool ggml_hip_env_override_available() {
#ifdef __linux__
    const char* env = std::getenv("LEMONADE_GGML_HIP_PATH");
    return env && *env && is_valid_ggml_hip_plugin_path(env);
#else
    return false;
#endif
}

inline bool ggml_hip_plugin_near_llama_server(const std::string& path_str) {
#ifdef __linux__
    size_t start = 0;
    while (start <= path_str.size()) {
        size_t end = path_str.find(':', start);
        std::string dir = path_str.substr(
            start,
            end == std::string::npos ? std::string::npos : end - start);
        if (!dir.empty()) {
            std::error_code ec;
            std::filesystem::path bin_dir(dir);
            std::filesystem::path llama_server = bin_dir / "llama-server";
            if (std::filesystem::is_regular_file(llama_server, ec) &&
                access(llama_server.c_str(), X_OK) == 0) {
                ec.clear();
                if (std::filesystem::exists(bin_dir / "libggml-hip.so", ec)) {
                    return true;
                }
                ec.clear();
                if (std::filesystem::exists(
                        bin_dir.parent_path() / "lib" / "libggml-hip.so", ec)) {
                    return true;
                }
                // Preserve the current resolver policy: once PATH selects an
                // executable llama-server, do not inspect later PATH entries.
                break;
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
#else
    (void)path_str;
#endif
    return false;
}

}  // namespace detail
}  // namespace llamacpp
}  // namespace backends
}  // namespace lemon
