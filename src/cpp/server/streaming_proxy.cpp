#include "lemon/streaming_proxy.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <curl/curl.h>
#include <lemon/utils/aixlog.hpp>

namespace lemon {

namespace {

void extract_telemetry_from_chunk(const nlohmann::json& chunk, StreamingProxy::TelemetryData& telemetry) {
    nlohmann::json usage;
    if (chunk.contains("usage")) {
        usage = chunk["usage"];
    } else if (chunk.contains("response") && chunk["response"].is_object() && chunk["response"].contains("usage")) {
        usage = chunk["response"]["usage"];
    }

    if (usage.is_object()) {
        if (usage.contains("prompt_tokens")) {
            telemetry.input_tokens = usage["prompt_tokens"].get<int>();
        } else if (usage.contains("input_tokens")) {
            telemetry.input_tokens = usage["input_tokens"].get<int>();
        }
        if (usage.contains("prompt_tokens") || usage.contains("input_tokens")) {
            telemetry.prompt_tokens = telemetry.input_tokens;
        }
        if (usage.contains("completion_tokens")) {
            telemetry.output_tokens = usage["completion_tokens"].get<int>();
        } else if (usage.contains("output_tokens")) {
            telemetry.output_tokens = usage["output_tokens"].get<int>();
        }
        if (usage.contains("prefill_duration_ttft")) {
            telemetry.time_to_first_token = usage["prefill_duration_ttft"].get<double>();
        }
        if (usage.contains("decoding_speed_tps")) {
            telemetry.tokens_per_second = usage["decoding_speed_tps"].get<double>();
        }
        if (usage.contains("prompt_tokens_details") && usage["prompt_tokens_details"].is_object() &&
            usage["prompt_tokens_details"].contains("cached_tokens") &&
            usage["prompt_tokens_details"]["cached_tokens"].is_number()) {
            telemetry.cache_tokens = usage["prompt_tokens_details"]["cached_tokens"].get<int>();
        } else if (usage.contains("input_tokens_details") && usage["input_tokens_details"].is_object() &&
                   usage["input_tokens_details"].contains("cached_tokens") &&
                   usage["input_tokens_details"]["cached_tokens"].is_number()) {
            // Responses API usage shape.
            telemetry.cache_tokens = usage["input_tokens_details"]["cached_tokens"].get<int>();
        }
    }

    nlohmann::json timings;
    if (chunk.contains("timings")) {
        timings = chunk["timings"];
    } else if (chunk.contains("response") && chunk["response"].is_object() && chunk["response"].contains("timings")) {
        timings = chunk["response"]["timings"];
    }

    if (timings.is_object()) {
        if (timings.contains("prompt_n")) {
            telemetry.input_tokens = timings["prompt_n"].get<int>();
        }
        if (timings.contains("predicted_n")) {
            telemetry.output_tokens = timings["predicted_n"].get<int>();
        }
        if (timings.contains("prompt_ms")) {
            telemetry.time_to_first_token = timings["prompt_ms"].get<double>() / 1000.0;
        }
        if (timings.contains("predicted_per_second")) {
            telemetry.tokens_per_second = timings["predicted_per_second"].get<double>();
        }
        if (timings.contains("cache_n") && timings["cache_n"].is_number()) {
            telemetry.cache_tokens = timings["cache_n"].get<int>();
        }
    }
}

std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool is_gpu_hang_or_compute_error(const std::string& message) {
    const std::string lowered = lower_copy(message);
    return lowered.find("compute error") != std::string::npos ||
           lowered.find("gpu hang") != std::string::npos;
}

std::optional<std::string> sse_event_data(const std::string& frame) {
    std::string event_data;
    bool has_data_field = false;
    std::size_t line_start = 0;
    while (line_start < frame.size()) {
        const std::size_t terminator =
            frame.find_first_of("\r\n", line_start);
        const std::size_t content_end =
            terminator == std::string::npos ? frame.size() : terminator;
        const std::size_t content_length = content_end - line_start;
        const bool empty_data_field =
            content_length == 4 &&
            frame.compare(line_start, content_length, "data") == 0;
        const bool data_field =
            content_length >= 5 && frame.compare(line_start, 5, "data:") == 0;
        if (empty_data_field || data_field) {
            if (has_data_field) {
                event_data.push_back('\n');
            }
            if (data_field) {
                std::size_t payload_start = line_start + 5;
                if (payload_start < content_end &&
                    frame[payload_start] == ' ') {
                    ++payload_start;
                }
                event_data.append(
                    frame, payload_start, content_end - payload_start);
            }
            has_data_field = true;
        }

        if (terminator == std::string::npos) {
            break;
        }
        line_start = terminator + 1;
        if (frame[terminator] == '\r' && line_start < frame.size() &&
            frame[line_start] == '\n') {
            ++line_start;
        }
    }

    if (!has_data_field) {
        return std::nullopt;
    }
    return event_data;
}

bool is_sse_done_frame(const std::string& frame) {
    const std::optional<std::string> event_data = sse_event_data(frame);
    return event_data.has_value() && *event_data == "[DONE]";
}

class SseFrameBuffer {
public:
    explicit SseFrameBuffer(std::size_t max_frame_bytes)
        : max_frame_bytes_(max_frame_bytes) {}

    template <typename FrameCallback, typename LineCallback>
    bool append(const char* data,
                std::size_t length,
                FrameCallback&& frame_callback,
                LineCallback&& line_callback) {
        std::size_t offset = 0;
        while (offset < length) {
            const std::size_t buffer_limit = max_frame_bytes_ + 1;
            const std::size_t available = buffer_limit - buffer_.size();
            const std::size_t append_size =
                std::min(length - offset, available);
            buffer_.append(data + offset, append_size);
            offset += append_size;
            if (!drain(false, frame_callback, line_callback)) {
                return false;
            }
        }
        return true;
    }

    template <typename FrameCallback, typename LineCallback>
    bool finish(FrameCallback&& frame_callback,
                LineCallback&& line_callback) {
        return drain(true, frame_callback, line_callback);
    }

    bool empty() const {
        return buffer_.empty();
    }

    void clear() {
        buffer_.clear();
        scan_position_ = 0;
        line_start_ = 0;
    }

private:
    bool prepare_stream_start(bool end_of_stream) {
        if (!stream_start_) {
            return true;
        }

        static constexpr char utf8_bom[] = "\xEF\xBB\xBF";
        constexpr std::size_t utf8_bom_size = sizeof(utf8_bom) - 1;
        const std::size_t comparison_size =
            std::min(buffer_.size(), utf8_bom_size);
        if (buffer_.compare(
                0, comparison_size, utf8_bom, comparison_size) != 0) {
            stream_start_ = false;
            return true;
        }
        if (buffer_.size() < utf8_bom_size && !end_of_stream) {
            return false;
        }
        if (buffer_.size() >= utf8_bom_size) {
            if (utf8_bom_removed_) {
                throw std::runtime_error(
                    "backend connection failed during SSE stream before DONE: "
                    "multiple leading UTF-8 BOMs");
            }
            buffer_.erase(0, utf8_bom_size);
            utf8_bom_removed_ = true;
            return prepare_stream_start(end_of_stream);
        }
        stream_start_ = false;
        return true;
    }

    template <typename FrameCallback, typename LineCallback>
    bool drain(bool end_of_stream,
               FrameCallback&& frame_callback,
               LineCallback&& line_callback) {
        if (!prepare_stream_start(end_of_stream)) {
            return true;
        }

        std::size_t consumed = 0;
        while (scan_position_ < buffer_.size()) {
            const std::size_t terminator =
                buffer_.find_first_of("\r\n", scan_position_);
            if (terminator == std::string::npos) {
                scan_position_ = buffer_.size();
                break;
            }

            std::size_t terminator_length = 1;
            if (buffer_[terminator] == '\r') {
                if (terminator + 1 == buffer_.size() && !end_of_stream) {
                    scan_position_ = terminator;
                    break;
                }
                if (terminator + 1 < buffer_.size() &&
                    buffer_[terminator + 1] == '\n') {
                    terminator_length = 2;
                }
            }

            const std::size_t terminated_end =
                terminator + terminator_length;
            if (terminated_end - consumed > max_frame_bytes_) {
                throw_frame_too_large();
            }

            line_callback(
                buffer_.substr(line_start_, terminator - line_start_));
            const bool frame_complete = terminator == line_start_;
            scan_position_ = terminated_end;
            line_start_ = terminated_end;
            if (!frame_complete) {
                continue;
            }

            if (!frame_callback(
                    buffer_.substr(consumed, terminated_end - consumed))) {
                clear();
                return false;
            }
            consumed = terminated_end;
        }

        if (buffer_.size() - consumed > max_frame_bytes_) {
            throw_frame_too_large();
        }
        if (consumed > 0) {
            buffer_.erase(0, consumed);
            scan_position_ -= consumed;
            line_start_ -= consumed;
        }
        return true;
    }

    [[noreturn]] void throw_frame_too_large() const {
        throw std::runtime_error(
            "backend connection failed during SSE stream before DONE: "
            "SSE frame exceeds the maximum size of " +
            std::to_string(max_frame_bytes_) + " bytes");
    }

    const std::size_t max_frame_bytes_;
    std::string buffer_;
    std::size_t scan_position_ = 0;
    std::size_t line_start_ = 0;
    bool stream_start_ = true;
    bool utf8_bom_removed_ = false;
};

} // namespace


void StreamingProxy::forward_sse_stream(
    const std::string& backend_url,
    const std::string& request_body,
    httplib::DataSink& sink,
    std::function<void(const TelemetryData&)> on_complete,
    long timeout_seconds,
    std::function<void()> on_backend_progress,
    long heartbeat_interval_ms,
    std::function<void()> on_frame) {
    forward_sse_stream_impl(
        backend_url, request_body, sink, nullptr, std::move(on_complete),
        timeout_seconds, std::move(on_backend_progress), heartbeat_interval_ms,
        std::move(on_frame));
}

void StreamingProxy::forward_transformed_sse_stream(
    const std::string& backend_url,
    const std::string& request_body,
    httplib::DataSink& sink,
    SseFrameTransform frame_transform,
    std::function<void(const TelemetryData&)> on_complete,
    long timeout_seconds,
    std::function<void()> on_backend_progress,
    long heartbeat_interval_ms,
    std::function<void()> on_frame) {
    forward_sse_stream_impl(
        backend_url, request_body, sink, std::move(frame_transform),
        std::move(on_complete), timeout_seconds,
        std::move(on_backend_progress), heartbeat_interval_ms,
        std::move(on_frame));
}

void StreamingProxy::forward_sse_stream_impl(
    const std::string& backend_url,
    const std::string& request_body,
    httplib::DataSink& sink,
    SseFrameTransform frame_transform,
    std::function<void(const TelemetryData&)> on_complete,
    long timeout_seconds,
    std::function<void()> on_backend_progress,
    long heartbeat_interval_ms,
    std::function<void()> on_frame) {

    TelemetryData telemetry;
    try {
        auto req_json = json::parse(request_body);
        if (req_json.contains("model") && req_json["model"].is_string()) {
            telemetry.model_name = req_json["model"].get<std::string>();
        }
    } catch (...) {}
    SseFrameBuffer frame_buffer(kMaxSseFrameBytes);
    std::exception_ptr frame_processing_error;
    bool stream_error = false;
    bool has_done_marker = false;
    bool has_first_token = false;
    double time_to_first_token = 0.0;
    const auto start_time = std::chrono::steady_clock::now();
    auto last_downstream_write_time = start_time;

    int backend_status = 200;
    std::string error_body;
    static constexpr size_t max_error_body = 64 * 1024;

    auto process_line = [&telemetry](const std::string& line) {
        std::string json_str;
        if (line.find("data: ") == 0) {
            json_str = line.substr(6);
        } else if (line.find("ChatCompletionChunk: ") == 0) {
            json_str = line.substr(21);
        }
        if (!json_str.empty() && json_str != "[DONE]") {
            try {
                auto chunk = json::parse(json_str);
                extract_telemetry_from_chunk(chunk, telemetry);
            } catch (...) {}
        }
    };

    auto process_frame = [&](const std::string& source_frame) {
        const bool is_done_frame = is_sse_done_frame(source_frame);
        has_done_marker = has_done_marker || is_done_frame;
        std::string frame = frame_transform
            ? frame_transform(source_frame)
            : source_frame;
        if (frame.empty()) {
            return true;
        }
        if (!sink.write(frame.data(), frame.size())) {
            return false;
        }
        last_downstream_write_time = std::chrono::steady_clock::now();

        if (on_frame) {
            on_frame();
        }
        if (!has_first_token &&
            frame.find("data: ") != std::string::npos) {
            has_first_token = true;
            time_to_first_token = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time)
                                      .count();
        }
        return true;
    };

    auto process_frame_bytes = [&](const char* data,
                                   std::size_t length,
                                   bool end_of_stream) {
        try {
            if (end_of_stream) {
                return frame_buffer.finish(process_frame, process_line);
            }
            return frame_buffer.append(
                data, length, process_frame, process_line);
        } catch (...) {
            frame_processing_error = std::current_exception();
            frame_buffer.clear();
            return false;
        }
    };

    utils::HttpResponse result = utils::HttpClient::post_stream(
        backend_url,
        request_body,
        [&backend_status, &error_body, &on_backend_progress,
         &process_frame_bytes](const char* data, size_t length) {
            if (backend_status != 200) {
                if (error_body.size() < max_error_body) {
                    error_body.append(data, std::min(length, max_error_body - error_body.size()));
                }
                return true;
            }

            if (length > 0 && on_backend_progress) {
                on_backend_progress();
            }

            return process_frame_bytes(data, length, false);
        },
        {},
        timeout_seconds,
        [&backend_status](int status) { backend_status = status; },
        utils::HttpSecurityPolicy::TrustedLoopback,
        [&sink, &last_downstream_write_time, &backend_status,
         heartbeat_interval_ms]() {
            if (sink.is_writable && !sink.is_writable()) {
                return true;
            }

            if (heartbeat_interval_ms > 0 && backend_status == 200) {
                const auto now = std::chrono::steady_clock::now();
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - last_downstream_write_time).count();
                if (elapsed >= heartbeat_interval_ms) {
                    static constexpr const char* heartbeat = ": ping\n\n";
                    if (!sink.write(heartbeat, std::strlen(heartbeat))) {
                        return true;
                    }
                    last_downstream_write_time = now;
                }
            }

            return false;
        }
    );

    const bool may_have_final_frame =
        result.curl_code == CURLE_OK ||
        result.curl_code == CURLE_PARTIAL_FILE ||
        result.curl_code == CURLE_RECV_ERROR;
    const bool eof_sink_rejected =
        backend_status == 200 && may_have_final_frame &&
        !process_frame_bytes(nullptr, 0, true);
    const bool has_incomplete_frame = !frame_buffer.empty();
    frame_buffer.clear();
    if (frame_processing_error) {
        std::rethrow_exception(frame_processing_error);
    }
    if (result.curl_code == CURLE_OK && backend_status == 200 &&
        has_incomplete_frame) {
        throw std::runtime_error(
            "backend connection failed during SSE stream before DONE: "
            "incomplete final event");
    }

    const bool client_disconnected =
        result.curl_code == CURLE_WRITE_ERROR ||
        result.curl_code == CURLE_ABORTED_BY_CALLBACK || eof_sink_rejected;
    const bool transport_interrupted =
        result.curl_code == CURLE_PARTIAL_FILE || result.curl_code == CURLE_RECV_ERROR;

    if (eof_sink_rejected && result.curl_code == CURLE_OK) {
        stream_error = true;
        telemetry.error_message = "Client disconnected during stream";
    } else if (result.curl_code != CURLE_OK) {
        if (client_disconnected) {
            stream_error = true;
            LOG(WARNING, "StreamingProxy") << "Client disconnected during SSE stream (CURL error: " << result.curl_error << ")" << std::endl;
            telemetry.error_message = "Client disconnected during stream";
        } else if (transport_interrupted) {
            if (!has_done_marker) {
                // This is the important crash path: HTTP headers may have been sent and
                // some bytes may even have reached the client, but the SSE protocol never
                // completed. Do not synthesize [DONE], because that hides backend crashes
                // from the router and leaves stale loaded-model state behind.
                throw std::runtime_error(
                    "backend connection failed during SSE stream before DONE: CURL error: " +
                    result.curl_error);
            }
        } else {
            stream_error = true;
            LOG(ERROR, "StreamingProxy") << "SSE stream failed: CURL error: " << result.curl_error << std::endl;
            telemetry.error_message = "SSE stream failed: CURL error: " + result.curl_error;
        }
    }

    if (!client_disconnected &&
        (result.status_code != 200 || backend_status != 200)) {
        const int status = backend_status != 200 ? backend_status : result.status_code;
        LOG(ERROR, "StreamingProxy") << "Backend returned error: " << status
                                     << (error_body.empty() ? "" : ": " + error_body) << std::endl;
        telemetry.error_message = "Backend returned error status code: " + std::to_string(status);

        if (is_gpu_hang_or_compute_error(error_body)) {
            throw std::runtime_error("backend compute error / GPU hang during streaming: " + error_body);
        }

        stream_error = true;

        // The response is already committed as 200 text/event-stream, so an
        // unframed error body is dropped by every spec-compliant client parser.
        // No [DONE] follows, matching OpenAI's behavior for in-stream errors.
        json payload;
        try {
            payload = json::parse(error_body);
        } catch (...) {
            payload = nullptr;
        }
        if (!payload.is_object() || !payload.contains("error")) {
            std::string message = error_body.empty()
                ? "backend returned HTTP " + std::to_string(status)
                : error_body;
            payload = json{{"error", {{"message", message},
                                      {"type", "backend_error"},
                                      {"status_code", status}}}};
        } else if (payload["error"].is_object() && !payload["error"].contains("status_code")) {
            // A backend's own error object carries no transport status, so
            // adapters downstream would have to guess one.
            payload["error"]["status_code"] = status;
        }
        const std::string event = "data: " + payload.dump() + "\n\n";
        sink.write(event.data(), event.size());
    }

    if (!stream_error) {
        // Ensure [DONE] marker is sent only for clean transports. If the transport
        // was interrupted before [DONE], the block above throws and recovery is
        // handled by WrappedServer/Router instead of pretending success.
        if (!has_done_marker) {
            LOG(WARNING, "StreamingProxy") << "WARNING: Backend did not send [DONE] marker, adding it" << std::endl;
            const char* done_marker = "data: [DONE]\n\n";
            sink.write(done_marker, strlen(done_marker));
        }

        sink.done();

        LOG(INFO, "Server") << "Streaming completed - 200 OK" << std::endl;

        if (telemetry.time_to_first_token <= 0.0) {
            telemetry.time_to_first_token = time_to_first_token;
        }
        if (telemetry.tokens_per_second <= 0.0 && telemetry.output_tokens > 0) {
            double total_duration = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_time).count();
            double decode_duration = total_duration - telemetry.time_to_first_token;
            if (decode_duration > 0.0) {
                telemetry.tokens_per_second = telemetry.output_tokens / decode_duration;
            }
        }
        telemetry.print();

        if (on_complete) {
            on_complete(telemetry);
        }
    } else {
        sink.done();
        if (on_complete) {
            on_complete(telemetry);
        }
    }
}

void StreamingProxy::forward_byte_stream(
    const std::string& backend_url,
    const std::string& request_body,
    httplib::DataSink& sink,
    long timeout_seconds,
    std::function<void()> on_chunk) {

    bool stream_error = false;

    // On a non-200 the backend body is an error description, not payload bytes:
    // divert it here instead of the client sink, so it can be reshaped into a
    // JSON error below rather than served as successful media.
    int backend_status = 200;
    std::string error_body;
    static constexpr size_t max_error_body = 64 * 1024;

    utils::HttpResponse result = utils::HttpClient::post_stream(
        backend_url,
        request_body,
        [&sink, &on_chunk, &backend_status, &error_body](const char* data, size_t length) {
            if (backend_status != 200) {
                if (error_body.size() < max_error_body) {
                    error_body.append(data, std::min(length, max_error_body - error_body.size()));
                }
                return true;
            }

            if (on_chunk) {
                on_chunk();
            }

            if (!sink.write(data, length)) {
                return false;
            }

            return true;
        },
        {},
        timeout_seconds,
        [&backend_status](int status) { backend_status = status; },
        utils::HttpSecurityPolicy::TrustedLoopback
    );

    const bool transport_interrupted =
        result.curl_code == CURLE_PARTIAL_FILE || result.curl_code == CURLE_RECV_ERROR;

    if (result.curl_code != CURLE_OK) {
        stream_error = true;
        if (result.curl_code == CURLE_WRITE_ERROR) {
            LOG(WARNING, "StreamingProxy") << "Client disconnected during byte stream (CURL error: " << result.curl_error << ")" << std::endl;
        } else if (transport_interrupted) {
            // Keep byte streams consistent with SSE: an interrupted transport is a
            // backend failure, not a clean stream completion. The caller will mark
            // the backend unavailable and reload after the current response unwinds.
            throw std::runtime_error(
                "backend connection failed during byte stream: CURL error: " +
                result.curl_error);
        } else {
            LOG(ERROR, "StreamingProxy") << "Byte stream failed: CURL error: " << result.curl_error << std::endl;
        }
    }

    if (result.status_code != 200 || backend_status != 200) {
        const int status = backend_status != 200 ? backend_status : result.status_code;
        LOG(ERROR, "StreamingProxy") << "Backend returned error " << status
                                     << (error_body.empty() ? "" : ": " + error_body) << std::endl;

        if (is_gpu_hang_or_compute_error(error_body)) {
            throw std::runtime_error("backend compute error / GPU hang during byte stream: " + error_body);
        }

        stream_error = true;

        json payload;
        try {
            payload = json::parse(error_body);
        } catch (...) {
            payload = nullptr;
        }
        if (!payload.is_object() || !payload.contains("error")) {
            std::string message = error_body.empty()
                ? "backend returned HTTP " + std::to_string(status)
                : error_body;
            payload = json{{"error", {{"message", message},
                                      {"type", "backend_error"},
                                      {"status_code", status}}}};
        } else if (payload["error"].is_object() && !payload["error"].contains("status_code")) {
            payload["error"]["status_code"] = status;
        }
        const std::string out = payload.dump();
        sink.write(out.data(), out.size());
    }

    if (!stream_error) {
        LOG(INFO, "Server") << "Streaming completed - 200 OK" << std::endl;
    }
    sink.done();
}

StreamingProxy::TelemetryData StreamingProxy::extract_telemetry(const nlohmann::json& payload) {
    TelemetryData telemetry;
    extract_telemetry_from_chunk(payload, telemetry);
    return telemetry;
}

StreamingProxy::TelemetryData StreamingProxy::parse_telemetry(const std::string& buffer) {
    TelemetryData telemetry;

    std::istringstream stream(buffer);
    std::string line;
    json last_chunk_with_usage;

    while (std::getline(stream, line)) {
        std::string json_str;
        if (line.find("data: ") == 0) {
            json_str = line.substr(6);
        } else if (line.find("ChatCompletionChunk: ") == 0) {
            json_str = line.substr(21);
        }

        if (!json_str.empty() && json_str != "[DONE]") {
            try {
                auto chunk = json::parse(json_str);
                bool has_usage = chunk.contains("usage") || chunk.contains("timings");
                if (!has_usage && chunk.contains("response") && chunk["response"].is_object()) {
                    has_usage = chunk["response"].contains("usage") || chunk["response"].contains("timings");
                }
                if (has_usage) {
                    last_chunk_with_usage = chunk;
                }
            } catch (...) {}
        }
    }

    if (!last_chunk_with_usage.empty()) {
        try {
            extract_telemetry_from_chunk(last_chunk_with_usage, telemetry);
        } catch (const std::exception& e) {
            LOG(ERROR, "StreamingProxy") << "Error parsing telemetry: " << e.what() << std::endl;
        }
    }

    return telemetry;
}

void StreamingProxy::process_sse_lines(std::string& line_buffer, std::function<void(const std::string&)> line_callback) {
    size_t pos;
    while ((pos = line_buffer.find('\n')) != std::string::npos) {
        std::string line = line_buffer.substr(0, pos);
        line_buffer.erase(0, pos + 1);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        line_callback(line);
    }
}

void StreamingProxy::accumulate_responses_delta(const nlohmann::json& parsed, std::string& accumulated_text) {
    if (parsed.contains("choices") && parsed["choices"].is_array() && !parsed["choices"].empty()) {
        auto choice = parsed["choices"][0];
        if (choice.is_object() && choice.contains("delta")) {
            auto delta = choice["delta"];
            if (delta.is_object() && delta.contains("content") && delta["content"].is_string()) {
                accumulated_text += delta["content"].get<std::string>();
            }
        }
    }
    if (parsed.contains("response") && parsed["response"].is_string()) {
        accumulated_text += parsed["response"].get<std::string>();
    }
    // Supports Responses API type-restricted delta vs. backward compatible fallback.
    if (parsed.contains("delta")) {
        bool should_extract_delta = true;
        if (parsed.contains("type")) {
            should_extract_delta = (parsed["type"] == "response.output_text.delta");
        }
        if (should_extract_delta) {
            if (parsed["delta"].is_string()) {
                accumulated_text += parsed["delta"].get<std::string>();
            } else if (parsed["delta"].is_object() && parsed["delta"].contains("text") && parsed["delta"]["text"].is_string()) {
                accumulated_text += parsed["delta"]["text"].get<std::string>();
            }
        }
    }
}

} // namespace lemon
