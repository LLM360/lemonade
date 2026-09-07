#include "lemon/backends/llamacpp/llamacpp_server.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

using namespace std::chrono_literals;

namespace {

constexpr const char* kRequestedModel = "builtin.K2-Horizon-0.9B-GGUF";
constexpr const char* kBackendModel = "/private/cache/models/K2-Horizon-1B-BF16.gguf";
constexpr std::size_t kMaxSseFrameBytes =
    lemon::StreamingProxy::kMaxSseFrameBytes;

int failures = 0;

void check(bool condition, const char* name) {
    if (condition) {
        std::printf("[PASS] %s\n", name);
        return;
    }
    std::printf("[FAIL] %s\n", name);
    ++failures;
}

class TestLlamaCppServer : public lemon::backends::LlamaCppServer {
public:
    explicit TestLlamaCppServer(int port)
        : LlamaCppServer("debug", nullptr, nullptr) {
        port_ = port;
    }

    bool is_backend_alive() const override {
        return true;
    }

    void forward_raw_sse(const std::string& endpoint,
                         const std::string& request_body,
                         httplib::DataSink& sink) {
        lemon::WrappedServer::forward_streaming_request(
            endpoint, request_body, sink, true, 5, nullptr);
    }
};

std::vector<nlohmann::json> data_events(const std::string& stream,
                                        int* malformed_events = nullptr) {
    std::vector<nlohmann::json> events;
    std::string buffer = stream;
    std::string event_data;
    bool has_data_line = false;
    auto finish_event = [&]() {
        if (event_data.empty() || event_data == "[DONE]") {
            event_data.clear();
            has_data_line = false;
            return;
        }
        const nlohmann::json parsed =
            nlohmann::json::parse(event_data, nullptr, false);
        event_data.clear();
        has_data_line = false;
        if (parsed.is_discarded()) {
            if (malformed_events) {
                ++*malformed_events;
            }
            return;
        }
        events.push_back(parsed);
    };
    lemon::StreamingProxy::process_sse_lines(
        buffer, [&event_data, &finish_event,
                 &has_data_line](const std::string& line) {
            if (line.empty()) {
                finish_event();
                return;
            }
            if (line.rfind("data:", 0) != 0) {
                return;
            }
            std::size_t payload_start = std::strlen("data:");
            while (payload_start < line.size() &&
                   (line[payload_start] == ' ' ||
                    line[payload_start] == '\t')) {
                ++payload_start;
            }
            if (has_data_line) {
                event_data.push_back('\n');
            }
            event_data.append(line, payload_start, std::string::npos);
            has_data_line = true;
        });
    return events;
}

std::size_t count_occurrences(const std::string& value,
                              const std::string& needle) {
    std::size_t count = 0;
    std::size_t offset = 0;
    while ((offset = value.find(needle, offset)) != std::string::npos) {
        ++count;
        offset += needle.size();
    }
    return count;
}

std::string sized_model_frame(std::size_t frame_size,
                              const std::string& separator) {
    const std::string prefix =
        "data: {\"id\":\"sized\",\"model\":\"" +
        std::string(kBackendModel) + "\",\"padding\":\"";
    const std::string suffix = "\"}" + separator;
    if (prefix.size() + suffix.size() > frame_size) {
        throw std::runtime_error("test frame size is too small");
    }
    return prefix +
           std::string(frame_size - prefix.size() - suffix.size(), 'x') +
           suffix;
}

std::string sized_incomplete_model_frame(std::size_t frame_size) {
    const std::string prefix =
        "data: {\"id\":\"sized\",\"model\":\"" +
        std::string(kBackendModel) + "\",\"padding\":\"";
    if (prefix.size() > frame_size) {
        throw std::runtime_error("test frame size is too small");
    }
    return prefix + std::string(frame_size - prefix.size(), 'x');
}

struct StreamResult {
    std::string output;
    bool retryable_reset = false;
    int done_calls = 0;
};

struct RawStreamResult {
    std::string output;
    bool interrupted = false;
    std::string error_message;
    int backend_progress_calls = 0;
};

StreamResult forward_stream(TestLlamaCppServer& server,
                            const std::string& endpoint,
                            const std::string& fault = "") {
    StreamResult result;
    httplib::DataSink sink;
    sink.write = [&result](const char* data, size_t length) {
        result.output.append(data, length);
        return true;
    };
    sink.done = [&result]() { ++result.done_calls; };
    sink.done_with_trailer = [&result](const httplib::Headers&) {
        ++result.done_calls;
    };
    sink.is_writable = []() { return true; };

    nlohmann::json request = {
        {"model", kRequestedModel},
        {"stream", true},
    };
    if (!fault.empty()) {
        request["fault"] = fault;
    }
    try {
        server.forward_streaming_request(
            endpoint, request.dump(), sink, true, 5, nullptr);
    } catch (const lemon::BackendStreamRetryableReset&) {
        result.retryable_reset = true;
    }
    return result;
}

StreamResult forward_wrapped_raw_stream(TestLlamaCppServer& server,
                                        const std::string& endpoint,
                                        const std::string& fault) {
    StreamResult result;
    httplib::DataSink sink;
    sink.write = [&result](const char* data, size_t length) {
        result.output.append(data, length);
        return true;
    };
    sink.done = [&result]() { ++result.done_calls; };
    sink.done_with_trailer = [&result](const httplib::Headers&) {
        ++result.done_calls;
    };
    sink.is_writable = []() { return true; };

    const nlohmann::json request = {
        {"model", kRequestedModel},
        {"stream", true},
        {"fault", fault},
    };
    try {
        server.forward_raw_sse(endpoint, request.dump(), sink);
    } catch (const lemon::BackendStreamRetryableReset&) {
        result.retryable_reset = true;
    }
    return result;
}

RawStreamResult forward_raw_stream(int port, const std::string& fault) {
    RawStreamResult result;
    httplib::DataSink sink;
    sink.write = [&result](const char* data, size_t length) {
        result.output.append(data, length);
        return true;
    };
    sink.done = []() {};
    sink.done_with_trailer = [](const httplib::Headers&) {};
    sink.is_writable = []() { return true; };

    const nlohmann::json request = {
        {"model", kRequestedModel},
        {"stream", true},
        {"fault", fault},
    };
    try {
        lemon::StreamingProxy::forward_sse_stream(
            "http://127.0.0.1:" + std::to_string(port) +
                "/v1/chat/completions",
            request.dump(), sink, nullptr, 5,
            [&result]() { ++result.backend_progress_calls; }, 0);
    } catch (const std::runtime_error& error) {
        result.interrupted = true;
        result.error_message = error.what();
    }
    return result;
}

int forward_stream_to_disconnected_client(TestLlamaCppServer& server) {
    int write_calls = 0;
    httplib::DataSink sink;
    sink.write = [&write_calls](const char*, size_t) {
        ++write_calls;
        return false;
    };
    sink.done = []() {};
    sink.done_with_trailer = [](const httplib::Headers&) {};
    sink.is_writable = []() { return true; };

    const nlohmann::json request = {
        {"model", kRequestedModel},
        {"stream", true},
    };
    server.forward_streaming_request(
        "/v1/chat/completions", request.dump(), sink, true, 5, nullptr);
    return write_calls;
}

}  // namespace

int main() {
    httplib::Server backend;
    std::atomic<int> partial_before_output_requests{0};
    std::atomic<int> partial_after_output_requests{0};
    const std::vector<std::string> endpoints = {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
    };

    for (const std::string& endpoint : endpoints) {
        backend.Post(endpoint, [endpoint, &partial_before_output_requests,
                                &partial_after_output_requests](
                                   const httplib::Request& request,
                                   httplib::Response& response) {
            const nlohmann::json body =
                nlohmann::json::parse(request.body, nullptr, false);
            const std::string fault =
                body.is_object() ? body.value("fault", "") : "";

            nlohmann::json event;
            if (endpoint == "/v1/responses") {
                event = {
                    {"type", "response.completed"},
                    {"response", {
                        {"id", "resp_mock"},
                        {"model", kBackendModel},
                        {"status", "completed"},
                    }},
                };
            } else {
                event = {
                    {"id", "cmpl_mock"},
                    {"model", kBackendModel},
                    {"choices", nlohmann::json::array()},
                };
            }

            const std::string separator =
                endpoint == "/v1/completions" ? "\r\n\r\n" : "\n\n";
            std::string frame =
                ": upstream-comment\n\n" +
                std::string("data: ") + event.dump() + separator;
            const std::string partial =
                "data: {\"id\":\"partial\",\"model\":\"" +
                std::string(kBackendModel) + "\"";

            if (fault == "done_literal_partial") {
                event["choices"] = nlohmann::json::array({
                    {{"delta", {{"content", "data: [DONE]"}}}},
                });
                frame = ": upstream-comment\n\n" +
                        std::string("data: ") + event.dump() + separator;
            }

            if (fault == "multiline_model") {
                const std::string multiline_stream =
                    "id: multiline\n"
                    "data: {\"id\":\"multi\",\n"
                    "data: \"model\":\"" +
                    std::string(kBackendModel) +
                    "\",\"choices\":[]}\n"
                    ": retained-comment\n\n"
                    "data: [DONE]\n\n";
                response.set_content(multiline_stream, "text/event-stream");
                return;
            }
            if (fault == "bom_model") {
                const std::string bom_stream =
                    std::string("\xEF\xBB\xBF", 3) + "data: " +
                    event.dump() + separator + "data: [DONE]\n\n";
                response.set_chunked_content_provider(
                    "text/event-stream",
                    [bom_stream](size_t, httplib::DataSink& sink) {
                        for (const char byte : bom_stream) {
                            if (!sink.write(&byte, 1)) {
                                return false;
                            }
                        }
                        sink.done();
                        return false;
                    });
                return;
            }
            if (fault == "bom_done_only") {
                response.set_content(
                    std::string("\xEF\xBB\xBF", 3) +
                        "data: [DONE]\n\n",
                    "text/event-stream");
                return;
            }
            if (fault == "double_bom_model") {
                const std::string double_bom_stream =
                    std::string("\xEF\xBB\xBF\xEF\xBB\xBF", 6) +
                    "data: " + event.dump() + separator +
                    "data: [DONE]\n\n";
                response.set_chunked_content_provider(
                    "text/event-stream",
                    [double_bom_stream](size_t, httplib::DataSink& sink) {
                        for (const char byte : double_bom_stream) {
                            if (!sink.write(&byte, 1)) {
                                return false;
                            }
                        }
                        sink.done();
                        return false;
                    });
                return;
            }
            if (fault == "double_bom_done_only") {
                response.set_content(
                    std::string("\xEF\xBB\xBF\xEF\xBB\xBF", 6) +
                        "data: [DONE]\n\n",
                    "text/event-stream");
                return;
            }
            if (fault == "terminal_cr") {
                response.set_content(
                    "data: " + event.dump() + "\r\n\r",
                    "text/event-stream");
                return;
            }
            if (fault == "malformed_model") {
                response.set_content(
                    "data: {\"id\":\"broken\",\"model\":\"" +
                        std::string(kBackendModel) + "\"\n\n",
                    "text/event-stream");
                return;
            }
            if (fault == "scalar_model") {
                response.set_content(
                    "data: \"" + std::string(kBackendModel) + "\"\n\n",
                    "text/event-stream");
                return;
            }
            if (fault == "frame_at_limit_lf") {
                response.set_content(
                    sized_model_frame(kMaxSseFrameBytes, "\n\n"),
                    "text/event-stream");
                return;
            }
            if (fault == "frame_at_limit_crlf") {
                response.set_content(
                    sized_model_frame(kMaxSseFrameBytes, "\r\n\r\n"),
                    "text/event-stream");
                return;
            }
            if (fault == "frame_over_limit") {
                response.set_content(
                    sized_model_frame(kMaxSseFrameBytes + 1, "\n\n"),
                    "text/event-stream");
                return;
            }
            if (fault == "frame_over_limit_after_frame") {
                const std::string safe_frame =
                    "data: {\"id\":\"safe\",\"model\":\"" +
                    std::string(kRequestedModel) + "\",\"choices\":[]}\n\n";
                response.set_content(
                    safe_frame +
                        sized_model_frame(kMaxSseFrameBytes + 1, "\n\n"),
                    "text/event-stream");
                return;
            }
            if (fault == "incomplete_frame_over_limit") {
                response.set_content(
                    sized_incomplete_model_frame(kMaxSseFrameBytes + 1),
                    "text/event-stream");
                return;
            }
            if (fault == "incomplete_frame_over_limit_after_frame") {
                const std::string safe_frame =
                    "data: {\"id\":\"safe\",\"model\":\"" +
                    std::string(kRequestedModel) + "\",\"choices\":[]}\n\n";
                response.set_content(
                    safe_frame +
                        sized_incomplete_model_frame(kMaxSseFrameBytes + 1),
                    "text/event-stream");
                return;
            }
            if (fault == "one_byte_fragmentation") {
                event["padding"] = std::string(4096, 'x');
                const std::string fragmented =
                    "data: " + event.dump() + "\r\n\r\n"
                    "data: [DONE]\r\r";
                response.set_chunked_content_provider(
                    "text/event-stream",
                    [fragmented](size_t, httplib::DataSink& sink) {
                        for (const char byte : fragmented) {
                            if (!sink.write(&byte, 1)) {
                                return false;
                            }
                        }
                        sink.done();
                        return false;
                    });
                return;
            }
            if (fault == "incomplete_progress") {
                const std::string fragmented =
                    "data: {\"id\":\"active\",\"model\":\"" +
                    std::string(kBackendModel) + "\"";
                response.set_chunked_content_provider(
                    "text/event-stream",
                    [fragmented](size_t, httplib::DataSink& sink) {
                        for (const char byte : fragmented) {
                            if (!sink.write(&byte, 1)) {
                                return false;
                            }
                            std::this_thread::sleep_for(1ms);
                        }
                        sink.done();
                        return false;
                    });
                return;
            }
            if (fault.rfind("split_done_", 0) == 0) {
                std::string done_separator = "\n\n";
                if (fault == "split_done_crlf") {
                    done_separator = "\r\n\r\n";
                } else if (fault == "split_done_cr") {
                    done_separator = "\r\r";
                }
                const std::string done = "data: [DONE]" + done_separator;
                response.set_chunked_content_provider(
                    "text/event-stream",
                    [frame, done](size_t, httplib::DataSink& sink) {
                        sink.write(frame.data(), frame.size());
                        const std::size_t split = done.size() / 2;
                        sink.write(done.data(), split);
                        std::this_thread::sleep_for(5ms);
                        sink.write(done.data() + split, done.size() - split);
                        sink.done();
                        return false;
                    });
                return;
            }
            if (fault == "clean_partial_before_output") {
                response.set_content(partial, "text/event-stream");
                return;
            }

            std::string truncated;
            if (fault == "partial_before_output") {
                const int attempt = partial_before_output_requests.fetch_add(1) + 1;
                if (attempt == 1) {
                    truncated = partial;
                }
            } else if (fault == "partial_after_output") {
                partial_after_output_requests.fetch_add(1);
                truncated = frame + partial;
            } else if (fault == "done_literal_partial") {
                truncated = frame + partial;
            }

            if (!truncated.empty()) {
                const std::size_t advertised_length = truncated.size() + 128;
                response.set_content_provider(
                    advertised_length,
                    "text/event-stream",
                    [truncated = std::move(truncated)](
                        size_t offset, size_t, httplib::DataSink& sink) {
                        if (offset == 0) {
                            sink.write(truncated.data(), truncated.size());
                        }
                        return false;
                    });
                return;
            }

            response.set_chunked_content_provider(
                "text/event-stream",
                [frame](size_t, httplib::DataSink& sink) {
                    const size_t split = frame.size() / 2;
                    sink.write(frame.data(), split);
                    std::this_thread::sleep_for(5ms);
                    sink.write(frame.data() + split, frame.size() - split);
                    const std::string done = "data: [DONE]\n\n";
                    const size_t done_split = done.size() / 2;
                    sink.write(done.data(), done_split);
                    sink.write(done.data() + done_split,
                               done.size() - done_split);
                    sink.done();
                    return false;
                });
        });
    }

    const int port = backend.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        std::printf("[FAIL] failed to bind mock backend\n");
        return 1;
    }

    std::thread backend_thread([&backend]() { backend.listen_after_bind(); });
    backend.wait_until_ready();

    TestLlamaCppServer server(port);

    const std::string chat =
        forward_stream(server, "/v1/chat/completions").output;
    const auto chat_events = data_events(chat);
    check(chat_events.size() == 1, "chat stream keeps one JSON event");
    check(!chat_events.empty() && chat_events[0].value("model", "") == kRequestedModel,
          "chat stream exposes the requested model identity");
    check(chat.find(kBackendModel) == std::string::npos,
          "chat stream does not expose the backend GGUF path");
    check(chat.find(": upstream-comment\n\n") != std::string::npos,
          "chat stream preserves complete non-data frames");
    check(count_occurrences(chat, "data: [DONE]\n\n") == 1,
          "chat stream preserves one split DONE frame");

    const std::string completion =
        forward_stream(server, "/v1/completions").output;
    const auto completion_events = data_events(completion);
    check(completion_events.size() == 1, "completion stream keeps one JSON event");
    check(!completion_events.empty() &&
              completion_events[0].value("model", "") == kRequestedModel,
          "completion stream exposes the requested model identity");
    check(completion.find(kBackendModel) == std::string::npos,
          "completion stream does not expose the backend GGUF path");
    check(completion.find("\r\n\r\n") != std::string::npos,
          "completion stream preserves CRLF frame boundaries");

    const std::string responses =
        forward_stream(server, "/v1/responses").output;
    const auto response_events = data_events(responses);
    check(response_events.size() == 1, "Responses stream keeps one JSON event");
    check(!response_events.empty() &&
              response_events[0].at("response").value("model", "") == kRequestedModel,
          "Responses stream exposes the requested model identity");
    check(responses.find(kBackendModel) == std::string::npos,
          "Responses stream does not expose the backend GGUF path");

    TestLlamaCppServer multiline_server(port);
    const StreamResult multiline = forward_stream(
        multiline_server, "/v1/chat/completions", "multiline_model");
    const auto multiline_events = data_events(multiline.output);
    check(multiline_events.size() == 1 &&
              multiline_events[0].value("model", "") == kRequestedModel,
          "multi-line SSE data is normalized as one JSON event");
    check(multiline.output.find(kBackendModel) == std::string::npos,
          "multi-line SSE data does not expose the backend GGUF path");
    check(multiline.output.find("id: multiline\n") != std::string::npos &&
              multiline.output.find(": retained-comment\n\n") !=
                  std::string::npos,
          "multi-line normalization preserves non-data fields");

    TestLlamaCppServer bom_server(port);
    const StreamResult bom = forward_stream(
        bom_server, "/v1/chat/completions", "bom_model");
    const auto bom_events = data_events(bom.output);
    check(!bom.retryable_reset && bom_events.size() == 1 &&
              bom_events[0].value("model", "") == kRequestedModel,
          "a stream-leading UTF-8 BOM preserves the public model identity");
    check(bom.output.find(kBackendModel) == std::string::npos,
          "a stream-leading UTF-8 BOM does not expose the backend GGUF path");

    TestLlamaCppServer bom_done_server(port);
    const StreamResult bom_done = forward_stream(
        bom_done_server, "/v1/chat/completions", "bom_done_only");
    check(!bom_done.retryable_reset &&
              count_occurrences(bom_done.output, "data: [DONE]\n\n") == 1,
          "a BOM-prefixed terminal event produces exactly one DONE frame");

    const RawStreamResult raw_bom_done =
        forward_raw_stream(port, "bom_done_only");
    check(!raw_bom_done.interrupted &&
              count_occurrences(
                  raw_bom_done.output, "data: [DONE]\n\n") == 1,
          "a raw BOM-prefixed terminal event produces exactly one DONE frame");

    TestLlamaCppServer double_bom_server(port);
    const StreamResult double_bom = forward_stream(
        double_bom_server, "/v1/chat/completions", "double_bom_model");
    check(double_bom.retryable_reset && double_bom.output.empty(),
          "two fragmented leading UTF-8 BOMs fail before client output");
    check(double_bom.output.find(kBackendModel) == std::string::npos,
          "two leading UTF-8 BOMs cannot expose the backend GGUF path");

    TestLlamaCppServer double_bom_done_server(port);
    const StreamResult double_bom_done = forward_stream(
        double_bom_done_server, "/v1/chat/completions",
        "double_bom_done_only");
    check(double_bom_done.retryable_reset && double_bom_done.output.empty(),
          "two leading UTF-8 BOMs cannot duplicate a transformed DONE frame");

    const RawStreamResult raw_double_bom_done =
        forward_raw_stream(port, "double_bom_done_only");
    check(raw_double_bom_done.interrupted &&
              raw_double_bom_done.output.empty(),
          "two leading UTF-8 BOMs cannot duplicate a raw DONE frame");

    TestLlamaCppServer terminal_cr_server(port);
    const StreamResult terminal_cr = forward_stream(
        terminal_cr_server, "/v1/chat/completions", "terminal_cr");
    const auto terminal_cr_events = data_events(terminal_cr.output);
    check(!terminal_cr.retryable_reset && terminal_cr_events.size() == 1 &&
              terminal_cr_events[0].value("model", "") == kRequestedModel,
          "a terminal bare CR completes a transformed SSE frame at EOF");
    check(terminal_cr.output.find(kBackendModel) == std::string::npos,
          "a terminal bare CR frame does not expose the backend GGUF path");

    TestLlamaCppServer malformed_server(port);
    const StreamResult malformed = forward_stream(
        malformed_server, "/v1/chat/completions", "malformed_model");
    check(malformed.retryable_reset && malformed.output.empty(),
          "a malformed OpenAI SSE object fails before client output");
    check(malformed.output.find(kBackendModel) == std::string::npos,
          "a malformed OpenAI SSE object does not expose the backend GGUF path");

    TestLlamaCppServer scalar_server(port);
    const StreamResult scalar = forward_stream(
        scalar_server, "/v1/chat/completions", "scalar_model");
    check(scalar.retryable_reset && scalar.output.empty(),
          "a non-object OpenAI SSE value fails before client output");
    check(scalar.output.find(kBackendModel) == std::string::npos,
          "a non-object OpenAI SSE value does not expose the backend GGUF path");

    const std::vector<std::string> exact_limit_faults = {
        "frame_at_limit_lf",
        "frame_at_limit_crlf",
    };
    for (const std::string& fault : exact_limit_faults) {
        TestLlamaCppServer exact_limit_server(port);
        const StreamResult transformed_exact_limit = forward_stream(
            exact_limit_server, "/v1/chat/completions", fault);
        check(!transformed_exact_limit.retryable_reset &&
                  transformed_exact_limit.output.find(kRequestedModel) !=
                      std::string::npos &&
                  transformed_exact_limit.output.find(kBackendModel) ==
                      std::string::npos,
              ("transformed stream accepts an exact-limit frame for " +
               fault).c_str());

        const RawStreamResult raw_exact_limit =
            forward_raw_stream(port, fault);
        check(!raw_exact_limit.interrupted,
              ("raw stream accepts an exact-limit frame for " + fault).c_str());
    }

    TestLlamaCppServer oversized_server(port);
    const StreamResult transformed_oversized = forward_stream(
        oversized_server, "/v1/chat/completions", "frame_over_limit");
    check(transformed_oversized.retryable_reset &&
              transformed_oversized.output.empty(),
          "a transformed oversized frame fails before client output");
    check(transformed_oversized.output.find(kBackendModel) ==
              std::string::npos,
          "a transformed oversized frame does not expose the backend path");

    const RawStreamResult raw_oversized =
        forward_raw_stream(port, "frame_over_limit");
    check(raw_oversized.interrupted && raw_oversized.output.empty(),
          "a raw oversized frame fails before client output");

    TestLlamaCppServer oversized_incomplete_server(port);
    const StreamResult transformed_oversized_incomplete = forward_stream(
        oversized_incomplete_server, "/v1/chat/completions",
        "incomplete_frame_over_limit");
    check(transformed_oversized_incomplete.retryable_reset &&
              transformed_oversized_incomplete.output.empty() &&
              transformed_oversized_incomplete.output.find(kBackendModel) ==
                  std::string::npos,
          "an incomplete transformed oversized frame fails closed");

    const RawStreamResult raw_oversized_incomplete =
        forward_raw_stream(port, "incomplete_frame_over_limit");
    check(raw_oversized_incomplete.interrupted &&
              raw_oversized_incomplete.output.empty(),
          "an incomplete raw oversized frame fails before client output");

    for (const std::string& fault : {
             "frame_over_limit_after_frame",
             "incomplete_frame_over_limit_after_frame",
         }) {
        TestLlamaCppServer raw_wrapper_server(port);
        const StreamResult raw_wrapper = forward_wrapped_raw_stream(
            raw_wrapper_server, "/v1/chat/completions", fault);
        int malformed_events = 0;
        const auto events = data_events(raw_wrapper.output, &malformed_events);
        check(!raw_wrapper.retryable_reset && raw_wrapper.done_calls == 1 &&
                  malformed_events == 0 && events.size() == 2 &&
                  events[0].value("model", "") == kRequestedModel &&
                  events[1].contains("error"),
              ("a wrapped raw oversized frame ends with a standalone error for " +
               fault).c_str());
        check(raw_wrapper.output.find(kBackendModel) == std::string::npos,
              ("a wrapped raw oversized frame does not expose buffered data for " +
               fault).c_str());
    }

    TestLlamaCppServer fragmented_server(port);
    const StreamResult transformed_fragmented = forward_stream(
        fragmented_server, "/v1/chat/completions",
        "one_byte_fragmentation");
    check(!transformed_fragmented.retryable_reset &&
              transformed_fragmented.output.find(kRequestedModel) !=
                  std::string::npos &&
              transformed_fragmented.output.find(kBackendModel) ==
                  std::string::npos &&
              count_occurrences(
                  transformed_fragmented.output, "data: [DONE]") == 1,
          "a transformed stream accepts one-byte SSE fragmentation");

    const RawStreamResult raw_fragmented =
        forward_raw_stream(port, "one_byte_fragmentation");
    check(!raw_fragmented.interrupted &&
              count_occurrences(raw_fragmented.output, "data: [DONE]") == 1,
          "a raw stream accepts one-byte SSE fragmentation");

    const RawStreamResult active_incomplete =
        forward_raw_stream(port, "incomplete_progress");
    check(active_incomplete.interrupted && active_incomplete.output.empty(),
          "an active incomplete frame remains retryable before client output");
    check(active_incomplete.backend_progress_calls > 1,
          "raw backend chunks refresh progress before an SSE frame completes");

    const std::vector<std::string> split_done_faults = {
        "split_done_lf",
        "split_done_crlf",
        "split_done_cr",
    };
    for (const std::string& fault : split_done_faults) {
        TestLlamaCppServer transformed_done_server(port);
        const StreamResult transformed_done = forward_stream(
            transformed_done_server, "/v1/chat/completions", fault);
        check(!transformed_done.retryable_reset &&
                  count_occurrences(
                      transformed_done.output, "data: [DONE]") == 1 &&
                  transformed_done.output.find(kRequestedModel) !=
                      std::string::npos &&
                  transformed_done.output.find(kBackendModel) ==
                      std::string::npos,
              ("transformed stream preserves one fragmented DONE for " +
               fault).c_str());

        const RawStreamResult raw_done = forward_raw_stream(port, fault);
        check(!raw_done.interrupted &&
                  count_occurrences(raw_done.output, "data: [DONE]") == 1,
              ("raw stream preserves one fragmented DONE for " + fault).c_str());
    }

    TestLlamaCppServer literal_done_server(port);
    const StreamResult literal_done = forward_stream(
        literal_done_server, "/v1/chat/completions",
        "done_literal_partial");
    int literal_done_malformed_events = 0;
    const auto literal_done_events =
        data_events(literal_done.output, &literal_done_malformed_events);
    check(!literal_done.retryable_reset &&
              literal_done_malformed_events == 0 &&
              literal_done_events.size() == 2 &&
              literal_done_events[0].value("model", "") == kRequestedModel &&
              literal_done_events[1].contains("error"),
          "DONE text inside JSON does not mask a transformed interruption");
    check(literal_done.output.find(kBackendModel) == std::string::npos,
          "a false-DONE interruption does not expose the backend GGUF path");

    const RawStreamResult raw_literal_done =
        forward_raw_stream(port, "done_literal_partial");
    check(raw_literal_done.interrupted,
          "DONE text inside JSON does not mask a raw-stream interruption");

    check(forward_stream_to_disconnected_client(server) == 1,
          "a disconnected client receives no buffered writes during cleanup");

    TestLlamaCppServer retry_server(port);
    const StreamResult interrupted_before_output = forward_stream(
        retry_server, "/v1/chat/completions", "partial_before_output");
    check(interrupted_before_output.retryable_reset,
          "a partial first event preserves the pre-output retry signal");
    check(interrupted_before_output.output.empty(),
          "a partial first event emits no client bytes");
    check(interrupted_before_output.output.find(kBackendModel) ==
              std::string::npos,
          "a partial first event does not expose the backend GGUF path");

    TestLlamaCppServer clean_eof_server(port);
    const StreamResult clean_eof_partial = forward_stream(
        clean_eof_server, "/v1/chat/completions",
        "clean_partial_before_output");
    check(clean_eof_partial.retryable_reset,
          "a clean HTTP EOF with an incomplete event remains retryable");
    check(clean_eof_partial.output.empty() &&
              clean_eof_partial.output.find(kBackendModel) ==
                  std::string::npos,
          "a clean HTTP EOF does not flush an incomplete event");

    StreamResult retried;
    if (interrupted_before_output.retryable_reset) {
        TestLlamaCppServer reloaded_server(port);
        retried = forward_stream(
            reloaded_server, "/v1/chat/completions", "partial_before_output");
    }
    int malformed_retried_events = 0;
    const auto retried_events =
        data_events(retried.output, &malformed_retried_events);
    check(partial_before_output_requests.load() == 2,
          "the caller can replay once after a pre-output interruption");
    check(retried_events.size() == 1 &&
              retried_events[0].value("model", "") == kRequestedModel,
          "the replay produces one normalized event");
    check(malformed_retried_events == 0 &&
              retried.output.find(kBackendModel) == std::string::npos &&
              retried.output.find("\"error\"") == std::string::npos &&
              count_occurrences(retried.output, "data: [DONE]\n\n") == 1,
          "the replay produces one complete error-free stream");

    TestLlamaCppServer no_replay_server(port);
    const StreamResult interrupted_after_output = forward_stream(
        no_replay_server, "/v1/chat/completions", "partial_after_output");
    int malformed_events = 0;
    const auto events_after_output =
        data_events(interrupted_after_output.output, &malformed_events);
    check(!interrupted_after_output.retryable_reset,
          "an interruption after client output is not replayed");
    check(partial_after_output_requests.load() == 1,
          "an interruption after client output makes one backend request");
    check(malformed_events == 0,
          "the post-output interruption emits only valid JSON events");
    check(events_after_output.size() == 2 &&
              events_after_output[0].value("model", "") == kRequestedModel &&
              events_after_output[1].contains("error"),
          "the partial event is replaced by a separate streaming error");
    check(interrupted_after_output.output.find(kBackendModel) ==
              std::string::npos,
          "the post-output interruption does not expose the backend GGUF path");

    backend.stop();
    backend_thread.join();

    return failures == 0 ? 0 : 1;
}
