#!/usr/bin/env python3
"""
Validate a llama.cpp backend release against all "hot" llamacpp models.

Usage:
    python test/validate_llamacpp.py --backend vulkan
    python test/validate_llamacpp.py --backend rocm
    python test/validate_llamacpp.py --backend vulkan --model MODEL_ID

This script expects `lemond` to already be running on the target port.

This script:
1. Queries `/api/v1/models?show_all=true` and selects either the requested
   llama.cpp models or all models with recipe `llamacpp` and label `hot`
2. Installs the requested llamacpp backend via `POST /api/v1/install`
3. For each selected model, loads it with the requested backend, sends a
   `chat/completions` request, optionally runs an API capability profile,
   queries `/api/v1/stats`, and unloads it
4. Outputs a JSON results file for CI consumption
"""

import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote

import requests

from utils.llamacpp_capability_validation import (
    CapabilityValidationError,
    K2_HORIZON_PROFILE,
    MAX_STREAM_EVENTS,
    MAX_STREAM_SSE_BYTES,
    STREAM_WALL_CLOCK_TIMEOUT_SECONDS,
    find_raw_ifm_control_marker,
    reject_duplicate_json_object,
    validate_capabilities,
)
from utils.server_base import _auth_headers
from utils.test_models import PORT, TIMEOUT_DEFAULT
from utils.validation_model_selection import (
    ModelSelectionError,
    add_model_selection_arguments,
    select_llamacpp_models,
)

TIMEOUT_HEALTH = 60
TIMEOUT_INFERENCE = 1800  # 30 minutes; large models may need 60+ GB download
PULL_WALL_CLOCK_TIMEOUT_SECONDS = 3 * 60 * 60
MAX_PULL_RESPONSE_BYTES = 64 * 1024
MAX_PULL_SSE_WIRE_BYTES = 1024 * 1024 * 1024
MAX_PULL_SSE_EVENTS = 4 * 1024 * 1024
MAX_PULL_SSE_LINE_BYTES = 64 * 1024
MAX_PULL_SSE_EVENT_DATA_BYTES = 64 * 1024
MAX_PULL_TERMINAL_SUMMARY_BYTES = 4096
SERVER_CONNECT_ATTEMPT_TIMEOUT_SECONDS = 1
SERVER_CONNECT_RETRY_DELAY_SECONDS = 1
VALIDATION_CTX_SIZE = 8192
VALIDATION_HOST = "127.0.0.1"
INTERNAL_STREAM_WORKER_ARGUMENT = "--internal-stream-worker"
STREAM_WORKER_PROTOCOL_VERSION = 2
STREAM_WORKER_MAGIC = b"LEMONADE_STREAM_WORKER_V2\n"
MAX_STREAM_WORKER_INPUT_BYTES = 1024 * 1024
MAX_STREAM_WORKER_HEADER_BYTES = 64 * 1024
MAX_STREAM_WORKER_ERROR_BYTES = 4096
STREAM_WORKER_CHUNK_BYTES = 64 * 1024
MAX_STREAM_SSE_LINE_BYTES = 64 * 1024
CHAT_PROMPT = [
    {"role": "user", "content": "What is 2+2? Reply in one sentence."},
]


def collect_server_logs(output_dir):
    """Collect Lemonade log files into the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = tempfile.gettempdir()
    patterns = ["lemonade*.log", "lemond*.log", "lemonade-server*.log"]
    copied = []
    for pattern in patterns:
        for log_file in glob.glob(os.path.join(temp_dir, pattern)):
            dest = os.path.join(output_dir, os.path.basename(log_file))
            try:
                shutil.copy2(log_file, dest)
                size = os.path.getsize(dest)
                copied.append(f"{os.path.basename(log_file)} ({size} bytes)")
            except Exception as exc:
                print(f"  Warning: Failed to copy {log_file}: {exc}", flush=True)
    if copied:
        print(f"Collected server logs: {', '.join(copied)}", flush=True)
    else:
        print("No server log files found to collect.", flush=True)
    return copied


def request_json(method, url, timeout, **kwargs):
    """Perform a bounded HTTP request and parse the JSON response when present."""
    deadline = kwargs.pop("wall_clock_deadline", None)
    if deadline is None:
        deadline = time.monotonic() + timeout
    unsupported_options = set(kwargs) - {"headers", "json"}
    if unsupported_options:
        raise CapabilityValidationError("JSON request options are invalid")
    auth_headers = _auth_headers()
    headers = {**(kwargs.get("headers") or {}), **auth_headers}
    response = _execute_http_worker(
        method,
        url,
        timeout,
        headers,
        kwargs.get("json"),
        "json",
        deadline,
        "HTTP response",
    )
    body = {}
    if response.content:
        try:
            body = response.json(object_pairs_hook=reject_duplicate_json_object)
        except ValueError:
            body = {"raw_text": response.text}
    if time.monotonic() >= deadline:
        raise CapabilityValidationError(
            "HTTP response exceeded its wall-clock deadline"
        )
    return response, body


class _BufferedResponse:
    def __init__(self, status_code, headers, encoding, content):
        self.status_code = status_code
        self.headers = headers
        self.encoding = encoding
        self.content = content

    @property
    def text(self):
        encoding = self.encoding or "utf-8"
        try:
            return self.content.decode(encoding, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")

    def json(self, **kwargs):
        return json.loads(self.text, **kwargs)

    def close(self):
        return None


def _encode_stream_worker_output(
    status_code,
    body,
    headers=None,
    encoding=None,
):
    header = json.dumps(
        {
            "version": STREAM_WORKER_PROTOCOL_VERSION,
            "status_code": status_code,
            "body_bytes": len(body),
            "headers": headers or {},
            "encoding": encoding,
        },
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    if len(header) > MAX_STREAM_WORKER_HEADER_BYTES:
        raise CapabilityValidationError("HTTP worker header exceeds bounds")
    return STREAM_WORKER_MAGIC + header + b"\n" + body


def _decode_http_worker_output(output):
    maximum_output_bytes = (
        len(STREAM_WORKER_MAGIC)
        + MAX_STREAM_WORKER_HEADER_BYTES
        + 1
        + MAX_STREAM_SSE_BYTES
    )
    if not isinstance(output, bytes) or len(output) > maximum_output_bytes:
        raise CapabilityValidationError("HTTP worker output exceeds bounds")
    if not output.startswith(STREAM_WORKER_MAGIC):
        raise CapabilityValidationError("HTTP worker output has invalid framing")
    raw_header, separator, body = output[len(STREAM_WORKER_MAGIC) :].partition(b"\n")
    if not separator or len(raw_header) > MAX_STREAM_WORKER_HEADER_BYTES:
        raise CapabilityValidationError("HTTP worker output has invalid framing")
    try:
        header = json.loads(
            raw_header.decode("ascii"),
            object_pairs_hook=reject_duplicate_json_object,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CapabilityValidationError(
            "HTTP worker output has an invalid header"
        ) from exc
    if not isinstance(header, dict) or set(header) != {
        "version",
        "status_code",
        "body_bytes",
        "headers",
        "encoding",
    }:
        raise CapabilityValidationError("HTTP worker output has an invalid header")
    status_code = header["status_code"]
    body_bytes = header["body_bytes"]
    headers = header["headers"]
    encoding = header["encoding"]
    if (
        header["version"] != STREAM_WORKER_PROTOCOL_VERSION
        or not isinstance(status_code, int)
        or isinstance(status_code, bool)
        or not 100 <= status_code <= 599
        or not isinstance(body_bytes, int)
        or isinstance(body_bytes, bool)
        or body_bytes != len(body)
        or body_bytes > MAX_STREAM_SSE_BYTES
        or not isinstance(headers, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in headers.items()
        )
        or not (encoding is None or isinstance(encoding, str))
    ):
        raise CapabilityValidationError("HTTP worker output has an invalid header")
    return _BufferedResponse(status_code, headers, encoding, body)


def _decode_stream_worker_output(output):
    response = _decode_http_worker_output(output)
    return response, _split_stream_lines(response.content)


def _split_stream_lines(body):
    lines = []
    event_count = 0
    line_start = 0
    while line_start < len(body):
        newline = body.find(b"\n", line_start)
        if newline < 0:
            line = body[line_start:]
            line_start = len(body)
        else:
            line = body[line_start:newline]
            line_start = newline + 1
        if len(line) > MAX_STREAM_SSE_LINE_BYTES:
            raise CapabilityValidationError("stream SSE line exceeds bounds")
        stripped = line.strip()
        if not stripped or stripped.startswith(b":"):
            continue
        if not stripped.startswith(b"data:"):
            raise CapabilityValidationError("stream line is not an SSE data frame")
        event_count += 1
        if event_count > MAX_STREAM_EVENTS:
            raise CapabilityValidationError("stream event count exceeds bounds")
        lines.append(line)
    return tuple(lines)


def _iter_bounded_response_lines(response, max_response_bytes):
    total_bytes = 0
    pending = bytearray()
    for chunk in response.iter_content(
        chunk_size=STREAM_WORKER_CHUNK_BYTES,
        decode_unicode=False,
    ):
        if not isinstance(chunk, bytes):
            raise CapabilityValidationError("HTTP response chunk must be bytes")
        total_bytes += len(chunk)
        if total_bytes > max_response_bytes:
            raise CapabilityValidationError(
                "model pull SSE wire byte count exceeds bounds"
            )
        start = 0
        while start < len(chunk):
            newline = chunk.find(b"\n", start)
            end = len(chunk) if newline < 0 else newline
            segment = chunk[start:end]
            if len(pending) + len(segment) > MAX_PULL_SSE_LINE_BYTES:
                raise CapabilityValidationError("model pull SSE line exceeds bounds")
            pending.extend(segment)
            if newline < 0:
                break
            line = bytes(pending)
            pending.clear()
            if line.endswith(b"\r"):
                line = line[:-1]
            yield line
            start = newline + 1
    if pending:
        raise CapabilityValidationError("model pull SSE ended with an incomplete line")


def _reject_non_finite_json(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _parse_pull_event_data(data):
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicate_json_object,
            parse_constant=_reject_non_finite_json,
        )
    except (UnicodeError, ValueError) as exc:
        raise CapabilityValidationError("model pull SSE event data is invalid") from exc
    if not isinstance(payload, dict):
        raise CapabilityValidationError("model pull SSE event data is invalid")
    return payload


def _pull_content_type(response):
    for key, value in response.headers.items():
        if str(key).lower() == "content-type":
            return str(value).split(";", maxsplit=1)[0].strip().lower()
    return ""


def _bounded_pull_error(error):
    encoded = str(error).encode("utf-8", errors="replace")[:1000]
    return encoded.decode("utf-8", errors="ignore")


def _reduce_pull_sse_response(response, max_response_bytes):
    if _pull_content_type(response) != "text/event-stream":
        raise CapabilityValidationError(
            "model pull returned an invalid SSE content type"
        )

    event_name = None
    event_data = bytearray()
    data_seen = False
    event_count = 0
    terminal = None

    for line in _iter_bounded_response_lines(response, max_response_bytes):
        if not line:
            if event_name is None and not data_seen:
                continue
            event_count += 1
            if event_count > MAX_PULL_SSE_EVENTS:
                raise CapabilityValidationError(
                    "model pull SSE event count exceeds bounds"
                )
            if terminal is not None:
                raise CapabilityValidationError(
                    "model pull SSE contained an event after its terminal event"
                )
            if event_name not in {"progress", "complete", "error"} or not data_seen:
                raise CapabilityValidationError("model pull SSE event is invalid")
            payload = _parse_pull_event_data(bytes(event_data))
            if event_name == "complete":
                terminal = {"terminal": "complete"}
            elif event_name == "error":
                error = payload.get("error")
                code = payload.get("code")
                if (
                    not isinstance(error, str)
                    or not error
                    or not (code is None or isinstance(code, str))
                ):
                    raise CapabilityValidationError(
                        "model pull SSE error event is invalid"
                    )
                message = _bounded_pull_error(error)
                terminal = {
                    "terminal": "error",
                    "status_code": (
                        400
                        if code == "unknown_model" or "unknown_model" in message
                        else 500
                    ),
                    "error": message,
                }
            event_name = None
            event_data.clear()
            data_seen = False
            continue

        if terminal is not None:
            raise CapabilityValidationError(
                "model pull SSE contained data after its terminal event"
            )
        if line.startswith(b"event:"):
            if event_name is not None:
                raise CapabilityValidationError("model pull SSE event is invalid")
            raw_name = line[len(b"event:") :]
            if raw_name.startswith(b" "):
                raw_name = raw_name[1:]
            try:
                event_name = raw_name.decode("ascii")
            except UnicodeError as exc:
                raise CapabilityValidationError(
                    "model pull SSE event is invalid"
                ) from exc
        elif line.startswith(b"data:"):
            raw_data = line[len(b"data:") :]
            if raw_data.startswith(b" "):
                raw_data = raw_data[1:]
            additional_bytes = len(raw_data) + (1 if data_seen else 0)
            if len(event_data) + additional_bytes > MAX_PULL_SSE_EVENT_DATA_BYTES:
                raise CapabilityValidationError(
                    "model pull SSE event data exceeds bounds"
                )
            if data_seen:
                event_data.extend(b"\n")
            event_data.extend(raw_data)
            data_seen = True
        else:
            raise CapabilityValidationError("model pull SSE line is invalid")

    if event_name is not None or data_seen:
        raise CapabilityValidationError("model pull SSE ended with an incomplete event")
    if terminal is None:
        raise CapabilityValidationError("model pull SSE ended without a terminal event")
    summary = json.dumps(
        terminal,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(summary) > MAX_PULL_TERMINAL_SUMMARY_BYTES:
        raise CapabilityValidationError("model pull terminal summary exceeds bounds")
    return summary


def _buffer_response_body(response, max_response_bytes, failure_message):
    buffered_body = bytearray()
    for chunk in response.iter_content(
        chunk_size=STREAM_WORKER_CHUNK_BYTES,
        decode_unicode=False,
    ):
        if not isinstance(chunk, bytes):
            raise CapabilityValidationError("HTTP response chunk must be bytes")
        if len(buffered_body) + len(chunk) > max_response_bytes:
            raise CapabilityValidationError(failure_message)
        buffered_body.extend(chunk)
    return bytes(buffered_body)


def _load_stream_worker_request():
    raw_request = sys.stdin.buffer.read(MAX_STREAM_WORKER_INPUT_BYTES + 1)
    if len(raw_request) > MAX_STREAM_WORKER_INPUT_BYTES:
        raise CapabilityValidationError("HTTP worker input exceeds bounds")
    try:
        request = json.loads(
            raw_request.decode("utf-8"),
            object_pairs_hook=reject_duplicate_json_object,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CapabilityValidationError("HTTP worker input is invalid") from exc
    if not isinstance(request, dict) or set(request) != {
        "version",
        "response_kind",
        "max_response_bytes",
        "method",
        "url",
        "timeout",
        "headers",
        "json",
    }:
        raise CapabilityValidationError("HTTP worker input is invalid")
    response_kind = request["response_kind"]
    max_response_bytes = request["max_response_bytes"]
    method = request["method"]
    url = request["url"]
    timeout = request["timeout"]
    headers = request["headers"]
    payload = request["json"]
    if (
        request["version"] != STREAM_WORKER_PROTOCOL_VERSION
        or response_kind not in ("json", "stream", "pull_sse")
        or not isinstance(max_response_bytes, int)
        or isinstance(max_response_bytes, bool)
        or not 1
        <= max_response_bytes
        <= (
            MAX_PULL_SSE_WIRE_BYTES
            if response_kind == "pull_sse"
            else MAX_STREAM_SSE_BYTES
        )
        or not isinstance(method, str)
        or not method
        or not isinstance(url, str)
        or not url
        or not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or timeout <= 0
        or not isinstance(headers, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in headers.items()
        )
        or not (payload is None or isinstance(payload, dict))
    ):
        raise CapabilityValidationError("HTTP worker input is invalid")
    if response_kind in {"stream", "pull_sse"} and not isinstance(payload, dict):
        raise CapabilityValidationError("HTTP worker input is invalid")
    return (
        response_kind,
        method,
        url,
        timeout,
        headers,
        payload,
        max_response_bytes,
    )


def _run_stream_worker():
    response_kind, method, url, timeout, headers, payload, max_response_bytes = (
        _load_stream_worker_request()
    )
    request_options = {
        "timeout": timeout,
        "headers": headers,
        "stream": True,
    }
    if payload is not None:
        request_options["json"] = payload
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.request(method, url, **request_options)
        try:
            if response_kind == "pull_sse" and response.status_code == 200:
                body = _reduce_pull_sse_response(response, max_response_bytes)
            else:
                response_bound = (
                    min(max_response_bytes, MAX_PULL_RESPONSE_BYTES)
                    if response_kind == "pull_sse"
                    else max_response_bytes
                )
                failure_message = (
                    "stream SSE byte count exceeds bounds"
                    if response_kind == "stream"
                    else "HTTP response byte count exceeds bounds"
                )
                body = _buffer_response_body(
                    response,
                    response_bound,
                    failure_message,
                )
        finally:
            response.close()
        output = _encode_stream_worker_output(
            response.status_code,
            body,
            dict(response.headers),
            response.encoding,
        )
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
    finally:
        session.close()


def _loopback_pull_url(host, port):
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise CapabilityValidationError("model pull host must be loopback")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise CapabilityValidationError("model pull port is invalid")
    authority = f"[{host}]" if host == "::1" else host
    return f"http://{authority}:{port}/api/v1/pull"


def _stream_worker_main():
    try:
        _run_stream_worker()
    except BaseException as exc:  # pylint: disable=broad-exception-caught
        error = f"{exc.__class__.__name__}: {exc}".encode("utf-8", errors="replace")
        sys.stderr.buffer.write(error[:MAX_STREAM_WORKER_ERROR_BYTES])
        sys.stderr.buffer.flush()
        return 1
    return 0


def _kill_and_reap_stream_worker(process):
    try:
        if process.poll() is None:
            process.kill()
    finally:
        process.communicate()


def _encode_stream_worker_request(
    method,
    url,
    timeout,
    headers,
    payload,
    response_kind="stream",
    max_response_bytes=MAX_STREAM_SSE_BYTES,
):
    try:
        encoded = json.dumps(
            {
                "version": STREAM_WORKER_PROTOCOL_VERSION,
                "response_kind": response_kind,
                "max_response_bytes": max_response_bytes,
                "method": method,
                "url": url,
                "timeout": timeout,
                "headers": headers,
                "json": payload,
            },
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CapabilityValidationError("HTTP request cannot be encoded") from exc
    if len(encoded) > MAX_STREAM_WORKER_INPUT_BYTES:
        raise CapabilityValidationError("HTTP worker input exceeds bounds")
    return encoded


def _execute_http_worker(
    method,
    url,
    timeout,
    headers,
    payload,
    response_kind,
    deadline,
    deadline_subject,
    max_response_bytes=MAX_STREAM_SSE_BYTES,
):
    worker_input = _encode_stream_worker_request(
        method,
        url,
        timeout,
        headers,
        payload,
        response_kind,
        max_response_bytes,
    )
    deadline_message = f"{deadline_subject} exceeded its wall-clock deadline"
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CapabilityValidationError(deadline_message)
    process = None
    worker_reaped = False
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                os.path.realpath(__file__),
                INTERNAL_STREAM_WORKER_ARGUMENT,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CapabilityValidationError(deadline_message)
        try:
            output, error = process.communicate(input=worker_input, timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise CapabilityValidationError(deadline_message) from exc
        worker_reaped = True
        if time.monotonic() >= deadline:
            raise CapabilityValidationError(deadline_message)
        if process.returncode != 0:
            detail = error[:MAX_STREAM_WORKER_ERROR_BYTES].decode(
                "utf-8", errors="replace"
            )
            raise CapabilityValidationError(
                f"{deadline_subject} transport worker failed: "
                f"{detail or 'unknown error'}"
            )
        response = _decode_http_worker_output(output)
        if time.monotonic() >= deadline:
            raise CapabilityValidationError(deadline_message)
        return response
    except BaseException:  # pylint: disable=broad-exception-caught
        if process is not None and not worker_reaped:
            _kill_and_reap_stream_worker(process)
        raise


def request_stream(method, url, timeout, **kwargs):
    """Perform a bounded streaming HTTP request and return raw response lines."""
    deadline = kwargs.pop("wall_clock_deadline", None)
    if deadline is None:
        deadline = time.monotonic() + min(
            timeout,
            STREAM_WALL_CLOCK_TIMEOUT_SECONDS,
        )
    unsupported_options = set(kwargs) - {"headers", "json"}
    if unsupported_options or not isinstance(kwargs.get("json"), dict):
        raise CapabilityValidationError("stream request options are invalid")
    auth_headers = _auth_headers()
    headers = {**(kwargs.get("headers") or {}), **auth_headers}
    response = _execute_http_worker(
        method,
        url,
        timeout,
        headers,
        kwargs["json"],
        "stream",
        deadline,
        "stream",
    )
    response_and_lines = response, _split_stream_lines(response.content)
    if time.monotonic() >= deadline:
        raise CapabilityValidationError("stream exceeded its wall-clock deadline")
    return response_and_lines


def wait_for_server(port=PORT, timeout=TIMEOUT_HEALTH, host="localhost"):
    """Wait for a server while bounding every loopback connection attempt."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Server failed to start within {timeout} seconds")
        try:
            connection = socket.create_connection(
                (host, port),
                timeout=min(SERVER_CONNECT_ATTEMPT_TIMEOUT_SECONDS, remaining),
            )
            connection.close()
            return True
        except OSError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Server failed to start within {timeout} seconds"
                ) from None
            time.sleep(min(SERVER_CONNECT_RETRY_DELAY_SECONDS, remaining))


def unload_all_models(port=PORT, attempts=3, host="localhost"):
    """Unload all models through the bounded HTTP worker."""
    response = None
    for attempt in range(1, attempts + 1):
        try:
            response, _body = request_json(
                "POST",
                f"http://{host}:{port}/api/v1/unload",
                timeout=TIMEOUT_DEFAULT,
                json={},
            )
            if response.status_code in (200, 404):
                return response
        except (requests.RequestException, CapabilityValidationError):
            if attempt == attempts:
                raise
        if attempt < attempts:
            time.sleep(1)
    return response


def _is_transient_pull_status(status_code):
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def _pull_model_streaming(
    model_name,
    port,
    host=VALIDATION_HOST,
    *,
    wall_clock_deadline=None,
):
    if wall_clock_deadline is None:
        wall_clock_deadline = time.monotonic() + PULL_WALL_CLOCK_TIMEOUT_SECONDS
    pull_url = _loopback_pull_url(host, port)
    response = _execute_http_worker(
        "POST",
        pull_url,
        TIMEOUT_INFERENCE,
        _auth_headers(),
        {"model_name": model_name, "stream": True, "subscribe": True},
        "pull_sse",
        wall_clock_deadline,
        "model pull",
        MAX_PULL_SSE_WIRE_BYTES,
    )
    if response.status_code != 200:
        return response.status_code, response.text[:1000]
    try:
        summary = response.json(
            object_pairs_hook=reject_duplicate_json_object,
            parse_constant=_reject_non_finite_json,
        )
    except (UnicodeError, ValueError) as exc:
        raise CapabilityValidationError(
            "model pull worker returned an invalid terminal summary"
        ) from exc
    if summary == {"terminal": "complete"}:
        return 200, ""
    if (
        isinstance(summary, dict)
        and set(summary) == {"terminal", "status_code", "error"}
        and summary["terminal"] == "error"
        and summary["status_code"] in {400, 500}
        and not isinstance(summary["status_code"], bool)
        and isinstance(summary["error"], str)
        and summary["error"]
    ):
        return summary["status_code"], summary["error"]
    raise CapabilityValidationError(
        "model pull worker returned an invalid terminal summary"
    )


def pull_model_with_retry(model_name, attempts=3, port=PORT, host=VALIDATION_HOST):
    """Pull a model with bounded retry for transient setup failures."""
    last_status = None
    last_body = ""
    wall_clock_deadline = time.monotonic() + PULL_WALL_CLOCK_TIMEOUT_SECONDS

    for attempt in range(1, attempts + 1):
        if attempt > 1:
            remaining = wall_clock_deadline - time.monotonic()
            if remaining <= 0:
                raise CapabilityValidationError(
                    "model pull exceeded its wall-clock deadline"
                )
            time.sleep(min(30, 2 ** (attempt - 1), remaining))

        status, body = _pull_model_streaming(
            model_name,
            port,
            host=host,
            wall_clock_deadline=wall_clock_deadline,
        )

        if status == 200:
            return

        last_status = status
        last_body = body
        if _is_transient_pull_status(status) and attempt < attempts:
            print(
                f"Transient /pull setup failure for {model_name}: "
                f"status={status}, attempt={attempt}/{attempts}. Retrying..."
            )
            continue
        break

    raise AssertionError(
        f"Expected 200 from /api/v1/pull for {model_name} after "
        f"{attempts} attempt(s), got {last_status}. Body: {last_body}"
    )


def require_running_server(base_url, port):
    """Wait for a running server and confirm the health endpoint responds."""
    wait_for_server(port=port, timeout=TIMEOUT_HEALTH, host=VALIDATION_HOST)
    response, body = request_json(
        "GET",
        f"{base_url}/health",
        timeout=TIMEOUT_DEFAULT,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Server health check failed: HTTP {response.status_code} - {body}"
        )
    print(
        f"Server is reachable on port {port} (status={body.get('status', 'unknown')})",
        flush=True,
    )


def get_model_catalog(base_url):
    """Return all model catalog entries from the API."""
    response, body = request_json(
        "GET",
        f"{base_url}/models?show_all=true",
        timeout=TIMEOUT_DEFAULT,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to query model catalog: HTTP {response.status_code} - {body}"
        )

    return body.get("data", [])


def get_builtin_model(base_url, canonical_model_id):
    """Resolve one canonical built-in model from the API."""
    encoded_model_id = quote(canonical_model_id, safe="")
    response, body = request_json(
        "GET",
        f"{base_url}/models/{encoded_model_id}",
        timeout=TIMEOUT_DEFAULT,
    )
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to query built-in model '{canonical_model_id}': "
            f"HTTP {response.status_code} - {body}"
        )
    if not isinstance(body, dict):
        raise RuntimeError(
            f"Built-in model '{canonical_model_id}' returned an invalid response"
        )
    return body


def set_rocm_channel(base_url, channel):
    """Configure the ROCm channel in lemond via the internal config API."""
    print(f"Setting rocm_channel={channel} via /internal/set", flush=True)
    response, body = request_json(
        "POST",
        f"{base_url.rsplit('/api/v1', 1)[0]}/internal/set",
        timeout=TIMEOUT_DEFAULT,
        json={"rocm_channel": channel},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to set rocm_channel: HTTP {response.status_code} - {body}"
        )
    print(f"Params response: {body}", flush=True)


def install_backend(base_url, backend):
    """Install or update the requested llama.cpp backend through the API."""
    print(f"Installing llamacpp backend via /install: {backend}", flush=True)
    response, body = request_json(
        "POST",
        f"{base_url}/install",
        timeout=TIMEOUT_INFERENCE,
        json={"recipe": "llamacpp", "backend": backend, "stream": False},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Backend install failed: HTTP {response.status_code} - {body}"
        )
    print(f"Install response: {body}", flush=True)


def unload_model(base_url, model_name=None):
    """Unload a specific model or all models."""
    payload = {}
    if model_name:
        payload["model_name"] = model_name
    response, body = request_json(
        "POST",
        f"{base_url}/unload",
        timeout=TIMEOUT_DEFAULT,
        json=payload,
    )
    if response.status_code not in (200, 404):
        raise RuntimeError(f"Model unload failed: HTTP {response.status_code} - {body}")


def require_all_models_unloaded(port, phase):
    """Unload every model and reject an unsuccessful cleanup response."""
    response = unload_all_models(port=port, host=VALIDATION_HOST)
    if response is None or response.status_code not in (200, 404):
        status = "no response" if response is None else f"HTTP {response.status_code}"
        raise RuntimeError(f"{phase} model unload failed: {status}")


def canonical_model_id(model_name):
    """Return the canonical built-in identity used at API boundaries."""
    return model_name if model_name.startswith("builtin.") else f"builtin.{model_name}"


def test_model(
    base_url,
    model_name,
    backend,
    max_tokens=50,
    *,
    capability_profile="",
    capability_evidence=None,
):
    """Send a chat/completions request and return (success, response_text, stats)."""
    print(f"  Loading model: {model_name} (backend={backend})", flush=True)
    try:
        load_resp, load_body = request_json(
            "POST",
            f"{base_url}/load",
            timeout=TIMEOUT_INFERENCE,
            json={
                "model_name": model_name,
                "llamacpp_backend": backend,
                "ctx_size": VALIDATION_CTX_SIZE,
            },
        )
        if load_resp.status_code != 200:
            return (
                False,
                f"Load failed: HTTP {load_resp.status_code} - {load_body}",
                {},
            )

        health_resp, health_body = request_json(
            "GET",
            f"{base_url}/health",
            timeout=TIMEOUT_DEFAULT,
        )
        if health_resp.status_code != 200:
            return (
                False,
                f"Loaded-model health inventory returned HTTP "
                f"{health_resp.status_code}: {health_body}",
                {},
            )
        if not isinstance(health_body, dict):
            return False, "Loaded-model health response is not an object", {}
        if "all_models_loaded" not in health_body:
            return (
                False,
                "Loaded-model health response is missing all_models_loaded",
                {},
            )
        loaded_models = health_body["all_models_loaded"]
        if not isinstance(loaded_models, list) or not all(
            isinstance(model, dict) and isinstance(model.get("model_name"), str)
            for model in loaded_models
        ):
            return (
                False,
                "Loaded-model health response has an invalid model inventory",
                {},
            )

        requested_model_id = canonical_model_id(model_name)
        requested_models = [
            model
            for model in loaded_models
            if canonical_model_id(model["model_name"]) == requested_model_id
        ]
        if not requested_models:
            return (
                False,
                f"Model '{model_name}' is absent from the loaded inventory",
                {},
            )
        if len(loaded_models) != 1:
            inventory = ", ".join(model["model_name"] for model in loaded_models)
            return (
                False,
                f"Loaded inventory must contain exactly the requested model "
                f"'{model_name}', found: {inventory or '<empty>'}",
                {},
            )
        loaded_model = requested_models[0]

        if "recipe_options" not in loaded_model:
            return False, f"Loaded model '{model_name}' is missing recipe_options", {}
        recipe_options = loaded_model["recipe_options"]
        if not isinstance(recipe_options, dict):
            return False, f"Loaded model '{model_name}' has invalid recipe_options", {}
        actual_backend = recipe_options.get("llamacpp_backend")
        if not isinstance(actual_backend, str) or not actual_backend:
            return False, f"Loaded model '{model_name}' is missing llamacpp_backend", {}
        if actual_backend != backend:
            return (
                False,
                f"Model loaded with backend '{actual_backend}' instead of '{backend}'",
                {},
            )

        print("  Sending chat/completions request...", flush=True)
        chat_resp, chat_body = request_json(
            "POST",
            f"{base_url}/chat/completions",
            timeout=TIMEOUT_INFERENCE,
            json={
                "model": model_name,
                "messages": CHAT_PROMPT,
                "max_completion_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        if chat_resp.status_code != 200:
            return False, f"HTTP {chat_resp.status_code}: {chat_body}", {}

        response_model = chat_body.get("model")
        if not isinstance(response_model, str) or (
            canonical_model_id(response_model) != requested_model_id
        ):
            return (
                False,
                f"Chat response model '{response_model}' does not match "
                f"requested model '{model_name}'",
                {},
            )

        message = chat_body["choices"][0]["message"]
        marker = find_raw_ifm_control_marker(message, "message")
        if marker:
            marker_text, marker_path = marker
            return (
                False,
                f"Raw IFM control marker {marker_text!r} found in "
                f"response field {marker_path}",
                {},
            )

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            return False, "Plain chat response has no visible final content", {}

        if capability_profile:
            if not isinstance(capability_evidence, dict):
                return False, "Capability evidence output is required", {}
            matrix, capabilities_passed, capability_summary = validate_capabilities(
                capability_profile,
                base_url=base_url,
                model=model_name,
                request_json=request_json,
                request_stream=request_stream,
                timeout=TIMEOUT_INFERENCE,
            )
            capability_evidence.clear()
            capability_evidence.update(matrix)
            if not capabilities_passed:
                return False, f"Capability validation failed: {capability_summary}", {}
            print(f"  Capabilities: {capability_summary}", flush=True)

        stats = {}
        stats_resp, stats_body = request_json(
            "GET",
            f"{base_url}/stats",
            timeout=TIMEOUT_DEFAULT,
        )
        if stats_resp.status_code == 200:
            stats = stats_body
            print(f"  Stats: {json.dumps(stats)}", flush=True)
        else:
            print(
                f"  Warning: stats returned HTTP {stats_resp.status_code}: {stats_body}",
                flush=True,
            )

        return True, content, stats
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return False, f"Bad response format: {exc}", {}
    except requests.RequestException as exc:
        return False, f"Request failed: {exc}", {}
    finally:
        unload_model(base_url, model_name)


def validate_model_lifecycle(
    base_url,
    model_name,
    backend,
    reload_after_unload=False,
    *,
    capability_profile="",
    capability_evidence=None,
):
    """Validate one model and optionally validate a second load after unload."""
    capability_args = {}
    if capability_profile:
        capability_args = {
            "capability_profile": capability_profile,
            "capability_evidence": capability_evidence,
        }
    result = test_model(base_url, model_name, backend, **capability_args)
    if not reload_after_unload or not result[0]:
        return result

    try:
        health_response, health_body = request_json(
            "GET",
            f"{base_url}/health",
            timeout=TIMEOUT_DEFAULT,
        )
    except requests.RequestException as exc:
        return False, f"Could not verify unload before reload: {exc}", result[2]

    if health_response.status_code != 200:
        return (
            False,
            f"Could not verify unload before reload: HTTP "
            f"{health_response.status_code} - {health_body}",
            result[2],
        )

    if not isinstance(health_body, dict):
        return False, "Could not verify unload: invalid health response", result[2]
    loaded_models = health_body.get("all_models_loaded")
    if not isinstance(loaded_models, list) or not all(
        isinstance(model, dict) and isinstance(model.get("model_name"), str)
        for model in loaded_models
    ):
        return False, "Could not verify unload: invalid model inventory", result[2]

    if loaded_models:
        remaining_models = ", ".join(model["model_name"] for model in loaded_models)
        return (
            False,
            f"Models are still loaded after unloading '{model_name}': "
            f"{remaining_models}",
            result[2],
        )

    print("  Reloading model after unload...", flush=True)
    return test_model(base_url, model_name, backend)


def write_results_file(output_path, results):
    with open(output_path, "x", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(
        description="Validate a llama.cpp backend against selected Lemonade models"
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=[
            "vulkan",
            "rocm",
            "cuda",
            "cpu",
            "metal",
            "system",
        ],
        help="Backend to test (vulkan, rocm, cuda, cpu, metal, system)",
    )
    parser.add_argument(
        "--channel",
        default=None,
        choices=["stable", "nightly"],
        help="Channel for backends that support multiple releases (e.g. rocm)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"Server port (default: {PORT})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results file",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip backend installation step",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Install the backend and pull selected models without inference",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Directory to collect server log files into (for CI artifact upload)",
    )
    parser.add_argument(
        "--capability-profile",
        choices=[K2_HORIZON_PROFILE],
        default="",
        help="Optional API capability contract to validate",
    )
    parser.add_argument(
        "--capability-model",
        action="append",
        default=[],
        help="Selected validation model that must run the capability profile",
    )
    add_model_selection_arguments(parser)
    args = parser.parse_args()

    has_capability_profile = bool(args.capability_profile)
    has_capability_models = bool(args.capability_model)
    if has_capability_profile != has_capability_models:
        parser.error(
            "--capability-profile and --capability-model must be supplied together"
        )

    canonical_capability_models = [
        model_id.removeprefix("builtin.") for model_id in args.capability_model
    ]
    if len(set(canonical_capability_models)) != len(canonical_capability_models):
        parser.error("--capability-model values must be unique")

    # Label used for output filenames and artifact names
    label = f"{args.backend}-{args.channel}" if args.channel else args.backend

    base_url = f"http://{VALIDATION_HOST}:{args.port}/api/v1"
    output_path = args.output or f"llamacpp_validation_{label}.json"

    require_running_server(base_url, args.port)

    catalog = get_model_catalog(base_url) if not args.model else []
    try:
        selected_models = select_llamacpp_models(
            catalog,
            requested_model_ids=args.model,
            lite=args.lite,
            builtin_model_resolver=lambda model_id: get_builtin_model(
                base_url, model_id
            ),
        )
    except ModelSelectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    selected_model_ids = {
        canonical_model_id(model.get("load_id", model["id"])).removeprefix("builtin.")
        for model in selected_models
    }
    missing_capability_models = sorted(
        set(canonical_capability_models) - selected_model_ids
    )
    if missing_capability_models:
        print(
            "ERROR: Capability models are absent from the selected validation set: "
            + ", ".join(missing_capability_models),
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    capability_model_ids = set(canonical_capability_models)

    if args.channel:
        set_rocm_channel(base_url, args.channel)

    print("Unloading all models for clean state...", flush=True)
    require_all_models_unloaded(args.port, "Pre-validation clean state")

    selection_mode = "explicitly selected" if args.model else "hot"
    print(
        f"Found {len(selected_models)} {selection_mode} llamacpp models:",
        flush=True,
    )
    for model in selected_models:
        print(f"  - {model['id']} ({model.get('size', '?')} GB)", flush=True)

    if args.lite:
        selected = selected_models[0]
        print(
            f"Lite mode: testing only smallest model: "
            f"{selected['id']} ({selected.get('size', '?')} GB)",
            flush=True,
        )

    if not args.skip_install:
        install_backend(base_url, args.backend)

    if args.prepare_only:
        try:
            for model in selected_models:
                model_name = model.get("load_id", model["id"])
                print(f"Pulling model: {model_name}", flush=True)
                pull_model_with_retry(
                    model_name,
                    port=args.port,
                    host=VALIDATION_HOST,
                )
        finally:
            if args.logs_dir:
                collect_server_logs(args.logs_dir)
        return

    results = []
    all_passed = True
    try:
        for model in selected_models:
            model_name = model["id"]
            load_id = model.get("load_id", model_name)
            print(f"\nTesting: {model_name}", flush=True)
            runs_capability_profile = (
                canonical_model_id(load_id).removeprefix("builtin.")
                in capability_model_ids
            )
            capability_evidence = {} if runs_capability_profile else None
            success, response_text, stats = validate_model_lifecycle(
                base_url,
                load_id,
                args.backend,
                reload_after_unload=bool(args.model),
                capability_profile=(
                    args.capability_profile if runs_capability_profile else ""
                ),
                capability_evidence=capability_evidence,
            )
            result = {
                "model": model_name,
                "pass": success,
                "response": response_text,
                "input_tokens": stats.get("input_tokens", "N/A"),
                "output_tokens": stats.get("output_tokens", "N/A"),
                "time_to_first_token": stats.get("time_to_first_token", "N/A"),
                "tokens_per_second": stats.get("tokens_per_second", "N/A"),
            }
            if runs_capability_profile:
                result["capability_matrix"] = capability_evidence
            results.append(result)
            status = "PASS" if success else "FAIL"
            print(f"  Result: {status}", flush=True)
            if not success:
                all_passed = False
                print(f"  Error: {response_text}", flush=True)
    finally:
        try:
            require_all_models_unloaded(args.port, "Final cleanup")
        finally:
            if args.logs_dir:
                collect_server_logs(args.logs_dir)

    write_results_file(output_path, results)
    print(f"\nResults written to {output_path}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    passed = sum(1 for result in results if result["pass"])
    print(f"Results: {passed}/{len(results)} models passed", flush=True)
    for result in results:
        status = "PASS" if result["pass"] else "FAIL"
        print(f"  [{status}] {result['model']}", flush=True)
    print(f"{'=' * 60}", flush=True)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1:] == [INTERNAL_STREAM_WORKER_ARGUMENT]:
        sys.exit(_stream_worker_main())
    main()
