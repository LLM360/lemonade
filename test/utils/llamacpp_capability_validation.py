"""Fail-closed API capability validation for llama.cpp models."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from typing import Any

K2_HORIZON_PROFILE = "k2-horizon-v1"
K2_HORIZON_CASE_IDS = (
    "openai_plain_off_nonstream",
    "openai_plain_off_stream",
    "openai_reasoning_high",
    "openai_reasoning_high_stream",
    "openai_reasoning_medium",
    "openai_reasoning_low",
    "openai_tool_json",
    "openai_tool_xml",
    "openai_tool_xml_typed",
    "openai_tool_default_xml",
    "openai_tool_stream_xml",
    "openai_tool_result_followup",
    "ollama_thinking",
    "anthropic_translation",
)
CASE_CONTRACTS = {
    "openai_plain_off_nonstream": ("openai", False, "stop", True, False, 0, None),
    "openai_plain_off_stream": ("openai", True, "stop", True, False, 0, None),
    "openai_reasoning_high": ("openai", False, "stop", True, True, 0, None),
    "openai_reasoning_high_stream": (
        "openai",
        True,
        "stop",
        True,
        True,
        0,
        None,
    ),
    "openai_reasoning_medium": ("openai", False, "stop", True, True, 0, None),
    "openai_reasoning_low": ("openai", False, "stop", True, False, 0, None),
    "openai_tool_json": ("openai", False, "tool_calls", False, False, 1, None),
    "openai_tool_xml": ("openai", False, "tool_calls", False, False, 1, None),
    "openai_tool_xml_typed": (
        "openai",
        False,
        "tool_calls",
        False,
        False,
        1,
        None,
    ),
    "openai_tool_default_xml": (
        "openai",
        False,
        "tool_calls",
        False,
        False,
        1,
        "openai_tool_xml",
    ),
    "openai_tool_stream_xml": (
        "openai",
        True,
        "tool_calls",
        False,
        False,
        1,
        "openai_tool_xml",
    ),
    "openai_tool_result_followup": (
        "openai",
        False,
        "stop",
        True,
        False,
        0,
        None,
    ),
    "ollama_thinking": ("ollama", False, "stop", True, True, 0, None),
    "anthropic_translation": (
        "anthropic",
        False,
        "end_turn",
        True,
        False,
        0,
        None,
    ),
}
TOOL_NAME = "lookup_weather"
EXPECTED_TOOL_ARGUMENTS = {
    "city": "Paris",
    "unit": "celsius",
    "days": 2,
    "include_hourly": False,
}
IFM_CONTROL_PREFIXES = ("<ifm|", "</ifm|", "<|ifm|")
MAX_STREAM_EVENTS = 4096
MAX_STREAM_SSE_BYTES = 8 * 1024 * 1024
MAX_STREAM_TEXT_BYTES = 4 * 1024 * 1024
MAX_STREAM_TOOL_ARGUMENT_BYTES = 1024 * 1024
STREAM_WALL_CLOCK_TIMEOUT_SECONDS = 30 * 60
CASE_FIELDS = {
    "content_chars",
    "error",
    "finish_reason",
    "id",
    "parity_with",
    "pass",
    "protocol",
    "raw_ifm_marker",
    "reasoning_chars",
    "status_code",
    "stream",
    "terminal_frame",
    "tool_arguments",
    "tool_call_count",
    "tool_name",
}
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Look up a forecast.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
                "days": {"type": "integer"},
                "include_hourly": {"type": "boolean"},
            },
            "required": ["city", "unit", "days", "include_hourly"],
            "additionalProperties": False,
        },
    },
}
ARITHMETIC_PROMPT = "What is 17 plus 25? End the final answer with 42."
TOOL_PROMPT = (
    "Call lookup_weather exactly once with city Paris, unit celsius, days 2, "
    "and include_hourly false. Do not answer in text."
)


class CapabilityValidationError(ValueError):
    """Raised when a capability response or evidence record is invalid."""


def _marker_children(candidate: object):
    if isinstance(candidate, dict):
        for key, nested_value in candidate.items():
            yield f".{key}", nested_value
    elif isinstance(candidate, (list, tuple)):
        for index, nested_value in enumerate(candidate):
            yield f"[{index}]", nested_value


def find_raw_ifm_control_marker(value: object, field: str = "response"):
    """Return the first raw IFM marker and its response-field path."""
    path_components = [field]
    pending = [(iter((("", value),)), 1)]
    while pending:
        children, parent_path_length = pending[-1]
        try:
            path_component, candidate = next(children)
        except StopIteration:
            pending.pop()
            continue
        del path_components[parent_path_length:]
        if path_component:
            path_components.append(path_component)
        if isinstance(candidate, str):
            normalized = candidate.lower()
            offsets = [
                normalized.find(prefix)
                for prefix in IFM_CONTROL_PREFIXES
                if prefix in normalized
            ]
            if offsets:
                start = min(offsets)
                end = candidate.find(">", start)
                if end < 0:
                    end = min(start + 79, len(candidate) - 1)
                return candidate[start : end + 1], "".join(path_components)
        elif isinstance(candidate, (dict, list, tuple)):
            pending.append(
                (
                    iter(_marker_children(candidate)),
                    len(path_components),
                )
            )
    return None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CapabilityValidationError(message)


def _marker_free(value: object, field: str = "response") -> None:
    marker = find_raw_ifm_control_marker(value, field)
    if marker:
        marker_text, marker_path = marker
        raise CapabilityValidationError(
            f"raw IFM control marker {marker_text!r} in {marker_path}"
        )


def _status_code(response: object) -> int:
    status_code = getattr(response, "status_code", None)
    _require(
        isinstance(status_code, int) and not isinstance(status_code, bool),
        "response has no integer status code",
    )
    return status_code


def _canonical_model_id(model: str) -> str:
    return model.removeprefix("builtin.")


def _require_response_model(body: dict, expected_model: str, field: str) -> None:
    response_model = body.get("model")
    _require(
        isinstance(response_model, str) and bool(response_model),
        f"{field} model is missing",
    )
    _require(
        _canonical_model_id(response_model) == _canonical_model_id(expected_model),
        f"{field} model {response_model!r} does not match {expected_model!r}",
    )


def reject_duplicate_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CapabilityValidationError(f"JSON object repeats key {key!r}")
        result[key] = value
    return result


def _stream_event_size(event: object) -> int:
    if isinstance(event, bytes):
        return len(event) + 1
    if isinstance(event, str):
        return len(event.encode("utf-8")) + 1
    try:
        encoded = json.dumps(
            event,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CapabilityValidationError("stream event cannot be encoded") from exc
    return len(encoded) + 1


def _validate_tool_arguments(arguments: object) -> dict:
    _require(isinstance(arguments, dict), "tool arguments must be an object")
    _require(
        set(arguments) == set(EXPECTED_TOOL_ARGUMENTS),
        "tool argument keys do not match request",
    )
    for field in ("city", "unit"):
        _require(
            type(arguments[field]) is str
            and arguments[field] == EXPECTED_TOOL_ARGUMENTS[field],
            f"tool argument {field} has the wrong value or type",
        )
    _require(
        type(arguments["days"]) is int
        and arguments["days"] == EXPECTED_TOOL_ARGUMENTS["days"],
        "tool argument days has the wrong value or type",
    )
    _require(
        type(arguments["include_hourly"]) is bool
        and arguments["include_hourly"] is EXPECTED_TOOL_ARGUMENTS["include_hourly"],
        "tool argument include_hourly has the wrong value or type",
    )
    return arguments


def _decode_tool_arguments(arguments_text: str) -> dict:
    try:
        arguments = json.loads(
            arguments_text,
            object_pairs_hook=reject_duplicate_json_object,
        )
    except json.JSONDecodeError as exc:
        raise CapabilityValidationError(
            f"tool arguments are invalid JSON: {exc}"
        ) from exc
    return _validate_tool_arguments(arguments)


def _post_json(
    request_json: Callable[..., tuple[object, object]],
    url: str,
    timeout: int,
    payload: dict,
) -> tuple[dict, int]:
    response, body = request_json("POST", url, timeout=timeout, json=payload)
    status_code = _status_code(response)
    _require(status_code == 200, f"HTTP {status_code}: {body}")
    _require(isinstance(body, dict), "response body must be an object")
    _marker_free(body)
    expected_model = payload.get("model")
    _require(
        isinstance(expected_model, str) and expected_model, "request model is missing"
    )
    _require_response_model(body, expected_model, "response")
    return body, status_code


def _openai_message(body: dict, finish_reason: str) -> dict:
    choices = body.get("choices")
    _require(isinstance(choices, list) and len(choices) == 1, "expected one choice")
    choice = choices[0]
    _require(isinstance(choice, dict), "choice must be an object")
    _require(
        choice.get("finish_reason") == finish_reason,
        f"expected finish_reason {finish_reason!r}",
    )
    message = choice.get("message")
    _require(isinstance(message, dict), "choice.message must be an object")
    _require(
        message.get("role") in (None, "assistant"), "message role is not assistant"
    )
    _marker_free(message, "message")
    return message


def _visible_content(message: dict, token: str | None = None) -> str:
    content = message.get("content")
    _require(isinstance(content, str) and bool(content.strip()), "content is blank")
    if token is not None:
        _require(token in content, f"content does not contain {token!r}")
    return content


def _reasoning_content(message: dict) -> str:
    reasoning = message.get("reasoning_content")
    _require(
        isinstance(reasoning, str) and bool(reasoning.strip()),
        "reasoning_content is blank",
    )
    return reasoning


def _require_no_reasoning(message: dict) -> None:
    reasoning = message.get("reasoning_content")
    _require(
        reasoning is None or (isinstance(reasoning, str) and not reasoning.strip()),
        "thinking-disabled response contains reasoning_content",
    )


def _require_no_tool_calls(message: dict) -> None:
    tool_calls = message.get("tool_calls")
    _require(
        tool_calls is None or (isinstance(tool_calls, list) and not tool_calls),
        "ordinary response contains tool calls",
    )


def _parse_tool_call(message: dict) -> tuple[dict, dict]:
    content = message.get("content")
    _require(
        content is None or (isinstance(content, str) and not content.strip()),
        "tool response contains unstructured visible content",
    )
    calls = message.get("tool_calls")
    _require(isinstance(calls, list) and len(calls) == 1, "expected one tool call")
    call = calls[0]
    _require(isinstance(call, dict), "tool call must be an object")
    _require(isinstance(call.get("id"), str) and call["id"], "tool call id is missing")
    _require(call.get("type") == "function", "tool call type must be function")
    function = call.get("function")
    _require(isinstance(function, dict), "tool call function must be an object")
    _require(function.get("name") == TOOL_NAME, "unexpected tool function name")
    arguments_text = function.get("arguments")
    _require(isinstance(arguments_text, str), "tool arguments must be a JSON string")
    arguments = _decode_tool_arguments(arguments_text)
    return call, arguments


def _base_openai_payload(model: str, *, stream: bool, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": ARITHMETIC_PROMPT}],
        "max_completion_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "stream": stream,
    }


def _tool_payload(model: str, tool_format: str | None, *, stream: bool) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TOOL_PROMPT}],
        "tools": [TOOL_DEFINITION],
        "tool_choice": "required",
        "max_completion_tokens": 256,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tool_format is not None:
        payload["chat_template_kwargs"]["tool_call_format"] = tool_format
    return payload


def _decode_openai_stream(
    events: Iterable[object], *, deadline: float | None = None
) -> tuple[dict, str, bool, str]:
    if deadline is None:
        deadline = time.monotonic() + STREAM_WALL_CLOCK_TIMEOUT_SECONDS
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tools: dict[int, dict[str, Any]] = {}
    finish_reason = None
    stream_model = None
    saw_choice = False
    saw_done = False
    total_sse_bytes = 0
    semantic_event_count = 0
    total_text_bytes = 0
    total_tool_argument_bytes = 0

    for event_index, event in enumerate(events):
        _require(
            time.monotonic() < deadline,
            "stream exceeded its wall-clock deadline",
        )
        total_sse_bytes += _stream_event_size(event)
        _require(
            total_sse_bytes <= MAX_STREAM_SSE_BYTES,
            "stream SSE byte count exceeds bounds",
        )
        if isinstance(event, bytes):
            try:
                event = event.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CapabilityValidationError(
                    f"stream event {event_index} is not UTF-8"
                ) from exc
        if isinstance(event, str):
            line = event.strip()
            if not line or line.startswith(":"):
                continue
            semantic_event_count += 1
            _require(
                semantic_event_count <= MAX_STREAM_EVENTS,
                "stream event count exceeds bounds",
            )
            _require(
                not saw_done, "stream contains data after the terminal [DONE] frame"
            )
            _marker_free(line, f"stream[{event_index}]")
            _require(line.startswith("data:"), "stream line is not an SSE data frame")
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                _require(not saw_done, "stream contains duplicate [DONE] frames")
                _require(
                    isinstance(finish_reason, str),
                    "stream ended before a terminal finish_reason",
                )
                saw_done = True
                continue
            try:
                chunk = json.loads(
                    data,
                    object_pairs_hook=reject_duplicate_json_object,
                )
            except json.JSONDecodeError as exc:
                raise CapabilityValidationError(
                    f"stream event {event_index} contains invalid JSON: {exc}"
                ) from exc
        else:
            semantic_event_count += 1
            _require(
                semantic_event_count <= MAX_STREAM_EVENTS,
                "stream event count exceeds bounds",
            )
            _require(
                not saw_done, "stream contains data after the terminal [DONE] frame"
            )
            chunk = event
        _require(isinstance(chunk, dict), "stream chunk must be an object")
        _marker_free(chunk, f"stream[{event_index}]")
        _require(
            "error" not in chunk, f"stream returned an error: {chunk.get('error')}"
        )
        choices = chunk.get("choices")
        _require(isinstance(choices, list), "stream choices must be an array")
        if not choices:
            continue
        _require(
            finish_reason is None,
            "stream contains a choice chunk after a terminal finish_reason",
        )
        _require(
            len(choices) == 1 and isinstance(choices[0], dict), "invalid stream choice"
        )
        response_model = chunk.get("model")
        _require(
            isinstance(response_model, str) and bool(response_model.strip()),
            f"stream[{event_index}] model must be a nonempty string",
        )
        _require(
            stream_model in (None, response_model),
            f"stream model changed from {stream_model!r} to {response_model!r}",
        )
        stream_model = response_model
        saw_choice = True
        choice = choices[0]
        delta = choice.get("delta", {})
        _require(isinstance(delta, dict), "stream delta must be an object")
        for key, destination in (
            ("content", content_parts),
            ("reasoning_content", reasoning_parts),
        ):
            value = delta.get(key)
            if value is not None:
                _require(isinstance(value, str), f"stream {key} must be a string")
                total_text_bytes += len(value.encode("utf-8"))
                _require(
                    total_text_bytes <= MAX_STREAM_TEXT_BYTES,
                    "stream content and reasoning exceed bounds",
                )
                destination.append(value)

        tool_deltas = delta.get("tool_calls", [])
        _require(isinstance(tool_deltas, list), "stream tool_calls must be an array")
        for tool_delta in tool_deltas:
            _require(
                isinstance(tool_delta, dict), "stream tool delta must be an object"
            )
            index = tool_delta.get("index")
            _require(
                isinstance(index, int) and not isinstance(index, bool) and index >= 0,
                "stream tool delta has an invalid index",
            )
            target = tools.setdefault(
                index,
                {
                    "id": None,
                    "type": None,
                    "name_parts": [],
                    "argument_parts": [],
                },
            )
            for key in ("id", "type"):
                value = tool_delta.get(key)
                if value is not None:
                    _require(
                        isinstance(value, str) and value,
                        f"stream tool {key} is invalid",
                    )
                    _require(
                        target[key] in (None, value),
                        f"stream tool {key} changed between chunks",
                    )
                    target[key] = value
            function = tool_delta.get("function")
            if function is not None:
                _require(isinstance(function, dict), "stream tool function is invalid")
                name = function.get("name")
                if name is not None:
                    _require(isinstance(name, str), "stream tool name must be a string")
                    target["name_parts"].append(name)
                arguments = function.get("arguments")
                if arguments is not None:
                    _require(
                        isinstance(arguments, str),
                        "stream tool arguments must be a string",
                    )
                    total_tool_argument_bytes += len(arguments.encode("utf-8"))
                    _require(
                        total_tool_argument_bytes <= MAX_STREAM_TOOL_ARGUMENT_BYTES,
                        "stream tool arguments exceed bounds",
                    )
                    target["argument_parts"].append(arguments)

        current_finish = choice.get("finish_reason")
        if current_finish is not None:
            _require(isinstance(current_finish, str), "finish_reason must be a string")
            _require(
                finish_reason in (None, current_finish),
                "finish_reason changed between chunks",
            )
            finish_reason = current_finish

    _require(saw_choice, "stream contains no choice chunks")
    _require(saw_done, "stream is missing the terminal [DONE] frame")
    _require(isinstance(finish_reason, str), "stream has no terminal finish_reason")
    _require(isinstance(stream_model, str), "stream model is missing")
    tool_calls = []
    for index in sorted(tools):
        tool = tools[index]
        tool_calls.append(
            {
                "id": tool["id"],
                "type": tool["type"],
                "function": {
                    "name": "".join(tool["name_parts"]),
                    "arguments": "".join(tool["argument_parts"]),
                },
            }
        )
    message = {
        "role": "assistant",
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "tool_calls": tool_calls,
    }
    _marker_free(message, "assembled_message")
    return message, finish_reason, saw_done, stream_model


def _post_openai_stream(
    request_stream: Callable[..., tuple[object, Iterable[object]]],
    url: str,
    timeout: int,
    payload: dict,
) -> tuple[dict, str, int, bool]:
    deadline = time.monotonic() + min(timeout, STREAM_WALL_CLOCK_TIMEOUT_SECONDS)
    response, events = request_stream(
        "POST",
        url,
        timeout=timeout,
        wall_clock_deadline=deadline,
        json=payload,
    )
    status_code = _status_code(response)
    _require(status_code == 200, f"stream returned HTTP {status_code}")
    expected_model = payload.get("model")
    _require(
        isinstance(expected_model, str) and expected_model, "request model is missing"
    )
    message, finish_reason, terminal, response_model = _decode_openai_stream(
        events,
        deadline=deadline,
    )
    _require_response_model(
        {"model": response_model},
        expected_model,
        "stream response",
    )
    return message, finish_reason, status_code, terminal


def _new_case(case_id: str, protocol: str, stream: bool) -> dict:
    return {
        "id": case_id,
        "protocol": protocol,
        "stream": stream,
        "pass": False,
        "status_code": 0,
        "finish_reason": None,
        "content_chars": 0,
        "reasoning_chars": 0,
        "tool_call_count": 0,
        "terminal_frame": not stream,
        "tool_name": None,
        "tool_arguments": None,
        "parity_with": None,
        "raw_ifm_marker": None,
        "error": None,
    }


def _run_case(
    cases: list[dict],
    case_id: str,
    protocol: str,
    stream: bool,
    action: Callable[[], tuple[dict, object]],
):
    case = _new_case(case_id, protocol, stream)
    artifact = None
    try:
        values, artifact = action()
        _require(isinstance(values, dict), "case metrics must be an object")
        unknown = set(values) - CASE_FIELDS
        _require(not unknown, f"case returned unknown metrics: {sorted(unknown)}")
        case.update(values)
        case["pass"] = True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        marker = find_raw_ifm_control_marker(str(exc), "error")
        if marker:
            case["raw_ifm_marker"] = marker[0]
        case["error"] = str(exc)[:512] or exc.__class__.__name__
    cases.append(case)
    return artifact


def _openai_plain_case(request_json, url, timeout, payload):
    body, status = _post_json(request_json, url, timeout, payload)
    message = _openai_message(body, "stop")
    content = _visible_content(message, "42")
    _require_no_reasoning(message)
    _require_no_tool_calls(message)
    return {
        "status_code": status,
        "finish_reason": "stop",
        "content_chars": len(content),
    }, message


def _openai_plain_stream_case(request_stream, url, timeout, payload):
    message, finish, status, terminal = _post_openai_stream(
        request_stream, url, timeout, payload
    )
    _require(finish == "stop", "plain stream did not finish with stop")
    content = _visible_content(message, "42")
    _require_no_reasoning(message)
    _require_no_tool_calls(message)
    return {
        "status_code": status,
        "finish_reason": finish,
        "content_chars": len(content),
        "terminal_frame": terminal,
    }, message


def _openai_reasoning_case(
    request_json,
    url,
    timeout,
    payload,
    *,
    require_reasoning: bool,
):
    body, status = _post_json(request_json, url, timeout, payload)
    message = _openai_message(body, "stop")
    content = _visible_content(message, "42")
    if require_reasoning:
        reasoning = _reasoning_content(message)
    else:
        _require_no_reasoning(message)
        reasoning = ""
    _require_no_tool_calls(message)
    return {
        "status_code": status,
        "finish_reason": "stop",
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
    }, message


def _openai_reasoning_stream_case(request_stream, url, timeout, payload):
    message, finish, status, terminal = _post_openai_stream(
        request_stream, url, timeout, payload
    )
    _require(finish == "stop", "reasoning stream did not finish with stop")
    content = _visible_content(message, "42")
    reasoning = _reasoning_content(message)
    _require_no_tool_calls(message)
    return {
        "status_code": status,
        "finish_reason": finish,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "terminal_frame": terminal,
    }, message


def _openai_tool_case(request_json, url, timeout, payload):
    body, status = _post_json(request_json, url, timeout, payload)
    message = _openai_message(body, "tool_calls")
    _require_no_reasoning(message)
    call, arguments = _parse_tool_call(message)
    return {
        "status_code": status,
        "finish_reason": "tool_calls",
        "tool_call_count": 1,
        "tool_name": TOOL_NAME,
        "tool_arguments": arguments,
    }, call


def _openai_tool_parity_case(
    request_json,
    url,
    timeout,
    payload,
    expected_call,
    parity_with,
):
    _require(isinstance(expected_call, dict), f"{parity_with} tool case did not pass")
    values, call = _openai_tool_case(request_json, url, timeout, payload)
    expected_function = expected_call.get("function", {})
    _require(
        call["function"]["name"] == expected_function.get("name")
        and _decode_tool_arguments(call["function"]["arguments"])
        == _decode_tool_arguments(expected_function.get("arguments", "null")),
        f"tool call differs from {parity_with}",
    )
    values["parity_with"] = parity_with
    return values, call


def _openai_tool_stream_case(request_stream, url, timeout, payload, expected_call):
    _require(
        isinstance(expected_call, dict), "non-streaming XML tool case did not pass"
    )
    message, finish, status, terminal = _post_openai_stream(
        request_stream, url, timeout, payload
    )
    _require(finish == "tool_calls", "tool stream did not finish with tool_calls")
    _require_no_reasoning(message)
    call, arguments = _parse_tool_call(message)
    expected_function = expected_call.get("function", {})
    _require(
        call["function"]["name"] == expected_function.get("name")
        and arguments
        == _decode_tool_arguments(expected_function.get("arguments", "null")),
        "streaming tool call differs from non-streaming XML result",
    )
    return {
        "status_code": status,
        "finish_reason": finish,
        "tool_call_count": 1,
        "terminal_frame": terminal,
        "tool_name": TOOL_NAME,
        "tool_arguments": arguments,
        "parity_with": "openai_tool_xml",
    }, call


def _openai_followup_case(request_json, url, timeout, model, call):
    _require(isinstance(call, dict), "non-streaming XML tool case did not pass")
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": TOOL_PROMPT},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(
                    {
                        "validation_token": "K2-SUNNY-731",
                        "forecast": "sunny",
                    },
                    separators=(",", ":"),
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return only the validation_token from the tool result. "
                    "Do not call a tool."
                ),
            },
        ],
        "tools": [TOOL_DEFINITION],
        "tool_choice": "none",
        "max_completion_tokens": 128,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body, status = _post_json(request_json, url, timeout, payload)
    message = _openai_message(body, "stop")
    content = _visible_content(message, "K2-SUNNY-731")
    _require_no_tool_calls(message)
    _require_no_reasoning(message)
    return {
        "status_code": status,
        "finish_reason": "stop",
        "content_chars": len(content),
    }, message


def _ollama_case(request_json, root_url, timeout, model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": ARITHMETIC_PROMPT}],
        "think": True,
        "stream": False,
        "options": {
            "num_predict": 3072,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
        },
    }
    body, status = _post_json(request_json, f"{root_url}/api/chat", timeout, payload)
    _require(body.get("done") is True, "Ollama response is not complete")
    _require(body.get("done_reason") == "stop", "Ollama response did not stop")
    message = body.get("message")
    _require(isinstance(message, dict), "Ollama message must be an object")
    _require(message.get("role") == "assistant", "Ollama role is not assistant")
    content = _visible_content(message, "42")
    thinking = message.get("thinking")
    _require(isinstance(thinking, str) and thinking.strip(), "Ollama thinking is blank")
    _require_no_tool_calls(message)
    return {
        "status_code": status,
        "finish_reason": "stop",
        "content_chars": len(content),
        "reasoning_chars": len(thinking),
    }, message


def _anthropic_case(request_json, root_url, timeout, model):
    payload = {
        "model": model,
        "system": [{"type": "text", "text": "Be concise."}],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Reply with ANTHROPIC-OK."}],
            }
        ],
        "max_tokens": 128,
        "temperature": 0,
        "stream": False,
        "thinking": {"type": "disabled"},
    }
    body, status = _post_json(
        request_json, f"{root_url}/v1/messages?beta=true", timeout, payload
    )
    _require(body.get("type") == "message", "Anthropic type is not message")
    _require(body.get("role") == "assistant", "Anthropic role is not assistant")
    _require(isinstance(body.get("id"), str) and body["id"], "Anthropic id is missing")
    _require(
        isinstance(body.get("model"), str) and body["model"],
        "Anthropic model is missing",
    )
    _require(body.get("stop_reason") == "end_turn", "Anthropic response did not end")
    blocks = body.get("content")
    _require(isinstance(blocks, list) and blocks, "Anthropic content is empty")
    _require(
        not any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in blocks
        ),
        "ordinary Anthropic response contains tool use",
    )
    text = "".join(
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )
    _require("ANTHROPIC-OK" in text, "Anthropic text does not contain sentinel")
    usage = body.get("usage")
    _require(isinstance(usage, dict), "Anthropic usage is missing")
    for field in ("input_tokens", "output_tokens"):
        value = usage.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"Anthropic {field} is invalid",
        )
    return {
        "status_code": status,
        "finish_reason": "end_turn",
        "content_chars": len(text),
    }, body


def validate_capabilities(
    profile: str,
    *,
    base_url: str,
    model: str,
    request_json: Callable[..., tuple[object, object]],
    request_stream: Callable[..., tuple[object, Iterable[object]]],
    timeout: int,
) -> tuple[dict, bool, str]:
    """Run a deterministic capability profile and return compact evidence."""
    if profile != K2_HORIZON_PROFILE:
        raise CapabilityValidationError(f"Unsupported capability profile: {profile}")
    if not base_url.endswith("/api/v1"):
        raise CapabilityValidationError("base_url must end with /api/v1")
    if not isinstance(model, str) or not model:
        raise CapabilityValidationError("model must be a nonempty string")

    root_url = base_url[: -len("/api/v1")]
    openai_url = f"{base_url}/chat/completions"
    cases: list[dict] = []

    off_payload = _base_openai_payload(model, stream=False, max_tokens=256)
    off_payload["chat_template_kwargs"] = {"enable_thinking": False}
    _run_case(
        cases,
        "openai_plain_off_nonstream",
        "openai",
        False,
        lambda: _openai_plain_case(request_json, openai_url, timeout, off_payload),
    )

    off_stream_payload = _base_openai_payload(model, stream=True, max_tokens=256)
    off_stream_payload["stream_options"] = {"include_usage": True}
    off_stream_payload["chat_template_kwargs"] = {"enable_thinking": False}
    _run_case(
        cases,
        "openai_plain_off_stream",
        "openai",
        True,
        lambda: _openai_plain_stream_case(
            request_stream, openai_url, timeout, off_stream_payload
        ),
    )

    for effort in ("high", "medium", "low"):
        payload = _base_openai_payload(model, stream=False, max_tokens=3072)
        payload["chat_template_kwargs"] = {"reasoning_effort": effort}
        _run_case(
            cases,
            f"openai_reasoning_{effort}",
            "openai",
            False,
            lambda payload=payload, effort=effort: _openai_reasoning_case(
                request_json,
                openai_url,
                timeout,
                payload,
                require_reasoning=effort != "low",
            ),
        )
        if effort == "high":
            stream_payload = _base_openai_payload(
                model,
                stream=True,
                max_tokens=3072,
            )
            stream_payload["stream_options"] = {"include_usage": True}
            stream_payload["chat_template_kwargs"] = {"reasoning_effort": effort}
            _run_case(
                cases,
                "openai_reasoning_high_stream",
                "openai",
                True,
                lambda: _openai_reasoning_stream_case(
                    request_stream,
                    openai_url,
                    timeout,
                    stream_payload,
                ),
            )

    xml_call = None
    for tool_format in ("json", "xml", "xml_typed"):
        call = _run_case(
            cases,
            f"openai_tool_{tool_format}",
            "openai",
            False,
            lambda tool_format=tool_format: _openai_tool_case(
                request_json,
                openai_url,
                timeout,
                _tool_payload(model, tool_format, stream=False),
            ),
        )
        if tool_format == "xml":
            xml_call = call

    _run_case(
        cases,
        "openai_tool_default_xml",
        "openai",
        False,
        lambda: _openai_tool_parity_case(
            request_json,
            openai_url,
            timeout,
            _tool_payload(model, None, stream=False),
            xml_call,
            "openai_tool_xml",
        ),
    )

    _run_case(
        cases,
        "openai_tool_stream_xml",
        "openai",
        True,
        lambda: _openai_tool_stream_case(
            request_stream,
            openai_url,
            timeout,
            _tool_payload(model, "xml", stream=True),
            xml_call,
        ),
    )
    _run_case(
        cases,
        "openai_tool_result_followup",
        "openai",
        False,
        lambda: _openai_followup_case(
            request_json, openai_url, timeout, model, xml_call
        ),
    )
    _run_case(
        cases,
        "ollama_thinking",
        "ollama",
        False,
        lambda: _ollama_case(request_json, root_url, timeout, model),
    )
    _run_case(
        cases,
        "anthropic_translation",
        "anthropic",
        False,
        lambda: _anthropic_case(request_json, root_url, timeout, model),
    )

    passed = all(case["pass"] for case in cases)
    matrix = {
        "profile": profile,
        "model": model,
        "pass": passed,
        "cases": cases,
    }
    if passed:
        summary = f"{len(cases)}/{len(cases)} capability cases passed"
    else:
        failed = [case["id"] for case in cases if not case["pass"]]
        details = "; ".join(
            f"{case['id']}: {case['error']}" for case in cases if not case["pass"]
        )
        summary = f"Failed capability cases {', '.join(failed)}: {details}"
    return matrix, passed, summary[:1024]


def validate_capability_matrix_evidence(
    value: object,
    expected_profile: str,
    expected_model: str,
) -> None:
    """Validate one compact capability evidence envelope."""
    _require(
        expected_profile == K2_HORIZON_PROFILE,
        f"Unsupported capability profile: {expected_profile}",
    )
    _require(isinstance(value, dict), "capability matrix must be an object")
    _require(
        set(value) == {"profile", "model", "pass", "cases"},
        "capability matrix has unexpected fields",
    )
    _require(value["profile"] == expected_profile, "capability profile mismatch")
    _require(
        isinstance(expected_model, str) and expected_model, "expected model is missing"
    )
    _require_response_model(value, expected_model, "capability matrix")
    _require(value["pass"] is True, "capability matrix did not pass")
    cases = value["cases"]
    _require(isinstance(cases, list), "capability cases must be an array")
    _require(
        [case.get("id") if isinstance(case, dict) else None for case in cases]
        == list(K2_HORIZON_CASE_IDS),
        "capability cases do not match the profile contract",
    )

    for case in cases:
        _require(set(case) == CASE_FIELDS, f"{case['id']}: unexpected case fields")
        _require(case["pass"] is True, f"{case['id']}: case did not pass")
        _require(case["error"] is None, f"{case['id']}: case has an error")
        _require(
            case["raw_ifm_marker"] is None,
            f"{case['id']}: raw IFM marker was recorded",
        )
        _require(
            type(case["status_code"]) is int and case["status_code"] == 200,
            f"{case['id']}: HTTP status is not 200",
        )
        _require(
            case["terminal_frame"] is True,
            f"{case['id']}: response is not terminal",
        )
        for field in ("content_chars", "reasoning_chars", "tool_call_count"):
            value_field = case[field]
            _require(
                isinstance(value_field, int)
                and not isinstance(value_field, bool)
                and value_field >= 0,
                f"{case['id']}: {field} is invalid",
            )
        (
            protocol,
            stream,
            finish_reason,
            has_content,
            has_reasoning,
            tool_call_count,
            parity_with,
        ) = CASE_CONTRACTS[case["id"]]
        _require(case["protocol"] == protocol, f"{case['id']}: protocol mismatch")
        _require(case["stream"] is stream, f"{case['id']}: stream flag mismatch")
        _require(
            case["finish_reason"] == finish_reason,
            f"{case['id']}: finish reason mismatch",
        )
        _require(
            (case["content_chars"] > 0) is has_content,
            f"{case['id']}: content contract mismatch",
        )
        _require(
            (case["reasoning_chars"] > 0) is has_reasoning,
            f"{case['id']}: reasoning contract mismatch",
        )
        _require(
            case["tool_call_count"] == tool_call_count,
            f"{case['id']}: tool-call count mismatch",
        )
        _require(
            case["parity_with"] == parity_with,
            f"{case['id']}: parity binding mismatch",
        )
        if tool_call_count:
            _require(
                case["tool_name"] == TOOL_NAME, f"{case['id']}: tool name mismatch"
            )
            _validate_tool_arguments(case["tool_arguments"])
        else:
            _require(case["tool_name"] is None, f"{case['id']}: unexpected tool name")
            _require(
                case["tool_arguments"] is None,
                f"{case['id']}: unexpected tool arguments",
            )
