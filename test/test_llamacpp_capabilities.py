#!/usr/bin/env python3
"""Tests for the llama.cpp capability validation profile."""

from __future__ import annotations

import copy
import json
import types
import unittest

from test.utils import llamacpp_capability_validation as capabilities

MODEL = "builtin.K2-Horizon-0.9B-GGUF"


class CapabilityHarness:
    def __init__(self) -> None:
        self.requests = []
        self.reasoning_marker = None
        self.omit_stream_done = False
        self.early_stream_done = False
        self.response_model = MODEL
        self.stream_response_models = [MODEL, MODEL]
        self.stream_reasoning = "high reasoning"
        self.default_tool_response_is_text = False
        self.stop_tool_call = False
        self.tool_reasoning = False
        self.tool_arguments = json.dumps(
            capabilities.EXPECTED_TOOL_ARGUMENTS,
            separators=(",", ":"),
        )

    @staticmethod
    def response(status_code=200):
        return types.SimpleNamespace(status_code=status_code)

    def openai(self, message, finish_reason="stop"):
        return {
            "id": "chatcmpl-test",
            "model": self.response_model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", **message},
                }
            ],
        }

    def tool_call(self):
        return {
            "id": "call_weather",
            "type": "function",
            "function": {
                "name": capabilities.TOOL_NAME,
                "arguments": self.tool_arguments,
            },
        }

    def stream_chunk(self, index, choices):
        chunk = {"choices": choices}
        model = self.stream_response_models[index]
        if model is not None:
            chunk["model"] = model
        return chunk

    def request_json(self, method, url, timeout, **kwargs):
        self.requests.append((method, url, kwargs["json"]))
        payload = kwargs["json"]
        if url.endswith("/api/chat"):
            message = {
                "role": "assistant",
                "thinking": "Seventeen plus twenty-five is forty-two.",
                "content": "42",
            }
            if self.stop_tool_call:
                message["tool_calls"] = [self.tool_call()]
            return self.response(), {
                "model": self.response_model,
                "done": True,
                "done_reason": "stop",
                "message": message,
            }
        if url.endswith("/v1/messages?beta=true"):
            content = [{"type": "text", "text": "ANTHROPIC-OK"}]
            if self.stop_tool_call:
                content.append(
                    {
                        "type": "tool_use",
                        "id": "call_weather",
                        "name": capabilities.TOOL_NAME,
                        "input": capabilities.EXPECTED_TOOL_ARGUMENTS,
                    }
                )
            return self.response(), {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": self.response_model,
                "stop_reason": "end_turn",
                "content": content,
                "usage": {"input_tokens": 8, "output_tokens": 3},
            }

        kwargs_value = payload.get("chat_template_kwargs", {})
        if payload.get("tool_choice") == "none":
            return self.response(), self.openai({"content": "K2-SUNNY-731"})
        if payload.get("tools"):
            if (
                "tool_call_format" not in kwargs_value
                and self.default_tool_response_is_text
            ):
                return self.response(), self.openai({"content": "Not a tool call."})
            message = {"content": None, "tool_calls": [self.tool_call()]}
            if self.tool_reasoning:
                message["reasoning_content"] = "thinking was supposed to be disabled"
            return self.response(), self.openai(
                message,
                "tool_calls",
            )
        if "reasoning_effort" in kwargs_value:
            effort = kwargs_value["reasoning_effort"]
            reasoning = f"{effort} reasoning"
            if self.reasoning_marker == effort:
                reasoning += " <ifm|think>"
            message = {"content": "42", "reasoning_content": reasoning}
            if self.stop_tool_call:
                message["tool_calls"] = [self.tool_call()]
            return self.response(), self.openai(message)
        message = {"content": "42"}
        if self.stop_tool_call:
            message["tool_calls"] = [self.tool_call()]
        return self.response(), self.openai(message)

    def request_stream(self, method, url, timeout, **kwargs):
        self.requests.append((method, url, kwargs["json"]))
        payload = kwargs["json"]
        if payload.get("tools"):
            events = [
                "data: "
                + json.dumps(
                    self.stream_chunk(
                        0,
                        [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_stream",
                                            "type": "function",
                                            "function": {
                                                "name": "lookup_",
                                                "arguments": '{"city":"Paris","unit":"',
                                            },
                                        }
                                    ]
                                },
                            }
                        ],
                    ),
                    separators=(",", ":"),
                ),
                "data: "
                + json.dumps(
                    self.stream_chunk(
                        1,
                        [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "name": "weather",
                                                "arguments": (
                                                    'celsius","days":2,'
                                                    '"include_hourly":false}'
                                                ),
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    ),
                    separators=(",", ":"),
                ),
            ]
        elif "reasoning_effort" in payload.get("chat_template_kwargs", {}):
            events = [
                "data: "
                + json.dumps(
                    self.stream_chunk(
                        0,
                        [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": self.stream_reasoning},
                            }
                        ],
                    ),
                    separators=(",", ":"),
                ),
                "data: "
                + json.dumps(
                    self.stream_chunk(
                        1,
                        [
                            {
                                "index": 0,
                                "delta": {"content": "42"},
                                "finish_reason": "stop",
                            }
                        ],
                    ),
                    separators=(",", ":"),
                ),
            ]
        else:
            events = [
                ": keep-alive",
                "data: "
                + json.dumps(
                    self.stream_chunk(
                        0,
                        [{"index": 0, "delta": {"content": "4"}}],
                    ),
                    separators=(",", ":"),
                ),
                "data: "
                + json.dumps(
                    self.stream_chunk(
                        1,
                        [
                            {
                                "index": 0,
                                "delta": {"content": "2"},
                                "finish_reason": "stop",
                            }
                        ],
                    ),
                    separators=(",", ":"),
                ),
                'data: {"choices":[],"usage":{"completion_tokens":2}}',
            ]
        if self.early_stream_done:
            events.insert(len(events) - 1, "data: [DONE]")
        elif not self.omit_stream_done:
            events.append("data: [DONE]")
        return self.response(), events


class LlamaCppCapabilityTests(unittest.TestCase):
    def run_profile(self, harness=None):
        active = harness or CapabilityHarness()
        matrix, passed, summary = capabilities.validate_capabilities(
            capabilities.K2_HORIZON_PROFILE,
            base_url="http://localhost:13305/api/v1",
            model=MODEL,
            request_json=active.request_json,
            request_stream=active.request_stream,
            timeout=60,
        )
        return active, matrix, passed, summary

    def test_complete_profile_exercises_the_exact_contract(self) -> None:
        harness, matrix, passed, summary = self.run_profile()

        self.assertTrue(passed, summary)
        self.assertTrue(matrix["pass"])
        self.assertEqual(matrix["profile"], capabilities.K2_HORIZON_PROFILE)
        self.assertEqual(matrix["model"], MODEL)
        self.assertEqual(
            [case["id"] for case in matrix["cases"]],
            list(capabilities.K2_HORIZON_CASE_IDS),
        )
        capabilities.validate_capability_matrix_evidence(
            matrix,
            capabilities.K2_HORIZON_PROFILE,
            MODEL,
        )

        openai_payloads = [
            payload for _method, url, payload in harness.requests if "/api/v1/" in url
        ]
        self.assertEqual(
            [
                payload["chat_template_kwargs"]["reasoning_effort"]
                for payload in openai_payloads
                if "reasoning_effort" in payload.get("chat_template_kwargs", {})
            ],
            ["high", "high", "medium", "low"],
        )
        self.assertTrue(
            all(
                payload["max_completion_tokens"] == 3072
                for payload in openai_payloads
                if "reasoning_effort" in payload.get("chat_template_kwargs", {})
            )
        )
        self.assertEqual(
            [
                payload["chat_template_kwargs"]["tool_call_format"]
                for payload in openai_payloads
                if payload.get("tools")
                and payload.get("tool_choice") == "required"
                and "tool_call_format" in payload.get("chat_template_kwargs", {})
            ],
            ["json", "xml", "xml_typed", "xml"],
        )
        default_tool_payloads = [
            payload
            for payload in openai_payloads
            if payload.get("tools")
            and payload.get("tool_choice") == "required"
            and "tool_call_format" not in payload.get("chat_template_kwargs", {})
        ]
        self.assertEqual(len(default_tool_payloads), 1)
        self.assertEqual(
            default_tool_payloads[0]["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        followup = next(
            payload
            for payload in openai_payloads
            if payload.get("tool_choice") == "none"
        )
        assistant = followup["messages"][1]
        tool_result = followup["messages"][2]
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_weather")
        self.assertEqual(tool_result["tool_call_id"], "call_weather")
        self.assertIn("K2-SUNNY-731", tool_result["content"])
        ollama_payload = next(
            payload
            for _method, url, payload in harness.requests
            if url.endswith("/api/chat")
        )
        self.assertEqual(ollama_payload["options"]["num_predict"], 3072)

    def test_reasoning_marker_fails_the_named_case_and_matrix(self) -> None:
        harness = CapabilityHarness()
        harness.reasoning_marker = "medium"

        _harness, matrix, passed, summary = self.run_profile(harness)

        self.assertFalse(passed)
        self.assertFalse(matrix["pass"])
        failed = [case for case in matrix["cases"] if not case["pass"]]
        self.assertEqual([case["id"] for case in failed], ["openai_reasoning_medium"])
        self.assertRegex(summary, "medium.*IFM|IFM.*medium")
        with self.assertRaises(capabilities.CapabilityValidationError):
            capabilities.validate_capability_matrix_evidence(
                matrix,
                capabilities.K2_HORIZON_PROFILE,
                MODEL,
            )

    def test_stream_requires_a_terminal_done_frame(self) -> None:
        harness = CapabilityHarness()
        harness.omit_stream_done = True

        _harness, matrix, passed, _summary = self.run_profile(harness)

        self.assertFalse(passed)
        failed_ids = {case["id"] for case in matrix["cases"] if not case["pass"]}
        self.assertEqual(
            failed_ids,
            {
                "openai_plain_off_stream",
                "openai_reasoning_high_stream",
                "openai_tool_stream_xml",
            },
        )

    def test_stream_rejects_done_before_the_final_data_frame(self) -> None:
        harness = CapabilityHarness()
        harness.early_stream_done = True

        _harness, matrix, passed, _summary = self.run_profile(harness)

        self.assertFalse(passed)
        failed_ids = {case["id"] for case in matrix["cases"] if not case["pass"]}
        self.assertEqual(
            failed_ids,
            {
                "openai_plain_off_stream",
                "openai_reasoning_high_stream",
                "openai_tool_stream_xml",
            },
        )

    def test_nonstream_responses_are_bound_to_the_requested_model(self) -> None:
        harness = CapabilityHarness()
        harness.response_model = "builtin.Not-K2"

        _harness, matrix, passed, _summary = self.run_profile(harness)

        self.assertFalse(passed)
        failed_ids = {case["id"] for case in matrix["cases"] if not case["pass"]}
        self.assertTrue(
            {case["id"] for case in matrix["cases"] if not case["stream"]}.issubset(
                failed_ids
            )
        )
        self.assertNotIn("openai_plain_off_stream", failed_ids)

    def test_stream_reasoning_requires_separate_marker_free_content(self) -> None:
        for stream_reasoning in ("", "high <ifm|think> reasoning"):
            with self.subTest(stream_reasoning=stream_reasoning):
                harness = CapabilityHarness()
                harness.stream_reasoning = stream_reasoning

                _harness, matrix, passed, _summary = self.run_profile(harness)

                self.assertFalse(passed)
                self.assertEqual(
                    {case["id"] for case in matrix["cases"] if not case["pass"]},
                    {"openai_reasoning_high_stream"},
                )

    def test_stream_accepts_a_stable_backend_model_identity(self) -> None:
        harness = CapabilityHarness()
        harness.stream_response_models = [
            "K2-Horizon-1B-BF16.gguf",
            "K2-Horizon-1B-BF16.gguf",
        ]

        _harness, _matrix, passed, summary = self.run_profile(harness)

        self.assertTrue(passed, summary)

    def test_stream_rejects_changed_or_missing_backend_model_identity(self) -> None:
        cases = (
            ["K2-Horizon-1B-BF16.gguf", "different.gguf"],
            [None, "K2-Horizon-1B-BF16.gguf"],
            ["K2-Horizon-1B-BF16.gguf", None],
            ["K2-Horizon-1B-BF16.gguf", "   "],
        )
        for stream_models in cases:
            with self.subTest(stream_models=stream_models):
                harness = CapabilityHarness()
                harness.stream_response_models = stream_models

                _harness, matrix, passed, _summary = self.run_profile(harness)

                self.assertFalse(passed)
                self.assertEqual(
                    {case["id"] for case in matrix["cases"] if not case["pass"]},
                    {
                        "openai_plain_off_stream",
                        "openai_reasoning_high_stream",
                        "openai_tool_stream_xml",
                    },
                )

    def test_stream_rejects_duplicate_json_members(self) -> None:
        invalid_events = (
            (
                'data: {"model":"wrong","model":"backend",'
                '"choices":[{"index":0,"delta":{"content":"42"},'
                '"finish_reason":"stop"}]}',
                "model",
            ),
            (
                'data: {"model":"backend","choices":[],"choices":'
                '[{"index":0,"delta":{"content":"42"},'
                '"finish_reason":"stop"}]}',
                "choices",
            ),
        )
        for event, duplicate_key in invalid_events:
            with self.subTest(duplicate_key=duplicate_key):
                with self.assertRaisesRegex(
                    capabilities.CapabilityValidationError,
                    f"repeats key '{duplicate_key}'",
                ):
                    capabilities._decode_openai_stream([event, "data: [DONE]"])

    def test_stream_rejects_choice_chunks_after_finish_reason(self) -> None:
        later_choices = (
            {"index": 0, "delta": {"content": "2"}},
            {"index": 0, "delta": {"reasoning_content": "later"}},
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_late",
                            "type": "function",
                            "function": {
                                "name": capabilities.TOOL_NAME,
                                "arguments": "{}",
                            },
                        }
                    ]
                },
            },
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        )
        first = {
            "model": "backend",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "4"},
                    "finish_reason": "stop",
                }
            ],
        }
        for later_choice in later_choices:
            with self.subTest(later_choice=later_choice):
                later = {"model": "backend", "choices": [later_choice]}
                events = [
                    f"data: {json.dumps(first)}",
                    f"data: {json.dumps(later)}",
                    "data: [DONE]",
                ]

                with self.assertRaisesRegex(
                    capabilities.CapabilityValidationError,
                    "after a terminal finish_reason",
                ):
                    capabilities._decode_openai_stream(events)

    def test_tool_arguments_require_exact_json_keys_and_types(self) -> None:
        arguments = (
            '{"city":"London","city":"Paris","unit":"celsius",'
            '"days":2,"include_hourly":false}',
            '{"city":"Paris","unit":"celsius","days":2.0,' '"include_hourly":false}',
            '{"city":"Paris","unit":"celsius","days":2,' '"include_hourly":0}',
        )

        for tool_arguments in arguments:
            with self.subTest(tool_arguments=tool_arguments):
                harness = CapabilityHarness()
                harness.tool_arguments = tool_arguments

                _harness, matrix, passed, _summary = self.run_profile(harness)

                self.assertFalse(passed)
                failed_ids = {
                    case["id"] for case in matrix["cases"] if not case["pass"]
                }
                self.assertTrue(
                    {
                        "openai_tool_json",
                        "openai_tool_xml",
                        "openai_tool_xml_typed",
                    }.issubset(failed_ids)
                )

    def test_omitted_tool_format_requires_the_xml_default(self) -> None:
        harness = CapabilityHarness()
        harness.default_tool_response_is_text = True

        _harness, matrix, passed, _summary = self.run_profile(harness)

        self.assertFalse(passed)
        self.assertEqual(
            {case["id"] for case in matrix["cases"] if not case["pass"]},
            {"openai_tool_default_xml"},
        )

    def test_ordinary_stop_responses_reject_tool_calls(self) -> None:
        harness = CapabilityHarness()
        harness.stop_tool_call = True

        _harness, matrix, passed, _summary = self.run_profile(harness)

        self.assertFalse(passed)
        self.assertEqual(
            {case["id"] for case in matrix["cases"] if not case["pass"]},
            {
                "openai_plain_off_nonstream",
                "openai_reasoning_high",
                "openai_reasoning_medium",
                "openai_reasoning_low",
                "ollama_thinking",
                "anthropic_translation",
            },
        )

    def test_tool_requests_reject_reasoning_when_thinking_is_disabled(self) -> None:
        harness = CapabilityHarness()
        harness.tool_reasoning = True

        _harness, matrix, passed, _summary = self.run_profile(harness)

        self.assertFalse(passed)
        failed_ids = {case["id"] for case in matrix["cases"] if not case["pass"]}
        self.assertTrue(
            {
                "openai_tool_json",
                "openai_tool_xml",
                "openai_tool_xml_typed",
            }.issubset(failed_ids)
        )

    def test_evidence_rejects_missing_unknown_and_non_boolean_cases(self) -> None:
        _harness, matrix, passed, _summary = self.run_profile()
        self.assertTrue(passed)

        mutations = []
        missing = copy.deepcopy(matrix)
        missing["cases"].pop()
        mutations.append(missing)
        unknown = copy.deepcopy(matrix)
        unknown["cases"][0]["id"] = "unknown"
        mutations.append(unknown)
        non_boolean = copy.deepcopy(matrix)
        non_boolean["cases"][0]["pass"] = 1
        mutations.append(non_boolean)
        extra = copy.deepcopy(matrix)
        extra["unexpected"] = True
        mutations.append(extra)
        wrong_protocol = copy.deepcopy(matrix)
        wrong_protocol["cases"][0]["protocol"] = "ollama"
        mutations.append(wrong_protocol)
        wrong_stream = copy.deepcopy(matrix)
        wrong_stream["cases"][0]["stream"] = True
        mutations.append(wrong_stream)
        wrong_finish = copy.deepcopy(matrix)
        wrong_finish["cases"][0]["finish_reason"] = "tool_calls"
        mutations.append(wrong_finish)
        stray_reasoning = copy.deepcopy(matrix)
        stray_reasoning["cases"][0]["reasoning_chars"] = 1
        mutations.append(stray_reasoning)
        stray_tool = copy.deepcopy(matrix)
        stray_tool["cases"][0]["tool_name"] = capabilities.TOOL_NAME
        mutations.append(stray_tool)
        float_integer = copy.deepcopy(matrix)
        float_tool_case = next(
            case for case in float_integer["cases"] if case["id"] == "openai_tool_json"
        )
        float_tool_case["tool_arguments"]["days"] = 2.0
        mutations.append(float_integer)
        integer_boolean = copy.deepcopy(matrix)
        boolean_tool_case = next(
            case
            for case in integer_boolean["cases"]
            if case["id"] == "openai_tool_json"
        )
        boolean_tool_case["tool_arguments"]["include_hourly"] = 0
        mutations.append(integer_boolean)

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(capabilities.CapabilityValidationError):
                    capabilities.validate_capability_matrix_evidence(
                        mutation,
                        capabilities.K2_HORIZON_PROFILE,
                        MODEL,
                    )

    def test_unknown_profile_fails_without_requests(self) -> None:
        harness = CapabilityHarness()

        with self.assertRaisesRegex(
            capabilities.CapabilityValidationError,
            "Unsupported capability profile",
        ):
            capabilities.validate_capabilities(
                "unknown",
                base_url="http://localhost:13305/api/v1",
                model=MODEL,
                request_json=harness.request_json,
                request_stream=harness.request_stream,
                timeout=60,
            )

        self.assertEqual(harness.requests, [])


if __name__ == "__main__":
    unittest.main()
