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
import sys
import tempfile
from urllib.parse import quote

import requests

from utils.llamacpp_capability_validation import (
    K2_HORIZON_PROFILE,
    find_raw_ifm_control_marker,
    reject_duplicate_json_object,
    validate_capabilities,
)
from utils.server_base import _auth_headers, unload_all_models, wait_for_server
from utils.test_models import PORT, TIMEOUT_DEFAULT
from utils.validation_model_selection import (
    ModelSelectionError,
    add_model_selection_arguments,
    select_llamacpp_models,
)

TIMEOUT_HEALTH = 60
TIMEOUT_INFERENCE = 1800  # 30 minutes; large models may need 60+ GB download
VALIDATION_CTX_SIZE = 8192
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
    """Perform an HTTP request and parse the JSON response when present."""
    auth_headers = _auth_headers()
    headers = {**(kwargs.get("headers") or {}), **auth_headers}
    request_kwargs = {**kwargs, "headers": headers}
    response = requests.request(method, url, timeout=timeout, **request_kwargs)
    body = {}
    if response.content:
        try:
            body = response.json(object_pairs_hook=reject_duplicate_json_object)
        except ValueError:
            body = {"raw_text": response.text}
    return response, body


def request_stream(method, url, timeout, **kwargs):
    """Perform a streaming HTTP request and return decoded response lines."""
    auth_headers = _auth_headers()
    headers = {**(kwargs.get("headers") or {}), **auth_headers}
    request_kwargs = {**kwargs, "headers": headers, "stream": True}
    response = requests.request(method, url, timeout=timeout, **request_kwargs)

    def lines():
        try:
            yield from response.iter_lines(decode_unicode=True)
        finally:
            response.close()

    return response, lines()


def require_running_server(base_url, port):
    """Wait for a running server and confirm the health endpoint responds."""
    wait_for_server(port=port, timeout=TIMEOUT_HEALTH)
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
        print(
            f"  Warning: unload returned HTTP {response.status_code}: {body}",
            flush=True,
        )


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

        aliases = {model_name}
        if model_name.startswith("builtin."):
            aliases.add(model_name.removeprefix("builtin."))
        else:
            aliases.add(f"builtin.{model_name}")
        loaded_model = next(
            (model for model in loaded_models if model["model_name"] in aliases),
            None,
        )
        if loaded_model is None:
            return (
                False,
                f"Model '{model_name}' is absent from the loaded inventory",
                {},
            )

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
            },
        )
        if chat_resp.status_code != 200:
            return False, f"HTTP {chat_resp.status_code}: {chat_body}", {}

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
        try:
            unload_model(base_url, model_name)
        except requests.RequestException as exc:
            print(f"  Warning: failed to unload {model_name}: {exc}", flush=True)


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

    aliases = {model_name}
    if model_name.startswith("builtin."):
        aliases.add(model_name.removeprefix("builtin."))
    else:
        aliases.add(f"builtin.{model_name}")
    if any(model["model_name"] in aliases for model in loaded_models):
        return False, f"Model '{model_name}' is still loaded after unload", result[2]

    print("  Reloading model after unload...", flush=True)
    return test_model(base_url, model_name, backend)


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

    base_url = f"http://localhost:{args.port}/api/v1"
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
        candidate.removeprefix("builtin.")
        for model in selected_models
        for candidate in (model["id"], model.get("load_id", model["id"]))
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
    try:
        unload_all_models(port=args.port)
    except requests.RequestException as exc:
        print(f"Warning: failed to unload pre-existing models: {exc}", flush=True)

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

    results = []
    all_passed = True
    try:
        for model in selected_models:
            model_name = model["id"]
            print(f"\nTesting: {model_name}", flush=True)
            runs_capability_profile = (
                model_name.removeprefix("builtin.") in capability_model_ids
            )
            capability_evidence = {} if runs_capability_profile else None
            success, response_text, stats = validate_model_lifecycle(
                base_url,
                model.get("load_id", model_name),
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
            unload_all_models(port=args.port)
        except requests.RequestException as exc:
            print(f"Warning: failed to unload models during cleanup: {exc}", flush=True)
        if args.logs_dir:
            collect_server_logs(args.logs_dir)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
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
    main()
