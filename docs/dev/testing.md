# Testing Guide

Lemonade is tested primarily by Python integration suites that run against a live server, plus C++ unit tests for pure logic and a small regression suite for the app. A feature isn't done until a test that could catch its regression runs in CI.

This guide covers what reviewers expect from the tests in a PR: where they go, how to run them, and what gets a PR sent back.

- [Principles](#principles)
- [Where Tests Go](#where-tests-go)
- [Running Tests Locally](#running-tests-locally)
- [Writing a Good Test](#writing-a-good-test)
- [CI Expectations](#ci-expectations)
- [What Reviewers Reject](#what-reviewers-reject)

---

## Principles

### Tests ship with the feature

New endpoints, commands, and backends need at least one test that exercises them, in the same PR as the code. Bug fixes get a regression test in the suite that owns the surface. Name the new test methods in your PR description or review replies so reviewers can find them.

### A test that isn't in CI doesn't exist

A committed test file that no workflow runs is an incomplete contribution. New Python suites must be added to a CI job; new C++ tests must be declared with `add_cpp_ci_test(<Name> CI ON COMMAND <target>)` in `CMakeLists.txt` so the `cpp-ci` CTest label picks them up. Guard the block with `if(BUILD_TESTING AND ...)` — distro packaging configures with `BUILD_TESTING=OFF` to avoid building test binaries it discards, and an unguarded declaration fails that build at configure time.

Most suites join the existing endpoint/CLI or hardware-matrix jobs in `cpp_server_build_test_release.yml`. A dedicated workflow is appropriate only when the suite has environment needs the existing jobs can't meet (real network downloads, a container, path-filtered smoke tests).

### A test must be able to fail

Before trusting a regression test, verify it fails on the pre-fix code and say so in the PR ("verified this test fails on `main`"). Assert the behavior the test name claims — a stats test that only checks counters are `>= 0` passes even when every counter is zeroed. For numeric outputs (classifiers, rerankers), structural checks (labels present, scores in range) pass on garbage; compare against a golden reference instead.

### Extend existing suites

Test suites are organized by modality and API surface, not by backend or device. A new backend or device for an existing modality — image generation on NPU, a new LLM engine — is covered by the existing modality suite through `--wrapped-server` / `--backend` flags and `test/utils/capabilities.py`, not by a new test file. Only a genuinely new surface (new endpoints, a new protocol) gets a new file.

### Test the mechanism, not the data

When a change adds rows to a data table an existing mechanism already consumes — a new GPU architecture, a model registry entry, a backend version pin — the mechanism's existing tests plus the live CI checks (artifact 404 probes, `test/server_gfx_topology.py`, the scheduled `validate_*` workflows) are the coverage. Don't add a test that asserts the new row exists. Verify the change on real hardware where you can, and state what you could and couldn't verify in the PR description.

### Deterministic over clever

No timers or sleeps as success signals — monitor logs or process IDs instead. Assert output is non-empty rather than coupling to a specific length or wording. Never add product or API surface just to make a test deterministic; configure the model (for example, llamacpp args in the test) instead.

### Respect the CI budget

Use the smallest model that exercises the code path. Suites that run on GitHub-hosted runners download models fresh every run: aim for under 1 GB, and treat 5 GB as a hard cap. The self-hosted hardware runners keep persistent model caches, so inference suites there may use larger models when the modality has no small variant. Scenarios that would meaningfully extend every PR's runtime (upgrade paths, long-idle behavior) belong in scheduled or manually-triggered workflows, not the per-PR gate.

---

## Where Tests Go

| If your PR... | Add or update tests in |
|---|---|
| Adds or changes an HTTP endpoint | `test/server_endpoints.py` |
| Adds or changes a CLI command | `test/server_cli2.py` |
| Adds a backend for an existing modality (LLM, image, audio, TTS) | The existing modality suite (`test/server_llm.py`, `test/server_sd.py`, ...): register the backend in `test/utils/capabilities.py` and `test/utils/test_models.py` so `--wrapped-server` / `--backend` cover it, and add a CI matrix row exercising it. See [adding a backend](./adding-a-backend.md). |
| Adds a backend with a new modality (new endpoints) | New `test/server_<modality>.py` modeled on `test/server_sd.py`; register it in `test/utils/capabilities.py` and `test/utils/test_models.py`; wire it into both the Windows and Linux test blocks of `cpp_server_build_test_release.yml` |
| Changes LLM inference (llamacpp, RyzenAI, FLM, vLLM) | `test/server_llm.py`, run per backend with `--wrapped-server` / `--backend` |
| Touches the Ollama-compatible API | `test/test_ollama.py` |
| Touches the Anthropic-compatible API | `test/test_ollama.py` (despite the name, this suite owns both the Ollama- and Anthropic-compatible API tests) |
| Touches the MCP gateway | `test/server_mcp.py` |
| Changes audio transcription | `test/server_whisper.py` or `test/server_moonshine.py` |
| Changes text-to-speech | `test/server_tts.py` or `test/server_tts_openmoss.py` |
| Changes image generation | `test/server_sd.py` — a new device is a flag on this suite, not a new file |
| Changes the router or routing policies | `test/cpp/test_routing_*.cpp` and `test/server_router.py`; keep `test/test_schema_lock.py` and `test/test_routing_fixtures.py` green |
| Changes the jobs engine | `test/cpp/test_job_*.cpp` and `test/server_jobs.py` |
| Changes WebSocket / Realtime behavior | `test/test_websocket_idle.py`, `test/server_websocket_auth.py` |
| Changes streaming error handling | `test/server_streaming_errors.py` |
| Changes model downloads or registry search | `test/server_downloads.py` |
| Changes API key authentication | `test/server_cli_apikey.py`, `test/server_websocket_auth.py` |
| Adds pure C++ logic (parsers, arg resolvers, utilities) | `test/cpp/test_<thing>.cpp`, declared via `add_cpp_ci_test()` in `CMakeLists.txt` |
| Changes `server_models.json` | Update `test/utils/test_models.py` and `test/utils/capabilities.py` if tests reference the affected models |
| Changes the desktop or web UI | `npm run typecheck` must pass; add a `test/app/app-regression/*.test.cjs` regression test where practical |
| Fixes a bug | A numbered regression test in whichever suite above owns the surface |
| Changes a persisted JSON format | A schema-version assertion in the owning suite, so accidental format bumps are caught |
| Docs only | No tests; `markdown-link-check` must pass |

---

## Running Tests Locally

Most Python suites expect a server already running on port `13305` (override with `LEMONADE_TEST_PORT`); they do not start one. That includes `test/server_cli2.py`, which fails fast in `setUpClass` when no server is reachable. Exceptions like `test/server_jobs.py` launch their own `lemond` from the build directory.

```bash
pip install -r test/requirements.txt
python test/server_endpoints.py
python test/server_cli2.py
python test/server_llm.py --wrapped-server llamacpp --backend vulkan
```

The `lemonade` CLI binary is auto-discovered from your CMake build directory; override with `--cli-binary`.

C++ unit tests — configure the build first (`./setup.sh` on Linux/macOS, `./setup.ps1` on Windows) if you haven't already.

Linux / macOS:

```bash
cmake --build --preset default --target cpp-ci-tests
ctest --test-dir build -L "^cpp-ci$" --output-on-failure
```

Windows (generator-independent, so it works whether `setup.ps1` configured the `windows` (VS 2022) or `vs18` (VS 2026) preset; Visual Studio builds are multi-config, so `ctest` needs `-C Release`):

```powershell
cmake --build build --config Release --target cpp-ci-tests
ctest --test-dir build -C Release -L "^cpp-ci$" --output-on-failure
```

App typecheck and regression tests:

```bash
cd src/app && npm ci && npm run typecheck
cd ../..
node test/app/run-app-regression-tests.cjs
```

Routing schema checks (pure Python, no server needed):

```bash
python test/test_routing_fixtures.py
python test/test_schema_lock.py
```

---

## Writing a Good Test

- Server integration suites (`test/server_endpoints.py`, `test/server_llm.py`, and most other `test/server_*.py` files) extend `ServerTestBase` (`test/utils/server_base.py`) and end with `run_server_tests(...)`. Suites that manage their own `lemond` processes (`test/server_jobs.py`), suites that drive the CLI against a persistent external server (`test/server_cli2.py`), and pure Python unit, fixture, and schema tests (`test/test_routing_fixtures.py`, `test/test_schema_lock.py`) are plain `unittest.TestCase` classes.
- Test methods are numbered to enforce order: `test_020_...`, with letter suffixes to insert between existing numbers (`test_021a_...`). Follow the natural sequence of the suite.
- Use the real client SDKs (`openai`, `ollama`) rather than raw HTTP where a suite is proving API compatibility.
- Gate tests on declared server capabilities with `@skip_if_unsupported` from `test/utils/capabilities.py`; it skips based on what the configured `--wrapped-server` / `--backend` reports supporting, not on detected hardware. Use `@requires_backend(...)` to gate on a specific backend. In practice, capability- and backend-gated tests execute on the self-hosted hardware runners and skip elsewhere.
- Mock external services in-process (for example, the mock cloud provider) so tests run in CI without secrets or network dependencies.
- Clean up after yourself: restore environment variables with `self.addCleanup(...)`, terminate any subprocess the test starts, and never leave a server running on a hardcoded port.

---

## CI Expectations

Every PR runs the C++ `cpp-ci` tests, the endpoint/CLI suites on Windows and Linux, routing schema checks, app typecheck and regression tests, and the docs drift and link checks. Inference suites run on self-hosted AMD hardware runners ([details](./self-hosted-runners.md)); [What defers to the merge queue](#what-defers-to-the-merge-queue) covers when they run.

- Relevant local tests should pass before requesting review. All required CI must be green before final approval and merge.
- Claiming a failure is a pre-existing flake requires evidence: link a `main` run with the identical failure signature. Fix flaky tests at the root cause; don't widen thresholds or add retries.
- The PR description states how the change was tested and which platforms you could not cover. Ask in the [Discord](https://discord.gg/5xXzkMu8Zk) for help testing on hardware you don't have.
- A silently-skipped test is a bug: if your change should be exercised by an existing CI job, confirm the job actually ran it rather than skipping.

### What defers to the merge queue

Packaging, distro, PPA, backend-validation, self-hosted inference and most macOS jobs do **not** run on PR pushes. They run in the merge queue, so a break there blocks the merge rather than every push. Each group reports an aggregate gate signal. The llama.cpp custom checks are diagnostic only; the trusted wrapper workflow is their enforcement boundary.

| Group | Gate signal | Opt in on a PR with |
|---|---|---|
| Fedora RPM, Debian 13, Arch, openSUSE, Launchpad PPA, `Build Lemonade Desktop Installer` | `Packaging builds`, `Linux distro builds`, `Launchpad PPA builds` | `ci:distros` |
| macOS `.dmg`, `Test CLI/Endpoints (macos-latest)`, `Test Embeddable (macOS)`, `Test .dmg - macOS inference` | `macOS builds` | `ci:macos` |
| llama.cpp, vLLM, stable-diffusion.cpp validation | Required workflow `Validate llama.cpp protected change`; `vLLM validation`; `stable-diffusion.cpp validation` | `ci:upgrades` |
| `Test .exe - *` and `Test .deb - *` inference suites on the self-hosted rigs | `Inference backend tests` | `ci:backends` |

The base-branch llama.cpp wrapper reports `llama.cpp authorized validation v2` for required default-event and merge-queue decisions, and `llama.cpp authorization diagnostics v2` for label and lifecycle metadata. Each check targets GitHub's synthetic test-merge commit when one is available and falls back to the immutable current head while GitHub is still computing mergeability or the PR conflicts; protected validation never executes without a test-merge commit. Each provides exact-candidate diagnostics and stale-run reconciliation, while the wrapper's own conclusion fails whenever policy resolution, authorization, validation, or reconciliation fails. Changes confined to `docs/`, `src/app/`, or `src/web-app/` receive a neutral success; `src/app/src/renderer/utils/toolDefinitions.json` is an exception because it is copied into the server resources. Every other path is treated as protected, including unknown future paths, because the validation builds or executes candidate-controlled inputs. Protected changes use the authorization and rerun procedure below. Shared CMake or runtime changes may need both `ci:upgrades` and the other matching surface label.

Dependabot-authored `pull_request_target` runs receive a read-only `GITHUB_TOKEN`, including when a maintainer re-runs them. These runs rely solely on the organization required-workflow conclusion and intentionally omit custom diagnostic Checks API runs. Authorization-required and awaiting-rerun states still fail the required workflow, while an authorized same-run rerun can pass.

Do not require `llama.cpp validation`; that is the reusable workflow's internal aggregate job and does not prove that a PR was authorized. Do **not** require `llama.cpp authorized validation v2` during bootstrap, and do not require `llama.cpp authorization diagnostics v2` either. Never configure `llama.cpp authorized validation v2` as a required status check, including with **GitHub Actions** selected as its expected source: the shared `github-actions` App can also report a same-name check from candidate-controlled workflow code, so its App identity does not identify the workflow or event that produced the result. Both custom checks are diagnostic only.

The validation-infrastructure change that introduces the trusted wrapper and policy helper must land on the default branch as a separate bootstrap merge before an organization required-workflow ruleset is created or activated; a rule cannot enforce a workflow that is not yet on the default branch. Enforcement therefore starts with later protected pull requests, not with the bootstrap pull request itself. Before relying on that enforcement, create the `ci:upgrades` label and provision ephemeral protected self-hosted runners that are discarded after every job.

The current `.github/CODEOWNERS` covers only agent configuration files. It does not cover the trusted llama.cpp workflows, policy and publication scripts, validation runner, validator, or trusted test helpers and fixtures. A maintainer must add explicit entries for every trusted validation and publication surface, using real eligible owners, and a repository administrator must enforce code-owner review on the protected branch. Do not invent an owner in repository code. Missing exact CODEOWNERS coverage or missing enforced code-owner review is an external activation **NO-GO**.

Scheduled publication has two additional external activation blockers. Add the repository secret `LLAMACPP_UPDATE_TOKEN` containing a separately provisioned, repository-scoped fine-grained PAT with **Contents: read and write** and **Pull requests: read and write**. Add the repository variable `LLAMACPP_UPDATE_ACTOR` containing that PAT's exact `viewer.login`. The current workflow does not support storing a GitHub App installation token in this secret: installation tokens expire after one hour, so App support would have to mint a fresh token inside each publication job. The workflow and publisher both fail before mutation when either value is absent or the authenticated actor differs. They use only this credential for checkout, release and pull-request APIs, push, pull-request creation, and draft promotion. Never substitute the repository's `GITHUB_TOKEN` or enable **Allow GitHub Actions to create and approve pull requests** as a workaround: that token does not trigger the required workflow for the pull request it creates. The publisher deliberately creates a draft first so its `opened` event records the expected initial failed required-workflow run, then marks it ready for a maintainer to authorize with the procedure below. Provisioning and rotation of the dedicated credential cannot be completed by repository code, so missing secret or variable configuration is an activation **NO-GO**.

The validation runner uses Linux child-subreaper adoption plus process groups. On Windows it starts a trusted gate wrapper, assigns the wrapper to a kill-on-close Job Object before releasing candidate execution, and relies on Job inheritance for candidate descendants. macOS has no equivalent kernel API that can retain a fast double-fork after it creates a new session, so Darwin validation fails closed by default. The trusted core workflow may pass `--allow-darwin-github-hosted-ephemeral-runner` only for its static `macos-latest` Metal lane when GitHub reports `runner.environment == 'github-hosted'`; for that lane, disposal of the GitHub-hosted VM is the final containment boundary. Ephemeral workers remain part of the Windows trust boundary because process containment is not an adversarial sandbox.

For protected validation, the base-branch checkout supplies the setup and cleanup actions, Python runner, validator, fixtures, and every imported `test.utils` helper. The candidate supplies only its build inputs, built `lemond` and adjacent build resources (including `defaults.json`), plus its base model catalog before the trusted overlay is applied. This prevents a pull request from substituting validation code, but it does **not** make candidate native code an adversarial sandbox: candidate build scripts and `lemond` execute as the runner's OS user and can access anything that user can. Jobs that execute protected candidate code must therefore use disposable GitHub-hosted workers or single-job ephemeral protected self-hosted workers, expose no repository secrets, and grant only a read-only repository token. The protected GitHub-hosted `macos-latest` lane receives no Hugging Face credential and runs no artifact upload, cache upload, or other token-bearing step after candidate runtime begins.

Before K2 promotion, all three pinned release repositories must provide immutable K2-capable backend releases. The Lemonade-controlled publishers must enable release immutability. Because the current `ggml-org/llama.cpp` b-release history is not immutable, upstream must enable immutable releases or Lemonade must use a controlled immutable mirror and revise the trust contract accordingly. The `lemonade-sdk/llama.cpp` publisher must verify its complete CUDA matrix before making a release immutable, including all three `sm_121` artifacts: Windows x64 plus Ubuntu x64 and arm64.

The `lemonade-sdk/llamacpp-rocm` publisher must emit the full advertised nightly matrix and immutable source sidecars that bind each exact asset to its concrete ROCm build targets; the `gfx103X` assets must cover `gfx1033`, `gfx1035`, and `gfx1036`. In particular, it currently needs Windows `gfx1152` plus Ubuntu `gfx1152` and `gfx942` artifacts. The scheduled updater intentionally keeps ROCm nightly ineligible while an asset or required build-target attestation is absent; do not weaken the matrix to make an incomplete release pass. Land and validate the matching managed pins plus validation-only model catalog (or temporarily phase K2 out of the gate).

Only after those prerequisites pass may an LLM360 organization owner create an organization-level branch ruleset for `LLM360/lemonade`. Target the protected `main` branch and add **Require workflows to pass before merging** with source repository `LLM360/lemonade`, source branch `main`, and workflow `.github/workflows/validate_llamacpp_pr.yml`. That workflow declares the supported `pull_request_target` and `merge_group` events, and its failing reconciliation job makes a policy failure fail the required workflow itself. Do not substitute a required status-check name for this workflow-bound rule. See GitHub's documentation for [organization ruleset setup](https://docs.github.com/en/enterprise-cloud@latest/organizations/managing-organization-settings/creating-rulesets-for-repositories-in-your-organization) and [workflow-rule event semantics](https://docs.github.com/en/enterprise-cloud@latest/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets#require-workflows-to-pass-before-merging).

GitHub ignores configured activity filters for a ruleset workflow. It starts required `pull_request_target` runs only for `opened`, `synchronize`, and `reopened`. Thus, an authorization-label run cannot satisfy the workflow rule. Use this procedure for a protected PR:

1. Find the newest failed **Validate llama.cpp protected change** run that GitHub created for `opened`, `synchronize`, or `reopened`. Do not use a run that a label or metadata change created.
2. If `ci:upgrades` is present, remove it. Apply `ci:upgrades` strictly after the original attempt's recorded run start time, not its queued creation time. A label applied while the run was queued is stale. The maintainer who applies the label must have current write permission.
3. In the selected run, select **Re-run all jobs**. The same maintainer must start the rerun. Do not select **Re-run failed jobs**, because that option does not rerun the successful policy job.
4. If the head, base branch, base commit, merge commit, or authorization actor changes, or a newer `opened`, `synchronize`, or `reopened` run starts, stop. Use the newest failed default-event run and repeat the procedure.

Metadata diagnostic runs do not supersede the selected required default-event run. `llama.cpp authorization diagnostics v2` uses a distinct visible name, so a red metadata result does not conflict with a recovered required result after the label is reapplied and the selected run is rerun. The `unlabeled` and `labeled` runs produced by step 2 are expected. If authorization is removed later, apply a fresh label and rerun the same selected default-event run, provided no newer default-event run exists. A removal during validation makes that attempt fail its live authorization recheck.

Removal after a completed green rerun is diagnostic only: because the workflow rule ignores `unlabeled`, that metadata run may not revoke the already-green organization workflow result. The authorization epoch applies only to reruns of one exact default-event run and candidate while the live identity remains current; do not describe it as durable merge state. Activation is **NO-GO** unless Evaluate mode proves that post-success removal revokes the required result, or an exclusive GitHub App or equivalent separately trusted status primitive is provisioned to enforce revocation.

A base-branch retarget starts an `edited` run, which the workflow rule cannot require. After a retarget, push a new commit or close and reopen the PR to create a supported default-event run. Then, apply a new label and rerun all jobs on that run. The policy rejects attempt 1, a rerun of a label event, a label at or before the original run start time, a rerun by a different actor, and an older run after an identity or lifecycle change.

The synthetic merge SHA proves one exact base-plus-head candidate. Before activation, require a merge queue, strict up-to-date enforcement, or a live-proven equivalent so the exact validated base-plus-head candidate is the candidate that will eventually merge. In Evaluate mode, test what happens when the default branch advances after a green rerun: the PR must become unmergeable until the new candidate passes a fresh required workflow, or a mandatory merge queue must validate the eventual merge-group candidate. Activation is **NO-GO** if a prior green result can carry over to a different base-plus-head candidate on the ordinary merge path.

After the bootstrap merge is on the default branch, an organization owner can create the rule in **Evaluate** mode. Exercise it with separate, unmerged live fork PRs before making it **Active** with no bypass for ordinary merges. The live checks must cover a neutral docs change, an unauthorized protected change, the full authorization and rerun procedure, label removal both during and after a successful run, a subsequent head update, a default-branch advance after a green rerun, a base retarget followed by a supported retrigger, and a merge-queue entry. Do not combine validation-infrastructure bootstrap and the protected feature change that depends on it into one activation step. Ruleset activation is an external deployment blocker; repository code and a repository administrator cannot complete it. If the organization-level workflow rule cannot enforce this lifecycle, the alternative is a separately provisioned GitHub App whose identity is exclusive to this gate, not the repository's ordinary `GITHUB_TOKEN` or the shared GitHub Actions expected source.

Every gated job is reachable from a PR by label — nothing is merge-queue-only. Apply the label when your change plausibly affects that surface (a backend version pin wants `ci:upgrades`, packaging or install-path changes want `ci:distros`, `#ifdef __APPLE__` or CMake changes want `ci:macos`, a wrapped-server or inference-path change wants `ci:backends`). For llama.cpp, complete the required-workflow rerun procedure after you apply `ci:upgrades`. Other surface labels take effect immediately without a push. Fork contributors cannot apply labels themselves — ask a maintainer (or the [Discord](https://discord.gg/5xXzkMu8Zk)) to add one. Note that `Test .dmg - macOS inference` exercises several of the same wrapped servers (llama.cpp, whisper.cpp, moonshine, kokoro) on Metal but lives in the macOS group — a change to one of those that could break on Metal wants `ci:macos` too. `Build Embeddable Lemonade (macOS)` still runs on every PR as the AppleClang compile check.

Two consequences worth knowing:

- **A green PR does not mean macOS, packaging or hardware inference are green.** If your change touches those surfaces, label it rather than discovering the break in the queue.
- **Adding a new suite to a gated job means it only runs in the queue by default.** Say so in the PR description.

### macOS specifics

The macOS `.pkg` suites run against an installed package, whether or not Apple signing secrets are present (`Test Embeddable (macOS)` is separate — it tests the embeddable tarball) — without them the installer is simply unsigned and notarization is skipped. There is no separate fork-PR test path, so a macOS test runs the same way everywhere. Use the `disable_macos_signing` input on a `workflow_dispatch` run to reproduce the unsigned path on demand.

---

## What Reviewers Reject

| Anti-pattern | Do instead |
|---|---|
| Reimplementing C++ logic in Python and asserting against the replica | Test the real code path, or probe the real artifact (e.g. CI 404-checks on release URLs) |
| Asserting a hardcoded list won't drift | Delete the test; validate against the live source of truth |
| Timers or sleeps as success signals | Monitor logs or track process IDs |
| Assertions coupled to model output length or wording | Assert non-empty output |
| New API surface added only for testability | Configure the model or server through existing options in the test |
| A new test file for a device variant | A flag on the existing suite |
| Committing a test no CI workflow runs | Wire it into a workflow or `add_cpp_ci_test()` in the same PR |
| Touching a merge-queue-gated surface and shipping on a green unlabeled PR | Apply the matching label from the [deferred-groups table](#what-defers-to-the-merge-queue) so the gated jobs actually run |
| Large models in CI jobs that download fresh every run | Use a sub-1 GB model and note the substitution in a comment |
| Negative tests that don't reset state (env vars, loaded models) | `self.addCleanup(...)`; verify the test still tests what it claims |
| Structural-only assertions on numeric outputs | Golden-reference comparison |
| Dismissing red CI as "flaky" without evidence | Link the identical failure on a `main` run |
