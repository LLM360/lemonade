#!/usr/bin/env bash

set -euo pipefail

versions_path=${VERSIONS_PATH:-src/cpp/resources/backend_versions.json}
manifest_path=${RELEASE_MANIFEST_PATH:-.github/llamacpp_release_manifest.json}
body_file=${PR_BODY_FILE:-pr_body.md}
base_branch=${BASE_BRANCH:-main}
repository=${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}
repository_owner=${GITHUB_REPOSITORY_OWNER:?GITHUB_REPOSITORY_OWNER is required}
validated_base_sha=${VALIDATED_BASE_SHA:?VALIDATED_BASE_SHA is required}
llamacpp_release=${LLAMACPP_RELEASE:?LLAMACPP_RELEASE is required}
lemonade_release=${LLAMACPP_LEMONADE_RELEASE:?LLAMACPP_LEMONADE_RELEASE is required}
rocm_release=${LLAMACPP_ROCM_RELEASE:?LLAMACPP_ROCM_RELEASE is required}
expected_release_manifest=${EXPECTED_RELEASE_ASSET_MANIFEST:?EXPECTED_RELEASE_ASSET_MANIFEST is required}
: "${GH_TOKEN:?GH_TOKEN must contain a separately provisioned publication token}"
publication_actor=${EXPECTED_PUBLICATION_ACTOR:?EXPECTED_PUBLICATION_ACTOR is required}

if ! authenticated_actor=$(gh api graphql \
    -f 'query=query { viewer { login } }' \
    --jq '.data.viewer.login'); then
    echo "Could not authenticate the dedicated publication token." >&2
    exit 1
fi
if [[ -z "$authenticated_actor" ||
      "$authenticated_actor" != "$publication_actor" ]]; then
    echo "The authenticated publication actor does not match EXPECTED_PUBLICATION_ACTOR." >&2
    exit 1
fi

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_directory}/../.." && pwd)
manifest_tool="${repository_root}/test/utils/llamacpp_release_manifest.py"
capture_tool="${script_directory}/capture_llamacpp_release_manifest.sh"
snapshot_root=$(mktemp -d)
git_askpass_path="${snapshot_root}/git-askpass.sh"
# The generated helper expands these variables only when Git invokes it.
# shellcheck disable=SC2016
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'case "$1" in' \
    '    *Username*) printf "%s\n" x-access-token ;;' \
    '    *Password*) printf "%s\n" "$GH_TOKEN" ;;' \
    '    *) exit 1 ;;' \
    'esac' > "$git_askpass_path"
chmod 0700 "$git_askpass_path"

manifest_digest=$(python "$manifest_tool" \
    --materialize-manifest "$manifest_path" \
    --manifest-json "$expected_release_manifest" \
    --append-summary-to "$body_file")

manifest_tag() {
    jq -er --arg repository "$1" \
        '.releases | map(select(.repository == $repository))
         | if length == 1 then .[0].tag_name else error("missing release") end' \
        "$manifest_path"
}

if [[ $(manifest_tag ggml-org/llama.cpp) != "$llamacpp_release" ||
      $(manifest_tag lemonade-sdk/llamacpp-rocm) != "$rocm_release" ||
      $(manifest_tag lemonade-sdk/llama.cpp) != "$lemonade_release" ]]; then
    echo "The release manifest does not match the validated candidate tags." >&2
    exit 1
fi

snapshot_number=0
recheck_release_manifest() {
    snapshot_number=$((snapshot_number + 1))
    LLAMACPP_RELEASE_MANIFEST_TOOL="$manifest_tool" \
        bash "$capture_tool" "$manifest_path" \
        "${snapshot_root}/snapshot-${snapshot_number}" >/dev/null
}

require_managed_pin_manifest() {
    python "$manifest_tool" \
        --require-managed-pin-manifest \
        "$base_versions_path" "$versions_path" "$manifest_path" >/dev/null
}

release_pattern='^b[0-9]+$'
for release in "$llamacpp_release" "$lemonade_release" "$rocm_release"; do
    if [[ ! "$release" =~ $release_pattern ]]; then
        echo "Invalid llama.cpp release tag: ${release}" >&2
        exit 1
    fi
done

local_base_sha=$(git rev-parse HEAD)
if [[ "$local_base_sha" != "$validated_base_sha" ]]; then
    echo "Local HEAD does not match the validated base SHA." >&2
    exit 1
fi

base_versions_path="${snapshot_root}/base-backend-versions.json"
git show "${validated_base_sha}:${versions_path}" > "$base_versions_path"
python - "$base_versions_path" "$versions_path" <<'PY'
import json
import re
import sys

managed_backends = ("cpu", "cuda", "metal", "rocm-nightly", "rocm-stable", "vulkan")
release_pattern = re.compile(r"b[0-9]+")

try:
    with open(sys.argv[1], encoding="utf-8") as base_file:
        base_versions = json.load(base_file)["llamacpp"]
    with open(sys.argv[2], encoding="utf-8") as candidate_file:
        candidate_versions = json.load(candidate_file)["llamacpp"]
except (KeyError, OSError, TypeError, ValueError) as error:
    raise SystemExit(
        "Scheduled llama.cpp update found invalid llamacpp version mappings."
    ) from error

if not isinstance(base_versions, dict) or not isinstance(candidate_versions, dict):
    raise SystemExit(
        "Scheduled llama.cpp update found invalid llamacpp version mappings."
    )

for backend in managed_backends:
    if backend not in base_versions or backend not in candidate_versions:
        raise SystemExit(
            f"Scheduled llama.cpp update changed the managed llamacpp.{backend} pin shape."
        )
    base_release = base_versions[backend]
    candidate_release = candidate_versions[backend]
    if (
        not isinstance(base_release, str)
        or release_pattern.fullmatch(base_release) is None
        or not isinstance(candidate_release, str)
        or release_pattern.fullmatch(candidate_release) is None
    ):
        raise SystemExit(
            f"Scheduled llama.cpp update found an invalid llamacpp.{backend} release tag."
        )
    if int(candidate_release[1:]) < int(base_release[1:]):
        raise SystemExit(
            f"Scheduled llama.cpp update would downgrade llamacpp.{backend} "
            f"from {base_release} to {candidate_release}."
        )
PY

if [[ $(git rev-parse --is-shallow-repository) == "true" ]]; then
    git fetch --unshallow --no-tags origin "refs/heads/${base_branch}"
else
    git fetch --no-tags origin "refs/heads/${base_branch}"
fi
fetched_base_sha=$(git rev-parse FETCH_HEAD)
if [[ "$fetched_base_sha" != "$validated_base_sha" ]]; then
    echo "The ${base_branch} branch advanced after validation; refusing publication." >&2
    exit 1
fi

assert_candidate_releases_current() {
    local fresh_llamacpp_release fresh_rocm_release fresh_lemonade_release
    fresh_llamacpp_release=$(gh api repos/ggml-org/llama.cpp/releases \
        --jq '[.[] | select(.draft | not) | .tag_name | select(test("^b[0-9]+$"))][0] // empty')
    fresh_rocm_release=$(gh api \
        repos/lemonade-sdk/llamacpp-rocm/releases/latest --jq '.tag_name')
    fresh_lemonade_release=$(gh api \
        repos/lemonade-sdk/llama.cpp/releases/latest --jq '.tag_name')

    if [[ "$fresh_llamacpp_release" != "$llamacpp_release" ||
          "$fresh_rocm_release" != "$rocm_release" ||
          "$fresh_lemonade_release" != "$lemonade_release" ]]; then
        echo "The candidate releases changed during validation; refusing publication." >&2
        exit 1
    fi
}

assert_candidate_releases_current

recheck_release_manifest

has_update=true
if git diff --quiet HEAD -- "$versions_path"; then
    has_update=false
    branch=""
else
    versions_blob=$(git hash-object "$versions_path")
    branch="auto/llamacpp-update-${llamacpp_release}-${lemonade_release}-${rocm_release}-${validated_base_sha}-${versions_blob}-${manifest_digest}"
fi

open_pr_records=$(gh api "repos/${repository}/pulls" \
    --method GET \
    -f state=open \
    -f per_page=100 \
    --paginate \
    --jq '.[] | select(.head.ref | startswith("auto/llamacpp-update-")) | [.number, .head.ref, (.head.repo.full_name // "<deleted>"), .user.login, .base.ref] | @tsv')

open_pr=""
stale_prs=()
while IFS=$'\t' read -r number head_ref head_repository author base_ref; do
    if [[ -z "$number" ]]; then
        continue
    fi

    trusted=false
    if [[ "$head_repository" == "$repository" &&
          "$author" == "$publication_actor" &&
          "$base_ref" == "$base_branch" ]]; then
        trusted=true
    fi

    if [[ "$has_update" == "true" &&
          "$head_ref" == "$branch" &&
          "$head_repository" == "$repository" ]]; then
        if [[ "$trusted" != "true" ]]; then
            echo "Found an untrusted open pull request for ${repository_owner}:${branch}." >&2
            exit 1
        fi
        if [[ -n "$open_pr" ]]; then
            echo "Multiple open pull requests use ${repository_owner}:${branch}; refusing publication." >&2
            exit 1
        fi
        open_pr=$number
    elif [[ "$trusted" == "true" ]]; then
        stale_prs+=("${number}"$'\t'"${head_ref}")
    fi
done <<< "$open_pr_records"

assert_base_is_current() {
    local remote_base_sha
    remote_base_sha=$(git ls-remote --heads origin "refs/heads/${base_branch}" |
        awk 'NR == 1 { print $1 }')
    if [[ -z "$remote_base_sha" || "$remote_base_sha" != "$validated_base_sha" ]]; then
        echo "The ${base_branch} branch advanced after validation; refusing publication." >&2
        exit 1
    fi
}

assert_publication_branch() {
    local expected_oid=$1
    local observed_oid
    observed_oid=$(git ls-remote --heads origin "refs/heads/${branch}" |
        awk 'NR == 1 { print $1 }')
    if [[ -z "$observed_oid" || "$observed_oid" != "$expected_oid" ]]; then
        echo "The publication branch changed after push." >&2
        exit 1
    fi
}

require_pull_request_identity() {
    local number=$1
    local expected_head_ref=$2
    local expected_head_sha=${3:-}
    local expected_state=${4:-open}
    local expected_base_sha=${5:-}
    local pull_request_json

    if ! [[ "$number" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid pull request number: ${number}" >&2
        exit 1
    fi
    pull_request_json=$(gh api "repos/${repository}/pulls/${number}")
    if ! pull_request_is_draft=$(jq -r \
        'if (.draft | type) == "boolean" then .draft else error("invalid draft") end' \
        <<< "$pull_request_json"); then
        echo "Pull request #${number} has invalid draft state." >&2
        exit 1
    fi
    if ! jq -e \
        --arg state "$expected_state" \
        --arg repository "$repository" \
        --arg head_ref "$expected_head_ref" \
        --arg head_sha "$expected_head_sha" \
        --arg actor "$publication_actor" \
        --arg base_ref "$base_branch" \
        --arg base_sha "$expected_base_sha" \
        '.state == $state and
         .head.repo.full_name == $repository and
         .head.ref == $head_ref and
         ($head_sha == "" or .head.sha == $head_sha) and
         .user.login == $actor and
         .base.ref == $base_ref and
         ($base_sha == "" or .base.sha == $base_sha)' \
        <<< "$pull_request_json" >/dev/null; then
        echo "Pull request #${number} identity changed before publication." >&2
        exit 1
    fi
}

report_stale_prs() {
    local stale_record stale_pr stale_head_ref
    for stale_record in "${stale_prs[@]}"; do
        IFS=$'\t' read -r stale_pr stale_head_ref <<< "$stale_record"
        echo "Superseded pull request #${stale_pr} (${stale_head_ref}) requires manual cleanup."
    done
}

reconcile_failed_pull_request_creation() {
    local number head_ref head_repository author base_ref head_sha records
    local matching_pr=""
    local record_count=0

    failed_create_reconciliation=unknown
    failed_create_pr=""
    if ! records=$(gh api "repos/${repository}/pulls" \
        --method GET \
        -f state=open \
        -f head="${repository_owner}:${branch}" \
        -f base="$base_branch" \
        --paginate \
        --jq '.[] | [.number, .head.ref, (.head.repo.full_name // "<deleted>"), .user.login, .base.ref, .head.sha] | @tsv'); then
        echo "Pull request creation failed, and its result could not be reconciled; preserving ${branch}." >&2
        return
    fi

    while IFS=$'\t' read -r number head_ref head_repository author base_ref head_sha; do
        if [[ -z "$number" ]]; then
            continue
        fi
        record_count=$((record_count + 1))
        if ! [[ "$number" =~ ^[1-9][0-9]*$ ]] ||
            [[ "$head_ref" != "$branch" ]] ||
            [[ "$head_repository" != "$repository" ]] ||
            [[ "$author" != "$publication_actor" ]] ||
            [[ "$base_ref" != "$base_branch" ]] ||
            [[ "$head_sha" != "$published_oid" ]]; then
            echo "Pull request creation failed, and its result was ambiguous; preserving ${branch}." >&2
            return
        fi
        matching_pr=$number
    done <<< "$records"

    if [[ "$record_count" -eq 0 ]]; then
        failed_create_reconciliation=absent
    elif [[ "$record_count" -eq 1 ]]; then
        failed_create_reconciliation=adopted
        failed_create_pr=$matching_pr
    else
        echo "Pull request creation failed, and multiple matching pull requests were found; preserving ${branch}." >&2
    fi
}

if [[ "$has_update" != "true" ]]; then
    assert_base_is_current
    recheck_release_manifest
    assert_candidate_releases_current
    report_stale_prs
    echo "backend_versions.json is unchanged; nothing to update."
    exit 0
fi

assert_base_is_current
title="Update llama.cpp to ${llamacpp_release}"
remote_oid=$(git ls-remote --heads origin "refs/heads/$branch" | awk 'NR == 1 { print $1 }')
remote_tree=""
if [[ -n "$remote_oid" ]]; then
    git fetch --no-tags origin "refs/heads/$branch"
    fetched_oid=$(git rev-parse FETCH_HEAD)
    if [[ "$fetched_oid" != "$remote_oid" ]]; then
        echo "The publication branch changed while it was inspected." >&2
        exit 1
    fi
    remote_tree=$(git rev-parse "${remote_oid}^{tree}")
    if ! merge_base=$(git merge-base "$validated_base_sha" "$remote_oid"); then
        echo "Publication branch ${branch} has unrelated history." >&2
        exit 1
    fi
    if [[ "$merge_base" != "$validated_base_sha" ]]; then
        echo "Publication branch ${branch} is not based on the validated commit." >&2
        exit 1
    fi
    unexpected_path=$(git diff --name-only "$merge_base" "$remote_oid" -- |
        awk -v versions="$versions_path" -v manifest="$manifest_path" \
            '$0 != versions && $0 != manifest { print; exit }')
    if [[ -n "$unexpected_path" ]]; then
        echo "Publication branch ${branch} changes unexpected path ${unexpected_path}." >&2
        exit 1
    fi
elif [[ -n "$open_pr" ]]; then
    echo "Trusted pull request #${open_pr} has no publication branch." >&2
    exit 1
fi

git config user.name "$publication_actor"
git config user.email "${publication_actor}@users.noreply.github.com"
git checkout -b "$branch"
recheck_release_manifest
git add "$versions_path" "$manifest_path"
git commit \
    -m "Update llama.cpp to ${llamacpp_release}, rocm-stable to ${lemonade_release}, rocm-nightly to ${rocm_release}" \
    -m "LlamaCpp-Release-Manifest-SHA256: ${manifest_digest}"
desired_tree=$(git rev-parse 'HEAD^{tree}')

if [[ -n "$remote_oid" && -z "$open_pr" && "$remote_tree" != "$desired_tree" ]]; then
    echo "Existing publication branch does not match the desired update." >&2
    exit 1
fi
if [[ -n "$remote_oid" && -n "$open_pr" && "$remote_tree" != "$desired_tree" ]]; then
    echo "Trusted pull request #${open_pr} does not match the desired update." >&2
    exit 1
fi

assert_base_is_current
recheck_release_manifest
require_managed_pin_manifest
if [[ -n "$remote_oid" ]]; then
    lease="refs/heads/${branch}:${remote_oid}"
    push_source="$remote_oid"
    published_oid=$remote_oid
else
    lease="refs/heads/${branch}:"
    push_source=HEAD
    published_oid=$(git rev-parse HEAD)
fi
assert_candidate_releases_current
GIT_ASKPASS="$git_askpass_path" \
    GIT_TERMINAL_PROMPT=0 \
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0=credential.helper \
    GIT_CONFIG_VALUE_0='' \
    git push --force-with-lease="$lease" origin \
    "${push_source}:refs/heads/$branch"

assert_base_is_current
assert_publication_branch "$published_oid"
if [[ -n "$open_pr" ]]; then
    require_pull_request_identity \
        "$open_pr" "$branch" "$published_oid" open "$validated_base_sha"
    gh pr edit "$open_pr" \
        --repo "$repository" \
        --title "$title" \
        --body-file "$body_file"
    current_pr=$open_pr
    echo "Refreshed pull request #${open_pr}."
else
    if ! created_pr_url=$(gh pr create \
        --repo "$repository" \
        --title "$title" \
        --body-file "$body_file" \
        --base "$base_branch" \
        --head "$branch" \
        --draft); then
        assert_candidate_releases_current
        reconcile_failed_pull_request_creation
        if [[ "$failed_create_reconciliation" == "adopted" ]]; then
            current_pr=$failed_create_pr
            echo "Recovered pull request #${current_pr} after an ambiguous creation result."
        else
            if [[ "$failed_create_reconciliation" == "absent" ]]; then
                echo "Pull request creation failed; preserving ${branch}." >&2
            fi
            exit 1
        fi
    else
        created_pr_url=${created_pr_url%/}
        current_pr=${created_pr_url##*/}
        created_draft_pr=true
    fi
fi

require_pull_request_identity \
    "$current_pr" "$branch" "$published_oid" open "$validated_base_sha"
if [[ "${created_draft_pr:-false}" == "true" &&
      "$pull_request_is_draft" != "true" ]]; then
    echo "New pull request #${current_pr} was not created as a draft." >&2
    exit 1
fi
assert_base_is_current
assert_publication_branch "$published_oid"
recheck_release_manifest
assert_candidate_releases_current
assert_base_is_current
assert_publication_branch "$published_oid"
require_pull_request_identity \
    "$current_pr" "$branch" "$published_oid" open "$validated_base_sha"
if [[ "$pull_request_is_draft" == "true" ]]; then
    gh pr ready "$current_pr" --repo "$repository"
    echo "Marked pull request #${current_pr} ready for review."
fi
require_pull_request_identity \
    "$current_pr" "$branch" "$published_oid" open "$validated_base_sha"
if [[ "$pull_request_is_draft" != "false" ]]; then
    echo "Pull request #${current_pr} remained a draft after promotion." >&2
    exit 1
fi
report_stale_prs
