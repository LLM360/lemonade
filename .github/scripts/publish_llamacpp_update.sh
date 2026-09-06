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

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_directory}/../.." && pwd)
manifest_tool="${repository_root}/test/utils/llamacpp_release_manifest.py"
capture_tool="${script_directory}/capture_llamacpp_release_manifest.sh"
snapshot_root=$(mktemp -d)

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
publication_actor='github-actions[bot]'
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

close_stale_prs() {
    local stale_record stale_pr stale_head_ref
    for stale_record in "${stale_prs[@]}"; do
        IFS=$'\t' read -r stale_pr stale_head_ref <<< "$stale_record"
        require_pull_request_identity "$stale_pr" "$stale_head_ref"
        gh api "repos/${repository}/pulls/${stale_pr}" \
            --method PATCH \
            -f state=closed >/dev/null
        require_pull_request_identity "$stale_pr" "$stale_head_ref" "" closed
        echo "Closed superseded pull request #${stale_pr}."
    done
}

if [[ "$has_update" != "true" ]]; then
    assert_base_is_current
    recheck_release_manifest
    close_stale_prs
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

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
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
if [[ -n "$remote_oid" ]]; then
    lease="refs/heads/${branch}:${remote_oid}"
    push_source="$remote_oid"
    published_oid=$remote_oid
else
    lease="refs/heads/${branch}:"
    push_source=HEAD
    published_oid=$(git rev-parse HEAD)
fi
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
    created_pr_url=$(gh pr create \
        --repo "$repository" \
        --title "$title" \
        --body-file "$body_file" \
        --base "$base_branch" \
        --head "$branch")
    created_pr_url=${created_pr_url%/}
    current_pr=${created_pr_url##*/}
fi

require_pull_request_identity \
    "$current_pr" "$branch" "$published_oid" open "$validated_base_sha"
close_stale_prs
assert_base_is_current
assert_publication_branch "$published_oid"
require_pull_request_identity \
    "$current_pr" "$branch" "$published_oid" open "$validated_base_sha"
