#!/usr/bin/env bash

set -euo pipefail

versions_path=${VERSIONS_PATH:-src/cpp/resources/backend_versions.json}
body_file=${PR_BODY_FILE:-pr_body.md}
base_branch=${BASE_BRANCH:-main}
repository=${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}
repository_owner=${GITHUB_REPOSITORY_OWNER:?GITHUB_REPOSITORY_OWNER is required}
validated_base_sha=${VALIDATED_BASE_SHA:?VALIDATED_BASE_SHA is required}
llamacpp_release=${LLAMACPP_RELEASE:?LLAMACPP_RELEASE is required}
lemonade_release=${LLAMACPP_LEMONADE_RELEASE:?LLAMACPP_LEMONADE_RELEASE is required}
rocm_release=${LLAMACPP_ROCM_RELEASE:?LLAMACPP_ROCM_RELEASE is required}

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

has_update=true
if git diff --quiet HEAD -- "$versions_path"; then
    has_update=false
    branch=""
else
    versions_blob=$(git hash-object "$versions_path")
    branch="auto/llamacpp-update-${llamacpp_release}-${lemonade_release}-${rocm_release}-${validated_base_sha}-${versions_blob}"
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
          "$author" == "github-actions[bot]" &&
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
        stale_prs+=("$number")
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

assert_base_is_current
for stale_pr in "${stale_prs[@]}"; do
    gh api "repos/${repository}/pulls/${stale_pr}" \
        --method PATCH \
        -f state=closed >/dev/null
    echo "Closed superseded pull request #${stale_pr}."
done

if [[ "$has_update" != "true" ]]; then
    echo "backend_versions.json is unchanged; nothing to update."
    exit 0
fi

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
        awk -v allowed="$versions_path" '$0 != allowed { print; exit }')
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
git add "$versions_path"
git commit -m \
    "Update llama.cpp to ${llamacpp_release}, rocm-stable to ${lemonade_release}, rocm-nightly to ${rocm_release}"
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
if [[ -n "$remote_oid" ]]; then
    lease="refs/heads/${branch}:${remote_oid}"
    push_source="$remote_oid"
else
    lease="refs/heads/${branch}:"
    push_source=HEAD
fi
git push --force-with-lease="$lease" origin \
    "${push_source}:refs/heads/$branch"

if [[ -n "$open_pr" ]]; then
    gh pr edit "$open_pr" \
        --repo "$repository" \
        --title "$title" \
        --body-file "$body_file"
    echo "Refreshed pull request #${open_pr}."
else
    gh pr create \
        --repo "$repository" \
        --title "$title" \
        --body-file "$body_file" \
        --base "$base_branch" \
        --head "$branch"
fi
