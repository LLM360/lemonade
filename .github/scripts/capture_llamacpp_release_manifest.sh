#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 EXPECTED_MANIFEST_PATH SNAPSHOT_DIRECTORY" >&2
    exit 2
fi

expected_manifest_path=$1
snapshot_directory=$2
manifest_tool=${LLAMACPP_RELEASE_MANIFEST_TOOL:-test/utils/llamacpp_release_manifest.py}
trusted_source_repository=ggml-org/llama.cpp
source_manifest_max_bytes=65536
source_manifest_directory=$(mktemp -d)

mkdir -p "$snapshot_directory"

canonical_expected_path="${snapshot_directory}/expected.json"
python "$manifest_tool" \
    --materialize-manifest "$canonical_expected_path" \
    --manifest-json "$(<"$expected_manifest_path")" >/dev/null

source_repository=$(jq -er '.releases[0].source_repository' "$canonical_expected_path")
upstream_reference=$(jq -er '.releases[0].upstream_reference' "$canonical_expected_path")
upstream_reference_head=$(jq -er \
    '.releases[0].upstream_reference_head' "$canonical_expected_path")
if [[ "$source_repository" != "$trusted_source_repository" ]]; then
    echo "Manifest source repository must be ${trusted_source_repository}." >&2
    exit 1
fi

tag_for_repository() {
    jq -er --arg repository "$1" \
        '.releases | map(select(.repository == $repository))
         | if length == 1 then .[0].tag_name else error("missing release") end' \
        "$canonical_expected_path"
}

ggml_release=$(tag_for_repository ggml-org/llama.cpp)
rocm_release=$(tag_for_repository lemonade-sdk/llamacpp-rocm)
lemonade_release=$(tag_for_repository lemonade-sdk/llama.cpp)

release_pattern='^b[0-9]+$'
for release in "$ggml_release" "$rocm_release" "$lemonade_release"; do
    if [[ ! "$release" =~ $release_pattern ]]; then
        echo "Manifest release tag must match bNNNN: ${release}" >&2
        exit 1
    fi
done

source_default_branch=$(gh api "repos/${trusted_source_repository}" \
    --jq '.default_branch')
if [[ -z "$source_default_branch" ]]; then
    echo "Could not resolve the trusted source repository default branch." >&2
    exit 1
fi
trusted_upstream_reference="refs/heads/${source_default_branch}"
if [[ "$upstream_reference" != "$trusted_upstream_reference" ]]; then
    echo "Manifest upstream reference is not the trusted default branch." >&2
    exit 1
fi
live_upstream_head=$(gh api \
    "repos/${trusted_source_repository}/commits/heads/${source_default_branch}" \
    --jq '.sha')
if ! [[ "$live_upstream_head" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Could not resolve the trusted upstream reference to a full commit." >&2
    exit 1
fi

anchor_comparison_path="${snapshot_directory}/upstream-anchor-comparison.json"
gh api \
    "repos/${trusted_source_repository}/compare/${upstream_reference_head}...${live_upstream_head}" \
    >"$anchor_comparison_path"
python "$manifest_tool" \
    --require-upstream-ancestry "$anchor_comparison_path" \
    "$trusted_source_repository" "$upstream_reference_head" "$live_upstream_head"

ggml_path="${snapshot_directory}/ggml.json"
rocm_path="${snapshot_directory}/rocm.json"
lemonade_path="${snapshot_directory}/lemonade.json"

gh api "repos/ggml-org/llama.cpp/releases/tags/${ggml_release}" >"$ggml_path"
gh api "repos/lemonade-sdk/llamacpp-rocm/releases/tags/${rocm_release}" >"$rocm_path"
gh api "repos/lemonade-sdk/llama.cpp/releases/tags/${lemonade_release}" \
    >"$lemonade_path"

attach_release_claims() {
    local repository=$1
    local tag=$2
    local path=$3
    local release_tag_commit publisher_claim_type
    local publisher_claimed_source_commit comparison_path
    local source_manifest_asset_id source_manifest_path
    local build_target_attestation_path

    release_tag_commit=$(gh api \
        "repos/${repository}/commits/tags/${tag}" --jq '.sha')
    if ! [[ "$release_tag_commit" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "Could not resolve ${repository} ${tag} to a release tag commit." >&2
        exit 1
    fi
    if [[ "$repository" == "$source_repository" ]]; then
        publisher_claim_type="source-release-tag"
        publisher_claimed_source_commit=$release_tag_commit
        build_target_attestation_path="${source_manifest_directory}/source-release-targets.json"
        printf '[]\n' >"$build_target_attestation_path"
    else
        publisher_claim_type="immutable-source-manifest"
        source_manifest_asset_id=$(python "$manifest_tool" \
            --source-manifest-asset-id "$path" "$repository")
        source_manifest_path="${source_manifest_directory}/source-${source_manifest_asset_id}.json"
        build_target_attestation_path="${source_manifest_path}.build-targets"
        gh api "repos/${repository}/releases/assets/${source_manifest_asset_id}" \
            -H "Accept: application/octet-stream" |
            head -c "$((source_manifest_max_bytes + 1))" >"$source_manifest_path"
        publisher_claimed_source_commit=$(python "$manifest_tool" \
            --validate-source-manifest "$source_manifest_path" "$path" \
            "$repository" "$tag" "$release_tag_commit" \
            --build-target-attestation-output "$build_target_attestation_path")
    fi
    if ! [[ "$publisher_claimed_source_commit" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "${repository} ${tag} did not bind one full upstream source commit." >&2
        exit 1
    fi
    comparison_path="${path}.upstream-comparison.json"
    gh api \
        "repos/${source_repository}/compare/${publisher_claimed_source_commit}...${upstream_reference_head}" \
        >"$comparison_path"
    python "$manifest_tool" \
        --require-upstream-ancestry "$comparison_path" "$source_repository" \
        "$publisher_claimed_source_commit" "$upstream_reference_head"
    jq \
        --slurpfile build_target_attestations "$build_target_attestation_path" \
        --arg publisher_claim_type "$publisher_claim_type" \
        --arg publisher_claimed_source_commit "$publisher_claimed_source_commit" \
        --arg release_tag_commit "$release_tag_commit" \
        --arg source_repository "$source_repository" \
        --arg upstream_reference "$upstream_reference" \
        --arg upstream_reference_head "$upstream_reference_head" \
        '.build_target_attestations = $build_target_attestations[0]
         | .publisher_claim_type = $publisher_claim_type
         | .publisher_claimed_source_commit = $publisher_claimed_source_commit
         | .release_tag_commit = $release_tag_commit
         | .source_repository = $source_repository
         | .upstream_reference = $upstream_reference
         | .upstream_reference_head = $upstream_reference_head
         | del(.source_commit)' \
        "$path" >"${path}.with-source"
    mv "${path}.with-source" "$path"
}

attach_release_claims ggml-org/llama.cpp "$ggml_release" "$ggml_path"
attach_release_claims lemonade-sdk/llamacpp-rocm "$rocm_release" "$rocm_path"
attach_release_claims lemonade-sdk/llama.cpp "$lemonade_release" "$lemonade_path"

python "$manifest_tool" \
    --release ggml-org/llama.cpp "$ggml_release" "$ggml_path" \
    --release lemonade-sdk/llamacpp-rocm "$rocm_release" "$rocm_path" \
    --release lemonade-sdk/llama.cpp "$lemonade_release" "$lemonade_path" \
    --expected-json "$(<"$canonical_expected_path")"
