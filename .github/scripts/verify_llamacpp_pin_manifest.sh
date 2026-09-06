#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 TRUSTED_REPOSITORY CANDIDATE_REPOSITORY BASE_SHA CANDIDATE_SHA" >&2
    exit 2
fi

trusted_repository=$1
candidate_repository=$2
base_sha=$3
candidate_sha=$4
oid_pattern='^[0-9a-fA-F]{40}$'

if ! [[ "$base_sha" =~ $oid_pattern && "$candidate_sha" =~ $oid_pattern ]]; then
    echo "Protected validation commits must be full Git OIDs." >&2
    exit 1
fi
if [[ $(git -C "$trusted_repository" rev-parse HEAD) != "$base_sha" ]]; then
    echo "Trusted checkout does not match the protected base commit." >&2
    exit 1
fi
if [[ $(git -C "$candidate_repository" rev-parse HEAD) != "$candidate_sha" ]]; then
    echo "Candidate checkout does not match the protected merge commit." >&2
    exit 1
fi

script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_directory}/../.." && pwd)
manifest_tool="${repository_root}/test/utils/llamacpp_release_manifest.py"
capture_tool="${script_directory}/capture_llamacpp_release_manifest.sh"
snapshot_directory=$(mktemp -d)
base_versions="${snapshot_directory}/base-versions.json"
candidate_versions="${snapshot_directory}/candidate-versions.json"
candidate_manifest="${snapshot_directory}/candidate-manifest.json"
manifest_path=.github/llamacpp_release_manifest.json

git -C "$trusted_repository" show \
    "${base_sha}:src/cpp/resources/backend_versions.json" >"$base_versions"
git -C "$candidate_repository" show \
    "${candidate_sha}:src/cpp/resources/backend_versions.json" >"$candidate_versions"

changes=$(python "$manifest_tool" \
    --managed-pin-changes "$base_versions" "$candidate_versions")
if [[ "$changes" == '{}' ]]; then
    base_manifest_entry=$(git -C "$trusted_repository" ls-tree \
        "$base_sha" -- "$manifest_path")
    candidate_manifest_entry=$(git -C "$candidate_repository" ls-tree \
        "$candidate_sha" -- "$manifest_path")
    if [[ "$base_manifest_entry" != "$candidate_manifest_entry" ]]; then
        echo "Release manifest changes require changed managed llama.cpp pins." >&2
        exit 1
    fi
    echo "No changed managed llama.cpp pins; no release manifest is required."
    exit 0
fi

if ! git -C "$candidate_repository" show \
    "${candidate_sha}:${manifest_path}" \
    >"$candidate_manifest"; then
    echo "Changed managed llama.cpp pins require a committed release manifest." >&2
    exit 1
fi
python "$manifest_tool" \
    --require-managed-pin-manifest \
    "$base_versions" "$candidate_versions" "$candidate_manifest" >/dev/null
LLAMACPP_RELEASE_MANIFEST_TOOL="$manifest_tool" \
    bash "$capture_tool" "$candidate_manifest" "${snapshot_directory}/live" \
    >/dev/null

echo "Verified publisher source claims and asset metadata for changed managed llama.cpp pins."
