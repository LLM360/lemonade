#!/usr/bin/env bash

set -euo pipefail

release=${1:?release is required}

if git describe --tags --abbrev=0 >/dev/null 2>&1; then
    version=$(git describe --tags --always)
else
    if [[ ! -f CMakeLists.txt ]]; then
        echo "Could not extract version from CMakeLists.txt" >&2
        exit 1
    fi

    version=$(sed -n \
        's/^project(lemon_cpp VERSION \([0-9]*\.[0-9]*\.[0-9]*\).*/\1/p' \
        CMakeLists.txt)
    if [[ -z "$version" ]]; then
        echo "Could not extract version from CMakeLists.txt" >&2
        exit 1
    fi

    commit_count=$(git rev-list --count HEAD)
    version="${version}+git${commit_count}.g$(git rev-parse --short HEAD)"
fi

debian_version=${version#v}
dpkg --validate-version "${debian_version}~${release}"
printf '%s\n' "$debian_version"
