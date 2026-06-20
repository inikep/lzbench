#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.


set -euxo pipefail

ZSTRONG_ROOT="$(dirname "${BASH_SOURCE[0]}")/.."

FROM_REL=$1
TO_REL=$2

FROM_ABS="$ZSTRONG_ROOT/$FROM_REL"
TO_ABS="$ZSTRONG_ROOT/$TO_REL"

if [ -d "$TO_ABS" ]; then
    file="$(basename "$FROM_REL")"
    TO_REL="$TO_REL/$file"
    TO_ABS="$TO_ABS/$file"
fi

# If the file has already been moved, skip it
if [ -e "$FROM_ABS" ] || [ ! -f "$TO_ABS" ]; then
    if [ ! -f "$FROM_ABS" ]; then
        echo "$FROM_REL does not exist!"
        exit 1
    fi

    if [ -e "$TO_ABS" ]; then
        echo "$TO_REL already exists!"
        exit 1
    fi

    mkdir -p "$(dirname "$TO_ABS")"
    hg mv "$FROM_ABS" "$TO_ABS"
fi

find "$ZSTRONG_ROOT" -name "*.[hc]" -or -name "*.cpp" | xargs sed -i "s,\"$FROM_REL\",\"$TO_REL\",g"
