#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.

# Note : this test only works with gcc compiler, not clang.
# If `gcc` is redirected to `clang` on running system (like OS-X), this test will fail
GCC=${GCC:-gcc}  # conditional setting of GCC (can also be set manually using environment variable)
COMPILE="$GCC -c -Wall -Wextra -Werror -I../include -I../src"

set -e # exit immediately if error

println() {
    printf '%b\n' "${*}"
}

die() {
    println "$@" 1>&2
    exit 1
}

# check that this is not run on clang, as the test would necessarily fail
$GCC -v 2>&1 | grep clang && println "this is clang compiler, aborting" && exit 0

println "basic compilation, no compile-time check, compiles successfully"
$COMPILE test_ensure.c

println "enable framework => success, as test condition is respected"
$COMPILE -O1 -DZS_ENABLE_ENSURE test_ensure.c

println "miss error checking => absence of check detected, generates a warning"
$COMPILE -O1 -DZS_ENABLE_ENSURE -DTEST_ENSURE_WILL_FAIL test_ensure.c && die "compilation should have failed" || echo "compilation failed as expected: detected absence of error checking"
