#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.


import os
import subprocess

MOVE_FILE = os.path.join(os.path.dirname(__file__), "move_file.sh")

COMMON_SRCS = [
    ("zstrong/common/detail/pdqsort-inl.h", "zstrong/shared/detail/pdqsort-inl.h"),
    ("zstrong/common/detail/pdqsort1.c", "zstrong/shared/detail/pdqsort1.c"),
    ("zstrong/common/detail/pdqsort2.c", "zstrong/shared/detail/pdqsort2.c"),
    ("zstrong/common/detail/pdqsort4.c", "zstrong/shared/detail/pdqsort4.c"),
    ("zstrong/common/detail/pdqsort8.c", "zstrong/shared/detail/pdqsort8.c"),
    "zstrong/common/portability.h",
    "zstrong/common/bits.h",
    "zstrong/common/clustering.h",
    "zstrong/common/clustering_common.c",
    ("zstrong/common/copy.h", "zstrong/transforms/common/copy.h"),
    ("zstrong/common/count.h", "zstrong/transforms/common/count.h"),
    ("zstrong/common/window.h", "zstrong/transforms/common/window.h"),
    ("zstrong/common/window.c", "zstrong/transforms/common/window.c"),
    (
        "zstrong/common/bitstream/bf_bitstream.h",
        "zstrong/transforms/common/bitstream/bf_bitstream.h",
    ),
    (
        "zstrong/common/bitstream/ff_bitstream.h",
        "zstrong/transforms/common/bitstream/ff_bitstream.h",
    ),
    "zstrong/common/cpu.h",
    "zstrong/common/data_stats.c",
    "zstrong/common/data_stats.h",
    "zstrong/common/hash.h",
    "zstrong/common/histogram.c",
    "zstrong/common/histogram.h",
    "zstrong/common/mem.h",
    "zstrong/common/numeric_operations.c",
    "zstrong/common/numeric_operations.h",
    "zstrong/common/overflow.h",
    "zstrong/common/pdqsort.h",
    "zstrong/common/utils.h",
    "zstrong/common/varint.h",
    "zstrong/common/xxhash.c",
    "zstrong/common/xxhash.h",
    "zstrong/common/zs_xxhash.h",
]

COMPRESS_SRCS = [
    "zstrong/compress/clustering_compress.c",
    "zstrong/compress/estimate.c",
    "zstrong/compress/estimate.h",
    (
        "zstrong/compress/match_finder/fast_table.c",
        "zstrong/transforms/common/fast_table.c",
    ),
    (
        "zstrong/compress/match_finder/fast_table.h",
        "zstrong/transforms/common/fast_table.h",
    ),
    (
        "zstrong/compress/match_finder/row_table.h",
        "zstrong/transforms/common/row_table.h",
    ),
    "zstrong/compress/match_finder/simd_wrapper.h",
]


SRCS = COMMON_SRCS + COMPRESS_SRCS

for src in SRCS:
    if isinstance(src, str):
        dst = os.path.join("zstrong/shared", os.path.basename(src))
    else:
        src, dst = src

    print(f"{src} -> {dst}")
    subprocess.check_call([MOVE_FILE, src, dst], stderr=subprocess.DEVNULL)
