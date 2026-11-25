// misa77 - A codec optimized for decompression throughput
// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Shreyas Ghildiyal <nonadhocproblems@gmail.com>

#define KERNEL_COMMANDS misa77_entropy_safe_kernel_neon
#define KERNEL_HUFFMAN misa77_entropy_huffman_kernel_neon
#define KERNEL_DELTA misa77_entropy_delta_kernel_neon
#define KERNEL_MERGE misa77_entropy_merge_kernel_neon
#include "aarch64.inc"
