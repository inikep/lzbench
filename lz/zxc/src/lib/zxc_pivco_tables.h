/*
 * ZXC - High-performance lossless compression
 *
 * Copyright (c) 2025-2026 Bertrand Lebonnois and contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file zxc_pivco_tables.h
 * @brief Shuffle-index tables for the PivCo merge routines (declarations).
 *
 * The definitions live in zxc_pivco_tables.c, compiled ONCE outside the
 * per-architecture function-multiversioning variants: the tables are pure
 * data, so duplicating them per variant would only bloat the binary
 * (22 KB x number of variants).
 *
 * For a control byte b (bits LSB-first; bit==0 -> next element of L, bit==1 ->
 * next of R), over a 32-lane view {L[0..15], R[0..15]} (L lanes 0..15,
 * R lanes 16..31):
 *
 *  - zxc_pivco_idxa_u8[b][j]: FINAL lane for output j of the first half of a
 *    16-output merge step. Also serves the portable 8-output merge routine (a
 *    24-byte scratch with R copied at offset 16).
 *
 *  - zxc_pivco_idxb_pre[pc0][b][j]: FINAL lane for output 8+j, pre-offset for
 *    pc0 = popcount(first control byte): L entries already skip the
 *    (8 - pc0) left elements consumed by the first half, R entries the pc0
 *    right ones. Indexing the row by pc0 folds this fixup into a single
 *    table load.
 */

#ifndef ZXC_PIVCO_TABLES_H
#define ZXC_PIVCO_TABLES_H

#include <stdint.h>

/* ELF/Mach-O visibility only: Windows PE/COFF targets (MinGW/MSYS2 GCC,
 * clang-on-Windows) do not support it and warn -Wattributes, so gate on
 * !_WIN32 exactly like ZXC_NO_EXPORT in zxc_export.h. */
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32)
#define ZXC_HIDDEN __attribute__((visibility("hidden")))
#else
#define ZXC_HIDDEN
#endif

extern ZXC_HIDDEN const uint8_t zxc_pivco_idxa_u8[256][8];
extern ZXC_HIDDEN const uint8_t zxc_pivco_idxb_pre[9][256][8];

#endif /* ZXC_PIVCO_TABLES_H */
