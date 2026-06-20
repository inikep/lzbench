#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.


"""
TYPE[2-bits]  = {0 = rep0, 1 = rep1, 2 = rep2, 3 = offset}
MASK[8-bits]  = [type0, type1, type2, type4]
VEC[256-bits] = [rep0, rep1, rep2, ____, off0, off1, off2, off3]

TODO: Can we load shuffle from a U64? Load -> shuffle to unpack u8 to u32 -> use it...
      Broadcast + shuffle

      We could also load from a u32 for a few more instructions

      This will require some benchmarking...
"""

# Set to true to generate 256-bit shuffle
# Otherwise we generate a 64-bit shuffle that
# needs to be unpacked to 256-bits at runtime
#
# The benefit is that we use 4x less memory
OFFSET_SHUFFLE_BITS = 64

assert OFFSET_SHUFFLE_BITS == 64 or OFFSET_SHUFFLE_BITS == 256


def shuffle32to8(shuffle32):
    assert len(shuffle32) == 8
    shuffle = []
    for x in shuffle32:
        assert x < 8
        shuffle += [x, 0, 0, 0]
    assert len(shuffle) == 32
    return shuffle


# SHUFFLE_LO = [[0] * 32 for _ in range(16)]
# SHUFFLE_HI = [[255] * 32 for _ in range(16)]
SHUFFLE = []
NUM_OFFSETS = []

for mask in range(256):
    reps = [0, 1, 2]
    offs = []
    num_offsets = 0
    for i in range(4):
        off_type = (mask >> (2 * i)) & 3
        if off_type == 0:
            off = reps[0]
            # repcodes unchanged
        if off_type == 1:
            off = reps[1]
            reps[1] = reps[0]
            reps[0] = off
        if off_type == 2:
            off = reps[2]
            reps[2] = reps[1]
            reps[1] = reps[0]
            reps[0] = off
        if off_type == 3:
            off = 4 + num_offsets
            reps[2] = reps[1]
            reps[1] = reps[0]
            reps[0] = off
            num_offsets += 1

        offs.append(off)

    assert len(reps) == 3
    assert len(offs) == 4

    shuffle = reps + [3] + offs

    if OFFSET_SHUFFLE_BITS == 256:
        shuffle = shuffle32to8(shuffle)

    SHUFFLE.append(shuffle)
    NUM_OFFSETS.append(num_offsets)


def header_guard(path):
    return path.replace("/", "_").replace(".", "_").upper()


def hexify(num):
    assert num < 256
    h = hex(num)
    if len(h) == 3:
        return "0x0" + h[-1]
    else:
        return h


def print_shuffle(shuffle):
    return f"{{ {', '.join(hexify(x) for x in shuffle)} }}"


SHUFFLES_TEMPLATE = """static uint8_t {name}[{num_shuffles}][{shuffle_len}] __attribute__((aligned({shuffle_len}))) = {{
    {shuffles}
}};"""


def print_shuffles(name, shuffles):
    return SHUFFLES_TEMPLATE.format(
        name=name,
        num_shuffles=len(shuffles),
        shuffle_len=len(shuffles[0]),
        shuffles=",\n    ".join(print_shuffle(r) for r in shuffles),
    )


TABLE_TEMPLATE = """static uint8_t {name}[{length}] = {{
    {data}
}};"""


def print_table(name, data):
    return TABLE_TEMPLATE.format(
        name=name, length=len(data), data=",\n    ".join(hexify(x) for x in data)
    )


PATH = "zstrong/transforms/lz/decode_field_lz_offset_tables.h"
HEADER_GUARD_MACRO = header_guard(PATH)

# Make a f-string so the script doesn't get marked as generated
GENERATED = "generated"

TEMPLATE = f"""// Copyright (c) Meta Platforms, Inc. and affiliates.
#ifndef {HEADER_GUARD_MACRO}
#define {HEADER_GUARD_MACRO}

/**
 * @{GENERATED} by scripts/gen_decompress_field_lz_offset_tables.py
 * Please don't modify this file directly!
 */

#include "zstrong/common/portability.h"

ZL_BEGIN_C_DECLS

#define ZL_OFFSET_SHUFFLE_BITS {OFFSET_SHUFFLE_BITS}

// clang-format off
{print_shuffles("ZS_kOffsetShuffle", SHUFFLE)}

{print_table("ZS_kNumOffsets", NUM_OFFSETS)}
// clang-format on

ZL_END_C_DECLS

#endif // {HEADER_GUARD_MACRO}
"""
# {print_shuffles("ZS_kOffsetShuffleLo", SHUFFLE_LO)}
# {print_shuffles("ZS_kOffsetShuffleHi", SHUFFLE_HI)}

with open(PATH, "w") as f:
    f.write(TEMPLATE)

print(TEMPLATE)
