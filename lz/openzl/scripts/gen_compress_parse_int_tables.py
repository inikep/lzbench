# Copyright (c) Meta Platforms, Inc. and affiliates.


def header_guard(path):
    return path.replace("/", "_").replace(".", "_").upper()


def create_non_zero_mask_table():
    rows, cols = (21, 32)
    table = [[0 for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        leadingZeros = 32 - row
        for i in range(leadingZeros):
            table[row][i] = 0
        for i in range(leadingZeros, 32):
            table[row][i] = 0xFF
    return table


def create_addition_lut():
    rows, cols = (4, 10)
    table = [[0 for _ in range(cols)] for _ in range(rows)]
    mult = 1
    for idx in range(rows - 1, -1, -1):
        for val in range(cols):
            table[idx][val] = val * mult
        mult = mult * 10
    return table


ADDITION_LUT_TEMPLATE = """static const uint16_t  {name}[{rows}][{cols}] = {{
    {table}
}};"""

NON_ZERO_MASK_TEMPLATE = """alignas(32) static const uint8_t {name}[{rows}][{cols}] = {{
    {table}
}};"""


def hexify(num):
    assert num < 256
    h = hex(num)
    if len(h) == 3:
        return "0x0" + h[-1]
    else:
        return h


def print_addition_lut():
    return ADDITION_LUT_TEMPLATE.format(
        name="kLookup",
        rows=4,
        cols=10,
        table=",\n    ".join(
            f"{{ {', '.join(str(val) for val in row)} }}"
            for row in create_addition_lut()
        ),
    )


def print_non_zero_masks_table():
    return NON_ZERO_MASK_TEMPLATE.format(
        name="kNonZeroMask",
        rows=21,
        cols=32,
        table=",\n    ".join(
            f"{{ {', '.join(hexify(val) for val in row)} }}"
            for row in create_non_zero_mask_table()
        ),
    )


PATH = "../src/zstrong/transforms/parse_int/encode_parse_int_gen_lut.h"
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

#include <stdalign.h>
#include "zstrong/shared/portability.h"

ZL_BEGIN_C_DECLS

// clang-format off
{print_addition_lut()}

{print_non_zero_masks_table()}
// clang-format on

ZL_END_C_DECLS

#endif // {HEADER_GUARD_MACRO}
"""

with open(PATH, "w") as f:
    f.write(TEMPLATE)

print(TEMPLATE)
