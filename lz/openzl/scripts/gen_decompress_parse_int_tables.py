# Copyright (c) Meta Platforms, Inc. and affiliates.

import math

INT64_MAX = 9223372036854775807


def header_guard(path):
    return path.replace("/", "_").replace(".", "_").upper()


def print_character(num):
    return "'" + str(num) + ("'")


def format_large_number(num):
    if num > INT64_MAX:
        return str(num) + "u"
    else:
        return str(num)


def create_ten_power_table():
    cols = 64
    table = [0 for _ in range(cols)]
    for i in range(cols - 1):
        table[i] = 10 ** math.floor(math.log10((1 << (64 - i)) - 1))
    table[cols - 1] = 0
    return table


def create_charater_table():
    cols = 40000
    table = [0 for _ in range(cols)]
    for i in range(10000):
        x = i
        for j in range(3, -1, -1):
            table[i * 4 + j] = x % 10
            x //= 10
    return table


CHARACTER_TABLE_TEMPLATE = """static const char  {name}[{cols}] = {{
 {table}
}};"""

TEN_POWERS_TEMPLATE = """static const uint64_t {name}[{cols}] = {{
 {table}
}};"""


def print_charater_table():
    table = create_charater_table()
    return CHARACTER_TABLE_TEMPLATE.format(
        name="characterTable",
        cols=40000,
        table=",\n ".join(
            f"{', '.join(print_character(table[i * 25 + j]) for j in range(25))}"
            for i in range(1600)
        ),
    )


def print_ten_power_table():
    return TEN_POWERS_TEMPLATE.format(
        name="tenPower",
        cols=64,
        table=",\n ".join(format_large_number(val) for val in create_ten_power_table()),
    )


PATH = "../src/zstrong/transforms/parse_int/decode_parse_int_gen_lut.h"
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

#include "zstrong/shared/portability.h"

ZL_BEGIN_C_DECLS

// clang-format off
{print_ten_power_table()}

{print_charater_table()}
// clang-format on

ZL_END_C_DECLS

#endif // {HEADER_GUARD_MACRO}
"""

with open(PATH, "w") as f:
    f.write(TEMPLATE)

print(TEMPLATE)
