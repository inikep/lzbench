#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.


import random
import struct
import sys


def generate_time_series_data(
    num_elts: int, max_delta: int, max_value: int
) -> list[int]:
    """
    Generates time series data starting from 0, with the maximum time difference
    specificed by 'max_delta'. The time series is returned modulo 'max_value'
    which can be used to ensure values fit within the specified integer width
    of the undelying formats.
    """
    initial = 0
    values = [0] * num_elts
    if num_elts == 0:
        return values
    values[0] = initial
    for i in range(1, num_elts):
        values[i] = (values[i - 1] + random.randint(0, max_delta)) % max_value
    return values


def generate_correlated_series(values: list[int], max_value: int) -> list[int]:
    """
    Generates a list of values closely correlated with 'values' and returns it.
    The new list of values is a copy of 'values', where each value is mutated
    with 20% probability by incrementing or decrementing the value. If the modified
    value is 0, it will not be decremented. If the modified value is 'max_value', it
    will not be incremented.
    """
    result = [0] * len(values)
    for i, value in enumerate(values):
        result[i] = value
        rng = random.randint(0, 9)
        if rng == 0 and result[i] < max_value:
            result[i] += 1
        elif rng == 1 and result[i] > 0:
            result[i] -= 1
    return result


def write_formatted_data_to_file(file: str, seed: int) -> None:
    """
    Writes the data to the file following a specific format and data generation pattern
    """
    random.seed(seed)
    FMT = {
        1: "B",
        2: "H",
        4: "I",
        8: "Q",
    }
    with open(file, "wb") as f:
        all_bytes = []
        # Create 60 arrays that originate from 30 sources arrays and 1 associated correlated
        # series for each source array
        max_values = [255, 65535, 4294967295, 18446744073709551615]
        elt_widths = [1, 2, 4, 8]
        input_tag = 0
        for i in range(30):
            width_idx = i % 3
            elt_width = elt_widths[width_idx]
            max_value = max_values[width_idx]
            num_elts = random.randint(1000, 10000)
            max_delta = random.randint(1, 10)
            num_bytes = num_elts * elt_width
            values = generate_time_series_data(num_elts, max_delta, max_value)
            segments = [
                values,
                generate_correlated_series(values, max_value),
            ]
            for segment in segments:
                all_bytes += [
                    struct.pack(
                        f"<IBI{num_elts}{FMT[elt_width]}",
                        num_bytes,
                        elt_width,
                        input_tag,
                        *segment,
                    )
                ]
                input_tag += 1
        # Reorder the byte segments before writing
        random.shuffle(all_bytes)
        for struct_bytes in all_bytes:
            f.write(struct_bytes)


if __name__ == "__main__":
    # Change file to populate a directory
    FILE = sys.argv[1]

    # Write formatted data to file with a fixed seed
    write_formatted_data_to_file(FILE, 0)
