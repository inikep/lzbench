# Copyright (c) Meta Platforms, Inc. and affiliates.

# --8<-- [start:imports]
import random
import struct
from typing import List

import numpy as np
import openzl.ext as zl

# --8<-- [end:imports]

"""
Setup from PyPI:

    pip install openzl

Setup from OpenZL repo:

    cd py
    pip install .
"""


# --8<-- [start:generate_test_data]

NUM_SAMPLES = 100
SAMPLE_SIZE = 100000


def generate_low_cardinality_sample(rng: random.Random) -> bytes:
    alphabet_size = rng.randint(1, 1000)
    alphabet = [rng.randint(0, 2**64 - 1) for _ in range(alphabet_size)]
    values = random.choices(alphabet, k=SAMPLE_SIZE)
    return b"".join(struct.pack("<Q", v) for v in values)


def generate_sorted_sample(rng: random.Random) -> bytes:
    MAX_INCREASE = 10000
    start = rng.randint(0, 2**63 - 1)
    values = [start]
    for _ in range(SAMPLE_SIZE - 1):
        values.append(values[-1] + random.randint(0, MAX_INCREASE))
    return b"".join(struct.pack("<Q", v) for v in values)


def generate_test_data() -> List[bytes]:
    rng = random.Random(0)
    samples = []
    for _ in range(NUM_SAMPLES):
        if rng.random() < 0.5:
            s = generate_low_cardinality_sample(rng)
        else:
            s = generate_sorted_sample(rng)
        samples.append(s)
    return samples


TEST_DATA_SET = generate_test_data()
TEST_DATA_SET_SIZE = sum(len(s) for s in TEST_DATA_SET)


# --8<-- [end:generate_test_data]


# --8<-- [start:simple_compressor]
def build_simple_compressor() -> zl.Compressor:
    compressor = zl.Compressor()

    # Set the compression graph to pass the input data to
    # OpenZL's generic compression backend.
    graph = zl.graphs.Compress()(compressor)
    compressor.select_starting_graph(graph)

    return compressor


# --8<-- [end:simple_compressor]


# --8<-- [start:compress_bytes]
def compress_bytes(compressor: zl.Compressor, data: bytes) -> bytes:
    cctx = zl.CCtx()

    # Tell the cctx to use the compressor we've built.
    cctx.ref_compressor(compressor)

    # Select the OpenZL format version to encode with.
    # This should be the latest version that the decompressor supports.
    cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

    # Compress a single input of serial data.
    return cctx.compress([zl.Input(zl.Type.Serial, data)])


def decompress_bytes(compressed: bytes) -> bytes:
    dctx = zl.DCtx()

    # Decompress the compressed data.
    # OpenZL compressed data can decompress to multiple outputs of several types.
    # Here, we expect to receive a single output of serial data.
    outputs = dctx.decompress(compressed)
    if len(outputs) != 1 or outputs[0].type != zl.Type.Serial:
        raise RuntimeError("Only one serial output supported")

    # Convert the OpenZL output to bytes.
    # This costs a copy, but is simple. The output data can also be accessed as a NumPy array with zero copies.
    return outputs[0].content.as_bytes()


# --8<-- [end:compress_bytes]


# --8<-- [start:test_simple_compressor]
def test_compressor(compressor: zl.Compressor, dataset: List[bytes]) -> int:
    compressed_size = 0
    for data in dataset:
        compressed = compress_bytes(compressor, data)
        decompressed = decompress_bytes(compressed)
        if decompressed != data:
            raise RuntimeError("Corruption!")
        compressed_size += len(compressed)
    return compressed_size


compressor = build_simple_compressor()
compressed_size = test_compressor(compressor, TEST_DATA_SET)
print(f"Simple bytes compressor ratio: {TEST_DATA_SET_SIZE / compressed_size:.2f}")


# --8<-- [end:test_simple_compressor]


# --8<-- [start:simple_int64_compressor]
def build_simple_int64_compressor() -> zl.Compressor:
    compressor = zl.Compressor()
    graph = zl.nodes.ConvertSerialToNumLE64()(
        compressor, successor=zl.graphs.Compress()
    )
    compressor.select_starting_graph(graph)
    return compressor


compressor = build_simple_int64_compressor()
compressed_size = test_compressor(compressor, TEST_DATA_SET)
print(f"Simple Int64 compressor ratio: {TEST_DATA_SET_SIZE / compressed_size:.2f}")


# --8<-- [end:simple_int64_compressor]


# --8<-- [start:better_int64_compressor]
def build_sorted_graph(compressor: zl.Compressor) -> zl.GraphID:
    """
    Run a delta on the input and pass it to the generic compression graph.
    """
    return zl.nodes.DeltaInt()(compressor, zl.graphs.Compress())


def build_low_cardinality_graph(compressor: zl.Compressor) -> zl.GraphID:
    """
    Tokenize the input data, sorting the alphabet in ascending order.
    Send the alphabet to the sorted graph, and send the indices to
    the generic compression graph.
    """
    tokenize = zl.nodes.Tokenize(type=zl.Type.Numeric, sort=True)
    return tokenize(
        compressor,
        alphabet=build_sorted_graph(compressor),
        indices=zl.graphs.Compress(),
    )


class QuickStartSelector(zl.Selector):
    """
    If the data is sorted, pass the input to the sorted_graph.
    Otherwise, pass the input to the low_cardinality_graph.
    """

    def __init__(self, sorted_graph: zl.GraphID, low_cardinality_graph: zl.GraphID):
        super().__init__()
        self._sorted_graph = sorted_graph
        self._low_cardinality_graph = low_cardinality_graph

    def selector_description(self) -> zl.SelectorDescription:
        return zl.SelectorDescription(
            name="quick_start_selector",
            input_type_mask=zl.TypeMask.Numeric,
        )

    def select(self, state: zl.SelectorState, input: zl.Input) -> zl.GraphID:
        data = input.content.as_nparray()
        is_sorted = np.all(data[:-1] <= data[1:])
        if is_sorted:
            return self._sorted_graph
        else:
            return self._low_cardinality_graph


def build_better_int64_compressor() -> zl.Compressor:
    compressor = zl.Compressor()

    # Build the two backend numeric compression graphs.
    sorted_graph = build_sorted_graph(compressor)
    low_cardinality_graph = build_low_cardinality_graph(compressor)

    # Set up the selector graph which inspects the input data and selects the
    # correct backend graph.
    graph = compressor.register_selector_graph(
        QuickStartSelector(sorted_graph, low_cardinality_graph)
    )

    # Add a conversion to int64 in front of the selector.
    graph = zl.nodes.ConvertSerialToNumLE64()(compressor, graph)

    compressor.select_starting_graph(graph)
    return compressor


compressor = build_better_int64_compressor()
compressed_size = test_compressor(compressor, TEST_DATA_SET)
print(f"Better Int64 compressor ratio: {TEST_DATA_SET_SIZE / compressed_size:.2f}")

# --8<-- [end:better_int64_compressor]


# --8<-- [start:compress_numpy_array]
def compress_numpy_array(compressor: zl.Compressor, data: np.ndarray) -> bytes:
    """
    Compresses a single 1-D array of numeric data.
    """
    cctx = zl.CCtx()
    cctx.ref_compressor(compressor)
    cctx.set_parameter(zl.CParam.FormatVersion, zl.MAX_FORMAT_VERSION)

    # Compress a single input of numeric data.
    return cctx.compress([zl.Input(zl.Type.Numeric, data)])


def decompress_numpy_array(compressed: bytes) -> np.ndarray:
    dctx = zl.DCtx()
    outputs = dctx.decompress(compressed)
    if len(outputs) != 1 or outputs[0].type != zl.Type.Numeric:
        raise RuntimeError("Only one numeric output supported")

    # This is a zero-copy operation because it uses the buffer protocol
    return outputs[0].content.as_nparray()


def build_constant_compressor() -> zl.Compressor:
    compressor = zl.Compressor()
    graph = zl.graphs.Constant()(compressor)
    compressor.select_starting_graph(graph)
    return compressor


data = np.array([42] * 1000, dtype=np.uint32)
compressed = compress_numpy_array(build_constant_compressor(), data)
decompressed = decompress_numpy_array(compressed)
if not np.array_equal(data, decompressed):
    raise RuntimeError("Corruption")


# --8<-- [end:compress_numpy_array]
