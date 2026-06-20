# Numeric Array Compression

Unlike traditional compression algorithms which only compress bytes, OpenZL can compress typed input data. This example shows how to compress an array of numeric data using OpenZL.

## Setting up the compressor

The compressor is created with [ZL_Compressor_create()][ZL_Compressor_create], and must be freed with [ZL_Compressor_free()][ZL_Compressor_free].
Next, the format version must be selected.
Each OpenZL release that introduces a new feature bumps the format version.
It is the users responsibility to select a format version that they know all decompressors support.
Then any additional parameters, like the compression level, can be selected.

After the parameterization, the compression [graph][graphs] must be constructed & selected.
In this case, we're using `ZL_GRAPH_COMPRESS_GENERIC`, which is a [standard graph][standard-graph] that OpenZL provides.
We tell OpenZL which graph to use to compress with [ZL_Compressor_selectStartingGraphID()][ZL_Compressor_selectStartingGraphID].

```cpp
--8<-- "src/examples/numeric_array.cpp:setup-compressor"

--8<-- "src/examples/numeric_array.cpp:setup-compressor-build1"
--8<-- "src/examples/numeric_array.cpp:setup-compressor-build2"

--8<-- "src/examples/numeric_array.cpp:select-graph"
    // ...
}
```

## Compression

After the compressor is set up, we're ready to start compression.
First, we create the `ZL_CCtx`, and reference the `compressor` with [ZL_CCtx_refCompressor()][ZL_CCtx_refCompressor], which tells OpenZL to use `compressor` to compress the data.

Then, we wrap the input data in a `ZL_TypedRef` with [ZL_TypedRef_createNumeric()][ZL_TypedRef_createNumeric], which tells OpenZL the data is numeric with the given element `width`.
Finally, we create an output buffer which is guaranteed to be large enough with [ZL_compressBound()][ZL_compressBound], and call [ZL_CCtx_compressTypedRef()][ZL_CCtx_compressTypedRef] to compress the data.

```cpp
--8<-- "src/examples/numeric_array.cpp:compress"
```

## Decompression

Decompression works independently of which `compressor` was chosen to compress the data.

First, we find the output size with [ZL_getDecompressedSize()][ZL_getDecompressedSize].
In this case, we know the original width, but this isn't required to decompress.
Next, we create the output buffer and wrap it in a typed buffer with [ZL_TypedBuffer_createWrapNumeric()][ZL_TypedBuffer_createWrapNumeric].
Finally, we decompress the data with [ZL_DCtx_decompressTBuffer()][ZL_DCtx_decompressTBuffer].

```cpp
--8<-- "src/examples/numeric_array.cpp:decompress"
```

## Customizing the compressor

Previously, we went through a very simple example of how to set up the compressor using a standard graph.
In this section we go through several other ways to set up compressors.

```cpp
--8<-- "src/examples/numeric_array.cpp:customizing-compressor1"
```

The simplest graph is a [static graph][static-graph], which inserts a codec whose output goes to another [graph][graphs].
This graph runs the delta codec first, which subtracts the previous value from the current, then passes the result of that to the standard compression graph `ZL_GRAPH_COMPRESS_GENERIC`.
This graph works well when the data is sorted.

```cpp
--8<-- "src/examples/numeric_array.cpp:customizing-compressor2"
```

This graph uses FieldLZ, which is an LZ engine that specializes on numeric data.
This is another standard graph like `ZL_GRAPH_COMPRESS_GENERIC`, and is a fundamental component that is used by many other graphs, including `ZL_GRAPH_COMPRESS_GENERIC`.

```cpp
--8<-- "src/examples/numeric_array.cpp:customizing-compressor3"
```

These two graphs specialize in compressing `bfloat16` and `float32` data.
They separate the exponent from the sign & fraction bits, and pass the exponent to `FSE`, and store the sign & fraction bits as-is.

```cpp
--8<-- "src/examples/numeric_array.cpp:customizing-compressor4"
```

This is a slightly more complex graph that introduces [selectors][selector-graph].
This graph first passes the input to a selector that tries each possible successor graph, and selects the best graph.
This selector is making a dynamic decision at compression time about which successor to pass the input to.
The selector is not present during decompression, because it only sees the single successor that was selected.

```cpp
--8<-- "src/examples/numeric_array.cpp:customizing-compressor5"
```

Finally, we can wrap this section up by building a new `buildCompressor()` function which can choose any of these compressor builders.

## Full Example Source

```cpp title="examples/numeric_array.cpp"
--8<-- "src/examples/numeric_array.cpp"
```
