## Simple API

The simplest decompression function.

!!! note
    Rich error information only works with decompression operations that take a `ZL_DCtx*`, so if the operation fails this function won't provide details.

::: ZL_decompress

## Querying Compressed Frames

::: ZL_getCompressedSize

::: ZL_FrameInfo_create

::: ZL_FrameInfo_free

::: ZL_FrameInfo_getFormatVersion

::: ZL_FrameInfo_getNumOutputs

::: ZL_FrameInfo_getOutputType

::: ZL_FrameInfo_getDecompressedSize

### Helper Functions

These functions are wrappers around the `ZL_FrameInfo` functions that provide simple APIs for common cases.

::: ZL_getDecompressedSize

::: ZL_getNumOutputs

::: ZL_getOutputType

## Lifetime Management

::: ZL_DCtx_create

::: ZL_DCtx_free

## Parameterization

::: ZL_DCtx_setParameter

::: ZL_DCtx_getParameter

::: ZL_DCtx_resetParameters

## Errors and Warnings

::: ZL_DCtx_getErrorContextString

::: ZL_DCtx_getErrorContextString_fromError

::: ZL_DCtx_getWarnings

## Serial Decompression

::: ZL_DCtx_decompress

## Typed Decompression

::: ZL_DCtx_decompressTBuffer

::: ZL_DCtx_decompressMultiTBuffer

### ZL_TypedBuffer

!!! note
    `ZL_TypedBuffer` will be replaced with `ZL_Input` soon.
    So expect these function names to change.

::: ZL_TypedBuffer_create

::: ZL_TypedBuffer_createWrapSerial

::: ZL_TypedBuffer_createWrapStruct

::: ZL_TypedBuffer_createWrapNumeric

::: ZL_TypedBuffer_free

::: ZL_TypedBuffer_type

::: ZL_TypedBuffer_rPtr

::: ZL_TypedBuffer_byteSize

::: ZL_TypedBuffer_numElts

::: ZL_TypedBuffer_eltWidth

::: ZL_TypedBuffer_rStringLens
