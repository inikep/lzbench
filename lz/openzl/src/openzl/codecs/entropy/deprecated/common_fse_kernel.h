// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_ENTROPY_COMMON_FSE_KERNEL_H
#define ZSTRONG_TRANSFORMS_ENTROPY_COMMON_FSE_KERNEL_H

/**
 * The FSE library can't encode all streams. This enum is used to indicate when
 * an alternate (degenerate) encoding is used to represent streams that can't
 * be FSE-compressed.
 */
typedef enum {
    /**
     * The stream is FSE-encoded.
     */
    ZS_FseTransformPrefix_fse = 0,

    /**
     * The stream was uncompressible and is literally encoded.
     */
    ZS_FseTransformPrefix_lit = 1,

    /**
     * The stream was all one symbol. The symbol (uint8_t) and a length
     * (little-endian uint64_t) follow.
     */
    ZS_FseTransformPrefix_constant = 2,
} ZS_FseTransformPrefix_e;

#endif
