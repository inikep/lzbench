// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_TRANSFORMS_ENTROPY_COMMON_HUFFMAN_KERNEL_H
#define ZSTRONG_TRANSFORMS_ENTROPY_COMMON_HUFFMAN_KERNEL_H

/**
 * The Huffman library we use has a couple limitations. It cannot natively
 * represent uncompressible data, and it can't compress more than 128KB at a
 * time. We develop our own
 */
typedef enum {
    /**
     * The next block is huffman-coded. The uncompressed size of the block
     * follows, encoded as a little-endian 32-bit integer. The compressed size
     * of the block follows that, similarly encoded.
     */
    ZS_HufTransformPrefix_huf = 0,

    /**
     * The next block was uncompressible and is represented literally. The
     * compressed size is the same as the uncompressed size, both of which are
     * encoded as a little-endian int in the next 4 bytes of the stream.
     */
    ZS_HufTransformPrefix_lit = 1,

    /**
     * The next block is all one symbol. A length (little-endian 32-bit) and the
     * symbol (uint8_t) and a length follow.
     */
    ZS_HufTransformPrefix_constant = 2,
} ZS_HufTransformPrefix_e;

#endif
