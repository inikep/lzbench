// =============================================================================
// lz/aceapex/cuda/aceapex_cuda.h
// ACEAPEX CUDA decoder for lzbench — SAME .aet format as the merged CPU codec.
// Hybrid decompress: CPU entropy (existing in-tree lit/FSE) -> H2D -> GPU
// warp-per-block LZ match decode -> D2H. Zero external dependencies
// (CUDA Runtime only; no nvcomp).
// =============================================================================
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Raw decoded streams of one .aet container (filled by aceapex_decode_streams
// in lz/aceapex/aceapex_api.cpp — same TU as the codec internals).
typedef struct {
    uint8_t *lit, *off, *len, *cmd;     // concatenated raw streams (malloc'd)
    uint64_t lit_sz, off_sz, len_sz, cmd_sz;
    void    *boffs_vec;                 // owner (opaque), freed by streams_free
    const void *boffs;                  // BlockOffsets[num_blocks], 64B each
    size_t   num_blocks;
    uint32_t block_size;
    uint64_t orig_size;
} aceapex_streams_t;

// CPU side (lz/aceapex/aceapex_api.cpp): parse .aet header + entropy-decode
// the four streams. Returns 0 on success.
int  aceapex_decode_streams(const void* src, size_t src_size, aceapex_streams_t* out);
void aceapex_streams_free(aceapex_streams_t* s);

// GPU side (lz/aceapex/cuda/aceapex_cuda.cu):
// 1 if a CUDA device is usable at runtime, else 0.
int aceapex_cg_available(void);
// H2D(streams) -> warp-per-block LZ decode on device -> D2H(dst).
// Returns orig_size, or 0 on error. Env ACEAPEX_CUDA_TIMING=1 prints a
// one-time H2D/kernel/D2H breakdown to stderr.
int64_t aceapex_cg_match_decode(const aceapex_streams_t* s, void* dst, size_t dst_capacity);
// Free cached device buffers (deinit).
void aceapex_cg_release(void);

#ifdef __cplusplus
}
#endif
