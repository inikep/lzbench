/*
 * GPUCompact
 * lzbench glue wrapper
 */
#ifndef BENCH_REMOVE_GPUCOMPACT
#ifdef BENCH_HAS_CUDA
#include "bench/codecs.h"
#include "context.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#pragma pack(push, 1)
struct GpucompactChunkHeader {
  uint32_t comp_size;
  uint32_t uncomp_size;
  uint32_t primary_idx;
  uint8_t is_raw;
  uint8_t reserved[3];
};
#pragma pack(pop)

struct GpucompactState {
  CompressionContext *comp_ctx = nullptr;
  DecompressionContext *decomp_ctx = nullptr;
  LaunchConfig config;
  size_t macro_bytes = 0;
};

char *lzbench_gpucompact_init(size_t insize, size_t level, size_t threads) {
  (void)threads;
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    fprintf(stderr, "gpucompact: no CUDA device available\n");
    return NULL;
  }

  GpucompactState *state = new (std::nothrow) GpucompactState();
  if (!state)
    return NULL;

  if (level == 1) { // best_speed
    state->config.macro_mb = 4;
    state->config.mini_size = 256;
    state->config.L = 512;
    state->config.threads_comp = 32;
    state->config.threads_decomp = 32;
  } else if (level == 2) { // speed
    state->config.macro_mb = 4;
    state->config.mini_size = 512;
    state->config.L = 512;
    state->config.threads_comp = 32;
    state->config.threads_decomp = 32;
  } else if (level == 3) { // balanced
    state->config.macro_mb = 4;
    state->config.mini_size = 512;
    state->config.L = 1024;
    state->config.threads_comp = 32;
    state->config.threads_decomp = 64;
  } else if (level == 4) { // ratio
    state->config.macro_mb = 4;
    state->config.mini_size = 2048;
    state->config.L = 2048;
    state->config.threads_comp = 32;
    state->config.threads_decomp = 64;
  } else { // best_ratio (level >= 5)
    state->config.macro_mb = 8;
    state->config.mini_size = 8192;
    state->config.L = 2048;
    state->config.threads_comp = 32;
    state->config.threads_decomp = 128;
  }

  size_t full_macro = (size_t)state->config.macro_mb * 1024 * 1024;
  if (insize > 0) {
    state->macro_bytes = std::min(full_macro, std::max((size_t)65536, insize));
  } else {
    state->macro_bytes = full_macro;
  }

  try {
    state->comp_ctx = new CompressionContext(
        state->macro_bytes, state->config.mini_size, state->config.L);
    state->decomp_ctx = new DecompressionContext(
        state->macro_bytes, state->config.mini_size, state->config.L);

    // Warmup CUDA context, streams, CUB kernels & CUDA graph (matching
    // benchmark.cpp)
    int dummy_n = (int)std::min((size_t)1024 * 1024, state->macro_bytes);
    for (int i = 0; i < dummy_n; i++)
      state->comp_ctx->host_in[i] = (unsigned char)(i % 255);
    state->comp_ctx->bytes_read = dummy_n;
    state->comp_ctx->compress_chunk(state->config.threads_comp,
                                    state->config.mini_size);
    cudaDeviceSynchronize();

    state->decomp_ctx->uncomp_size = dummy_n;
    state->decomp_ctx->comp_size = state->comp_ctx->comp_size;
    state->decomp_ctx->is_raw = state->comp_ctx->is_raw;
    state->decomp_ctx->primary_idx = state->comp_ctx->primary_idx;
    state->decomp_ctx->num_chunks = state->comp_ctx->num_chunks;
    state->decomp_ctx->total_words = state->comp_ctx->total_words;

    std::memcpy(state->decomp_ctx->host_in, state->comp_ctx->host_out,
                state->comp_ctx->comp_size);
    state->decomp_ctx->decompress_chunk(state->config.threads_decomp,
                                        state->config.mini_size);
    cudaDeviceSynchronize();

  } catch (...) {
    if (state->comp_ctx)
      delete state->comp_ctx;
    if (state->decomp_ctx)
      delete state->decomp_ctx;
    delete state;
    return NULL;
  }

  return (char *)state;
}

void lzbench_gpucompact_deinit(char *workmem) {
  if (!workmem)
    return;
  GpucompactState *state = (GpucompactState *)workmem;
  if (state->comp_ctx)
    delete state->comp_ctx;
  if (state->decomp_ctx)
    delete state->decomp_ctx;
  delete state;
}

int64_t lzbench_gpucompact_compress(char *inbuf, size_t insize, char *outbuf,
                                    size_t outsize,
                                    codec_options_t *codec_options) {
  if (!codec_options || !codec_options->work_mem)
    return 0;
  GpucompactState *state = (GpucompactState *)codec_options->work_mem;
  CompressionContext *ctx = state->comp_ctx;

  try {
    size_t in_offset = 0;
    size_t out_offset = 0;

    while (in_offset < insize) {
      size_t chunk_in_size = std::min(insize - in_offset, state->macro_bytes);

      if (out_offset + sizeof(GpucompactChunkHeader) > outsize) {
        return 0;
      }

      std::memcpy(ctx->host_in, inbuf + in_offset, chunk_in_size);
      ctx->bytes_read = (int)chunk_in_size;

      ctx->compress_chunk(state->config.threads_comp, state->config.mini_size);
      cudaDeviceSynchronize();

      if (out_offset + sizeof(GpucompactChunkHeader) + ctx->comp_size > outsize) {
        return 0;
      }

      GpucompactChunkHeader header;
      header.comp_size = (uint32_t)ctx->comp_size;
      header.uncomp_size = (uint32_t)chunk_in_size;
      header.primary_idx = (uint32_t)ctx->primary_idx;
      header.is_raw = (uint8_t)ctx->is_raw;
      std::memset(header.reserved, 0, sizeof(header.reserved));

      std::memcpy(outbuf + out_offset, &header, sizeof(GpucompactChunkHeader));
      out_offset += sizeof(GpucompactChunkHeader);

      std::memcpy(outbuf + out_offset, ctx->host_out, ctx->comp_size);
      out_offset += ctx->comp_size;

      in_offset += chunk_in_size;
    }

    return (int64_t)out_offset;
  } catch (...) {
    return 0;
  }
}

int64_t lzbench_gpucompact_decompress(char *inbuf, size_t insize, char *outbuf,
                                      size_t outsize,
                                      codec_options_t *codec_options) {
  if (!codec_options || !codec_options->work_mem)
    return 0;
  GpucompactState *state = (GpucompactState *)codec_options->work_mem;
  DecompressionContext *ctx = state->decomp_ctx;

  try {
    size_t in_offset = 0;
    size_t out_offset = 0;

    while (in_offset < insize) {
      if (in_offset + sizeof(GpucompactChunkHeader) > insize) {
        return 0;
      }

      GpucompactChunkHeader header;
      std::memcpy(&header, inbuf + in_offset, sizeof(GpucompactChunkHeader));
      in_offset += sizeof(GpucompactChunkHeader);

      if (in_offset + header.comp_size > insize) {
        return 0;
      }

      if (out_offset + header.uncomp_size > outsize) {
        return 0;
      }

      std::memcpy(ctx->host_in, inbuf + in_offset, header.comp_size);
      ctx->comp_size = (int)header.comp_size;
      ctx->uncomp_size = (int)header.uncomp_size;
      ctx->primary_idx = (int)header.primary_idx;
      ctx->is_raw = (int)header.is_raw;

      cudaDeviceSynchronize();

      if (!ctx->decompress_chunk(state->config.threads_decomp,
                                 state->config.mini_size)) {
        return 0;
      }
      cudaDeviceSynchronize();

      std::memcpy(outbuf + out_offset, ctx->host_out, header.uncomp_size);
      out_offset += header.uncomp_size;
      in_offset += header.comp_size;
    }

    return (int64_t)out_offset;
  } catch (...) {
    return 0;
  }
}

#endif // BENCH_HAS_CUDA
#endif // BENCH_REMOVE_GPUCOMPACT
