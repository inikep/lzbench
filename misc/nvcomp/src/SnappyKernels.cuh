/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "SnappyBlockUtils.cuh"
#include "SnappyTypes.h"
#include "CudaUtils.h"

namespace nvcomp {

#define HASH_BITS 12

// TBD: Tentatively limits to 2-byte codes to prevent long copy search followed by long literal
// encoding
#define MAX_LITERAL_LENGTH 256

#define MAX_COPY_LENGTH 64       // Syntax limit
#define MAX_COPY_DISTANCE 32768  // Matches encoder limit as described in snappy format description

const int COMP_THREADS_PER_BLOCK = 64; // 2 warps per stream, 1 stream per block
const int DECOMP_THREADS_PER_BLOCK = 96; // 3 warps per stream, 1 stream per block

/**
 * @brief snappy compressor state
 **/
struct snap_state_s {
  const uint8_t *src;                 ///< Ptr to uncompressed data
  uint32_t src_len;                   ///< Uncompressed data length
  uint8_t *dst_base;                  ///< Base ptr to output compressed data
  uint8_t *dst;                       ///< Current ptr to uncompressed data
  uint8_t *end;                       ///< End of uncompressed data buffer
  volatile uint32_t literal_length;   ///< Number of literal bytes
  volatile uint32_t copy_length;      ///< Number of copy bytes
  volatile uint32_t copy_distance;    ///< Distance for copy bytes
  uint16_t hash_map[1 << HASH_BITS];  ///< Low 16-bit offset from hash
};

static inline __device__ uint32_t get_max_compressed_length(uint32_t source_bytes)
{
  // This is an estimate from the original snappy library 
  return 32 + source_bytes + source_bytes / 6;
}

/**
 * @brief 12-bit hash from four consecutive bytes
 **/
static inline __device__ uint32_t snap_hash(uint32_t v)
{
  return (v * ((1 << 20) + (0x2a00) + (0x6a) + 1)) >> (32 - HASH_BITS);
}

/**
 * @brief Outputs a snappy literal symbol
 *
 * @param dst Destination compressed byte stream
 * @param end End of compressed data buffer
 * @param src Pointer to literal bytes
 * @param len_minus1 Number of literal bytes minus 1
 * @param t Thread in warp
 *
 * @return Updated pointer to compressed byte stream
 **/
static inline __device__ uint8_t *StoreLiterals(
  uint8_t *dst, uint8_t *end, const uint8_t *src, uint32_t len_minus1, uint32_t t)
{
  if (len_minus1 < 60) {
    if (!t && dst < end) dst[0] = (len_minus1 << 2);
    dst += 1;
  } else if (len_minus1 <= 0xff) {
    if (!t && dst + 1 < end) {
      dst[0] = 60 << 2;
      dst[1] = len_minus1;
    }
    dst += 2;
  } else if (len_minus1 <= 0xffff) {
    if (!t && dst + 2 < end) {
      dst[0] = 61 << 2;
      dst[1] = len_minus1;
      dst[2] = len_minus1 >> 8;
    }
    dst += 3;
  } else if (len_minus1 <= 0xffffff) {
    if (!t && dst + 3 < end) {
      dst[0] = 62 << 2;
      dst[1] = len_minus1;
      dst[2] = len_minus1 >> 8;
      dst[3] = len_minus1 >> 16;
    }
    dst += 4;
  } else {
    if (!t && dst + 4 < end) {
      dst[0] = 63 << 2;
      dst[1] = len_minus1;
      dst[2] = len_minus1 >> 8;
      dst[3] = len_minus1 >> 16;
      dst[4] = len_minus1 >> 24;
    }
    dst += 5;
  }
  for (uint32_t i = t; i <= len_minus1; i += 32) {
    if (dst + i < end) dst[i] = src[i];
  }
  return dst + len_minus1 + 1;
}

/**
 * @brief Outputs a snappy copy symbol (assumed to be called by a single thread)
 *
 * @param dst Destination compressed byte stream
 * @param end End of compressed data buffer
 * @param copy_len Copy length
 * @param distance Copy distance
 *
 * @return Updated pointer to compressed byte stream
 **/
static inline __device__ uint8_t *StoreCopy(
    uint8_t *dst,
    uint8_t *end,
    uint32_t copy_len,
    uint32_t distance)
{
  if (copy_len < 12 && distance < 2048) {
    // xxxxxx01.oooooooo: copy with 3-bit length, 11-bit offset
    if (dst + 2 <= end) {
      dst[0] = ((distance & 0x700) >> 3) | ((copy_len - 4) << 2) | 0x01;
      dst[1] = distance;
    }
    return dst + 2;
  } else {
    // xxxxxx1x: copy with 6-bit length, 16-bit offset
    if (dst + 3 <= end) {
      dst[0] = ((copy_len - 1) << 2) | 0x2;
      dst[1] = distance;
      dst[2] = distance >> 8;
    }
    return dst + 3;
  }
}

/**
 * @brief Returns mask of any thread in the warp that has a hash value
 * equal to that of the calling thread
 **/
static inline __device__ uint32_t HashMatchAny(uint32_t v, uint32_t t)
{
#if (__CUDA_ARCH__ >= 700)
  return __match_any_sync(~0, v);
#else
  uint32_t err_map = 0;
  for (uint32_t i = 0; i < HASH_BITS; i++, v >>= 1) {
    uint32_t b       = v & 1;
    uint32_t match_b = BALLOT(b);
    err_map |= match_b ^ -(int32_t)b;
  }
  return ~err_map;
#endif
}

/**
 * @brief Finds the first occurence of a consecutive 4-byte match in the input sequence,
 * or at most MAX_LITERAL_LENGTH bytes
 *
 * @param s Compressor state (copy_length set to 4 if a match is found, zero otherwise)
 * @param src Uncompressed buffer
 * @param pos0 Position in uncompressed buffer
 * @param t thread in warp
 *
 * @return Number of bytes before first match (literal length)
 **/
static __device__ inline uint32_t FindFourByteMatch(
    snap_state_s *s,
    const uint8_t *src,
    uint32_t pos0,
    uint32_t t)
{
  uint32_t len    = s->src_len;
  uint32_t pos    = pos0;
  uint32_t maxpos = pos0 + MAX_LITERAL_LENGTH - 31;
  uint32_t match_mask, literal_cnt;
  if (t == 0) { s->copy_length = 0; }
  do {
    bool valid4               = (pos + t + 4 <= len);
    uint32_t data32           = (valid4) ? unaligned_load32(src + pos + t) : 0;
    uint32_t hash             = (valid4) ? snap_hash(data32) : 0;
    uint32_t local_match      = HashMatchAny(hash, t);
    uint32_t local_match_lane = 31 - __clz(local_match & ((1 << t) - 1));
    uint32_t local_match_data = SHFL(data32, min(local_match_lane, t));
    uint32_t offset, match;
    if (valid4) {
      if (local_match_lane < t && local_match_data == data32) {
        match  = 1;
        offset = pos + local_match_lane;
      } else {
        offset = (pos & ~0xffff) | s->hash_map[hash];
        if (offset >= pos) { offset = (offset >= 0x10000) ? offset - 0x10000 : pos; }
        match =
          (offset < pos && offset + MAX_COPY_DISTANCE >= pos + t && unaligned_load32(src + offset) == data32);
      }
    } else {
      match       = 0;
      local_match = 0;
      offset      = pos + t;
    }
    match_mask = BALLOT(match);
    if (match_mask != 0) {
      literal_cnt = __ffs(match_mask) - 1;
      if (t == literal_cnt) {
        s->copy_distance = pos + t - offset;
        s->copy_length   = 4;
      }
    } else {
      literal_cnt = 32;
    }
    // Update hash up to the first 4 bytes of the copy length
    local_match &= (0x2 << literal_cnt) - 1;
    if (t <= literal_cnt && t == 31 - __clz(local_match)) { s->hash_map[hash] = pos + t; }
    pos += literal_cnt;
  } while (literal_cnt == 32 && pos < maxpos);
  return min(pos, len) - pos0;
}

/// @brief Returns the number of matching bytes for two byte sequences up to 63 bytes
static __device__ inline uint32_t Match60(const uint8_t *src1,
                                   const uint8_t *src2,
                                   uint32_t len,
                                   uint32_t t)
{
  uint32_t mismatch = BALLOT(t >= len || src1[t] != src2[t]);
  if (mismatch == 0) {
    mismatch = BALLOT(32 + t >= len || src1[32 + t] != src2[32 + t]);
    return 31 + __ffs(mismatch);  // mismatch cannot be zero here if len <= 63
  } else {
    return __ffs(mismatch) - 1;
  }
}

/**
 * @brief Snappy compression device function
 * See http://github.com/google/snappy/blob/master/format_description.txt
 *
 * Device helper function that can be used to 
 *
 * @param[in] inputs Source/Destination buffer information per block
 * @param[out] outputs Compression status per block
 * @param[in] count Number of blocks to compress
 **/
__device__ inline void
do_snap(
  const uint8_t* __restrict__ device_in_ptr,
  const uint64_t device_in_bytes,
  uint8_t* const __restrict__ device_out_ptr,
  const uint64_t device_out_available_bytes,
  gpu_snappy_status_s* __restrict__ outputs,
	uint64_t* device_out_bytes)
{
  __shared__ __align__(16) snap_state_s state_g;

  snap_state_s *const s = &state_g;
  uint32_t t            = threadIdx.x;
  uint32_t pos;
  const uint8_t *src;

  if (!t) {
    const uint8_t *src = device_in_ptr;
    uint32_t src_len   = static_cast<uint32_t>(device_in_bytes);
    uint8_t *dst       = device_out_ptr;
    uint32_t dst_len   = device_out_available_bytes;
    if (dst_len == 0)
      dst_len = get_max_compressed_length(src_len);

    uint8_t *end       = dst + dst_len;
    s->src             = src;
    s->src_len         = src_len;
    s->dst_base        = dst;
    s->end             = end;
    while (src_len > 0x7f) {
      if (dst < end) { dst[0] = src_len | 0x80; }
      dst++;
      src_len >>= 7;
    }
    if (dst < end) { dst[0] = src_len; }
    s->dst            = dst + 1;
    s->literal_length = 0;
    s->copy_length    = 0;
    s->copy_distance  = 0;
  }
  for (uint32_t i = t; i < sizeof(s->hash_map) / sizeof(uint32_t); i += 128) {
    *reinterpret_cast<volatile uint32_t *>(&s->hash_map[i * 2]) = 0;
  }
  __syncthreads();
  src = s->src;
  pos = 0;
  while (pos < s->src_len) {
    uint32_t literal_len = s->literal_length;
    uint32_t copy_len    = s->copy_length;
    uint32_t distance    = s->copy_distance;
    __syncthreads();
    if (t < 32) {
      // WARP0: Encode literals and copies
      uint8_t *dst = s->dst;
      uint8_t *end = s->end;
      if (literal_len > 0) {
        dst = StoreLiterals(dst, end, src + pos, literal_len - 1, t);
        pos += literal_len;
      }
      if (copy_len > 0) {
        if (t == 0) { dst = StoreCopy(dst, end, copy_len, distance); }
        pos += copy_len;
      }
      SYNCWARP();
      if (t == 0) { s->dst = dst; }
    } else {
      pos += literal_len + copy_len;
      if (t < 32 * 2) {
        // WARP1: Find a match using 12-bit hashes of 4-byte blocks
        uint32_t t5 = t & 0x1f;
        literal_len = FindFourByteMatch(s, src, pos, t5);
        if (t5 == 0) { s->literal_length = literal_len; }
        copy_len = s->copy_length;
        if (copy_len != 0) {
          uint32_t match_pos = pos + literal_len + copy_len;  // NOTE: copy_len is always 4 here
          copy_len += Match60(src + match_pos,
                              src + match_pos - s->copy_distance,
                              min(s->src_len - match_pos, 64 - copy_len),
                              t5);
          if (t5 == 0) { s->copy_length = copy_len; }
        }
      }
    }
    __syncthreads();
  }
  __syncthreads();
  if (!t) {
    *device_out_bytes = s->dst - s->dst_base;
    if (outputs)
      outputs->status = (s->dst > s->end) ? 1 : 0;
  }
}

// Not supporting streams longer than this (not what snappy is intended for)
#define SNAPPY_MAX_STREAM_SIZE 0x7fffffff

#define LOG2_BATCH_SIZE 5
#define BATCH_SIZE (1 << LOG2_BATCH_SIZE)
#define LOG2_BATCH_COUNT 2
#define BATCH_COUNT (1 << LOG2_BATCH_COUNT)
#define LOG2_PREFETCH_SIZE 12
#define PREFETCH_SIZE (1 << LOG2_PREFETCH_SIZE)  // 4KB, in 32B chunks
#define PREFETCH_SECTORS 8 // How many loads in flight when prefetching
#define LITERAL_SECTORS 4 // How many loads in flight when processing the literal

#define LOG_CYCLECOUNT 0

/**
 * @brief Describes a single LZ77 symbol (single entry in batch)
 **/
struct unsnap_batch_s {
  int32_t len;  // 1..64 = Number of bytes
  uint32_t
    offset;  // copy distance if greater than zero or negative of literal offset in byte stream
};

/**
 * @brief Queue structure used to exchange data between warps
 **/
struct unsnap_queue_s {
  uint32_t prefetch_wrpos;         ///< Prefetcher write position
  uint32_t prefetch_rdpos;         ///< Prefetch consumer read position
  int32_t prefetch_end;            ///< Prefetch enable flag (nonzero stops prefetcher)
  int32_t batch_len[BATCH_COUNT];  ///< Length of each batch - <0:end, 0:not ready, >0:symbol count
  uint32_t batch_prefetch_rdpos[BATCH_COUNT];
  unsnap_batch_s batch[BATCH_COUNT * BATCH_SIZE];  ///< LZ77 batch data
  uint8_t buf[PREFETCH_SIZE];                      ///< Prefetch buffer
};

/**
 * @brief Input parameters for the decompression interface
 **/
 struct gpu_input_parameters {
  const void *srcDevice;
  uint64_t srcSize;
  void *dstDevice;
  uint64_t dstSize;
};

/**
 * @brief snappy decompression state
 **/
struct unsnap_state_s {
  const uint8_t *base;         ///< base ptr of compressed stream
  const uint8_t *end;          ///< end of compressed stream
  uint32_t uncompressed_size;  ///< uncompressed stream size
  uint32_t bytes_left;         ///< bytes to uncompressed remaining
  int32_t error;               ///< current error status
  uint32_t tstart;             ///< start time for perf logging
  volatile unsnap_queue_s q;   ///< queue for cross-warp communication
  gpu_input_parameters in;      ///< input parameters for current block
};

/**
 * @brief prefetches data for the symbol decoding stage
 *
 * @param s decompression state
 * @param t warp lane id
 **/
__device__ inline void snappy_prefetch_bytestream(unsnap_state_s *s, int t)
{
  const uint8_t *base  = s->base;
  uint32_t end         = (uint32_t)(s->end - base);
  uint32_t align_bytes = (uint32_t)(0x20 - (0x1f & reinterpret_cast<uintptr_t>(base)));
  int32_t pos          = min(align_bytes, end);
  int32_t blen;
  // Start by prefetching up to the next a 32B-aligned location
  if (t < pos) { s->q.buf[t] = base[t]; }
  blen = 0;
  do {
    SYNCWARP();
    if (!t) {
      uint32_t minrdpos;
      s->q.prefetch_wrpos = pos;
      minrdpos            = pos - min(pos, PREFETCH_SIZE - PREFETCH_SECTORS * 32u);
      blen                = (int)min(PREFETCH_SECTORS * 32u, end - pos);
      for (;;) {
        uint32_t rdpos = s->q.prefetch_rdpos;
        if (rdpos >= minrdpos) break;
        if (s->q.prefetch_end) {
          blen = 0;
          break;
        }
        NANOSLEEP(1600);
      }
    }
    blen = SHFL0(blen);
    if (blen == PREFETCH_SECTORS * 32u) {
      uint8_t vals[PREFETCH_SECTORS];
      for(int i = 0; i < PREFETCH_SECTORS; ++i)
        vals[i] = base[pos + t + i * 32u];
      for(int i = 0; i < PREFETCH_SECTORS; ++i)
        s->q.buf[(pos + t + i * 32u) & (PREFETCH_SIZE - 1)] = vals[i];
    } else {
#pragma unroll 1
      for(int elem = t; elem < blen; elem += 32) {
        s->q.buf[(pos + elem) & (PREFETCH_SIZE - 1)] = base[pos + elem];
      }
    }
    pos += blen;
  } while (blen > 0);
}

/**
 * @brief Lookup table for get_len3_mask()
 *
 * Indexed by a 10-bit pattern, contains the corresponding 4-bit mask of
 * 3-byte code lengths in the lower 4 bits, along with the total number of
 * bytes used for coding the four lengths in the upper 4 bits.
 * The upper 4-bit value could also be obtained by 8+__popc(mask4)
 *
 *   for (uint32_t k = 0; k < 1024; k++)
 *   {
 *       for (uint32_t i = 0, v = 0, b = k, n = 0; i < 4; i++)
 *       {
 *           v |= (b & 1) << i;
 *           n += (b & 1) + 2;
 *           b >>= (b & 1) + 2;
 *       }
 *       k_len3lut[k] = v | (n << 4);
 *   }
 *
 **/
static const uint8_t __device__ __constant__ k_len3lut[1 << 10] = {
  0x80, 0x91, 0x80, 0x91, 0x92, 0x91, 0x92, 0x91, 0x80, 0xa3, 0x80, 0xa3, 0x92, 0xa3, 0x92, 0xa3,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xa3, 0x94, 0xa3, 0x92, 0xa3, 0x92, 0xa3,
  0x80, 0xa5, 0x80, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x80, 0xa3, 0x80, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x94, 0xa5, 0x94, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x94, 0xa3, 0x94, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x98, 0x91, 0x98, 0x91, 0x92, 0x91, 0x92, 0x91, 0x98, 0xb7, 0x98, 0xb7, 0x92, 0xb7, 0x92, 0xb7,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xb7, 0x94, 0xb7, 0x92, 0xb7, 0x92, 0xb7,
  0x98, 0xa5, 0x98, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x98, 0xb7, 0x98, 0xb7, 0xa6, 0xb7, 0xa6, 0xb7,
  0x94, 0xa5, 0x94, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x94, 0xb7, 0x94, 0xb7, 0xa6, 0xb7, 0xa6, 0xb7,
  0x80, 0xa9, 0x80, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x80, 0xa3, 0x80, 0xa3, 0xaa, 0xa3, 0xaa, 0xa3,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xa3, 0xac, 0xa3, 0xaa, 0xa3, 0xaa, 0xa3,
  0x80, 0xa5, 0x80, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x80, 0xa3, 0x80, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0xac, 0xa5, 0xac, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0xac, 0xa3, 0xac, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x98, 0xa9, 0x98, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x98, 0xb7, 0x98, 0xb7, 0xaa, 0xb7, 0xaa, 0xb7,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xb7, 0xac, 0xb7, 0xaa, 0xb7, 0xaa, 0xb7,
  0x98, 0xa5, 0x98, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x98, 0xb7, 0x98, 0xb7, 0xa6, 0xb7, 0xa6, 0xb7,
  0xac, 0xa5, 0xac, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0xac, 0xb7, 0xac, 0xb7, 0xa6, 0xb7, 0xa6, 0xb7,
  0x80, 0x91, 0x80, 0x91, 0x92, 0x91, 0x92, 0x91, 0x80, 0xbb, 0x80, 0xbb, 0x92, 0xbb, 0x92, 0xbb,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xbb, 0x94, 0xbb, 0x92, 0xbb, 0x92, 0xbb,
  0x80, 0xbd, 0x80, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x80, 0xbb, 0x80, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x94, 0xbd, 0x94, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x94, 0xbb, 0x94, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x98, 0x91, 0x98, 0x91, 0x92, 0x91, 0x92, 0x91, 0x98, 0xb7, 0x98, 0xb7, 0x92, 0xb7, 0x92, 0xb7,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xb7, 0x94, 0xb7, 0x92, 0xb7, 0x92, 0xb7,
  0x98, 0xbd, 0x98, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x98, 0xb7, 0x98, 0xb7, 0xbe, 0xb7, 0xbe, 0xb7,
  0x94, 0xbd, 0x94, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x94, 0xb7, 0x94, 0xb7, 0xbe, 0xb7, 0xbe, 0xb7,
  0x80, 0xa9, 0x80, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x80, 0xbb, 0x80, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xbb, 0xac, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb,
  0x80, 0xbd, 0x80, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x80, 0xbb, 0x80, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0xac, 0xbd, 0xac, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0xac, 0xbb, 0xac, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x98, 0xa9, 0x98, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x98, 0xb7, 0x98, 0xb7, 0xaa, 0xb7, 0xaa, 0xb7,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xb7, 0xac, 0xb7, 0xaa, 0xb7, 0xaa, 0xb7,
  0x98, 0xbd, 0x98, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x98, 0xb7, 0x98, 0xb7, 0xbe, 0xb7, 0xbe, 0xb7,
  0xac, 0xbd, 0xac, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0xac, 0xb7, 0xac, 0xb7, 0xbe, 0xb7, 0xbe, 0xb7,
  0x80, 0x91, 0x80, 0x91, 0x92, 0x91, 0x92, 0x91, 0x80, 0xa3, 0x80, 0xa3, 0x92, 0xa3, 0x92, 0xa3,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xa3, 0x94, 0xa3, 0x92, 0xa3, 0x92, 0xa3,
  0x80, 0xa5, 0x80, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x80, 0xa3, 0x80, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x94, 0xa5, 0x94, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x94, 0xa3, 0x94, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x98, 0x91, 0x98, 0x91, 0x92, 0x91, 0x92, 0x91, 0x98, 0xcf, 0x98, 0xcf, 0x92, 0xcf, 0x92, 0xcf,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xcf, 0x94, 0xcf, 0x92, 0xcf, 0x92, 0xcf,
  0x98, 0xa5, 0x98, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x98, 0xcf, 0x98, 0xcf, 0xa6, 0xcf, 0xa6, 0xcf,
  0x94, 0xa5, 0x94, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x94, 0xcf, 0x94, 0xcf, 0xa6, 0xcf, 0xa6, 0xcf,
  0x80, 0xa9, 0x80, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x80, 0xa3, 0x80, 0xa3, 0xaa, 0xa3, 0xaa, 0xa3,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xa3, 0xac, 0xa3, 0xaa, 0xa3, 0xaa, 0xa3,
  0x80, 0xa5, 0x80, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x80, 0xa3, 0x80, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0xac, 0xa5, 0xac, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0xac, 0xa3, 0xac, 0xa3, 0xa6, 0xa3, 0xa6, 0xa3,
  0x98, 0xa9, 0x98, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x98, 0xcf, 0x98, 0xcf, 0xaa, 0xcf, 0xaa, 0xcf,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xcf, 0xac, 0xcf, 0xaa, 0xcf, 0xaa, 0xcf,
  0x98, 0xa5, 0x98, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0x98, 0xcf, 0x98, 0xcf, 0xa6, 0xcf, 0xa6, 0xcf,
  0xac, 0xa5, 0xac, 0xa5, 0xa6, 0xa5, 0xa6, 0xa5, 0xac, 0xcf, 0xac, 0xcf, 0xa6, 0xcf, 0xa6, 0xcf,
  0x80, 0x91, 0x80, 0x91, 0x92, 0x91, 0x92, 0x91, 0x80, 0xbb, 0x80, 0xbb, 0x92, 0xbb, 0x92, 0xbb,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xbb, 0x94, 0xbb, 0x92, 0xbb, 0x92, 0xbb,
  0x80, 0xbd, 0x80, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x80, 0xbb, 0x80, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x94, 0xbd, 0x94, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x94, 0xbb, 0x94, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x98, 0x91, 0x98, 0x91, 0x92, 0x91, 0x92, 0x91, 0x98, 0xcf, 0x98, 0xcf, 0x92, 0xcf, 0x92, 0xcf,
  0x94, 0x91, 0x94, 0x91, 0x92, 0x91, 0x92, 0x91, 0x94, 0xcf, 0x94, 0xcf, 0x92, 0xcf, 0x92, 0xcf,
  0x98, 0xbd, 0x98, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x98, 0xcf, 0x98, 0xcf, 0xbe, 0xcf, 0xbe, 0xcf,
  0x94, 0xbd, 0x94, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x94, 0xcf, 0x94, 0xcf, 0xbe, 0xcf, 0xbe, 0xcf,
  0x80, 0xa9, 0x80, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x80, 0xbb, 0x80, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xbb, 0xac, 0xbb, 0xaa, 0xbb, 0xaa, 0xbb,
  0x80, 0xbd, 0x80, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x80, 0xbb, 0x80, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0xac, 0xbd, 0xac, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0xac, 0xbb, 0xac, 0xbb, 0xbe, 0xbb, 0xbe, 0xbb,
  0x98, 0xa9, 0x98, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0x98, 0xcf, 0x98, 0xcf, 0xaa, 0xcf, 0xaa, 0xcf,
  0xac, 0xa9, 0xac, 0xa9, 0xaa, 0xa9, 0xaa, 0xa9, 0xac, 0xcf, 0xac, 0xcf, 0xaa, 0xcf, 0xaa, 0xcf,
  0x98, 0xbd, 0x98, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0x98, 0xcf, 0x98, 0xcf, 0xbe, 0xcf, 0xbe, 0xcf,
  0xac, 0xbd, 0xac, 0xbd, 0xbe, 0xbd, 0xbe, 0xbd, 0xac, 0xcf, 0xac, 0xcf, 0xbe, 0xcf, 0xbe, 0xcf};

/**
 * @brief Returns a 32-bit mask where 1 means 3-byte code length and 0 means 2-byte
 * code length, given an input mask of up to 96 bits.
 *
 * Implemented by doing 8 consecutive lookups, building the result 4-bit at a time
 **/
inline __device__ uint32_t get_len3_mask(uint32_t v0, uint32_t v1, uint32_t v2)
{
  uint32_t m, v, m4, n;
  v  = v0;
  m4 = k_len3lut[v & 0x3ff];
  m  = m4 & 0xf;
  n  = m4 >> 4;  // 8..12
  v  = v0 >> n;
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 4;
  n += m4 >> 4;  // 16..24
  v  = __funnelshift_r(v0, v1, n);
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 8;
  n += m4 >> 4;  // 24..36
  v >>= (m4 >> 4);
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 12;
  n  = (n + (m4 >> 4)) & 0x1f;  // (32..48) % 32 = 0..16
  v1 = __funnelshift_r(v1, v2, n);
  v2 >>= n;
  v  = v1;
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 16;
  n  = m4 >> 4;  // 8..12
  v  = v1 >> n;
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 20;
  n += m4 >> 4;  // 16..24
  v  = __funnelshift_r(v1, v2, n);
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 24;
  n += m4 >> 4;  // 24..36
  v >>= (m4 >> 4);
  m4 = k_len3lut[v & 0x3ff];
  m |= (m4 & 0xf) << 28;
  return m;
}

/**
 * @brief Returns a 32-bit mask where each 2-bit pair contains the symbol length
 * minus 2, given two input masks each containing bit0 or bit1 of the corresponding
 * code length minus 2 for up to 32 bytes
 **/
inline __device__ uint32_t get_len5_mask(uint32_t v0, uint32_t v1)
{
  uint32_t m;
  m = (v1 & 1) * 2 + (v0 & 1);
  v0 >>= (m + 2);
  v1 >>= (m + 1);
  for (uint32_t i = 1; i < 16; i++) {
    uint32_t m2 = (v1 & 2) | (v0 & 1);
    uint32_t n  = m2 + 2;
    m |= m2 << (i * 2);
    v0 >>= n;
    v1 >>= n;
  }
  return m;
}

#define READ_BYTE(pos) s->q.buf[(pos) & (PREFETCH_SIZE - 1)]

/**
 * @brief decode symbols and output LZ77 batches (single-warp)
 *
 * @param s decompression state
 * @param t warp lane id
 **/
__device__ inline void snappy_decode_symbols(unsnap_state_s *s, uint32_t t)
{
  uint32_t cur        = 0;
  uint32_t end        = static_cast<uint32_t>(s->end - s->base);
  uint32_t bytes_left = s->uncompressed_size;
  uint32_t dst_pos    = 0;
  int32_t batch       = 0;

  for (;;) {
    int32_t batch_len = 0;
    volatile unsnap_batch_s *b;

    // Wait for prefetcher
    if (t == 0) {
#pragma unroll(1)  // We don't want unrolling here
      while (s->q.prefetch_wrpos < min(cur + 5 * BATCH_SIZE, end)) { NANOSLEEP(50); }
      b = &s->q.batch[batch * BATCH_SIZE];
    }
    // Process small symbols in parallel: for data that does not get good compression,
    // the stream will consist of a large number of short literals (1-byte or 2-byte)
    // followed by short repeat runs. This results in many 2-byte or 3-byte symbols
    // that can all be decoded in parallel once we know the symbol length.
    {
      uint32_t v0, v1, v2, len3_mask, cur_t, is_long_sym, short_sym_mask;
      uint32_t b0;
      cur            = SHFL0(cur);
      cur_t          = cur + t;
      b0             = READ_BYTE(cur_t);
      v0             = BALLOT((b0 == 4) || (b0 & 2));
      b0             = READ_BYTE(cur_t + 32);
      v1             = BALLOT((b0 == 4) || (b0 & 2));
      b0             = READ_BYTE(cur_t + 64);
      v2             = BALLOT((b0 == 4) || (b0 & 2));
      len3_mask      = SHFL0((t == 0) ? get_len3_mask(v0, v1, v2) : 0);
      cur_t          = cur + 2 * t + __popc(len3_mask & ((1 << t) - 1));
      b0             = READ_BYTE(cur_t);
      is_long_sym    = ((b0 & ~4) != 0) && (((b0 + 1) & 2) == 0);
      short_sym_mask = BALLOT(is_long_sym);
      batch_len      = 0;
      b = reinterpret_cast<volatile unsnap_batch_s *>(SHFL0(reinterpret_cast<uintptr_t>(b)));
      if (!(short_sym_mask & 1)) {
        batch_len = SHFL0((t == 0) ? (short_sym_mask) ? __ffs(short_sym_mask) - 1 : 32 : 0);
        if (batch_len != 0) {
          uint32_t blen = 0;
          int32_t ofs   = 0;
          if (t < batch_len) {
            blen = (b0 & 1) ? ((b0 >> 2) & 7) + 4 : ((b0 >> 2) + 1);
            ofs  = (b0 & 1) ? ((b0 & 0xe0) << 3) | READ_BYTE(cur_t + 1)
                           : (b0 & 2) ? READ_BYTE(cur_t + 1) | (READ_BYTE(cur_t + 2) << 8)
                                      : -(int32_t)(cur_t + 1);
            b[t].len    = blen;
            b[t].offset = ofs;
            ofs += blen;  // for correct out-of-range detection below
          }
          blen           = WarpReducePos32(blen, t);
          bytes_left     = SHFL0(bytes_left);
          dst_pos        = SHFL0(dst_pos);
          short_sym_mask = __ffs(BALLOT(blen > bytes_left || ofs > (int32_t)(dst_pos + blen)));
          if (short_sym_mask != 0) { batch_len = min(batch_len, short_sym_mask - 1); }
          if (batch_len != 0) {
            blen = SHFL(blen, batch_len - 1);
            cur  = SHFL(cur_t, batch_len - 1) + 2 + ((len3_mask >> (batch_len - 1)) & 1);
            if (t == 0) {
              dst_pos += blen;
              bytes_left -= blen;
            }
          }
        }
      }
      // Check if the batch was stopped by a 3-byte or 4-byte literal
      if (batch_len < BATCH_SIZE - 2 && SHFL(b0 & ~4, batch_len) == 8) {
        // If so, run a slower version of the above that can also handle 3/4-byte literal sequences
        uint32_t batch_add;
        do {
          uint32_t clen, mask_t;
          cur_t     = cur + t;
          b0        = READ_BYTE(cur_t);
          clen      = (b0 & 3) ? (b0 & 2) ? 1 : 0 : (b0 >> 2);  // symbol length minus 2
          v0        = BALLOT(clen & 1);
          v1        = BALLOT((clen >> 1) & 1);
          len3_mask = SHFL0((t == 0) ? get_len5_mask(v0, v1) : 0);
          mask_t    = (1 << (2 * t)) - 1;
          cur_t     = cur + 2 * t + 2 * __popc((len3_mask & 0xaaaaaaaa) & mask_t) +
                  __popc((len3_mask & 0x55555555) & mask_t);
          b0          = READ_BYTE(cur_t);
          is_long_sym = ((b0 & 3) ? ((b0 & 3) == 3) : (b0 > 3 * 4)) || (cur_t >= cur + 32) ||
                        (batch_len + t >= BATCH_SIZE);
          batch_add = __ffs(BALLOT(is_long_sym)) - 1;
          if (batch_add != 0) {
            uint32_t blen = 0;
            int32_t ofs   = 0;
            if (t < batch_add) {
              blen = (b0 & 1) ? ((b0 >> 2) & 7) + 4 : ((b0 >> 2) + 1);
              ofs  = (b0 & 1) ? ((b0 & 0xe0) << 3) | READ_BYTE(cur_t + 1)
                             : (b0 & 2) ? READ_BYTE(cur_t + 1) | (READ_BYTE(cur_t + 2) << 8)
                                        : -(int32_t)(cur_t + 1);
              b[batch_len + t].len    = blen;
              b[batch_len + t].offset = ofs;
              ofs += blen;  // for correct out-of-range detection below
            }
            blen           = WarpReducePos32(blen, t);
            bytes_left     = SHFL0(bytes_left);
            dst_pos        = SHFL0(dst_pos);
            short_sym_mask = __ffs(BALLOT(blen > bytes_left || ofs > (int32_t)(dst_pos + blen)));
            if (short_sym_mask != 0) { batch_add = min(batch_add, short_sym_mask - 1); }
            if (batch_add != 0) {
              blen = SHFL(blen, batch_add - 1);
              cur  = SHFL(cur_t, batch_add - 1) + 2 + ((len3_mask >> ((batch_add - 1) * 2)) & 3);
              if (t == 0) {
                dst_pos += blen;
                bytes_left -= blen;
              }
              batch_len += batch_add;
            }
          }
        } while (batch_add >= 6 && batch_len < BATCH_SIZE - 2);
      }
    }
    if (t == 0) {
      uint32_t current_prefetch_wrpos = s->q.prefetch_wrpos;
      while (bytes_left > 0 && batch_len < BATCH_SIZE && min(cur + 5, end) <= current_prefetch_wrpos) {
        uint32_t blen, offset;
        uint8_t b0 = READ_BYTE(cur);
        if (b0 & 3) {
          uint8_t b1 = READ_BYTE(cur + 1);
          if (!(b0 & 2)) {
            // xxxxxx01.oooooooo: copy with 3-bit length, 11-bit offset
            offset = ((b0 & 0xe0) << 3) | b1;
            blen   = ((b0 >> 2) & 7) + 4;
            cur += 2;
          } else {
            // xxxxxx1x: copy with 6-bit length, 2-byte or 4-byte offset
            offset = b1 | (READ_BYTE(cur + 2) << 8);
            if (b0 & 1)  // 4-byte offset
            {
              offset |= (READ_BYTE(cur + 3) << 16) | (READ_BYTE(cur + 4) << 24);
              cur += 5;
            } else {
              cur += 3;
            }
            blen = (b0 >> 2) + 1;
          }
          dst_pos += blen;
          if (offset - 1u >= dst_pos || bytes_left < blen) break;
          bytes_left -= blen;
        } else if (b0 < 4 * 4) {
          // 0000xx00: short literal
          blen   = (b0 >> 2) + 1;
          offset = -(int32_t)(cur + 1);
          cur += 1 + blen;
          dst_pos += blen;
          if (bytes_left < blen) break;
          bytes_left -= blen;
        } else {
          // xxxxxx00: literal
          blen = b0 >> 2;
          if (blen >= 60) {
            uint32_t num_bytes = blen - 59;
            blen               = READ_BYTE(cur + 1);
            if (num_bytes > 1) {
              blen |= READ_BYTE(cur + 2) << 8;
              if (num_bytes > 2) {
                blen |= READ_BYTE(cur + 3) << 16;
                if (num_bytes > 3) { blen |= READ_BYTE(cur + 4) << 24; }
              }
            }
            cur += num_bytes;
          }
          cur += 1;
          blen += 1;
          offset = -(int32_t)cur;
          cur += blen;
          dst_pos += blen;
          if (bytes_left < blen) break;
          bytes_left -= blen;
        }
        b[batch_len].len    = blen;
        b[batch_len].offset = offset;
        batch_len++;
      }
      if (batch_len != 0) {
        s->q.batch_len[batch] = batch_len;
        s->q.batch_prefetch_rdpos[batch] = cur;
        batch                 = (batch + 1) & (BATCH_COUNT - 1);
      }
    }
    batch_len = SHFL0(batch_len);
    bytes_left = SHFL0(bytes_left);
    if (t == 0) {
      while (s->q.batch_len[batch] != 0) { NANOSLEEP(100); }
    }
    if (bytes_left <= 0) { break; }
  }
  if (!t) {
    s->q.prefetch_end     = 1;
    s->q.batch_len[batch] = -1;
    s->bytes_left         = bytes_left;
    if (bytes_left != 0) { s->error = -2; }
  }
}

/**
 * @brief process LZ77 symbols and output uncompressed stream
 *
 * @param s decompression state
 * @param t thread id within participating group (lane id)
 *
 * NOTE: No error checks at this stage (WARP0 responsible for not sending offsets and lengths that
 *would result in out-of-bounds accesses)
 **/
__device__ inline void snappy_process_symbols(unsnap_state_s *s, int t)
{
  const uint8_t *literal_base = s->base;
  uint8_t *out                = reinterpret_cast<uint8_t *>(s->in.dstDevice);
  int batch                   = 0;

  do {
    volatile unsnap_batch_s *b = &s->q.batch[batch * BATCH_SIZE];
    int32_t batch_len, blen_t, dist_t;

    if (t == 0) {
      while ((batch_len = s->q.batch_len[batch]) == 0) { NANOSLEEP(100); }
    } else {
      batch_len = 0;
    }
    batch_len = SHFL0(batch_len);
    if (batch_len <= 0) { break; }
    if (t < batch_len) {
      blen_t = b[t].len;
      dist_t = b[t].offset;
    } else {
      blen_t = dist_t = 0;
    }
    // Try to combine as many small entries as possible, but try to avoid doing that
    // if we see a small repeat distance 8 bytes or less
    if (SHFL0(min((uint32_t)dist_t, (uint32_t)SHFL_XOR(dist_t, 1))) > 8) {
      uint32_t n;
      do {
        uint32_t bofs       = WarpReducePos32(blen_t, t);
        uint32_t stop_mask  = BALLOT((uint32_t)dist_t < bofs);
        uint32_t start_mask = WarpReduceSum32((bofs < 32 && t < batch_len) ? 1 << bofs : 0);
        n = min(min((uint32_t)__popc(start_mask), (uint32_t)(__ffs(stop_mask) - 1u)),
                (uint32_t)batch_len);
        if (n != 0) {
          uint32_t it  = __popc(start_mask & ((2 << t) - 1));
          uint32_t tr  = t - SHFL(bofs - blen_t, it);
          int32_t dist = SHFL(dist_t, it);
          if (it < n) {
            const uint8_t *src = (dist > 0) ? (out + t - dist) : (literal_base + tr - dist);
            out[t]             = *src;
          }
          out += SHFL(bofs, n - 1);
          blen_t = SHFL(blen_t, (n + t) & 0x1f);
          dist_t = SHFL(dist_t, (n + t) & 0x1f);
          batch_len -= n;
        }
      } while (n >= 4);
    }
    uint32_t current_prefetch_wrpos = s->q.prefetch_wrpos;
    for (int i = 0; i < batch_len; i++) {
      int32_t blen  = SHFL(blen_t, i);
      int32_t dist  = SHFL(dist_t, i);
      int32_t blen2 = (i + 1 < batch_len) ? SHFL(blen_t, i + 1) : 32;
      // Try to combine consecutive small entries if they are independent
      if ((uint32_t)dist >= (uint32_t)blen && blen + blen2 <= 32) {
        int32_t dist2 = SHFL(dist_t, i + 1);
        if ((uint32_t)dist2 >= (uint32_t)(blen + blen2)) {
          int32_t d;
          if (t < blen) {
            d = dist;
          } else {
            dist = dist2;
            d    = (dist2 <= 0) ? dist2 + blen : dist2;
          }
          blen += blen2;
          if (t < blen) {
            const uint8_t *src = (dist > 0) ? (out - d) : (literal_base - d);
            out[t]             = src[t];
          }
          out += blen;
          i++;
          continue;
        }
      }
      if (dist > 0) {
        // Copy
        uint8_t b0, b1;
        if (t < blen) {
          uint32_t pos       = t;
          const uint8_t *src = out + ((pos >= dist) ? (pos % dist) : pos) - dist;
          b0                 = *src;
        }
        if (32 + t < blen) {
          uint32_t pos       = 32 + t;
          const uint8_t *src = out + ((pos >= dist) ? (pos % dist) : pos) - dist;
          b1                 = *src;
        }
        if (t < blen) { out[t] = b0; }
        if (32 + t < blen) { out[32 + t] = b1; }
      } else {
        // Literal
        uint8_t b[LITERAL_SECTORS];
        dist = -dist;
#pragma unroll 1
        for(int k = 0; k < blen / (LITERAL_SECTORS * 32u); ++k) {
          if (dist + LITERAL_SECTORS * 32u < current_prefetch_wrpos) {
            for(int i = 0; i < LITERAL_SECTORS; ++i)
              b[i] = READ_BYTE(dist + i * 32u + t);
          } else {
            for(int i = 0; i < LITERAL_SECTORS; ++i)
              b[i] = literal_base[dist + i * 32u + t];
          }
          for(int i = 0; i < LITERAL_SECTORS; ++i)
            out[i * 32u + t] = b[i];
          dist += LITERAL_SECTORS * 32u;
          out += LITERAL_SECTORS * 32u;
        }
        blen %= LITERAL_SECTORS * 32u;
        if (dist + blen < current_prefetch_wrpos) {
          for(int i = 0; i < LITERAL_SECTORS; ++i)
            if (i * 32u + t < blen)
              b[i] = READ_BYTE(dist + i * 32u + t);
        } else {
          for(int i = 0; i < LITERAL_SECTORS; ++i)
            if (i * 32u + t < blen)
              b[i] = literal_base[dist + i * 32u + t];
        }
        for(int i = 0; i < LITERAL_SECTORS; ++i)
          if (i * 32u + t < blen)
            out[i * 32u + t] = b[i];
      }
      out += blen;
    }
    SYNCWARP();
    if (t == 0) {
      s->q.prefetch_rdpos = s->q.batch_prefetch_rdpos[batch];
      s->q.batch_len[batch] = 0;
    }
    batch = (batch + 1) & (BATCH_COUNT - 1);
  } while (1);
}

/**
 * @brief Snappy decompression device function
 **/ 
__device__ inline void do_unsnap(
    const uint8_t* const __restrict__ device_in_ptr,
    const uint64_t device_in_bytes,
    uint8_t* const __restrict__ device_out_ptr,
    const uint64_t device_out_available_bytes,
    nvcompStatus_t* const __restrict__ outputs,
    uint64_t* __restrict__ device_out_bytes)
{
  __shared__ __align__(16) unsnap_state_s state_g;

  int t             = threadIdx.x;
  unsnap_state_s *s = &state_g;

  if (!t) {
    s->in.srcDevice = device_in_ptr;
    s->in.srcSize = device_in_bytes;
    s->in.dstDevice = device_out_ptr;
    s->in.dstSize = device_out_available_bytes;
  }
  if (t < BATCH_COUNT) { s->q.batch_len[t] = 0; }
  __syncthreads();
  if (!t) {
    const uint8_t *cur = reinterpret_cast<const uint8_t *>(s->in.srcDevice);
    const uint8_t *end = cur + s->in.srcSize;
    s->error           = 0;
#if LOG_CYCLECOUNT
    s->tstart = clock();
#endif
    if (cur < end) {
      // Read uncompressed size (varint), limited to 32-bit
      uint32_t uncompressed_size = *cur++;
      if (uncompressed_size > 0x7f) {
        uint32_t c        = (cur < end) ? *cur++ : 0;
        uncompressed_size = (uncompressed_size & 0x7f) | (c << 7);
        if (uncompressed_size >= (0x80 << 7)) {
          c                 = (cur < end) ? *cur++ : 0;
          uncompressed_size = (uncompressed_size & ((0x7f << 7) | 0x7f)) | (c << 14);
          if (uncompressed_size >= (0x80 << 14)) {
            c = (cur < end) ? *cur++ : 0;
            uncompressed_size =
              (uncompressed_size & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | (c << 21);
            if (uncompressed_size >= (0x80 << 21)) {
              c = (cur < end) ? *cur++ : 0;
              if (c < 0x8)
                uncompressed_size =
                  (uncompressed_size & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) |
                  (c << 28);
              else
                s->error = -1;
            }
          }
        }
      }
      s->uncompressed_size = uncompressed_size;
      s->bytes_left        = uncompressed_size;
      s->base              = cur;
      s->end               = end;
      if (s->in.dstSize == 0)
        s->in.dstSize = uncompressed_size;
      if ((cur >= end && uncompressed_size != 0) || (uncompressed_size > s->in.dstSize)) {
        s->error = -1;
      }
    } else {
      s->error = -1;
    }
    s->q.prefetch_end   = 0;
    s->q.prefetch_wrpos = 0;
    s->q.prefetch_rdpos = 0;
  }
  __syncthreads();
  if (!s->error) {
    if (t < 32) {
      // WARP0: decode lengths and offsets
      snappy_decode_symbols(s, t);
    } else if (t < 64) {
      // WARP1: prefetch byte stream for WARP0
      snappy_prefetch_bytestream(s, t & 0x1f);
    } else if (t < 96) {
      // WARP2: LZ77
      snappy_process_symbols(s, t & 0x1f);
    }
    __syncthreads();
  }
  if (!t) {
    if (device_out_bytes)
      *device_out_bytes = s->uncompressed_size - s->bytes_left;
    if (outputs)
      *outputs = s->error ? nvcompErrorCannotDecompress : nvcompSuccess;
  }
}

} // nvcomp namespace
