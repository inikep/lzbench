/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "CudaUtils.h"
#include "LZ4CompressionKernels.h"
#include "TempSpaceBroker.h"
#include "common.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <cub/cub.cuh>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "cuda_runtime.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

using offset_type = uint16_t;
using word_type = uint32_t;
using position_type = size_t;
using double_word_type = uint64_t;
using item_type = uint32_t;

namespace nvcomp {

constexpr const int COMP_THREADS = 32;
constexpr const int DECOMP_THREADS = 32;
constexpr const int Y_DIM = 2;
constexpr const position_type BUFFER_SIZE
    = DECOMP_THREADS * sizeof(double_word_type);
constexpr const position_type PREFETCH_DIST = BUFFER_SIZE / 2;

constexpr const position_type HASH_TABLE_SIZE = 1U << 14;
constexpr const offset_type NULL_OFFSET = static_cast<offset_type>(-1);
constexpr const position_type MAX_OFFSET = (1U << 16) - 1;

constexpr const size_t MIN_CHUNK_SIZE = sizeof(offset_type) * HASH_TABLE_SIZE;
constexpr const size_t MAX_CHUNK_SIZE = 1U << 24; // 16 MB

// ideally this would fit in a quad-word -- right now though it spills into
// 24-bytes (instead of 16-bytes).
struct chunk_header
{
  const uint8_t* src;
  uint8_t* dst;
  uint32_t size;
};

struct compression_chunk_header
{
  const uint8_t* src;
  uint8_t* dst;
  offset_type* hash;
  size_t* comp_size;
  uint32_t size;
};

/******************************************************************************
 * DEVICE FUNCTIONS AND KERNELS ***********************************************
 *****************************************************************************/

inline __device__ __host__ size_t maxSizeOfStream(const size_t size)
{
  const size_t expansion = size + 1 + roundUpDiv(size, 255);
  return roundUpTo(expansion, sizeof(size_t));
}

inline __device__ void syncCTA()
{
  if (DECOMP_THREADS > 32) {
    __syncthreads();
  } else {
    __syncwarp();
  }
}

inline __device__ int warpBallot(int vote)
{
  return __ballot_sync(0xffffffff, vote);
}

template <typename T>
inline __device__ int warpMatchAny(const int participants, T val)
{
#if __CUDA_ARCH__ >= 700
  return __match_any_sync(participants, val);
#else
  int mask = 0;

  // full search
  __shared__ T values[32];
  assert(blockDim.x == 32);
  assert(blockDim.y == 1);
  if ((1 << threadIdx.x) & participants) {
    values[threadIdx.x] = val;
    __syncwarp(participants);
    for (int d = 0; d < 32; ++d) {
      const int nbr_id = (threadIdx.x + d) & 31;
      if ((1 << nbr_id) & participants) {
        const T nbr_val = values[nbr_id];
        mask |= (val == nbr_val) << nbr_id;
      }
    }
    __syncwarp(participants);
  }

  return mask;
#endif
}

template <typename T>
inline __device__ void writeWord(uint8_t* const address, const T word)
{
#pragma unroll
  for (size_t i = 0; i < sizeof(T); ++i) {
    address[i] = static_cast<uint8_t>((word >> (8 * i)) & 0xff);
  }
}

template <typename T>
inline __device__ T readWord(const uint8_t* const address)
{
  T word = 0;
  for (size_t i = 0; i < sizeof(T); ++i) {
    word |= address[i] << (8 * i);
  }

  return word;
}

template<int BLOCK_SIZE>
inline __device__ void writeLSIC(uint8_t* const out, const position_type number)
{
  assert(BLOCK_SIZE == blockDim.x);

  const position_type num = (number / 0xffu) + 1;
  const uint8_t leftOver = number % 0xffu;
  for (size_t i = threadIdx.x; i < num; i += BLOCK_SIZE) {
    const uint8_t val = i + 1 < num ? 0xffu : leftOver;
    out[i] = val;
  }
}

struct token_type
{
  position_type num_literals;
  position_type num_matches;

   __device__ bool hasNumLiteralsOverflow() const
  {
    return num_literals >= 15;
  }

   __device__ bool hasNumMatchesOverflow() const
  {
    return num_matches >= 19;
  }

  __device__ position_type numLiteralsOverflow() const
  {
    if (hasNumLiteralsOverflow()) {
      return num_literals - 15;
    } else {
      return 0;
    }
  }

  __device__ uint8_t numLiteralsForHeader() const
  {
    if (hasNumLiteralsOverflow()) {
      return 15;
    } else {
      return num_literals;
    }
  }

  __device__ position_type numMatchesOverflow() const
  {
    if (hasNumMatchesOverflow()) {
      assert(num_matches >= 19);
      return num_matches - 19;
    } else {
      assert(num_matches < 19);
      return 0;
    }
  }

  __device__ uint8_t numMatchesForHeader() const
  {
    if (hasNumMatchesOverflow()) {
      return 15;
    } else {
      return num_matches - 4;
    }
  }
  __device__ position_type lengthOfLiteralEncoding() const
  {
    if (hasNumLiteralsOverflow()) {
      const position_type num = numLiteralsOverflow();
      const position_type length = (num / 0xff) + 1;
      return length;
    }
    return 0;
  }

  __device__ position_type lengthOfMatchEncoding() const
  {
    if (hasNumMatchesOverflow()) {
      const position_type num = numMatchesOverflow();
      const position_type length = (num / 0xff) + 1;
      return length;
    }
    return 0;
  }
};

class BufferControl
{
public:

  __device__ BufferControl(
      uint8_t* const buffer, const uint8_t* const compData, const position_type length) :
      m_offset(0),
      m_length(length),
      m_buffer(buffer),
      m_compData(compData)
  {
    // do nothing
  }

  #ifdef WARP_READ_LSIC
    // this is currently unused as its slower
  inline __device__ position_type queryLSIC(const position_type idx) const
  {
    if (idx + DECOMP_THREADS <= end()) {
      // most likely case
      const uint8_t byte = rawAt(idx)[threadIdx.x];

      uint32_t mask = warpBallot(byte != 0xff);
      mask = __brev(mask);
  
      const position_type fullBytes = __clz(mask);
  
      if (fullBytes < DECOMP_THREADS) {
        return fullBytes * 0xff + rawAt(idx)[fullBytes];
      } else {
        return DECOMP_THREADS * 0xff;
      }
    } else {
      uint8_t byte;
      if (idx + threadIdx.x < end()) {
        byte = rawAt(idx)[threadIdx.x];
      } else {
        byte = m_compData[idx + threadIdx.x];
      }

      uint32_t mask = warpBallot(byte != 0xff);
      mask = __brev(mask);
  
      const position_type fullBytes = __clz(mask);
  
      if (fullBytes < DECOMP_THREADS) {
        return fullBytes * 0xff + __shfl_sync(0xffffffff, byte, fullBytes);
      } else {
        return DECOMP_THREADS * 0xff;
      }
    }
  }
  #endif
  
  inline __device__ position_type readLSIC(position_type& idx) const
  {
  #ifdef WARP_READ_LSIC
    position_type num = 0;
    while (true) {
      const position_type block = queryLSIC(idx);
      num += block;
  
      if (block < DECOMP_THREADS * 0xff) {
        idx += (block / 0xff) + 1;
        break;
      } else {
        idx += DECOMP_THREADS;
      }
    }
    return num;
  #else
    position_type num = 0;
    uint8_t next = 0xff;
    // read from the buffer
    while (next == 0xff && idx < end()) {
      next = rawAt(idx)[0];
      ++idx;
      num += next;
    }
      // read from global memory
    while (next == 0xff) {
      next = m_compData[idx];
      ++idx;
      num += next;
    }
    return num;
  #endif
  }
  
    inline __device__ const uint8_t* raw() const
    {
      return m_buffer;
    }
  
    inline __device__ const uint8_t* rawAt(const position_type i) const
    {
      return raw() + (i - begin());
    }
    inline __device__ uint8_t operator[](const position_type i) const
    {
      if (i >= m_offset && i - m_offset < BUFFER_SIZE) {
        return m_buffer[i - m_offset];
      } else {
        return m_compData[i];
      }
    }

    inline __device__ void setAndAlignOffset(const position_type offset)
    {
      static_assert(
          sizeof(size_t) == sizeof(const uint8_t*),
          "Size of pointer must be equal to size_t.");

      const uint8_t* const alignedPtr = reinterpret_cast<const uint8_t*>(
          (reinterpret_cast<size_t>(m_compData + offset)
           / sizeof(double_word_type))
          * sizeof(double_word_type));

      m_offset = alignedPtr - m_compData;
    }

    inline __device__ void loadAt(const position_type offset)
    {
      setAndAlignOffset(offset);

      if (m_offset + BUFFER_SIZE <= m_length) {
        assert(
            reinterpret_cast<size_t>(m_compData + m_offset)
                % sizeof(double_word_type)
            == 0);
        assert(BUFFER_SIZE == DECOMP_THREADS * sizeof(double_word_type));
        const double_word_type* const word_data
            = reinterpret_cast<const double_word_type*>(m_compData + m_offset);
        double_word_type* const word_buffer
            = reinterpret_cast<double_word_type*>(m_buffer);
        word_buffer[threadIdx.x] = word_data[threadIdx.x];
      } else {
  #pragma unroll
      for (int i = threadIdx.x; i < BUFFER_SIZE; i += DECOMP_THREADS) {
        if (m_offset + i < m_length) {
          m_buffer[i] = m_compData[m_offset + i];
        }
      }
    }
  
    syncCTA();
  }
  
  inline __device__ position_type begin() const
  {
    return m_offset;
  }
  
  
  inline __device__ position_type end() const
  {
    return m_offset + BUFFER_SIZE;
  }

private:
  position_type m_offset;
  const position_type m_length;
  uint8_t* const m_buffer;
  const uint8_t* const m_compData;
}; //End BufferControl Class


inline __device__ void coopCopyNoOverlap(
    uint8_t* const dest, const uint8_t* const source, const size_t length)
{
  for (size_t i = threadIdx.x; i < length; i += blockDim.x) {
    dest[i] = source[i];
  }
}

inline __device__ void coopCopyRepeat(
    uint8_t* const dest,
    const uint8_t* const source,
    const position_type dist,
    const position_type length)
{
// if there is overlap, it means we repeat, so we just
// need to organize our copy around that
  for (position_type i = threadIdx.x; i < length; i += blockDim.x) {
    dest[i] = source[i % dist];
  }
}

inline __device__ void coopCopyOverlap(
    uint8_t* const dest,
    const uint8_t* const source,
    const position_type dist,
    const position_type length)
{
  if (dist < length) {
    coopCopyRepeat(dest, source, dist, length);
  } else {
    coopCopyNoOverlap(dest, source, length);
  }
}

inline __device__ position_type hash(const word_type key)
{
  // needs to be 12 bits
  return (__brev(key) + (key^0xc375)) & (HASH_TABLE_SIZE - 1);
}

inline __device__ uint8_t encodePair(const uint8_t t1, const uint8_t t2)
{
  return ((t1 & 0x0f) << 4) | (t2 & 0x0f);
}

inline __device__ token_type decodePair(const uint8_t num)
{
  return token_type{static_cast<uint8_t>((num & 0xf0) >> 4),
                    static_cast<uint8_t>(num & 0x0f)};
}

template<int BLOCK_SIZE>
inline __device__ void copyLiterals(
    uint8_t* const dest, const uint8_t* const source, const size_t length)
{
  assert(BLOCK_SIZE == blockDim.x);
  for (size_t i = threadIdx.x; i < length; i += BLOCK_SIZE) {
    dest[i] = source[i];
  }
}

inline __device__ position_type lengthOfMatch(
    const uint8_t* const data,
    const position_type prev_location,
    const position_type next_location,
    const position_type length)
{
  assert(prev_location < next_location);

  position_type match_length = length - next_location - 5;
  for (position_type j = 0; j + next_location + 5 < length; j += blockDim.x) {
    const position_type i = threadIdx.x + j;
    int match = i + next_location + 5 < length
                    ? (data[prev_location + i] != data[next_location + i])
                    : 1;
    match = warpBallot(match);
    if (match) {
      match_length = j + __clz(__brev(match));
      break;
    }
  }

  return match_length;
}

inline __device__ position_type
convertIdx(const offset_type offset, const position_type pos)
{
  constexpr const position_type OFFSET_SIZE = MAX_OFFSET + 1;

  assert(offset <= pos);

  position_type realPos = (pos / OFFSET_SIZE) * OFFSET_SIZE + offset;
  if (realPos >= pos) {
    realPos -= OFFSET_SIZE;
  }
  assert(realPos < pos);

  return realPos;
}

inline __device__ bool isValidHash(
    const uint8_t* const data,
    const offset_type* const hashTable,
    const position_type key,
    const position_type hashPos,
    const position_type decomp_idx)
{
  const position_type hashed_offset = hashTable[hashPos];

  if (hashed_offset == NULL_OFFSET) {
    return false;
  }

  const position_type offset = convertIdx(hashed_offset, decomp_idx);

  if (decomp_idx - offset > MAX_OFFSET) {
    // can't match current position, ahead, or NULL_OFFSET
    return false;
  }

  const word_type hashKey = readWord<word_type>(data + offset);

  if (hashKey != key) {
    return false;
  }

  return true;
}

template<int BLOCK_SIZE>
inline __device__ void writeSequenceData(
    uint8_t* const compData,
    const uint8_t* const decompData,
    const token_type token,
    const offset_type offset,
    const position_type decomp_idx,
    position_type& comp_idx)
{
  assert(token.num_matches == 0 || token.num_matches >= 4);

  // -> add token
  if (threadIdx.x == 0) {
    compData[comp_idx]
        = encodePair(token.numLiteralsForHeader(), token.numMatchesForHeader());
  }
  ++comp_idx;

  // -> add literal length
  const position_type literalEncodingLength = token.lengthOfLiteralEncoding();
  if (literalEncodingLength) {
    writeLSIC<BLOCK_SIZE>(compData + comp_idx, token.numLiteralsOverflow());
    comp_idx += literalEncodingLength;
  }

  // -> add literals
  copyLiterals<BLOCK_SIZE>(
      compData + comp_idx, decompData + decomp_idx, token.num_literals);
  comp_idx += token.num_literals;

  // -> add offset
  if (token.num_matches > 0) {
    assert(offset > 0);

    if (threadIdx.x == 0) {
      writeWord(compData + comp_idx, offset);
    }
    comp_idx += sizeof(offset);

    // -> add match length
    if (token.hasNumMatchesOverflow()) {
      writeLSIC<BLOCK_SIZE>(compData + comp_idx, token.numMatchesOverflow());
      comp_idx += token.lengthOfMatchEncoding();
    }
  }
}

inline __device__ int numValidThreadsToMask(
    const int numValidThreads)
{
  return 0xffffffff >> (32-numValidThreads);
}

inline __device__ void insertHashTableWarp(
    offset_type* hashTable,
    const offset_type pos,
    const word_type next,
    const int numValidThreads)
{
  position_type hashPos = hash(next);

  if (threadIdx.x < numValidThreads) {
    const int match = warpMatchAny(numValidThreadsToMask(numValidThreads), hashPos);
    if (!match || 31 - __clz(match) == threadIdx.x) {
      // I'm the last match -- can insert
      hashTable[hashPos] = pos & MAX_OFFSET;
    }
  }

  __syncwarp();
}

__device__ void compressStream(
    uint8_t* compData,
    const uint8_t* decompData,
    offset_type* const hashTable,
    const size_t length,
    size_t* comp_length)
{
  assert(blockDim.x == COMP_THREADS);
  static_assert(COMP_THREADS <= 32, "Compression can be done with at "
      "most one warp");

  position_type decomp_idx = 0;
  position_type comp_idx = 0;

  for (position_type i = threadIdx.x; i < HASH_TABLE_SIZE; i += COMP_THREADS) {
    hashTable[i] = NULL_OFFSET;
  }

  __syncwarp();

  while (decomp_idx < length) {
    const position_type tokenStart = decomp_idx;
    while (true) {
      if (decomp_idx + 5 + 4 >= length) {
        // jump to end
        decomp_idx = length;

        // no match -- literals to the end
        token_type tok;
        tok.num_literals = length - tokenStart;
        tok.num_matches = 0;
        writeSequenceData<COMP_THREADS>(compData, decompData, tok, 0, tokenStart, comp_idx);
        break;
      }

      // begin adding tokens to the hash table until we find a match
      uint8_t byte = 0;
      if (decomp_idx + 5 + threadIdx.x < length) {
        byte = decompData[decomp_idx + threadIdx.x];
      }

      // each thread needs a four byte word, but only separated by a byte e.g.:
      // for two threads, the five bytes [ 0x12 0x34 0x56 0x78 0x9a ] would
      // be assigned as [0x78563412 0x9a785634 ] to the two threads
      // (little-endian). That means when reading 32 bytes, we can only fill
      // the first 29 thread's 4-byte words.
      word_type next = byte;
      // collect second byte
      next |= __shfl_down_sync(0xffffffff, byte, 1) << 8;
      // collect third and fourth bytes
      next |= __shfl_down_sync(0xffffffff, next, 2) << 16;

      // since we do not have valid data for the last 3 threads (or more if
      // we're at the end of the data), mark them as inactive.
      const int numValidThreads
          = min(static_cast<size_t>(COMP_THREADS - 3), length - decomp_idx - 9);

      // first try to find a local match
      position_type match_location = length;
      int match_mask_self = 0;
      if (threadIdx.x < numValidThreads) {
        match_mask_self = warpMatchAny(numValidThreadsToMask(numValidThreads), next);
      }

      // each thread has a mask of other threads with matches, next we need
      // to find the first thread with a match before it
      const int match_mask_warp = warpBallot(
          match_mask_self && __clz(__brev(match_mask_self)) != threadIdx.x);

      int first_match_thread;
      if (match_mask_warp) {
        // find the byte offset (thread id) within the warp where the first
        // match is located
        first_match_thread = __clz(__brev(match_mask_warp));

        // determine the global position for the finding thread
        match_location = __clz(__brev(match_mask_self)) + decomp_idx;

        // comunicate the global position of the match to other threads
        match_location = __shfl_sync(0xffffffff, match_location, first_match_thread);
      } else {
        first_match_thread = COMP_THREADS;
      }

      // only go to the hash table, if there is a possibility of a finding an
      // earlier match
      if (first_match_thread > 0) {
        // go to hash table for an earlier match
        position_type hashPos = hash(next);
        const int match_found = threadIdx.x < numValidThreads
                                    ? isValidHash(
                                          decompData,
                                          hashTable,
                                          next,
                                          hashPos,
                                          decomp_idx + threadIdx.x)
                                    : 0;

        // determine the first thread to find a match
        const int match = warpBallot(match_found);
        const int candidate_first_match_thread = __clz(__brev(match));

        assert(candidate_first_match_thread != threadIdx.x || match_found);
        assert(!match_found || candidate_first_match_thread <= threadIdx.x);

        if (candidate_first_match_thread < first_match_thread) {
          // if we found a valid match, and it occurs before a previously found
          // match, use that
          first_match_thread = candidate_first_match_thread;
          hashPos = __shfl_sync(0xffffffff, hashPos, first_match_thread);
          match_location
              = convertIdx(hashTable[hashPos], decomp_idx + first_match_thread);
        }
      }

      if (match_location != length) {
        // insert up to the match into the hash table
        insertHashTableWarp(
            hashTable, decomp_idx + threadIdx.x, next, first_match_thread);

        const position_type pos = decomp_idx + first_match_thread;
        assert(match_location < pos);
        assert(pos - match_location <= MAX_OFFSET);

        // we found a match
        const offset_type match_offset = pos - match_location;
        assert(match_offset > 0);
        assert(match_offset <= pos);
        const position_type num_literals = pos - tokenStart;

        // compute match length
        const position_type num_matches
            = lengthOfMatch(decompData, match_location, pos, length);

        // -> write our token and literal length
        token_type tok;
        tok.num_literals = num_literals;
        tok.num_matches = num_matches;

        // update our position
        decomp_idx = tokenStart + num_matches + num_literals;

        // insert only the literals into the hash table
        writeSequenceData<COMP_THREADS>(
            compData, decompData, tok, match_offset, tokenStart, comp_idx);
        break;
      }

      // insert everything into hash table
      insertHashTableWarp(
          hashTable, decomp_idx + threadIdx.x, next, numValidThreads);

      decomp_idx += numValidThreads;
    }
  }

  if (threadIdx.x == 0) {
    *comp_length = comp_idx;
  }
}

inline __device__ void decompressStream(
    uint8_t* buffer,
    uint8_t* decompData,
    const uint8_t* compData,
    position_type length)
{
  position_type comp_end = length;

  BufferControl ctrl(buffer, compData, comp_end);
  ctrl.loadAt(0);

  position_type decomp_idx = 0;
  position_type comp_idx = 0;
  while (comp_idx < comp_end) {
    if (comp_idx + PREFETCH_DIST > ctrl.end()) {
      ctrl.loadAt(comp_idx);
    }

    // read header byte
    token_type tok = decodePair(*ctrl.rawAt(comp_idx));
    ++comp_idx;

    // read the length of the literals
    position_type num_literals = tok.num_literals;
    if (tok.num_literals == 15) {
      num_literals += ctrl.readLSIC(comp_idx);
    }
    const position_type literalStart = comp_idx;

    // copy the literals to the out stream
    if (num_literals + comp_idx > ctrl.end()) {
      coopCopyNoOverlap(
          decompData + decomp_idx, compData + comp_idx, num_literals);
    } else {
      // our buffer can copy
      coopCopyNoOverlap(
          decompData + decomp_idx, ctrl.rawAt(comp_idx), num_literals);
    }

    comp_idx += num_literals;
    decomp_idx += num_literals;

    // Note that the last sequence stops right after literals field.
    // There are specific parsing rules to respect to be compatible with the
    // reference decoder : 1) The last 5 bytes are always literals 2) The last
    // match cannot start within the last 12 bytes Consequently, a file with
    // less then 13 bytes can only be represented as literals These rules are in
    // place to benefit speed and ensure buffer limits are never crossed.
    if (comp_idx < comp_end) {

      // read the offset
      offset_type offset;
      if (comp_idx + sizeof(offset_type) > ctrl.end()) {
        offset = readWord<offset_type>(compData + comp_idx);
      } else {
        offset = readWord<offset_type>(ctrl.rawAt(comp_idx));
      }

      comp_idx += sizeof(offset_type);

      // read the match length
      position_type match = 4 + tok.num_matches;
      if (tok.num_matches == 15) {
        match += ctrl.readLSIC(comp_idx);
      }

      // copy match
      if (offset <= num_literals
          && (ctrl.begin() <= literalStart
              && ctrl.end() >= literalStart + num_literals)) {
        // we are using literals already present in our buffer

        coopCopyOverlap(
            decompData + decomp_idx,
            ctrl.rawAt(literalStart + (num_literals - offset)),
            offset,
            match);
        // we need to sync after we copy since we use the buffer
        syncCTA();
      } else {
        // we need to sync before we copy since we use decomp
        syncCTA();

        coopCopyOverlap(
            decompData + decomp_idx,
            decompData + decomp_idx - offset,
            offset,
            match);
      }
      decomp_idx += match;
    }
  }
  assert(comp_idx == comp_end);
}

template <typename T>
struct BlockPrefixCallbackOp
{
  T m_running_total;

  __device__ BlockPrefixCallbackOp(const T running_total) :
      m_running_total(running_total)
  {
  }

  __device__ T operator()(const T block_aggregate)
  {
    const T old_prefix = m_running_total;
    m_running_total += block_aggregate;
    return old_prefix;
  }

  __device__ T total() const
  {
    return m_running_total;
  }
};

template <int BLOCK_SIZE>
inline __device__ void generateItemChunkMappings(
    const size_t* const decomp_sizes,
    const size_t target_chunk,
    const size_t batch_size,
    const int max_chunk_size,
    item_type& item,
    size_t& local_chunk)
{
  using BlockScan = typename cub::BlockScan<size_t, BLOCK_SIZE>;

  // each thread is assigned a chunk, and they cooperatively prefix sum
  // the items, and then write out the their chunk's item. The first thread
  // assigned to any given item also writes out it's prefix

  __shared__ typename BlockScan::TempStorage temp_space;
  __shared__ size_t local_prefix[BLOCK_SIZE + 1];

  BlockPrefixCallbackOp<size_t> prefix_op(0);

  item = static_cast<item_type>(-1);

  // we have thread per item computing the prefix sum
  for (size_t item_start = 0; item_start < batch_size;
       item_start += BLOCK_SIZE) {
    const size_t i = item_start + threadIdx.x;
    const size_t item_chunks
        = i < batch_size ? roundUpDiv(decomp_sizes[i], max_chunk_size) : 0;

    BlockScan(temp_space)
        .ExclusiveSum(item_chunks, local_prefix[threadIdx.x], prefix_op);
    if (threadIdx.x == 0) {
      local_prefix[BLOCK_SIZE] = prefix_op.total();
    }
    __syncthreads();

    // if a thread's chunk lies in this set of items
    if (target_chunk >= local_prefix[0]
        && target_chunk < local_prefix[BLOCK_SIZE]) {
      int beg = item_start;
      int end = min(item_start + BLOCK_SIZE, batch_size) - 1;

      // Binary search for the right chunk -- we know it exists, so we don't
      // have to handle cases of before or after the sequence. We find the
      // first index the target chunk is less than
      item = (end + beg) / 2;
      while (beg < end) {
        const size_t chunk = local_prefix[item + 1 - item_start];

        if (chunk <= target_chunk) {
          assert(beg != chunk);
          // the current mid-point works as a lower bound
          beg = item + 1;
        } else {
          // the current mid-point does not work as a lower bound, so it must
          // work as an upper bound
          end = item;
        }
        item = (end + beg) / 2;
      }

      // the target for this thread is here
      local_chunk = target_chunk - local_prefix[item - item_start];
    }

    __syncthreads();
  }
}

template <int BLOCK_SIZE>
__global__ void lz4CompressGenerateHeaders(
    const uint8_t* const* const decomp_data,
    const size_t* const decomp_sizes,
    uint8_t* const comp_data,
    size_t* const* const comp_sizes,
    const size_t batch_size,
    const int max_chunk_size,
    const size_t total_chunks,
    const size_t* const scratch_space_offset,
    uint8_t* const* const scratch_space,
    compression_chunk_header* const headers,
    size_t* const item_prefix,
    item_type* const item_map)
{
  const size_t target_chunk = BLOCK_SIZE * blockIdx.x + threadIdx.x;

  item_type item;
  size_t local_chunk;

  generateItemChunkMappings<BLOCK_SIZE>(
      decomp_sizes,
      target_chunk,
      batch_size,
      max_chunk_size,
      item,
      local_chunk);

  // write out items and chunk id's
  if (target_chunk < total_chunks) {
    if (local_chunk == 0) {
      item_prefix[item] = target_chunk;
    }

    assert(item < batch_size);
    item_map[target_chunk] = item;

    const size_t chunk_offset
        = local_chunk * static_cast<size_t>(max_chunk_size);
    const size_t chunk_end = chunk_offset + max_chunk_size;

    const size_t comp_offset = maxSizeOfStream(max_chunk_size) * target_chunk;

    compression_chunk_header h;
    h.src = decomp_data[item] + chunk_offset;
    h.dst = comp_data + comp_offset;
    h.hash = reinterpret_cast<offset_type*>(
        scratch_space[item] + scratch_space_offset[item]
        + (HASH_TABLE_SIZE * sizeof(offset_type)) * local_chunk);
    h.comp_size = comp_sizes[item] + local_chunk;
    h.size = min(chunk_end, decomp_sizes[item]) - chunk_offset;

    headers[target_chunk] = h;
  }
}

__global__ void
lz4CompressMultistreamKernel(const compression_chunk_header* const headers)
{
  const int bidx = blockIdx.x * blockDim.y + threadIdx.y;

  const uint8_t* decomp_ptr = headers[bidx].src;
  const size_t decomp_length = headers[bidx].size;

  uint8_t* comp_ptr = headers[bidx].dst;
  size_t* const comp_length = headers[bidx].comp_size;

  compressStream(
      comp_ptr, decomp_ptr, headers[bidx].hash, decomp_length, comp_length);
}

template <int BLOCK_SIZE>
__global__ void lz4CompressSumSizes(
    size_t* const* const sizes,
    const size_t* const offsets,
    const size_t* const decomp_sizes,
    const size_t chunk_size)
{
  using BlockScan = typename cub::BlockScan<size_t, BLOCK_SIZE>;

  __shared__ typename BlockScan::TempStorage temp_space;

  BlockPrefixCallbackOp<size_t> prefix_op(offsets[blockIdx.x]);

  const size_t num = roundUpDiv(decomp_sizes[blockIdx.x], chunk_size);

  size_t size = 0;
  for (size_t i = 0; i < num; i += BLOCK_SIZE) {
    const size_t index = i + threadIdx.x;
    if (index < num) {
      size = sizes[blockIdx.x][index];
    } else {
      size = 0;
    }
    BlockScan(temp_space).ExclusiveSum(size, size, prefix_op);

    if (index < num) {
      sizes[blockIdx.x][index] = size;
    }

    __syncthreads();
  }

  if (threadIdx.x == 0) {
    sizes[blockIdx.x][num] = prefix_op.total();
  }
}

template <int BLOCK_SIZE>
__global__ void copyToContig(
    const item_type* const item_map,
    const size_t* const item_prefix,
    const uint8_t* const temp_data,
    const int stride,
    const size_t* const* const comp_prefix,
    uint8_t* const* const comp_data)
{
  const size_t global_chunk = blockIdx.x;
  const size_t item = item_map[global_chunk];

  // we assume there are no empty items
  assert(item <= global_chunk);

  const size_t local_chunk = global_chunk - item_prefix[item];
  const size_t offset = comp_prefix[item][local_chunk];
  const size_t size = comp_prefix[item][local_chunk + 1] - offset;

  for (size_t i = threadIdx.x; i < size; i += BLOCK_SIZE) {
    comp_data[item][offset + i] = temp_data[stride * global_chunk + i];
  }
}

__global__ void lz4DecompressMultistreamKernel(
    const chunk_header* const headers, const int num_chunks)
{
  const int bid = blockIdx.x * Y_DIM + threadIdx.y;

  __shared__ uint8_t buffer[BUFFER_SIZE * Y_DIM];

  if (bid < num_chunks) {
    uint8_t* const decomp_ptr = headers[bid].dst;
    const uint8_t* const comp_ptr = headers[bid].src;
    const size_t chunk_length = headers[bid].size;

    decompressStream(
        buffer + threadIdx.y * BUFFER_SIZE, decomp_ptr, comp_ptr, chunk_length);
  }
}

__global__ void lz4DecompressGenerateHeaders(
    uint8_t* const decomp_data,
    const uint8_t* const comp_data,
    const size_t* const comp_chunk_prefix,
    const size_t decomp_chunk_size,
    const size_t num_chunks,
    chunk_header* const headers)
{
  const int chunk = threadIdx.x + blockIdx.x * blockDim.x;

  if (chunk < num_chunks) {
    const size_t comp_chunk_offset = comp_chunk_prefix[chunk];
    const size_t decomp_chunk_offset = chunk * decomp_chunk_size;

    chunk_header h;
    h.src = comp_data + comp_chunk_offset;
    h.dst = decomp_data + decomp_chunk_offset;
    h.size = comp_chunk_prefix[chunk + 1] - comp_chunk_prefix[chunk];

    headers[chunk] = h;
  }
}

/******************************************************************************
 * PUBLIC FUNCTIONS ***********************************************************
 *****************************************************************************/

void lz4CompressBatch(
    const uint8_t* const* const decomp_data_device,
    const size_t* const decomp_prefixes_device,
    const size_t* const decomp_sizes_host,
    const size_t batch_size,
    const size_t max_chunk_size,
    uint8_t* const temp_data_device,
    const size_t temp_bytes,
    uint8_t* const* const comp_data_device,
    size_t* const* const comp_prefixes_device,
    const size_t* const comp_prefix_offset_device,
    cudaStream_t stream)
{
  if (max_chunk_size < lz4MinChunkSize()) {
    throw std::runtime_error(
        "Minimum chunk size for LZ4 is " + std::to_string(MIN_CHUNK_SIZE));
  } else if (max_chunk_size > lz4MaxChunkSize()) {
    throw std::runtime_error(
        "Maximum chunk size for LZ4 is " + std::to_string(MAX_CHUNK_SIZE));
  }

  // most of the kernels take a negligible amount of time, so by default we
  // just use 128 threads. this value, however is choose arbitrarily, and
  // has not been tuned for any architecture or dataset size.
  constexpr const int BLOCK_SIZE = 128;

  const size_t stride = lz4ComputeMaxSize(max_chunk_size);

  const size_t chunks_in_batch
      = lz4ComputeChunksInBatch(decomp_sizes_host, batch_size, max_chunk_size);

  TempSpaceBroker broker(temp_data_device, temp_bytes);

  uint8_t* staging_space;
  broker.reserve(&staging_space, chunks_in_batch * stride);

  compression_chunk_header* headers;
  broker.reserve(&headers, chunks_in_batch);

  // look up starting chunk per item
  size_t* item_prefix;
  broker.reserve(&item_prefix, batch_size);

  // look up item per chunk
  item_type* item_map;
  broker.reserve(&item_map, chunks_in_batch);

  // setup headers
  {
    const dim3 grid(roundUpDiv(chunks_in_batch, BLOCK_SIZE));
    const dim3 block(BLOCK_SIZE);

    lz4CompressGenerateHeaders<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        decomp_data_device,
        decomp_prefixes_device,
        staging_space,
        comp_prefixes_device,
        batch_size,
        max_chunk_size,
        chunks_in_batch,
        comp_prefix_offset_device,
        comp_data_device,
        headers,
        item_prefix,
        item_map);
    CudaUtils::check_last_error();
  }

  // perform compression
  {
    const dim3 grid(chunks_in_batch);
    const dim3 block(COMP_THREADS);

    lz4CompressMultistreamKernel<<<grid, block, 0, stream>>>(headers);
    CudaUtils::check_last_error();
  }

  // perform prefix sum
  {
    const dim3 grid(batch_size);
    const dim3 block(BLOCK_SIZE);

    lz4CompressSumSizes<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        comp_prefixes_device,
        comp_prefix_offset_device,
        decomp_prefixes_device,
        max_chunk_size);
    CudaUtils::check_last_error();
  }

  {
    const dim3 grid(chunks_in_batch);
    // Since we are copying a whole chunk per thread block, maximize the number
    // of threads we have copying each block
    const dim3 block(1024);

    // Copy prefix sums values to metadata header and copy compressed data into
    // contiguous space
    copyToContig<1024><<<grid, block, 0, stream>>>(
        item_map,
        item_prefix,
        staging_space,
        stride,
        comp_prefixes_device,
        comp_data_device);
    CudaUtils::check_last_error();
  }
}

void lz4DecompressBatches(
    void* const temp_space,
    const size_t temp_size,
    void* const* decompData,
    const uint8_t* const* compData,
    int batch_size,
    const size_t** compPrefix,
    int chunk_size,
    int* chunks_in_item,
    cudaStream_t stream)
{   
  TempSpaceBroker broker(temp_space, temp_size);

  int total_chunks=0;
  for(int i=0; i<batch_size; i++) {
    total_chunks += chunks_in_item[i];
  }

  chunk_header* headers;
  broker.reserve(&headers, total_chunks);

  int chunk_start=0;

  for(int i=0; i<batch_size; i++) {

    const dim3 header_block(128);
    const dim3 header_grid(roundUpDiv(chunks_in_item[i], header_block.x));

    lz4DecompressGenerateHeaders<<<header_grid, header_block, 0, stream>>>(
        static_cast<uint8_t*>(decompData[i]),
        compData[i],
        compPrefix[i],
        chunk_size,
        chunks_in_item[i],
        &headers[chunk_start]);

    chunk_start += chunks_in_item[i];
  }

  lz4DecompressMultistreamKernel<<<
      roundUpDiv(total_chunks, Y_DIM),
      dim3(DECOMP_THREADS, Y_DIM, 1),
      0,
      stream>>>(headers, total_chunks);


}


size_t lz4ComputeChunksInBatch(
    const size_t* const decomp_data_size,
    const size_t batch_size,
    const size_t chunk_size)
{
  size_t num_chunks = 0;

  for (size_t i = 0; i < batch_size; ++i) {
    num_chunks += roundUpDiv(decomp_data_size[i], chunk_size);
  }

  return num_chunks;
}

size_t lz4CompressComputeTempSize(
    const size_t maxChunksInBatch, const size_t chunkSize)
{
  const size_t batch_size = 1;

  size_t prefix_temp_size;
  cudaError_t err = cub::DeviceScan::InclusiveSum(
      NULL,
      prefix_temp_size,
      static_cast<const size_t*>(nullptr),
      static_cast<size_t*>(nullptr),
      maxChunksInBatch + 1);
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "Failed to get space for cub inclusive sub: " + std::to_string(err));
  }

  const size_t strideSize = lz4ComputeMaxSize(chunkSize);

  const size_t staging_size
      = roundUpTo(strideSize * maxChunksInBatch, sizeof(size_t));

  const size_t prefix_out_size = sizeof(size_t) * (maxChunksInBatch + 1);
  const size_t header_size = roundUpTo(
      sizeof(compression_chunk_header) * maxChunksInBatch, sizeof(size_t));
  const size_t map_size
      = roundUpTo(sizeof(uint32_t) * maxChunksInBatch, sizeof(size_t));
  const size_t prefix_size = sizeof(size_t) * batch_size;

  return prefix_temp_size + prefix_out_size + staging_size + +header_size
         + map_size + prefix_size;
}

size_t lz4DecompressComputeTempSize(
    const size_t maxChunksInBatch, const size_t /* chunkSize */)
{
  const size_t header_size = sizeof(chunk_header) * maxChunksInBatch;

  return roundUpTo(header_size, sizeof(size_t));
}

size_t lz4ComputeMaxSize(const size_t size)
{
  return maxSizeOfStream(size);
}

size_t lz4MinChunkSize()
{
  return MIN_CHUNK_SIZE;
}

size_t lz4MaxChunkSize()
{
  return MAX_CHUNK_SIZE;
}

} // nvcomp namespace

