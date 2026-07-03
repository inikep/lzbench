// Copyright 2005 Google Inc. All Rights Reserved.
// Modifications Copyright (C) 2022-2026, Advanced Micro Devices. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/*
* ISA selection C, AVX, BMI2, etc are done at compile time in the reference library
* for some functions.
* Such checks are done at runtime in AOCL. Multiple variants of these functions
* are created. Naming convention: 
*   <func_name>_c   : Variant of <func_name> without any advanced ISA
*   <func_name>_avx : Variant of <func_name> with AVX or above support
*   <func_name>_bmi : Variant of <func_name> with BMI2 support
* 
* AOCL_<func_name>  : AOCL optimized variant of <func_name>
*/

#include "snappy-internal.h"
#include "snappy-sinksource.h"
#include "snappy.h"

// ISA selection using compile time flags is replaced with runtime selection

//#if !defined(SNAPPY_HAVE_BMI2)
//// __BMI2__ is defined by GCC and Clang. Visual Studio doesn't target BMI2
// specifically, but it does define __AVX2__ when AVX2 support is available.
// Fortunately, AVX2 was introduced in Haswell, just like BMI2.
//
// BMI2 is not defined as a subset of AVX2 (unlike SSSE3 and AVX above). So,
// GCC and Clang can build code with AVX2 enabled but BMI2 disabled, in which
// case issuing BMI2 instructions results in a compiler error.
//#if defined(__BMI2__) || (defined(_MSC_VER) && defined(__AVX2__))
//#define SNAPPY_HAVE_BMI2 1
//#else
//#define SNAPPY_HAVE_BMI2 0
//#endif
//#endif  // !defined(SNAPPY_HAVE_BMI2)

#if !defined(SNAPPY_HAVE_X86_CRC32)
#if defined(__SSE4_2__)
#define SNAPPY_HAVE_X86_CRC32 1
#else
#define SNAPPY_HAVE_X86_CRC32 0
#endif
#endif  // !defined(SNAPPY_HAVE_X86_CRC32)

#if !defined(SNAPPY_HAVE_NEON_CRC32)
#if SNAPPY_HAVE_NEON && defined(__ARM_FEATURE_CRC32)
#define SNAPPY_HAVE_NEON_CRC32 1
#else
#define SNAPPY_HAVE_NEON_CRC32 0
#endif
#endif  // !defined(SNAPPY_HAVE_NEON_CRC32)

#if SNAPPY_HAVE_BMI2 || SNAPPY_HAVE_X86_CRC32
// Please do not replace with <x86intrin.h>. or with headers that assume more
// advanced SSE versions without checking with all the OWNERS.
#include <immintrin.h>
#elif SNAPPY_HAVE_NEON_CRC32
#include <arm_acle.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// AOCL definitions start
#ifdef AOCL_SNAPPY_AVX_OPT
#include<immintrin.h>
#define AOCL_SNAPPY_TARGET_AVX __attribute__((__target__("avx")))
#else
#define AOCL_SNAPPY_TARGET_AVX
#endif /* AOCL_SNAPPY_AVX_OPT */

#ifdef AOCL_SNAPPY_AVX2_OPT
#include<immintrin.h>
#define AOCL_SNAPPY_TARGET_AVX2 __attribute__((__target__("bmi2,avx2")))
#else
#define AOCL_SNAPPY_TARGET_AVX2
#endif /* AOCL_SNAPPY_AVX_OPT */

#include "utils/utils.h"
#include "algos/common/aoclAlgoLog.h"
#include "utils/dispatcher.h"

#ifdef AOCL_ENABLE_THREADS
#include "threads/threads.h"
#endif
// AOCL definitions end

namespace snappy {

#ifdef AOCL_SNAPPY_OPT
/* Dynamic dispatcher setup function for native APIs.
    * All native APIs that call aocl optimized functions within their call stack,
    * must call AOCL_SETUP_NATIVE() at the start of the function. This sets up
    * appropriate code paths to take based on user defined environment variables,
    * as well as cpu instruction set supported by the runtime machine. */
static void aocl_setup_native(void);
#define AOCL_SETUP_NATIVE() aocl_setup_native()
#else
#define AOCL_SETUP_NATIVE()
#endif
static int setup_ok_snappy = 0; // flag to indicate status of dynamic dispatcher setup
#ifndef AOCL_ENABLE_THREADS
static std::atomic_flag setup_snappy = ATOMIC_FLAG_INIT;
#endif

/* --------- Forward declarations for dynamic ISA selection - start ---------*/

class with_bmi_avx;
class with_avx;
class with_c;
class SnappyIOVecWriter;
class SnappyArrayWriter;
class SnappyDecompressionValidator;
class SnappySinkAllocator;
template <typename Allocator>
class SnappyScatteredWriter;

static inline uint32_t ExtractLowBytes_c(const uint32_t& v, int n);
void MemCopy64_c(char* dst, const void* src, size_t size);
void MemCopy64_c(ptrdiff_t dst, const void* src, size_t size);
bool Uncompress_c(Source* compressed, Sink* uncompressed);
bool SAW_RawUncompress_c(const char* compressed, size_t compressed_length, char* uncompressed);
namespace {
uint32_t CalculateTableSize(uint32_t input_size);
}

template <typename T>
bool InternalGetUncompressedLength(Source* source, uint32_t* result);
template <typename Writer>
static bool InternalUncompress_c(Source* r, Writer* writer);
template <typename T>
std::pair<const uint8_t*, ptrdiff_t> DecompressBranchless_c(
    const uint8_t* ip, const uint8_t* ip_limit, ptrdiff_t op, T op_base,
    ptrdiff_t op_limit_min_slop);
namespace internal {
char* CompressFragment_c(const char* input, size_t input_size, char* op,
    uint16_t* table, const int table_size);
}

uint32_t (*CalculateTableSize_fp)(uint32_t input_size) = CalculateTableSize;
static bool (*GetUncompressedLengthInternal_fp)(Source* source, uint32_t* result) 
    = InternalGetUncompressedLength<with_avx>;
static char* (*SNAPPY_compress_fragment_fp)(const char* input,
    size_t input_size, char* op, uint16_t* table, const int table_size) 
    = internal::CompressFragment_c;
static bool (*SNAPPY_SAW_raw_uncompress_fp)(const char* compressed,
    size_t compressed_length, char* uncompressed) 
    = SAW_RawUncompress_c;
bool (*InternalUncompressIOVec_fp)(Source* r, SnappyIOVecWriter* writer) 
    = InternalUncompress_c<SnappyIOVecWriter>;
bool (*InternalUncompressArray_fp)(Source* r, SnappyArrayWriter* writer) 
    = InternalUncompress_c<SnappyArrayWriter>;
bool (*InternalUncompressDecompression_fp)(Source* r, SnappyDecompressionValidator* writer) 
    = InternalUncompress_c<SnappyDecompressionValidator>;
bool (*InternalUncompressScattered_fp)(Source* r, SnappyScatteredWriter<SnappySinkAllocator>* writer) 
    = InternalUncompress_c<SnappyScatteredWriter<SnappySinkAllocator>>;
bool (*Uncompress_fp)(Source* compressed, Sink* uncompressed) 
    = Uncompress_c;

#ifdef AOCL_SNAPPY_AVX_OPT
template <typename Writer, typename T>
AOCL_SNAPPY_TARGET_AVX
static bool InternalUncompress_avx(Source* r, Writer* writer);
template <typename T>
AOCL_SNAPPY_TARGET_AVX
bool Uncompress_avx(Source* compressed, Sink* uncompressed);
namespace internal {
AOCL_SNAPPY_TARGET_AVX
char* CompressFragment_crc32(const char* input, size_t input_size, char* op,
    uint16_t* table, const int table_size);
}
#endif /* AOCL_SNAPPY_AVX_OPT */

#ifdef AOCL_SNAPPY_AVX2_OPT
AOCL_SNAPPY_TARGET_AVX2
void MemCopy64_avx2(char* dst, const void* src, size_t size);
AOCL_SNAPPY_TARGET_AVX2
void MemCopy64_avx2(ptrdiff_t dst, const void* src, size_t size);
template <typename T>
AOCL_SNAPPY_TARGET_AVX2
std::pair<const uint8_t*, ptrdiff_t> DecompressBranchless_avx2(
    const uint8_t* ip, const uint8_t* ip_limit, ptrdiff_t op, T op_base,
    ptrdiff_t op_limit_min_slop);
#endif /* AOCL_SNAPPY_AVX2_OPT */

#ifdef AOCL_ENABLE_THREADS
bool SAW_RawUncompressDirect(const char* compressed, size_t compressed_length,
    char* uncompressed, uint32_t uncompressed_len);
template <typename Writer, typename T>
static bool InternalUncompressDirect(Source* r, Writer* writer, uint32_t uncompressed_len);

static bool (*SNAPPY_SAW_raw_uncompress_direct_fp)(const char* compressed,
    size_t compressed_length, char* uncompressed, uint32_t uncompressed_len) 
    = SAW_RawUncompressDirect;
bool (*InternalUncompressDirectArray_fp)(Source* r, SnappyArrayWriter* writer,
    uint32_t uncompressed_len) 
    = InternalUncompressDirect<SnappyArrayWriter, with_avx>;
#endif /* AOCL_ENABLE_THREADS */

/* ---------- Forward declarations for dynamic ISA selection - end ----------*/

/* ----------- Forward declarations for AOCL optimizations - start ----------*/

size_t MaxCompressedLength_st(size_t source_bytes);

#ifdef AOCL_SNAPPY_OPT
#ifndef AOCL_SNAPPY_HIGH_COMPRESSION
namespace {
uint32_t AOCL_CalculateTableSize(uint32_t input_size);
}
#endif /* AOCL_SNAPPY_HIGH_COMPRESSION */
namespace internal {
char* AOCL_CompressFragment_c(const char* input, size_t input_size, char* op,
    uint16_t* table, const int table_size);
}
#endif /* AOCL_SNAPPY_OPT */

#ifdef AOCL_SNAPPY_AVX_OPT
class AOCL_SnappyArrayWriter_AVX;
namespace {
AOCL_SNAPPY_TARGET_AVX
    inline void AOCL_FastMemcopy64Bytes(char* dst, char* src);
}
AOCL_SNAPPY_TARGET_AVX
bool AOCL_SAW_RawUncompress_avx(const char* compressed, size_t compressed_length,
    char* uncompressed);
namespace internal {
AOCL_SNAPPY_TARGET_AVX
char* AOCL_CompressFragment_crc32(const char* input, size_t input_size, char* op,
    uint16_t* table, const int table_size);
}

bool (*InternalUncompressAOCLArray_fp)(Source* r, AOCL_SnappyArrayWriter_AVX* writer)
    = InternalUncompress_avx<AOCL_SnappyArrayWriter_AVX, with_avx>;
#endif /* AOCL_SNAPPY_AVX_OPT */

#ifdef AOCL_ENABLE_THREADS
size_t MaxCompressedLength_mt(size_t source_bytes);
void AOCL_RawCompress_mt(const char* input, size_t input_length, char* compressed,
    size_t* compressed_length, CompressionOptions options);
bool AOCL_RawUncompress_mt(const char* compressed, size_t compressed_length, char* uncompressed);
AOCL_INT32 AOCL_GetUncompressedLengthAndVarintByteWidth(const char* start,
    size_t n, uint32_t* result);
#ifdef AOCL_SNAPPY_AVX_OPT
bool AOCL_SAW_RawUncompressDirect(const char* compressed, size_t compressed_length,
    char* uncompressed, uint32_t uncompressed_len);
bool (*InternalUncompressDirectAOCLArray_fp)(Source* r, AOCL_SnappyArrayWriter_AVX* writer,
    uint32_t uncompressed_len)
    = InternalUncompressDirect<AOCL_SnappyArrayWriter_AVX, with_avx>;
#endif /* AOCL_SNAPPY_AVX_OPT */
#endif /* AOCL_ENABLE_THREADS */

/* ------------ Forward declarations for AOCL optimizations - end -----------*/

namespace {

// The amount of slop bytes writers are using for unconditional copies.
constexpr int kSlopBytes = 64;

using internal::char_table;
using internal::COPY_1_BYTE_OFFSET;
using internal::COPY_2_BYTE_OFFSET;
using internal::COPY_4_BYTE_OFFSET;
using internal::kMaximumTagLength;
using internal::LITERAL;
#if SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE
using internal::V128;
using internal::V128_Load;
using internal::V128_LoadU;
using internal::V128_Shuffle;
using internal::V128_StoreU;
using internal::V128_DupChar;
#endif

// We translate the information encoded in a tag through a lookup table to a
// format that requires fewer instructions to decode. Effectively we store
// the length minus the tag part of the offset. The lowest significant byte
// thus stores the length. While total length - offset is given by
// entry - ExtractOffset(type). The nice thing is that the subtraction
// immediately sets the flags for the necessary check that offset >= length.
// This folds the cmp with sub. We engineer the long literals and copy-4 to
// always fail this check, so their presence doesn't affect the fast path.
// To prevent literals from triggering the guard against offset < length (offset
// does not apply to literals) the table is giving them a spurious offset of
// 256.
inline constexpr int16_t MakeEntry(int16_t len, int16_t offset) {
  return len - (offset << 8);
}

inline constexpr int16_t LengthMinusOffset(int data, int type) {
  return type == 3   ? 0xFF                    // copy-4 (or type == 3)
         : type == 2 ? MakeEntry(data + 1, 0)  // copy-2
         : type == 1 ? MakeEntry((data & 7) + 4, data >> 3)  // copy-1
         : data < 60 ? MakeEntry(data + 1, 1)  // note spurious offset.
                     : 0xFF;                   // long literal
}

inline constexpr int16_t LengthMinusOffset(uint8_t tag) {
  return LengthMinusOffset(tag >> 2, tag & 3);
}

template <size_t... Ints>
struct index_sequence {};

template <std::size_t N, size_t... Is>
struct make_index_sequence : make_index_sequence<N - 1, N - 1, Is...> {};

template <size_t... Is>
struct make_index_sequence<0, Is...> : index_sequence<Is...> {};

template <size_t... seq>
constexpr std::array<int16_t, 256> MakeTable(index_sequence<seq...>) {
  return std::array<int16_t, 256>{LengthMinusOffset(seq)...};
}

alignas(64) const std::array<int16_t, 256> kLengthMinusOffset =
    MakeTable(make_index_sequence<256>{});

/*
// Given a table of uint16_t whose size is mask / 2 + 1, return a pointer to the
// relevant entry, if any, for the given bytes.  Any hash function will do,
// but a good hash function reduces the number of collisions and thus yields
// better compression for compressible input.
//
// REQUIRES: mask is 2 * (table_size - 1), and table_size is a power of two.
inline uint16_t* TableEntry(uint16_t* table, uint32_t bytes, uint32_t mask)
-> TableEntry_c, TableEntry_crc32
*/

inline uint16_t* TableEntry4ByteMatch(uint16_t* table, uint32_t bytes,
                                      uint32_t mask) {
  constexpr uint32_t kMagic = 2654435761U;
  const uint32_t hash = (kMagic * bytes) >> (32 - kMaxHashTableBits);
  return reinterpret_cast<uint16_t*>(reinterpret_cast<uintptr_t>(table) +
                                     (hash & mask));
}

inline uint16_t* TableEntry8ByteMatch(uint16_t* table, uint64_t bytes,
                                      uint32_t mask) {
  constexpr uint64_t kMagic = 58295818150454627ULL;
  const uint32_t hash = (kMagic * bytes) >> (64 - kMaxHashTableBits);
  return reinterpret_cast<uint16_t*>(reinterpret_cast<uintptr_t>(table) +
                                     (hash & mask));
}

}  // namespace

size_t MaxCompressedLength(size_t source_bytes) {
#ifdef AOCL_ENABLE_THREADS
    return MaxCompressedLength_mt(source_bytes);
#else
    return MaxCompressedLength_st(source_bytes);
#endif
}

namespace {

void UnalignedCopy64(const void* src, void* dst) {
  char tmp[8];
  std::memcpy(tmp, src, 8);
  std::memcpy(dst, tmp, 8);
}

void UnalignedCopy128(const void* src, void* dst) {
  // std::memcpy() gets vectorized when the appropriate compiler options are
  // used. For example, x86 compilers targeting SSE2+ will optimize to an SSE2
  // load and store.
  char tmp[16];
  std::memcpy(tmp, src, 16);
  std::memcpy(dst, tmp, 16);
}

template <bool use_16bytes_chunk>
inline void ConditionalUnalignedCopy128(const char* src, char* dst) {
  if (use_16bytes_chunk) {
    UnalignedCopy128(src, dst);
  } else {
    UnalignedCopy64(src, dst);
    UnalignedCopy64(src + 8, dst + 8);
  }
}

// Copy [src, src+(op_limit-op)) to [op, (op_limit-op)) a byte at a time. Used
// for handling COPY operations where the input and output regions may overlap.
// For example, suppose:
//    src       == "ab"
//    op        == src + 2
//    op_limit  == op + 20
// After IncrementalCopySlow(src, op, op_limit), the result will have eleven
// copies of "ab"
//    ababababababababababab
// Note that this does not match the semantics of either std::memcpy() or
// std::memmove().
inline char* IncrementalCopySlow(const char* src, char* op,
                                 char* const op_limit) {
  // TODO: Remove pragma when LLVM is aware this
  // function is only called in cold regions and when cold regions don't get
  // vectorized or unrolled.
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
  while (op < op_limit) {
    *op++ = *src++;
  }
  return op_limit;
}

#if SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE

// Computes the bytes for shuffle control mask (please read comments on
// 'pattern_generation_masks' as well) for the given index_offset and
// pattern_size. For example, when the 'offset' is 6, it will generate a
// repeating pattern of size 6. So, the first 16 byte indexes will correspond to
// the pattern-bytes {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3} and the
// next 16 byte indexes will correspond to the pattern-bytes {4, 5, 0, 1, 2, 3,
// 4, 5, 0, 1, 2, 3, 4, 5, 0, 1}. These byte index sequences are generated by
// calling MakePatternMaskBytes(0, 6, index_sequence<16>()) and
// MakePatternMaskBytes(16, 6, index_sequence<16>()) respectively.
template <size_t... indexes>
inline constexpr std::array<char, sizeof...(indexes)> MakePatternMaskBytes(
    int index_offset, int pattern_size, index_sequence<indexes...>) {
  return {static_cast<char>((index_offset + indexes) % pattern_size)...};
}

// Computes the shuffle control mask bytes array for given pattern-sizes and
// returns an array.
template <size_t... pattern_sizes_minus_one>
inline constexpr std::array<std::array<char, sizeof(V128)>,
                            sizeof...(pattern_sizes_minus_one)>
MakePatternMaskBytesTable(int index_offset,
                          index_sequence<pattern_sizes_minus_one...>) {
  return {
      MakePatternMaskBytes(index_offset, pattern_sizes_minus_one + 1,
                           make_index_sequence</*indexes=*/sizeof(V128)>())...};
}

// This is an array of shuffle control masks that can be used as the source
// operand for PSHUFB to permute the contents of the destination XMM register
// into a repeating byte pattern.
alignas(16) constexpr std::array<std::array<char, sizeof(V128)>,
                                 16> pattern_generation_masks =
    MakePatternMaskBytesTable(
        /*index_offset=*/0,
        /*pattern_sizes_minus_one=*/make_index_sequence<16>());

// Similar to 'pattern_generation_masks', this table is used to "rotate" the
// pattern so that we can copy the *next 16 bytes* consistent with the pattern.
// Basically, pattern_reshuffle_masks is a continuation of
// pattern_generation_masks. It follows that, pattern_reshuffle_masks is same as
// pattern_generation_masks for offsets 1, 2, 4, 8 and 16.
alignas(16) constexpr std::array<std::array<char, sizeof(V128)>,
                                 16> pattern_reshuffle_masks =
    MakePatternMaskBytesTable(
        /*index_offset=*/16,
        /*pattern_sizes_minus_one=*/make_index_sequence<16>());

SNAPPY_ATTRIBUTE_ALWAYS_INLINE
static inline V128 LoadPattern(const char* src, const size_t pattern_size) {
  V128 generation_mask = V128_Load(reinterpret_cast<const V128*>(
      pattern_generation_masks[pattern_size - 1].data()));
  // Uninitialized bytes are masked out by the shuffle mask.
  // TODO: remove annotation and macro defs once MSan is fixed.
  SNAPPY_ANNOTATE_MEMORY_IS_INITIALIZED(src + pattern_size, 16 - pattern_size);
  return V128_Shuffle(V128_LoadU(reinterpret_cast<const V128*>(src)),
                      generation_mask);
}

SNAPPY_ATTRIBUTE_ALWAYS_INLINE
static inline std::pair<V128 /* pattern */, V128 /* reshuffle_mask */>
LoadPatternAndReshuffleMask(const char* src, const size_t pattern_size) {
  V128 pattern = LoadPattern(src, pattern_size);

  // This mask will generate the next 16 bytes in-place. Doing so enables us to
  // write data by at most 4 V128_StoreU.
  //
  // For example, suppose pattern is:        abcdefabcdefabcd
  // Shuffling with this mask will generate: efabcdefabcdefab
  // Shuffling again will generate:          cdefabcdefabcdef
  V128 reshuffle_mask = V128_Load(reinterpret_cast<const V128*>(
      pattern_reshuffle_masks[pattern_size - 1].data()));
  return {pattern, reshuffle_mask};
}

#endif  // SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE

// Fallback for when we need to copy while extending the pattern, for example
// copying 10 bytes from 3 positions back abc -> abcabcabcabca.
//
// REQUIRES: [dst - offset, dst + 64) is a valid address range.
SNAPPY_ATTRIBUTE_ALWAYS_INLINE
static inline bool Copy64BytesWithPatternExtension(char* dst, size_t offset) {
#if SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE
  if (SNAPPY_PREDICT_TRUE(offset <= 16)) {
    switch (offset) {
      case 0:
        return false;
      case 1: {
        // TODO: Ideally we should memset, move back once the
        // codegen issues are fixed.
        V128 pattern = V128_DupChar(dst[-1]);
        for (int i = 0; i < 4; i++) {
          V128_StoreU(reinterpret_cast<V128*>(dst + 16 * i), pattern);
        }
        return true;
      }
      case 2:
      case 4:
      case 8:
      case 16: {
        V128 pattern = LoadPattern(dst - offset, offset);
        for (int i = 0; i < 4; i++) {
          V128_StoreU(reinterpret_cast<V128*>(dst + 16 * i), pattern);
        }
        return true;
      }
      default: {
        auto pattern_and_reshuffle_mask =
            LoadPatternAndReshuffleMask(dst - offset, offset);
        V128 pattern = pattern_and_reshuffle_mask.first;
        V128 reshuffle_mask = pattern_and_reshuffle_mask.second;
        for (int i = 0; i < 4; i++) {
          V128_StoreU(reinterpret_cast<V128*>(dst + 16 * i), pattern);
          pattern = V128_Shuffle(pattern, reshuffle_mask);
        }
        return true;
      }
    }
  }
#else
  if (SNAPPY_PREDICT_TRUE(offset < 16)) {
    if (SNAPPY_PREDICT_FALSE(offset == 0)) return false;
    // Extend the pattern to the first 16 bytes.
    // The simpler formulation of `dst[i - offset]` induces undefined behavior.
    for (int i = 0; i < 16; i++) dst[i] = (dst - offset)[i];
    // Find a multiple of pattern >= 16.
    static std::array<uint8_t, 16> pattern_sizes = []() {
      std::array<uint8_t, 16> res;
      for (int i = 1; i < 16; i++) res[i] = (16 / i + 1) * i;
      return res;
    }();
    offset = pattern_sizes[offset];
    for (int i = 1; i < 4; i++) {
      std::memcpy(dst + i * 16, dst + i * 16 - offset, 16);
    }
    return true;
  }
#endif  // SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE

  // Very rare.
  for (int i = 0; i < 4; i++) {
    std::memcpy(dst + i * 16, dst + i * 16 - offset, 16);
  }
  return true;
}

// Copy [src, src+(op_limit-op)) to [op, op_limit) but faster than
// IncrementalCopySlow. buf_limit is the address past the end of the writable
// region of the buffer.
inline char* IncrementalCopy(const char* src, char* op, char* const op_limit,
                             char* const buf_limit) {
#if SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE
  constexpr int big_pattern_size_lower_bound = 16;
#else
  constexpr int big_pattern_size_lower_bound = 8;
#endif

  // Terminology:
  //
  // slop = buf_limit - op
  // pat  = op - src
  // len  = op_limit - op
  assert(src < op);
  assert(op < op_limit);
  assert(op_limit <= buf_limit);
  // NOTE: The copy tags use 3 or 6 bits to store the copy length, so len <= 64.
  assert(op_limit - op <= 64);
  // NOTE: In practice the compressor always emits len >= 4, so it is ok to
  // assume that to optimize this function, but this is not guaranteed by the
  // compression format, so we have to also handle len < 4 in case the input
  // does not satisfy these conditions.

  size_t pattern_size = op - src;
  // The cases are split into different branches to allow the branch predictor,
  // FDO, and static prediction hints to work better. For each input we list the
  // ratio of invocations that match each condition.
  //
  // input        slop < 16   pat < 8  len > 16
  // ------------------------------------------
  // html|html4|cp   0%         1.01%    27.73%
  // urls            0%         0.88%    14.79%
  // jpg             0%        64.29%     7.14%
  // pdf             0%         2.56%    58.06%
  // txt[1-4]        0%         0.23%     0.97%
  // pb              0%         0.96%    13.88%
  // bin             0.01%     22.27%    41.17%
  //
  // It is very rare that we don't have enough slop for doing block copies. It
  // is also rare that we need to expand a pattern. Small patterns are common
  // for incompressible formats and for those we are plenty fast already.
  // Lengths are normally not greater than 16 but they vary depending on the
  // input. In general if we always predict len <= 16 it would be an ok
  // prediction.
  //
  // In order to be fast we want a pattern >= 16 bytes (or 8 bytes in non-SSE)
  // and an unrolled loop copying 1x 16 bytes (or 2x 8 bytes in non-SSE) at a
  // time.

  // Handle the uncommon case where pattern is less than 16 (or 8 in non-SSE)
  // bytes.
  if (pattern_size < big_pattern_size_lower_bound) {
#if SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE
    // Load the first eight bytes into an 128-bit XMM register, then use PSHUFB
    // to permute the register's contents in-place into a repeating sequence of
    // the first "pattern_size" bytes.
    // For example, suppose:
    //    src       == "abc"
    //    op        == op + 3
    // After V128_Shuffle(), "pattern" will have five copies of "abc"
    // followed by one byte of slop: abcabcabcabcabca.
    //
    // The non-SSE fallback implementation suffers from store-forwarding stalls
    // because its loads and stores partly overlap. By expanding the pattern
    // in-place, we avoid the penalty.

    // Typically, the op_limit is the gating factor so try to simplify the loop
    // based on that.
    if (SNAPPY_PREDICT_TRUE(op_limit <= buf_limit - 15)) {
      auto pattern_and_reshuffle_mask =
          LoadPatternAndReshuffleMask(src, pattern_size);
      V128 pattern = pattern_and_reshuffle_mask.first;
      V128 reshuffle_mask = pattern_and_reshuffle_mask.second;

      // There is at least one, and at most four 16-byte blocks. Writing four
      // conditionals instead of a loop allows FDO to layout the code with
      // respect to the actual probabilities of each length.
      // TODO: Replace with loop with trip count hint.
      V128_StoreU(reinterpret_cast<V128*>(op), pattern);

      if (op + 16 < op_limit) {
        pattern = V128_Shuffle(pattern, reshuffle_mask);
        V128_StoreU(reinterpret_cast<V128*>(op + 16), pattern);
      }
      if (op + 32 < op_limit) {
        pattern = V128_Shuffle(pattern, reshuffle_mask);
        V128_StoreU(reinterpret_cast<V128*>(op + 32), pattern);
      }
      if (op + 48 < op_limit) {
        pattern = V128_Shuffle(pattern, reshuffle_mask);
        V128_StoreU(reinterpret_cast<V128*>(op + 48), pattern);
      }
      return op_limit;
    }
    char* const op_end = buf_limit - 15;
    if (SNAPPY_PREDICT_TRUE(op < op_end)) {
      auto pattern_and_reshuffle_mask =
          LoadPatternAndReshuffleMask(src, pattern_size);
      V128 pattern = pattern_and_reshuffle_mask.first;
      V128 reshuffle_mask = pattern_and_reshuffle_mask.second;

      // This code path is relatively cold however so we save code size
      // by avoiding unrolling and vectorizing.
      //
      // TODO: Remove pragma when when cold regions don't get
      // vectorized or unrolled.
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
      do {
        V128_StoreU(reinterpret_cast<V128*>(op), pattern);
        pattern = V128_Shuffle(pattern, reshuffle_mask);
        op += 16;
      } while (SNAPPY_PREDICT_TRUE(op < op_end));
    }
    return IncrementalCopySlow(op - pattern_size, op, op_limit);
#else   // !SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE
    // If plenty of buffer space remains, expand the pattern to at least 8
    // bytes. The way the following loop is written, we need 8 bytes of buffer
    // space if pattern_size >= 4, 11 bytes if pattern_size is 1 or 3, and 10
    // bytes if pattern_size is 2.  Precisely encoding that is probably not
    // worthwhile; instead, invoke the slow path if we cannot write 11 bytes
    // (because 11 are required in the worst case).
    if (SNAPPY_PREDICT_TRUE(op <= buf_limit - 11)) {
      while (pattern_size < 8) {
        UnalignedCopy64(src, op);
        op += pattern_size;
        pattern_size *= 2;
      }
      if (SNAPPY_PREDICT_TRUE(op >= op_limit)) return op_limit;
    } else {
      return IncrementalCopySlow(src, op, op_limit);
    }
#endif  // SNAPPY_HAVE_VECTOR_BYTE_SHUFFLE
  }
  assert(pattern_size >= big_pattern_size_lower_bound);
  constexpr bool use_16bytes_chunk = big_pattern_size_lower_bound == 16;

  // Copy 1x 16 bytes (or 2x 8 bytes in non-SSE) at a time. Because op - src can
  // be < 16 in non-SSE, a single UnalignedCopy128 might overwrite data in op.
  // UnalignedCopy64 is safe because expanding the pattern to at least 8 bytes
  // guarantees that op - src >= 8.
  //
  // Typically, the op_limit is the gating factor so try to simplify the loop
  // based on that.
  if (SNAPPY_PREDICT_TRUE(op_limit <= buf_limit - 15)) {
    // There is at least one, and at most four 16-byte blocks. Writing four
    // conditionals instead of a loop allows FDO to layout the code with respect
    // to the actual probabilities of each length.
    // TODO: Replace with loop with trip count hint.
    ConditionalUnalignedCopy128<use_16bytes_chunk>(src, op);
    if (op + 16 < op_limit) {
      ConditionalUnalignedCopy128<use_16bytes_chunk>(src + 16, op + 16);
    }
    if (op + 32 < op_limit) {
      ConditionalUnalignedCopy128<use_16bytes_chunk>(src + 32, op + 32);
    }
    if (op + 48 < op_limit) {
      ConditionalUnalignedCopy128<use_16bytes_chunk>(src + 48, op + 48);
    }
    return op_limit;
  }

  // Fall back to doing as much as we can with the available slop in the
  // buffer. This code path is relatively cold however so we save code size by
  // avoiding unrolling and vectorizing.
  //
  // TODO: Remove pragma when when cold regions don't get vectorized
  // or unrolled.
#ifdef __clang__
#pragma clang loop unroll(disable)
#endif
  for (char* op_end = buf_limit - 16; op < op_end; op += 16, src += 16) {
    ConditionalUnalignedCopy128<use_16bytes_chunk>(src, op);
  }
  if (op >= op_limit) return op_limit;

  // We only take this branch if we didn't have enough slop and we can do a
  // single 8 byte copy.
  if (SNAPPY_PREDICT_FALSE(op <= buf_limit - 8)) {
    UnalignedCopy64(src, op);
    src += 8;
    op += 8;
  }
  return IncrementalCopySlow(src, op, op_limit);
}

}  // namespace

template <bool allow_fast_path>
static inline char* EmitLiteral(char* op, const char* literal, int len) {
  // The vast majority of copies are below 16 bytes, for which a
  // call to std::memcpy() is overkill. This fast path can sometimes
  // copy up to 15 bytes too much, but that is okay in the
  // main loop, since we have a bit to go on for both sides:
  //
  //   - The input will always have kInputMarginBytes = 15 extra
  //     available bytes, as long as we're in the main loop, and
  //     if not, allow_fast_path = false.
  //   - The output will always have 32 spare bytes (see
  //     MaxCompressedLength).
  assert(len > 0);  // Zero-length literals are disallowed
  int n = len - 1;
  if (allow_fast_path && len <= 16) {
    // Fits in tag byte
    *op++ = LITERAL | (n << 2);

    UnalignedCopy128(literal, op);
    return op + len;
  }

  if (n < 60) {
    // Fits in tag byte
    *op++ = LITERAL | (n << 2);
  } else {
    int count = (Bits::Log2Floor(n) >> 3) + 1;
    assert(count >= 1);
    assert(count <= 4);
    *op++ = LITERAL | ((59 + count) << 2);
    // Encode in upcoming bytes.
    // Write 4 bytes, though we may care about only 1 of them. The output buffer
    // is guaranteed to have at least 3 more spaces left as 'len >= 61' holds
    // here and there is a std::memcpy() of size 'len' below.
    LittleEndian::Store32(op, n);
    op += count;
  }
  // When allow_fast_path is true, we can overwrite up to 16 bytes.
  if (allow_fast_path) {
    char* destination = op;
    const char* source = literal;
    const char* end = destination + len;
    do {
      std::memcpy(destination, source, 16);
      destination += 16;
      source += 16;
    } while (destination < end);
  } else {
    std::memcpy(op, literal, len);
  }
  return op + len;
}

template <bool len_less_than_12>
static inline char* EmitCopyAtMost64(char* op, size_t offset, size_t len) {
  assert(len <= 64);
  assert(len >= 4);
  assert(offset < 65536);
  assert(len_less_than_12 == (len < 12));

  if (len_less_than_12) {
    uint32_t u = (len << 2) + (offset << 8);
    uint32_t copy1 = COPY_1_BYTE_OFFSET - (4 << 2) + ((offset >> 3) & 0xe0);
    uint32_t copy2 = COPY_2_BYTE_OFFSET - (1 << 2);
    // It turns out that offset < 2048 is a difficult to predict branch.
    // `perf record` shows this is the highest percentage of branch misses in
    // benchmarks. This code produces branch free code, the data dependency
    // chain that bottlenecks the throughput is so long that a few extra
    // instructions are completely free (IPC << 6 because of data deps).
    u += offset < 2048 ? copy1 : copy2;
    LittleEndian::Store32(op, u);
    op += offset < 2048 ? 2 : 3;
  } else {
    // Write 4 bytes, though we only care about 3 of them.  The output buffer
    // is required to have some slack, so the extra byte won't overrun it.
    uint32_t u = COPY_2_BYTE_OFFSET + ((len - 1) << 2) + (offset << 8);
    LittleEndian::Store32(op, u);
    op += 3;
  }
  return op;
}

template <bool len_less_than_12>
static inline char* EmitCopy(char* op, size_t offset, size_t len) {
  assert(len_less_than_12 == (len < 12));
  if (len_less_than_12) {
    return EmitCopyAtMost64</*len_less_than_12=*/true>(op, offset, len);
  } else {
    // A special case for len <= 64 might help, but so far measurements suggest
    // it's in the noise.

    // Emit 64 byte copies but make sure to keep at least four bytes reserved.
    while (SNAPPY_PREDICT_FALSE(len >= 68)) {
      op = EmitCopyAtMost64</*len_less_than_12=*/false>(op, offset, 64);
      len -= 64;
    }

    // One or two copies will now finish the job.
    if (len > 64) {
      op = EmitCopyAtMost64</*len_less_than_12=*/false>(op, offset, 60);
      len -= 60;
    }

    // Emit remainder.
    if (len < 12) {
      op = EmitCopyAtMost64</*len_less_than_12=*/true>(op, offset, len);
    } else {
      op = EmitCopyAtMost64</*len_less_than_12=*/false>(op, offset, len);
    }
    return op;
  }
}

bool GetUncompressedLength(const char* start, size_t n, size_t* result) {
  if (start == NULL || result == NULL) return false;
  uint32_t v = 0;
  const char* limit = start + n;
  if (Varint::Parse32WithLimit(start, limit, &v) != NULL) {
    *result = v;
    return true;
  } else {
    return false;
  }
}

namespace {
uint32_t CalculateTableSize(uint32_t input_size) {
  static_assert(
      kMaxHashTableSize >= kMinHashTableSize,
      "kMaxHashTableSize should be greater or equal to kMinHashTableSize.");
  if (input_size > kMaxHashTableSize) {
    return kMaxHashTableSize;
  }
  if (input_size < kMinHashTableSize) {
    return kMinHashTableSize;
  }
  // This is equivalent to Log2Ceiling(input_size), assuming input_size > 1.
  // 2 << Log2Floor(x - 1) is equivalent to 1 << (1 + Log2Floor(x - 1)).
  return 2u << Bits::Log2Floor(input_size - 1);
}
}  // namespace

namespace internal {
WorkingMemory::WorkingMemory(size_t input_size) {
  const size_t max_fragment_size = std::min(input_size, kBlockSize);
  const size_t table_size = CalculateTableSize_fp(max_fragment_size);
  size_ = table_size * sizeof(*table_) + max_fragment_size +
          MaxCompressedLength(max_fragment_size);
  mem_ = std::allocator<char>().allocate(size_);
  table_ = reinterpret_cast<uint16_t*>(mem_);
  input_ = mem_ + table_size * sizeof(*table_);
  output_ = input_ + max_fragment_size;
}

WorkingMemory::~WorkingMemory() {
  std::allocator<char>().deallocate(mem_, size_);
}

uint16_t* WorkingMemory::GetHashTable(size_t fragment_size,
                                      int* table_size) const {
  const size_t htsize = CalculateTableSize_fp(fragment_size);
  memset(table_, 0, htsize * sizeof(*table_));
  *table_size = htsize;
  return table_;
}
}  // end namespace internal

// Flat array compression that does not emit the "uncompressed length"
// prefix. Compresses "input" string to the "*op" buffer.
//
// REQUIRES: "input" is at most "kBlockSize" bytes long.
// REQUIRES: "op" points to an array of memory that is at least
// "MaxCompressedLength(input.size())" in size.
// REQUIRES: All elements in "table[0..table_size-1]" are initialized to zero.
// REQUIRES: "table_size" is a power of two
//
// Returns an "end" pointer into "op" buffer.
// "end - op" is the compressed size of "input".
namespace internal {
/*char* CompressFragment(const char* input, size_t input_size, char* op,
                       uint16_t* table, const int table_size);
-> CompressFragment_c, CompressFragment_crc32
*/

char* CompressFragmentDoubleHash(const char* input, size_t input_size, char* op,
                                 uint16_t* table, const int table_size,
                                 uint16_t* table2, const int table_size2) {
  (void)table_size2;
  assert(table_size == table_size2);
  // "ip" is the input pointer, and "op" is the output pointer.
  const char* ip = input;
  assert(input_size <= kBlockSize);
  assert((table_size & (table_size - 1)) == 0);  // table must be power of two
  const uint32_t mask = 2 * (table_size - 1);
  const char* ip_end = input + input_size;
  const char* base_ip = ip;

  const size_t kInputMarginBytes = 15;
  if (SNAPPY_PREDICT_TRUE(input_size >= kInputMarginBytes)) {
    const char* ip_limit = input + input_size - kInputMarginBytes;

    for (;;) {
      const char* next_emit = ip++;
      uint64_t data = LittleEndian::Load64(ip);
      uint32_t skip = 512;

      const char* candidate;
      uint32_t candidate_length;
      while (true) {
        assert(static_cast<uint32_t>(data) == LittleEndian::Load32(ip));
        uint16_t* table_entry2 = TableEntry8ByteMatch(table2, data, mask);
        uint32_t bytes_between_hash_lookups = skip >> 9;
        skip++;
        const char* next_ip = ip + bytes_between_hash_lookups;
        if (SNAPPY_PREDICT_FALSE(next_ip > ip_limit)) {
          ip = next_emit;
          goto emit_remainder;
        }
        candidate = base_ip + *table_entry2;
        assert(candidate >= base_ip);
        assert(candidate < ip);

        *table_entry2 = ip - base_ip;
        if (SNAPPY_PREDICT_FALSE(static_cast<uint32_t>(data) ==
                                LittleEndian::Load32(candidate))) {
          candidate_length =
              FindMatchLengthPlain(candidate + 4, ip + 4, ip_end) + 4;
          break;
        }

        uint16_t* table_entry = TableEntry4ByteMatch(table, data, mask);
        candidate = base_ip + *table_entry;
        assert(candidate >= base_ip);
        assert(candidate < ip);

        *table_entry = ip - base_ip;
        if (SNAPPY_PREDICT_FALSE(static_cast<uint32_t>(data) ==
                                LittleEndian::Load32(candidate))) {
          candidate_length =
              FindMatchLengthPlain(candidate + 4, ip + 4, ip_end) + 4;
          table_entry2 =
              TableEntry8ByteMatch(table2, LittleEndian::Load64(ip + 1), mask);
          auto candidate2 = base_ip + *table_entry2;
          size_t candidate_length2 =
              FindMatchLengthPlain(candidate2, ip + 1, ip_end);
          if (candidate_length2 > candidate_length) {
            *table_entry2 = ip - base_ip;
            candidate = candidate2;
            candidate_length = candidate_length2;
            ++ip;
          }
          break;
        }
        data = LittleEndian::Load64(next_ip);
        ip = next_ip;
      }
      // Backtrack to the point it matches fully.
      while (ip > next_emit && candidate > base_ip &&
             *(ip - 1) == *(candidate - 1)) {
        --ip;
        --candidate;
        ++candidate_length;
      }
      *TableEntry8ByteMatch(table2, LittleEndian::Load64(ip + 1), mask) =
          ip - base_ip + 1;
      *TableEntry8ByteMatch(table2, LittleEndian::Load64(ip + 2), mask) =
          ip - base_ip + 2;
      *TableEntry4ByteMatch(table, LittleEndian::Load32(ip + 1), mask) =
          ip - base_ip + 1;
      // Step 2: A 4-byte or 8-byte match has been found.
      // We'll later see if more than 4 bytes match.  But, prior to the match,
      // input bytes [next_emit, ip) are unmatched.  Emit them as
      // "literal bytes."
      assert(next_emit + 16 <= ip_end);
      if (ip - next_emit > 0) {
        op = EmitLiteral</*allow_fast_path=*/true>(op, next_emit,
                                                   ip - next_emit);
      }
      // Step 3: Call EmitCopy, and then see if another EmitCopy could
      // be our next move.  Repeat until we find no match for the
      // input immediately after what was consumed by the last EmitCopy call.
      //
      // If we exit this loop normally then we need to call EmitLiteral next,
      // though we don't yet know how big the literal will be.  We handle that
      // by proceeding to the next iteration of the main loop.  We also can exit
      // this loop via goto if we get close to exhausting the input.
      do {
        // We have a 4-byte match at ip, and no need to emit any
        // "literal bytes" prior to ip.
        const char* base = ip;
        ip += candidate_length;
        size_t offset = base - candidate;
        if (candidate_length < 12) {
          op =
              EmitCopy</*len_less_than_12=*/true>(op, offset, candidate_length);
        } else {
          op = EmitCopy</*len_less_than_12=*/false>(op, offset,
                                                    candidate_length);
        }
        if (SNAPPY_PREDICT_FALSE(ip >= ip_limit)) {
          goto emit_remainder;
        }
        // We are now looking for a 4-byte match again.  We read
        // table[Hash(ip, mask)] for that. To improve compression,
        // we also update several previous table entries.
        if (ip - base_ip > 7) {
          *TableEntry8ByteMatch(table2, LittleEndian::Load64(ip - 7), mask) =
              ip - base_ip - 7;
          *TableEntry8ByteMatch(table2, LittleEndian::Load64(ip - 4), mask) =
              ip - base_ip - 4;
        }
        *TableEntry8ByteMatch(table2, LittleEndian::Load64(ip - 3), mask) =
            ip - base_ip - 3;
        *TableEntry8ByteMatch(table2, LittleEndian::Load64(ip - 2), mask) =
            ip - base_ip - 2;
        *TableEntry4ByteMatch(table, LittleEndian::Load32(ip - 2), mask) =
            ip - base_ip - 2;
        *TableEntry4ByteMatch(table, LittleEndian::Load32(ip - 1), mask) =
            ip - base_ip - 1;

        uint16_t* table_entry =
            TableEntry8ByteMatch(table2, LittleEndian::Load64(ip), mask);
        candidate = base_ip + *table_entry;
        *table_entry = ip - base_ip;
        if (LittleEndian::Load32(ip) == LittleEndian::Load32(candidate)) {
          candidate_length =
              FindMatchLengthPlain(candidate + 4, ip + 4, ip_end) + 4;
          continue;
        }
        table_entry =
            TableEntry4ByteMatch(table, LittleEndian::Load32(ip), mask);
        candidate = base_ip + *table_entry;
        *table_entry = ip - base_ip;
        if (LittleEndian::Load32(ip) == LittleEndian::Load32(candidate)) {
          candidate_length =
              FindMatchLengthPlain(candidate + 4, ip + 4, ip_end) + 4;
          continue;
        }
        break;
      } while (true);
    }
  }

emit_remainder:
  // Emit the remaining bytes as a literal
  if (ip < ip_end) {
    op = EmitLiteral</*allow_fast_path=*/false>(op, ip, ip_end - ip);
  }

  return op;
}
}  // end namespace internal

static inline void Report(int token, const char *algorithm, size_t
compressed_size, size_t uncompressed_size) {
  // TODO: Switch to [[maybe_unused]] when we can assume C++17.
  (void)token;
  (void)algorithm;
  (void)compressed_size;
  (void)uncompressed_size;
}

// Signature of output types needed by decompression code.
// The decompression code is templatized on a type that obeys this
// signature so that we do not pay virtual function call overhead in
// the middle of a tight decompression loop.
//
// class DecompressionWriter {
//  public:
//   // Called before decompression
//   void SetExpectedLength(size_t length);
//
//   // For performance a writer may choose to donate the cursor variable to the
//   // decompression function. The decompression will inject it in all its
//   // function calls to the writer. Keeping the important output cursor as a
//   // function local stack variable allows the compiler to keep it in
//   // register, which greatly aids performance by avoiding loads and stores of
//   // this variable in the fast path loop iterations.
//   T GetOutputPtr() const;
//
//   // At end of decompression the loop donates the ownership of the cursor
//   // variable back to the writer by calling this function.
//   void SetOutputPtr(T op);
//
//   // Called after decompression
//   bool CheckLength() const;
//
//   // Called repeatedly during decompression
//   // Each function get a pointer to the op (output pointer), that the writer
//   // can use and update. Note it's important that these functions get fully
//   // inlined so that no actual address of the local variable needs to be
//   // taken.
//   bool Append(const char* ip, size_t length, T* op);
//   bool AppendFromSelf(uint32_t offset, size_t length, T* op);
//
//   // The rules for how TryFastAppend differs from Append are somewhat
//   // convoluted:
//   //
//   //  - TryFastAppend is allowed to decline (return false) at any
//   //    time, for any reason -- just "return false" would be
//   //    a perfectly legal implementation of TryFastAppend.
//   //    The intention is for TryFastAppend to allow a fast path
//   //    in the common case of a small append.
//   //  - TryFastAppend is allowed to read up to <available> bytes
//   //    from the input buffer, whereas Append is allowed to read
//   //    <length>. However, if it returns true, it must leave
//   //    at least five (kMaximumTagLength) bytes in the input buffer
//   //    afterwards, so that there is always enough space to read the
//   //    next tag without checking for a refill.
//   //  - TryFastAppend must always return decline (return false)
//   //    if <length> is 61 or more, as in this case the literal length is not
//   //    decoded fully. In practice, this should not be a big problem,
//   //    as it is unlikely that one would implement a fast path accepting
//   //    this much data.
//   //
//   bool TryFastAppend(const char* ip, size_t available, size_t length, T* op);
// };

//static inline uint32_t ExtractLowBytes(const uint32_t& v, int n) -> ExtractLowBytes_c, ExtractLowBytes_bmi

static inline bool LeftShiftOverflows(uint8_t value, uint32_t shift) {
  assert(shift < 32);
  static const uint8_t masks[] = {
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  //
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  //
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,  //
      0x00, 0x80, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc, 0xfe};
  return (value & masks[shift]) != 0;
}

inline bool Copy64BytesWithPatternExtension(ptrdiff_t dst, size_t offset) {
  // TODO: Switch to [[maybe_unused]] when we can assume C++17.
  (void)dst;
  return offset != 0;
}

//void MemCopy64(char* dst, const void* src, size_t size) -> MemCopy64_c, MemCopy64_avx2

//void MemCopy64(ptrdiff_t dst, const void* src, size_t size) -> MemCopy64_c, MemCopy64_avx2

inline void ClearDeferred(const void** deferred_src, size_t* deferred_length,
                   uint8_t* safe_source) {
  *deferred_src = safe_source;
  *deferred_length = 0;
}

inline void DeferMemCopy(const void** deferred_src, size_t* deferred_length,
                  const void* src, size_t length) {
  *deferred_src = src;
  *deferred_length = length;
}

SNAPPY_ATTRIBUTE_ALWAYS_INLINE
inline size_t AdvanceToNextTagARMOptimized(const uint8_t** ip_p, size_t* tag) {
  const uint8_t*& ip = *ip_p;
  // This section is crucial for the throughput of the decompression loop.
  // The latency of an iteration is fundamentally constrained by the
  // following data chain on ip.
  // ip -> c = Load(ip) -> delta1 = (c & 3)        -> ip += delta1 or delta2
  //                       delta2 = ((c >> 2) + 1)    ip++
  // This is different from X86 optimizations because ARM has conditional add
  // instruction (csinc) and it removes several register moves.
  const size_t tag_type = *tag & 3;
  const bool is_literal = (tag_type == 0);
  if (is_literal) {
    size_t next_literal_tag = (*tag >> 2) + 1;
    *tag = ip[next_literal_tag];
    ip += next_literal_tag + 1;
  } else {
    *tag = ip[tag_type];
    ip += tag_type + 1;
  }
  return tag_type;
}

SNAPPY_ATTRIBUTE_ALWAYS_INLINE
inline size_t AdvanceToNextTagX86Optimized(const uint8_t** ip_p, size_t* tag) {
  const uint8_t*& ip = *ip_p;
  // This section is crucial for the throughput of the decompression loop.
  // The latency of an iteration is fundamentally constrained by the
  // following data chain on ip.
  // ip -> c = Load(ip) -> ip1 = ip + 1 + (c & 3) -> ip = ip1 or ip2
  //                       ip2 = ip + 2 + (c >> 2)
  // This amounts to 8 cycles.
  // 5 (load) + 1 (c & 3) + 1 (lea ip1, [ip + (c & 3) + 1]) + 1 (cmov)
  size_t literal_len = *tag >> 2;
  size_t tag_type = *tag;
  bool is_literal;
#if defined(__GCC_ASM_FLAG_OUTPUTS__) && defined(__x86_64__)
  // TODO clang misses the fact that the (c & 3) already correctly
  // sets the zero flag.
  asm("and $3, %k[tag_type]\n\t"
      : [tag_type] "+r"(tag_type), "=@ccz"(is_literal)
      :: "cc");
#else
  tag_type &= 3;
  is_literal = (tag_type == 0);
#endif
  // TODO
  // This is code is subtle. Loading the values first and then cmov has less
  // latency then cmov ip and then load. However clang would move the loads
  // in an optimization phase, volatile prevents this transformation.
  // Note that we have enough slop bytes (64) that the loads are always valid.
  size_t tag_literal =
      static_cast<const volatile uint8_t*>(ip)[1 + literal_len];
  size_t tag_copy = static_cast<const volatile uint8_t*>(ip)[tag_type];
  *tag = is_literal ? tag_literal : tag_copy;
  const uint8_t* ip_copy = ip + 1 + tag_type;
  const uint8_t* ip_literal = ip + 2 + literal_len;
  ip = is_literal ? ip_literal : ip_copy;
#if defined(__GNUC__) && defined(__x86_64__)
  // TODO Clang is "optimizing" zero-extension (a totally free
  // operation) this means that after the cmov of tag, it emits another movzb
  // tag, byte(tag). It really matters as it's on the core chain. This dummy
  // asm, persuades clang to do the zero-extension at the load (it's automatic)
  // removing the expensive movzb.
  asm("" ::"r"(tag_copy));
#endif
  return tag_type;
}

// Extract the offset for copy-1 and copy-2 returns 0 for literals or copy-4.
inline uint32_t ExtractOffset(uint32_t val, size_t tag_type) {
  // For x86 non-static storage works better. For ARM static storage is better.
  // TODO: Once the array is recognized as a register, improve the
  // readability for x86.
#if defined(__x86_64__)
  constexpr uint64_t kExtractMasksCombined = 0x0000FFFF00FF0000ull;
  uint16_t result;
  memcpy(&result,
         reinterpret_cast<const char*>(&kExtractMasksCombined) + 2 * tag_type,
         sizeof(result));
  return val & result;
#elif defined(__aarch64__)
  constexpr uint64_t kExtractMasksCombined = 0x0000FFFF00FF0000ull;
  return val & static_cast<uint32_t>(
      (kExtractMasksCombined >> (tag_type * 16)) & 0xFFFF);
#else
  static constexpr uint32_t kExtractMasks[4] = {0, 0xFF, 0xFFFF, 0};
  return val & kExtractMasks[tag_type];
#endif
}

/*
std::pair<const uint8_t*, ptrdiff_t> DecompressBranchless(
    const uint8_t* ip, const uint8_t* ip_limit, ptrdiff_t op, T op_base,
    ptrdiff_t op_limit_min_slop)
-> DecompressBranchless_c, DecompressBranchless_avx2
*/

// Helper class for decompression
template<typename T>
class SnappyDecompressor {
 private:
  Source* reader_;        // Underlying source of bytes to decompress
  const char* ip_;        // Points to next buffered byte
  const char* ip_limit_;  // Points just past buffered bytes
  // If ip < ip_limit_min_maxtaglen_ it's safe to read kMaxTagLength from
  // buffer.
  const char* ip_limit_min_maxtaglen_;
  uint32_t peeked_;                  // Bytes peeked from reader (need to skip)
  bool eof_;                         // Hit end of input without an error?
  char scratch_[kMaximumTagLength];  // See RefillTag().

  // Ensure that all of the tag metadata for the next tag is available
  // in [ip_..ip_limit_-1].  Also ensures that [ip,ip+4] is readable even
  // if (ip_limit_ - ip_ < 5).
  //
  // Returns true on success, false on error or end of input.
  bool RefillTag();

  void ResetLimit(const char* ip) {
    ip_limit_min_maxtaglen_ =
        ip_limit_ - std::min<ptrdiff_t>(ip_limit_ - ip, kMaximumTagLength - 1);
  }

 public:
  explicit SnappyDecompressor(Source* reader)
      : reader_(reader), ip_(NULL), ip_limit_(NULL), ip_limit_min_maxtaglen_(NULL), peeked_(0), eof_(false) {}

  ~SnappyDecompressor() {
    // Advance past any bytes we peeked at from the reader
    reader_->Skip(peeked_);
  }

  // Returns true iff we have hit the end of the input without an error.
  bool eof() const { return eof_; }

  // Read the uncompressed length stored at the start of the compressed data.
  // On success, stores the length in *result and returns true.
  // On failure, returns false.
  bool ReadUncompressedLength(uint32_t* result) {
    assert(ip_ == NULL);  // Must not have read anything yet
    // Length is encoded in 1..5 bytes
    *result = 0;
    uint32_t shift = 0;
    while (true) {
      if (shift >= 32) return false;
      size_t n;
      const char* ip = reader_->Peek(&n);
      if (n == 0) return false;
      const unsigned char c = *(reinterpret_cast<const unsigned char*>(ip));
      reader_->Skip(1);
      uint32_t val = c & 0x7f;
      if (LeftShiftOverflows(static_cast<uint8_t>(val), shift)) return false;
      *result |= val << shift;
      if (c < 128) {
        break;
      }
      shift += 7;
    }
    return true;
  }

 /* Same as DecompressAllTags, but calls ExtractLowBytes_bmi and is built with
* target attribute bmi2 and avx. Attribute 'target' multiversioned functions do not support
* function templates in clang yet, hence DecompressAllTags_bmi and DecompressAllTags
* have to be maintained separately. */
#ifdef AOCL_SNAPPY_AVX2_OPT
    template <class Writer>
#if defined(__GNUC__) && defined(__x86_64__)
  __attribute__((aligned(32)))
#endif
AOCL_SNAPPY_TARGET_AVX2
  void
  DecompressAllTags_bmi(Writer* writer);
#endif /* AOCL_SNAPPY_AVX2_OPT */

/* Same as DecompressAllTags, but is built with target attribute avx. 
* Attribute 'target' multiversioned functions do not support
* function templates in clang yet, hence DecompressAllTags_avx and DecompressAllTags
* have to be maintained separately. */
  template <class Writer>
#if defined(__GNUC__) && defined(__x86_64__)
  __attribute__((aligned(32)))
#endif
  AOCL_SNAPPY_TARGET_AVX
  void
  DecompressAllTags_avx(Writer* writer);

  // Process the next item found in the input.
  // Returns true if successful, false on error or end of input.
  template <class Writer>
#if defined(__GNUC__) && defined(__x86_64__)
  __attribute__((aligned(32)))
#endif
  void
  DecompressAllTags(Writer* writer) {
    const char* ip = ip_;
    ResetLimit(ip);
    auto op = writer->GetOutputPtr();
    // We could have put this refill fragment only at the beginning of the loop.
    // However, duplicating it at the end of each branch gives the compiler more
    // scope to optimize the <ip_limit_ - ip> expression based on the local
    // context, which overall increases speed.
#define MAYBE_REFILL()                                      \
  if (SNAPPY_PREDICT_FALSE(ip >= ip_limit_min_maxtaglen_)) { \
    ip_ = ip;                                               \
    if (SNAPPY_PREDICT_FALSE(!RefillTag())) goto exit;       \
    ip = ip_;                                               \
    ResetLimit(ip);                                         \
  }                                                         \
  preload = static_cast<uint8_t>(*ip)

    // At the start of the for loop below the least significant byte of preload
    // contains the tag.
    uint32_t preload;
    MAYBE_REFILL();
    for (;;) {
      {
        ptrdiff_t op_limit_min_slop;
        auto op_base = writer->GetBase(&op_limit_min_slop);
        if (op_base) {
          auto res =
              DecompressBranchless_c<decltype(op_base)>(reinterpret_cast<const uint8_t*>(ip),
                                   reinterpret_cast<const uint8_t*>(ip_limit_),
                                   op - op_base, op_base, op_limit_min_slop);
          ip = reinterpret_cast<const char*>(res.first);
          op = op_base + res.second;
          MAYBE_REFILL();
        }
      }
      const uint8_t c = static_cast<uint8_t>(preload);
      ip++;

      // Ratio of iterations that have LITERAL vs non-LITERAL for different
      // inputs.
      //
      // input          LITERAL  NON_LITERAL
      // -----------------------------------
      // html|html4|cp   23%        77%
      // urls            36%        64%
      // jpg             47%        53%
      // pdf             19%        81%
      // txt[1-4]        25%        75%
      // pb              24%        76%
      // bin             24%        76%
      if (SNAPPY_PREDICT_FALSE((c & 0x3) == LITERAL)) {
        size_t literal_length = (c >> 2) + 1u;
        if (writer->TryFastAppend(ip, ip_limit_ - ip, literal_length, &op)) {
          assert(literal_length < 61);
          ip += literal_length;
          // NOTE: There is no MAYBE_REFILL() here, as TryFastAppend()
          // will not return true unless there's already at least five spare
          // bytes in addition to the literal.
          preload = static_cast<uint8_t>(*ip);
          continue;
        }
        if (SNAPPY_PREDICT_FALSE(literal_length >= 61)) {
          // Long literal.
          const size_t literal_length_length = literal_length - 60;
          literal_length =
              ExtractLowBytes_c(LittleEndian::Load32(ip), literal_length_length) +
              1;
          ip += literal_length_length;
        }

        size_t avail = ip_limit_ - ip;
        while (avail < literal_length) {
          if (!writer->Append(ip, avail, &op)) goto exit;
          literal_length -= avail;
          reader_->Skip(peeked_);
          size_t n;
          ip = reader_->Peek(&n);
          avail = n;
          peeked_ = avail;
          if (avail == 0) goto exit;
          ip_limit_ = ip + avail;
          ResetLimit(ip);
        }
        if (!writer->Append(ip, literal_length, &op)) goto exit;
        ip += literal_length;
        MAYBE_REFILL();
      } else {
        if (SNAPPY_PREDICT_FALSE((c & 3) == COPY_4_BYTE_OFFSET)) {
          const size_t copy_offset = LittleEndian::Load32(ip);
          const size_t length = (c >> 2) + 1;
          ip += 4;

          if (!writer->AppendFromSelf(copy_offset, length, &op)) goto exit;
        } else {
          const ptrdiff_t entry = kLengthMinusOffset[c];
          preload = LittleEndian::Load32(ip);
          const uint32_t trailer = ExtractLowBytes_c(preload, c & 3);
          const uint32_t length = entry & 0xff;
          assert(length > 0);

          // copy_offset/256 is encoded in bits 8..10.  By just fetching
          // those bits, we get copy_offset (since the bit-field starts at
          // bit 8).
          const uint32_t copy_offset = trailer - entry + length;
          if (!writer->AppendFromSelf(copy_offset, length, &op)) goto exit;

          ip += (c & 3);
          // By using the result of the previous load we reduce the critical
          // dependency chain of ip to 4 cycles.
          preload >>= (c & 3) * 8;
          if (ip < ip_limit_min_maxtaglen_) continue;
        }
        MAYBE_REFILL();
      }
    }
#undef MAYBE_REFILL
  exit:
    writer->SetOutputPtr(op);
  }
};

constexpr uint32_t CalculateNeeded(uint8_t tag) {
  return ((tag & 3) == 0 && tag >= (60 * 4))
             ? (tag >> 2) - 58
             : (0x05030201 >> ((tag * 8) & 31)) & 0xFF;
}

#if __cplusplus >= 201402L
constexpr bool VerifyCalculateNeeded() {
  for (int i = 0; i < 1; i++) {
    if (CalculateNeeded(i) != (uint32_t)(char_table[i] >> 11) + 1) return false;
  }
  return true;
}

// Make sure CalculateNeeded is correct by verifying it against the established
// table encoding the number of added bytes needed.
static_assert(VerifyCalculateNeeded(), "");
#endif  // c++14

template<typename T>
bool SnappyDecompressor<T>::RefillTag() {
  const char* ip = ip_;
  if (ip == ip_limit_) {
    // Fetch a new fragment from the reader
    reader_->Skip(peeked_);  // All peeked bytes are used up
    size_t n;
    ip = reader_->Peek(&n);
    peeked_ = n;
    eof_ = (n == 0);
    if (eof_) return false;
    ip_limit_ = ip + n;
  }

  // Read the tag character
  assert(ip < ip_limit_);
  const unsigned char c = *(reinterpret_cast<const unsigned char*>(ip));
  // At this point make sure that the data for the next tag is consecutive.
  // For copy 1 this means the next 2 bytes (tag and 1 byte offset)
  // For copy 2 the next 3 bytes (tag and 2 byte offset)
  // For copy 4 the next 5 bytes (tag and 4 byte offset)
  // For all small literals we only need 1 byte buf for literals 60...63 the
  // length is encoded in 1...4 extra bytes.
  const uint32_t needed = CalculateNeeded(c);
  assert(needed <= sizeof(scratch_));

  // Read more bytes from reader if needed
  uint32_t nbuf = ip_limit_ - ip;
  if (nbuf < needed) {
    // Stitch together bytes from ip and reader to form the word
    // contents.  We store the needed bytes in "scratch_".  They
    // will be consumed immediately by the caller since we do not
    // read more than we need.
    std::memmove(scratch_, ip, nbuf);
    reader_->Skip(peeked_);  // All peeked bytes are used up
    peeked_ = 0;
    while (nbuf < needed) {
      size_t length;
      const char* src = reader_->Peek(&length);
      if (length == 0) return false;
      uint32_t to_add = std::min<uint32_t>(needed - nbuf, length);
      std::memcpy(scratch_ + nbuf, src, to_add);
      nbuf += to_add;
      reader_->Skip(to_add);
    }
    assert(nbuf == needed);
    ip_ = scratch_;
    ip_limit_ = scratch_ + needed;
  } else if (nbuf < kMaximumTagLength) {
    // Have enough bytes, but move into scratch_ so that we do not
    // read past end of input
    std::memmove(scratch_, ip, nbuf);
    reader_->Skip(peeked_);  // All peeked bytes are used up
    peeked_ = 0;
    ip_ = scratch_;
    ip_limit_ = scratch_ + nbuf;
  } else {
    // Pass pointer to buffer returned by reader_.
    ip_ = ip;
  }
  return true;
}

/*template <typename Writer>
static bool InternalUncompress(Source* r, Writer* writer)
-> InternalUncompress_c, InternalUncompress_avx
*/

/*template <typename Writer>
static bool InternalUncompressAllTags(SnappyDecompressor* decompressor,
                                      Writer* writer, uint32_t compressed_len,
                                      uint32_t uncompressed_len)
-> InternalUncompressAllTags_c, InternalUncompressAllTags_avx
*/

template <typename T>
bool InternalGetUncompressedLength(Source* source, uint32_t* result) {
    if (source == NULL || result == NULL) return false;
    SnappyDecompressor<T> decompressor(source);
    return decompressor.ReadUncompressedLength(result);
}

bool GetUncompressedLength(Source* source, uint32_t* result) {
  AOCL_SETUP_NATIVE();
  return GetUncompressedLengthInternal_fp(source, result);
}

size_t Compress(Source* reader, Sink* writer) {
  return Compress(reader, writer, CompressionOptions{});
}

size_t Compress(Source* reader, Sink* writer, CompressionOptions options) {
  assert(options.level == 1 || options.level == 2);
  AOCL_SETUP_NATIVE();
  if (reader == NULL || writer == NULL) return 0;
  int token = 0;
  size_t written = 0;
  size_t N = reader->Available();
  const size_t uncompressed_size = N;
  char ulength[Varint::kMax32];
  char* p = Varint::Encode32(ulength, N);
  writer->Append(ulength, p - ulength);
  written += (p - ulength);

  internal::WorkingMemory wmem(N);

  while (N > 0) {
    // Get next block to compress (without copying if possible)
    size_t fragment_size;
    const char* fragment = reader->Peek(&fragment_size);
    assert(fragment_size != 0);  // premature end of input
    const size_t num_to_read = std::min(N, kBlockSize);
    size_t bytes_read = fragment_size;

    size_t pending_advance = 0;
    if (bytes_read >= num_to_read) {
      // Buffer returned by reader is large enough
      pending_advance = num_to_read;
      fragment_size = num_to_read;
    } else {
      char* scratch = wmem.GetScratchInput();
      std::memcpy(scratch, fragment, bytes_read);
      reader->Skip(bytes_read);

      while (bytes_read < num_to_read) {
        fragment = reader->Peek(&fragment_size);
        size_t n = std::min<size_t>(fragment_size, num_to_read - bytes_read);
        std::memcpy(scratch + bytes_read, fragment, n);
        bytes_read += n;
        reader->Skip(n);
      }
      assert(bytes_read == num_to_read);
      fragment = scratch;
      fragment_size = num_to_read;
    }
    assert(fragment_size == num_to_read);

    // Get encoding table for compression
    int table_size;
    uint16_t* table = wmem.GetHashTable(num_to_read, &table_size);

    // Compress input_fragment and append to dest
    int max_output = MaxCompressedLength(num_to_read);

    // Since we encode kBlockSize regions followed by a region
    // which is <= kBlockSize in length, a previously allocated
    // scratch_output[] region is big enough for this iteration.
    // Need a scratch buffer for the output, in case the byte sink doesn't
    // have room for us directly.
    char* dest = writer->GetAppendBuffer(max_output, wmem.GetScratchOutput());
    char* end = nullptr;
    if (options.level == 1) {
      end = SNAPPY_compress_fragment_fp(fragment, fragment_size, dest, table,
                                       table_size);
    } else if (options.level == 2) {
      end = internal::CompressFragmentDoubleHash(
          fragment, fragment_size, dest, table, table_size >> 1,
          table + (table_size >> 1), table_size >> 1);
    }
    writer->Append(dest, end - dest);
    written += (end - dest);

    N -= num_to_read;
    reader->Skip(pending_advance);
  }

  Report(token, "snappy_compress", written, uncompressed_size);
  AOCL_LOG_API_SUMMARY(options.level, uncompressed_size, written);
  return written;
}

// -----------------------------------------------------------------------
// IOVec interfaces
// -----------------------------------------------------------------------

// A `Source` implementation that yields the contents of an `iovec` array. Note
// that `total_size` is the total number of bytes to be read from the elements
// of `iov` (_not_ the total number of elements in `iov`).
class SnappyIOVecReader : public Source {
 public:
  SnappyIOVecReader(const struct iovec* iov, size_t total_size)
      : curr_iov_(iov),
        curr_pos_(total_size > 0 ? reinterpret_cast<const char*>(iov->iov_base)
                                 : nullptr),
        curr_size_remaining_(total_size > 0 ? iov->iov_len : 0),
        total_size_remaining_(total_size) {
    // Skip empty leading `iovec`s.
    if (total_size > 0 && curr_size_remaining_ == 0) Advance();
  }

  ~SnappyIOVecReader() override = default;

  size_t Available() const override { return total_size_remaining_; }

  const char* Peek(size_t* len) override {
    *len = curr_size_remaining_;
    return curr_pos_;
  }

  void Skip(size_t n) override {
    while (n >= curr_size_remaining_ && n > 0) {
      n -= curr_size_remaining_;
      Advance();
    }
    curr_size_remaining_ -= n;
    total_size_remaining_ -= n;
    curr_pos_ += n;
  }

 private:
  // Advances to the next nonempty `iovec` and updates related variables.
  void Advance() {
    do {
      assert(total_size_remaining_ >= curr_size_remaining_);
      total_size_remaining_ -= curr_size_remaining_;
      if (total_size_remaining_ == 0) {
        curr_pos_ = nullptr;
        curr_size_remaining_ = 0;
        return;
      }
      ++curr_iov_;
      curr_pos_ = reinterpret_cast<const char*>(curr_iov_->iov_base);
      curr_size_remaining_ = curr_iov_->iov_len;
    } while (curr_size_remaining_ == 0);
  }

  // The `iovec` currently being read.
  const struct iovec* curr_iov_;
  // The location in `curr_iov_` currently being read.
  const char* curr_pos_;
  // The amount of unread data in `curr_iov_`.
  size_t curr_size_remaining_;
  // The amount of unread data in the entire input array.
  size_t total_size_remaining_;
};

// A type that writes to an iovec.
// Note that this is not a "ByteSink", but a type that matches the
// Writer template argument to SnappyDecompressor::DecompressAllTags().
class SnappyIOVecWriter {
 private:
  // output_iov_end_ is set to iov + count and used to determine when
  // the end of the iovs is reached.
  const struct iovec* output_iov_end_;

#if !defined(NDEBUG)
  const struct iovec* output_iov_;
#endif  // !defined(NDEBUG)

  // Current iov that is being written into.
  const struct iovec* curr_iov_;

  // Pointer to current iov's write location.
  char* curr_iov_output_;

  // Remaining bytes to write into curr_iov_output.
  size_t curr_iov_remaining_;

  // Total bytes decompressed into output_iov_ so far.
  size_t total_written_;

  // Maximum number of bytes that will be decompressed into output_iov_.
  size_t output_limit_;

  static inline char* GetIOVecPointer(const struct iovec* iov, size_t offset) {
    return reinterpret_cast<char*>(iov->iov_base) + offset;
  }

 public:
  // Does not take ownership of iov. iov must be valid during the
  // entire lifetime of the SnappyIOVecWriter.
  inline SnappyIOVecWriter(const struct iovec* iov, size_t iov_count)
      : output_iov_end_(iov + iov_count),
#if !defined(NDEBUG)
        output_iov_(iov),
#endif  // !defined(NDEBUG)
        curr_iov_(iov),
        curr_iov_output_(iov_count ? reinterpret_cast<char*>(iov->iov_base)
                                   : nullptr),
        curr_iov_remaining_(iov_count ? iov->iov_len : 0),
        total_written_(0),
        output_limit_(-1) {
  }

  inline void SetExpectedLength(size_t len) { output_limit_ = len; }

  inline bool CheckLength() const { return total_written_ == output_limit_; }

  inline bool Append(const char* ip, size_t len, char**) {
    if (total_written_ + len > output_limit_) {
      return false;
    }

    return AppendNoCheck(ip, len);
  }

  char* GetOutputPtr() { return nullptr; }
  char* GetBase(ptrdiff_t*) { return nullptr; }
  void SetOutputPtr(char* op) {
    // TODO: Switch to [[maybe_unused]] when we can assume C++17.
    (void)op;
  }

  inline bool AppendNoCheck(const char* ip, size_t len) {
    while (len > 0) {
      if (curr_iov_remaining_ == 0) {
        // This iovec is full. Go to the next one.
        if (curr_iov_ + 1 >= output_iov_end_) {
          return false;
        }
        ++curr_iov_;
        curr_iov_output_ = reinterpret_cast<char*>(curr_iov_->iov_base);
        curr_iov_remaining_ = curr_iov_->iov_len;
      }

      const size_t to_write = std::min(len, curr_iov_remaining_);
      std::memcpy(curr_iov_output_, ip, to_write);
      curr_iov_output_ += to_write;
      curr_iov_remaining_ -= to_write;
      total_written_ += to_write;
      ip += to_write;
      len -= to_write;
    }

    return true;
  }

  inline bool TryFastAppend(const char* ip, size_t available, size_t len,
                            char**) {
    const size_t space_left = output_limit_ - total_written_;
    if (len <= 16 && available >= 16 + kMaximumTagLength && space_left >= 16 &&
        curr_iov_remaining_ >= 16) {
      // Fast path, used for the majority (about 95%) of invocations.
      UnalignedCopy128(ip, curr_iov_output_);
      curr_iov_output_ += len;
      curr_iov_remaining_ -= len;
      total_written_ += len;
      return true;
    }

    return false;
  }

  inline bool AppendFromSelf(size_t offset, size_t len, char**) {
    // See SnappyArrayWriter::AppendFromSelf for an explanation of
    // the "offset - 1u" trick.
    if (offset - 1u >= total_written_) {
      return false;
    }
    const size_t space_left = output_limit_ - total_written_;
    if (len > space_left) {
      return false;
    }

    // Locate the iovec from which we need to start the copy.
    const iovec* from_iov = curr_iov_;
    size_t from_iov_offset = curr_iov_->iov_len - curr_iov_remaining_;
    while (offset > 0) {
      if (from_iov_offset >= offset) {
        from_iov_offset -= offset;
        break;
      }

      offset -= from_iov_offset;
      --from_iov;
#if !defined(NDEBUG)
      assert(from_iov >= output_iov_);
#endif  // !defined(NDEBUG)
      from_iov_offset = from_iov->iov_len;
    }

    // Copy <len> bytes starting from the iovec pointed to by from_iov_index to
    // the current iovec.
    while (len > 0) {
      assert(from_iov <= curr_iov_);
      if (from_iov != curr_iov_) {
        const size_t to_copy =
            std::min(from_iov->iov_len - from_iov_offset, len);
        AppendNoCheck(GetIOVecPointer(from_iov, from_iov_offset), to_copy);
        len -= to_copy;
        if (len > 0) {
          ++from_iov;
          from_iov_offset = 0;
        }
      } else {
        size_t to_copy = curr_iov_remaining_;
        if (to_copy == 0) {
          // This iovec is full. Go to the next one.
          if (curr_iov_ + 1 >= output_iov_end_) {
            return false;
          }
          ++curr_iov_;
          curr_iov_output_ = reinterpret_cast<char*>(curr_iov_->iov_base);
          curr_iov_remaining_ = curr_iov_->iov_len;
          continue;
        }
        if (to_copy > len) {
          to_copy = len;
        }
        assert(to_copy > 0);

        IncrementalCopy(GetIOVecPointer(from_iov, from_iov_offset),
                        curr_iov_output_, curr_iov_output_ + to_copy,
                        curr_iov_output_ + curr_iov_remaining_);
        curr_iov_output_ += to_copy;
        curr_iov_remaining_ -= to_copy;
        from_iov_offset += to_copy;
        total_written_ += to_copy;
        len -= to_copy;
      }
    }

    return true;
  }

  inline void Flush() {}
};

bool RawUncompressToIOVec(const char* compressed, size_t compressed_length,
                          const struct iovec* iov, size_t iov_cnt) {
  AOCL_SETUP_NATIVE();
  if (compressed == NULL || iov == NULL) return false;
  ByteArraySource reader(compressed, compressed_length);
  return RawUncompressToIOVec(&reader, iov, iov_cnt);
}

bool RawUncompressToIOVec(Source* compressed, const struct iovec* iov,
                          size_t iov_cnt) {
  AOCL_SETUP_NATIVE();
  if (compressed == NULL || iov == NULL) return false;
  SnappyIOVecWriter output(iov, iov_cnt);
  return InternalUncompressIOVec_fp(compressed, &output);
}

// -----------------------------------------------------------------------
// Flat array interfaces
// -----------------------------------------------------------------------

// A type that writes to a flat array.
// Note that this is not a "ByteSink", but a type that matches the
// Writer template argument to SnappyDecompressor::DecompressAllTags().
class SnappyArrayWriter {
 private:
  char* base_;
  char* op_;
  char* op_limit_;
  // If op < op_limit_min_slop_ then it's safe to unconditionally write
  // kSlopBytes starting at op.
  char* op_limit_min_slop_;

 public:
  inline explicit SnappyArrayWriter(char* dst)
      : base_(dst),
        op_(dst),
        op_limit_(dst),
        op_limit_min_slop_(dst) {}  // Safe default see invariant.

  inline void SetExpectedLength(size_t len) {
    op_limit_ = op_ + len;
    // Prevent pointer from being past the buffer.
    op_limit_min_slop_ = op_limit_ - std::min<size_t>(kSlopBytes - 1, len);
  }

  inline bool CheckLength() const { return op_ == op_limit_; }

  char* GetOutputPtr() { return op_; }
  char* GetBase(ptrdiff_t* op_limit_min_slop) {
    *op_limit_min_slop = op_limit_min_slop_ - base_;
    return base_;
  }
  void SetOutputPtr(char* op) { op_ = op; }

  inline bool Append(const char* ip, size_t len, char** op_p) {
    char* op = *op_p;
    const size_t space_left = op_limit_ - op;
    if (space_left < len) return false;
    std::memcpy(op, ip, len);
    *op_p = op + len;
    return true;
  }

  inline bool TryFastAppend(const char* ip, size_t available, size_t len,
                            char** op_p) {
    char* op = *op_p;
    const size_t space_left = op_limit_ - op;
    if (len <= 16 && available >= 16 + kMaximumTagLength && space_left >= 16) {
      // Fast path, used for the majority (about 95%) of invocations.
      UnalignedCopy128(ip, op);
      *op_p = op + len;
      return true;
    } else {
      return false;
    }
  }

  SNAPPY_ATTRIBUTE_ALWAYS_INLINE
  inline bool AppendFromSelf(size_t offset, size_t len, char** op_p) {
    assert(len > 0);
    char* const op = *op_p;
    assert(op >= base_);
    char* const op_end = op + len;

    // Check if we try to append from before the start of the buffer.
    if (SNAPPY_PREDICT_FALSE(static_cast<size_t>(op - base_) < offset))
      return false;

    if (SNAPPY_PREDICT_FALSE((kSlopBytes < 64 && len > kSlopBytes) ||
                            op >= op_limit_min_slop_ || offset < len)) {
      if (op_end > op_limit_ || offset == 0) return false;
      *op_p = IncrementalCopy(op - offset, op, op_end, op_limit_);
      return true;
    }
    std::memmove(op, op - offset, kSlopBytes);
    *op_p = op_end;
    return true;
  }
  inline size_t Produced() const {
    assert(op_ >= base_);
    return op_ - base_;
  }
  inline void Flush() {}
};

// "SAW" stands for "SnappyArrayWriter" as that is the
// class used for decompression of flat buffers to flat buffers in this library.
bool SAW_RawUncompress_c(const char* compressed, size_t compressed_length, char* uncompressed) {
    ByteArraySource reader(compressed, compressed_length);
    SnappyArrayWriter output(uncompressed);
    return InternalUncompressArray_fp(&reader, &output);
}

#ifdef AOCL_SNAPPY_OPT
bool RawUncompress(const char* compressed, size_t compressed_length, char* uncompressed) {
  LOG_UNFORMATTED(TRACE, logCtx, "Enter");
  AOCL_SETUP_NATIVE();
#ifdef AOCL_ENABLE_THREADS // Threaded
  return AOCL_RawUncompress_mt(compressed, compressed_length, uncompressed);
#else // Non-threaded
  // sanity checks ------------------------------------------------------------
     size_t ulength;
     if (!GetUncompressedLength(compressed, compressed_length, &ulength))
     {
        LOG_UNFORMATTED(TRACE, logCtx, "Exit");
        return false;
     }
     if (ulength != 0 && uncompressed == NULL)
     {
        LOG_UNFORMATTED(TRACE, logCtx, "Exit");
        return false;
     }
  // sanity checks ------------------------------------------------------------
  bool ret = false;
  ret = SNAPPY_SAW_raw_uncompress_fp(compressed, compressed_length, uncompressed);
  LOG_UNFORMATTED(TRACE, logCtx, "Exit");
  return ret;
#endif
}
#else
bool RawUncompress(const char* compressed, size_t compressed_length,
                   char* uncompressed) {
  ByteArraySource reader(compressed, compressed_length);
  return RawUncompress(&reader, uncompressed);
}
#endif /* AOCL_SNAPPY_OPT */

bool RawUncompress(Source* compressed, char* uncompressed) {
  // sanity checks ------------------------------------------------------------
  size_t _readable_length, ulength;
  const char* _compressed_buffer;
  if (compressed == NULL) return false;
  _compressed_buffer = compressed->Peek(&_readable_length);
  if (!GetUncompressedLength(_compressed_buffer, _readable_length, &ulength)) return false;
  if (ulength != 0 && uncompressed == NULL) return false;
  // sanity checks ------------------------------------------------------------

  SnappyArrayWriter output(uncompressed);
  return InternalUncompressArray_fp(compressed, &output);
}

bool Uncompress(const char* compressed, size_t compressed_length,
                std::string* uncompressed) {
  AOCL_SETUP_NATIVE();
  size_t ulength;
  if (!GetUncompressedLength(compressed, compressed_length, &ulength)) {
    return false;
  }
  // On 32-bit builds: max_size() < kuint32max.  Check for that instead
  // of crashing (e.g., consider externally specified compressed data).
  if (uncompressed == NULL || ulength > uncompressed->max_size()) {
    return false;
  }
  STLStringResizeUninitialized(uncompressed, ulength);
  return RawUncompress(compressed, compressed_length,
                       string_as_array(uncompressed));
}

// A Writer that drops everything on the floor and just does validation
class SnappyDecompressionValidator {
 private:
  size_t expected_;
  size_t produced_;

 public:
  inline SnappyDecompressionValidator() : expected_(0), produced_(0) {}
  inline void SetExpectedLength(size_t len) { expected_ = len; }
  size_t GetOutputPtr() { return produced_; }
  size_t GetBase(ptrdiff_t* op_limit_min_slop) {
    *op_limit_min_slop = std::numeric_limits<ptrdiff_t>::max() - kSlopBytes + 1;
    return 1;
  }
  void SetOutputPtr(size_t op) { produced_ = op; }
  inline bool CheckLength() const { return expected_ == produced_; }
  inline bool Append(const char* ip, size_t len, size_t* produced) {
    // TODO: Switch to [[maybe_unused]] when we can assume C++17.
    (void)ip;

    *produced += len;
    return *produced <= expected_;
  }
  inline bool TryFastAppend(const char* ip, size_t available, size_t length,
                            size_t* produced) {
    // TODO: Switch to [[maybe_unused]] when we can assume C++17.
    (void)ip;
    (void)available;
    (void)length;
    (void)produced;

    return false;
  }
  inline bool AppendFromSelf(size_t offset, size_t len, size_t* produced) {
    // See SnappyArrayWriter::AppendFromSelf for an explanation of
    // the "offset - 1u" trick.
    if (*produced <= offset - 1u) return false;
    *produced += len;
    return *produced <= expected_;
  }
  inline void Flush() {}
};

bool IsValidCompressedBuffer(const char* compressed, size_t compressed_length) {
  AOCL_SETUP_NATIVE();
  if (compressed == NULL) return false;
  ByteArraySource reader(compressed, compressed_length);
  SnappyDecompressionValidator writer;
  return InternalUncompressDecompression_fp(&reader, &writer);
}

bool IsValidCompressed(Source* compressed) {
  AOCL_SETUP_NATIVE();
  if (compressed == NULL) return false;
  SnappyDecompressionValidator writer;
  return InternalUncompressDecompression_fp(compressed, &writer);
}

void RawCompress(const char* input, size_t input_length, char* compressed,
                 size_t* compressed_length) {
  RawCompress(input, input_length, compressed, compressed_length,
              CompressionOptions{});
}

void RawCompress(const char* input, size_t input_length, char* compressed,
                 size_t* compressed_length, CompressionOptions options) {
  LOG_UNFORMATTED(TRACE, logCtx, "Enter");
  if ((input_length != 0 && input == NULL) || compressed == NULL || compressed_length == NULL)
  {
    LOG_UNFORMATTED(ERR, logCtx, "Invalid input");
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
    return;
  }
  AOCL_SETUP_NATIVE();

#ifdef AOCL_ENABLE_THREADS
  AOCL_RawCompress_mt(input, input_length, compressed, compressed_length, options);
#else
  ByteArraySource reader(input, input_length);
  UncheckedByteArraySink writer(compressed);
  Compress(&reader, &writer, options);

  // Compute how many bytes were added
  *compressed_length = (writer.CurrentDestination() - compressed);
#endif /* AOCL_ENABLE_THREADS */

  LOG_UNFORMATTED(TRACE, logCtx, "Exit");
}

void RawCompressFromIOVec(const struct iovec* iov, size_t uncompressed_length,
                          char* compressed, size_t* compressed_length) {
  RawCompressFromIOVec(iov, uncompressed_length, compressed, compressed_length,
                       CompressionOptions{});
}

void RawCompressFromIOVec(const struct iovec* iov, size_t uncompressed_length,
                          char* compressed, size_t* compressed_length,
                          CompressionOptions options) {
  SnappyIOVecReader reader(iov, uncompressed_length);
  UncheckedByteArraySink writer(compressed);
  Compress(&reader, &writer, options);

  // Compute how many bytes were added.
  *compressed_length = writer.CurrentDestination() - compressed;
}

size_t Compress(const char* input, size_t input_length,
                std::string* compressed) {
  return Compress(input, input_length, compressed, CompressionOptions{});
}

size_t Compress(const char* input, size_t input_length, std::string* compressed,
                CompressionOptions options) {
  AOCL_SETUP_NATIVE();
  if (input == NULL || compressed == NULL) return 0;
  // Pre-grow the buffer to the max length of the compressed output
  STLStringResizeUninitialized(compressed, MaxCompressedLength(input_length));

  size_t compressed_length=0;
  RawCompress(input, input_length, string_as_array(compressed),
              &compressed_length, options);
  compressed->erase(compressed_length);
  return compressed_length;
}

size_t CompressFromIOVec(const struct iovec* iov, size_t iov_cnt,
                         std::string* compressed) {
  return CompressFromIOVec(iov, iov_cnt, compressed, CompressionOptions{});
}

size_t CompressFromIOVec(const struct iovec* iov, size_t iov_cnt,
                         std::string* compressed, CompressionOptions options) {
  // Compute the number of bytes to be compressed.
  size_t uncompressed_length = 0;
  for (size_t i = 0; i < iov_cnt; ++i) {
    uncompressed_length += iov[i].iov_len;
  }

  // Pre-grow the buffer to the max length of the compressed output.
  STLStringResizeUninitialized(compressed, MaxCompressedLength(
      uncompressed_length));

  size_t compressed_length;
  RawCompressFromIOVec(iov, uncompressed_length, string_as_array(compressed),
                       &compressed_length, options);
  compressed->erase(compressed_length);
  return compressed_length;
}

// -----------------------------------------------------------------------
// Sink interface
// -----------------------------------------------------------------------

// A type that decompresses into a Sink. The template parameter
// Allocator must export one method "char* Allocate(int size);", which
// allocates a buffer of "size" and appends that to the destination.
template <typename Allocator>
class SnappyScatteredWriter {
  Allocator allocator_;

  // We need random access into the data generated so far.  Therefore
  // we keep track of all of the generated data as an array of blocks.
  // All of the blocks except the last have length kBlockSize.
  std::vector<char*> blocks_;
  size_t expected_;

  // Total size of all fully generated blocks so far
  size_t full_size_;

  // Pointer into current output block
  char* op_base_;   // Base of output block
  char* op_ptr_;    // Pointer to next unfilled byte in block
  char* op_limit_;  // Pointer just past block
  // If op < op_limit_min_slop_ then it's safe to unconditionally write
  // kSlopBytes starting at op.
  char* op_limit_min_slop_;

  inline size_t Size() const { return full_size_ + (op_ptr_ - op_base_); }

  bool SlowAppend(const char* ip, size_t len);
  bool SlowAppendFromSelf(size_t offset, size_t len);

 public:
  inline explicit SnappyScatteredWriter(const Allocator& allocator)
      : allocator_(allocator),
        expected_(0),
        full_size_(0),
        op_base_(NULL),
        op_ptr_(NULL),
        op_limit_(NULL),
        op_limit_min_slop_(NULL) {}
  char* GetOutputPtr() { return op_ptr_; }
  char* GetBase(ptrdiff_t* op_limit_min_slop) {
    *op_limit_min_slop = op_limit_min_slop_ - op_base_;
    return op_base_;
  }
  void SetOutputPtr(char* op) { op_ptr_ = op; }

  inline void SetExpectedLength(size_t len) {
    assert(blocks_.empty());
    expected_ = len;
  }

  inline bool CheckLength() const { return Size() == expected_; }

  // Return the number of bytes actually uncompressed so far
  inline size_t Produced() const { return Size(); }

  inline bool Append(const char* ip, size_t len, char** op_p) {
    char* op = *op_p;
    size_t avail = op_limit_ - op;
    if (len <= avail) {
      // Fast path
      std::memcpy(op, ip, len);
      *op_p = op + len;
      return true;
    } else {
      op_ptr_ = op;
      bool res = SlowAppend(ip, len);
      *op_p = op_ptr_;
      return res;
    }
  }

  inline bool TryFastAppend(const char* ip, size_t available, size_t length,
                            char** op_p) {
    char* op = *op_p;
    const int space_left = op_limit_ - op;
    if (length <= 16 && available >= 16 + kMaximumTagLength &&
        space_left >= 16) {
      // Fast path, used for the majority (about 95%) of invocations.
      UnalignedCopy128(ip, op);
      *op_p = op + length;
      return true;
    } else {
      return false;
    }
  }

  inline bool AppendFromSelf(size_t offset, size_t len, char** op_p) {
    char* op = *op_p;
    assert(op >= op_base_);
    // Check if we try to append from before the start of the buffer.
    if (SNAPPY_PREDICT_FALSE((kSlopBytes < 64 && len > kSlopBytes) ||
                            static_cast<size_t>(op - op_base_) < offset ||
                            op >= op_limit_min_slop_ || offset < len)) {
      if (offset == 0) return false;
      if (SNAPPY_PREDICT_FALSE(static_cast<size_t>(op - op_base_) < offset ||
                              op + len > op_limit_)) {
        op_ptr_ = op;
        bool res = SlowAppendFromSelf(offset, len);
        *op_p = op_ptr_;
        return res;
      }
      *op_p = IncrementalCopy(op - offset, op, op + len, op_limit_);
      return true;
    }
    // Fast path
    char* const op_end = op + len;
    std::memmove(op, op - offset, kSlopBytes);
    *op_p = op_end;
    return true;
  }

  // Called at the end of the decompress. We ask the allocator
  // write all blocks to the sink.
  inline void Flush() { allocator_.Flush(Produced()); }
};

template <typename Allocator>
bool SnappyScatteredWriter<Allocator>::SlowAppend(const char* ip, size_t len) {
  size_t avail = op_limit_ - op_ptr_;
  while (len > avail) {
    // Completely fill this block
    std::memcpy(op_ptr_, ip, avail);
    op_ptr_ += avail;
    assert(op_limit_ - op_ptr_ == 0);
    full_size_ += (op_ptr_ - op_base_);
    len -= avail;
    ip += avail;

    // Bounds check
    if (full_size_ + len > expected_) return false;

    // Make new block
    size_t bsize = std::min<size_t>(kBlockSize, expected_ - full_size_);
    op_base_ = allocator_.Allocate(bsize);
    op_ptr_ = op_base_;
    op_limit_ = op_base_ + bsize;
    op_limit_min_slop_ = op_limit_ - std::min<size_t>(kSlopBytes - 1, bsize);

    blocks_.push_back(op_base_);
    avail = bsize;
  }

  std::memcpy(op_ptr_, ip, len);
  op_ptr_ += len;
  return true;
}

template <typename Allocator>
bool SnappyScatteredWriter<Allocator>::SlowAppendFromSelf(size_t offset,
                                                         size_t len) {
  // Overflow check
  // See SnappyArrayWriter::AppendFromSelf for an explanation of
  // the "offset - 1u" trick.
  const size_t cur = Size();
  if (offset - 1u >= cur) return false;
  if (expected_ - cur < len) return false;

  // Currently we shouldn't ever hit this path because Compress() chops the
  // input into blocks and does not create cross-block copies. However, it is
  // nice if we do not rely on that, since we can get better compression if we
  // allow cross-block copies and thus might want to change the compressor in
  // the future.
  // TODO Replace this with a properly optimized path. This is not
  // triggered right now. But this is so super slow, that it would regress
  // performance unacceptably if triggered.
  size_t src = cur - offset;
  char* op = op_ptr_;
  while (len-- > 0) {
    char c = blocks_[src >> kBlockLog][src & (kBlockSize - 1)];
    if (!Append(&c, 1, &op)) {
      op_ptr_ = op;
      return false;
    }
    src++;
  }
  op_ptr_ = op;
  return true;
}

class SnappySinkAllocator {
 public:
  explicit SnappySinkAllocator(Sink* dest) : dest_(dest) {}

  char* Allocate(int size) {
    Datablock block(new char[size], size);
    blocks_.push_back(block);
    return block.data;
  }

  // We flush only at the end, because the writer wants
  // random access to the blocks and once we hand the
  // block over to the sink, we can't access it anymore.
  // Also we don't write more than has been actually written
  // to the blocks.
  void Flush(size_t size) {
    size_t size_written = 0;
    for (Datablock& block : blocks_) {
      size_t block_size = std::min<size_t>(block.size, size - size_written);
      dest_->AppendAndTakeOwnership(block.data, block_size,
                                    &SnappySinkAllocator::Deleter, NULL);
      size_written += block_size;
    }
    blocks_.clear();
  }

 private:
  struct Datablock {
    char* data;
    size_t size;
    Datablock(char* p, size_t s) : data(p), size(s) {}
  };

  static void Deleter(void* arg, const char* bytes, size_t size) {
    // TODO: Switch to [[maybe_unused]] when we can assume C++17.
    (void)arg;
    (void)size;

    delete[] bytes;
  }

  Sink* dest_;
  std::vector<Datablock> blocks_;

  // Note: copying this object is allowed
};

size_t UncompressAsMuchAsPossible(Source* compressed, Sink* uncompressed) {
  AOCL_SETUP_NATIVE();
  if (compressed == NULL || uncompressed == NULL) return 0;
  SnappySinkAllocator allocator(uncompressed);
  SnappyScatteredWriter<SnappySinkAllocator> writer(allocator);
  InternalUncompressScattered_fp(compressed, &writer);
  return writer.Produced();
}

bool Uncompress(Source* compressed, Sink* uncompressed) {
  AOCL_SETUP_NATIVE();
  return Uncompress_fp(compressed, uncompressed);
}

// This method is used by snappy_test.cc for creating an instance of
// ByteArraySource class, the constructor of this class can't be
// accessed by snappy_test.cc.
Source * SNAPPY_Gtest_Util::ByteArraySource_ext(const char *p, size_t n)
{
  return new ByteArraySource(p,n);
}

// This method is used by snappy_test.cc for creating an instance of
// UncheckedByteArraySink class, the constructor of this class can't be
// accessed by snappy_test.cc.
Sink * SNAPPY_Gtest_Util::UncheckedByteArraySink_ext(char *dest)
{
  return new UncheckedByteArraySink(dest);
}

// For making Varint::Append32 available for snappy_test.cc.
void SNAPPY_Gtest_Util::Append32(std::string* s, uint32_t value)
{
  return Varint::Append32(s,value);
}

#define SET_FP_TO_WITH_C \
InternalUncompressIOVec_fp           = InternalUncompress_c<SnappyIOVecWriter>;\
InternalUncompressArray_fp           = InternalUncompress_c<SnappyArrayWriter>;\
InternalUncompressDecompression_fp   = InternalUncompress_c<SnappyDecompressionValidator>;\
InternalUncompressScattered_fp       = InternalUncompress_c<SnappyScatteredWriter<SnappySinkAllocator>>;\
Uncompress_fp                        = Uncompress_c;\
GetUncompressedLengthInternal_fp     = InternalGetUncompressedLength<with_c>;\
SNAPPY_compress_fragment_fp          = internal::CompressFragment_c;

#define SET_FP_TO_WITH_AVX \
InternalUncompressIOVec_fp           = InternalUncompress_avx<SnappyIOVecWriter, with_avx>;\
InternalUncompressArray_fp           = InternalUncompress_avx<SnappyArrayWriter, with_avx>;\
InternalUncompressDecompression_fp   = InternalUncompress_avx<SnappyDecompressionValidator, with_avx>;\
InternalUncompressScattered_fp       = InternalUncompress_avx<SnappyScatteredWriter<SnappySinkAllocator>, with_avx>;\
Uncompress_fp                        = Uncompress_avx<with_avx>;\
GetUncompressedLengthInternal_fp     = InternalGetUncompressedLength<with_avx>;\
SNAPPY_compress_fragment_fp          = internal::CompressFragment_crc32;

#define SET_FP_TO_WITH_BMI_AVX \
InternalUncompressIOVec_fp           = InternalUncompress_avx<SnappyIOVecWriter, with_bmi_avx>;\
InternalUncompressArray_fp           = InternalUncompress_avx<SnappyArrayWriter, with_bmi_avx>;\
InternalUncompressDecompression_fp   = InternalUncompress_avx<SnappyDecompressionValidator, with_bmi_avx>;\
InternalUncompressScattered_fp       = InternalUncompress_avx<SnappyScatteredWriter<SnappySinkAllocator>, with_bmi_avx>;\
Uncompress_fp                        = Uncompress_avx<with_bmi_avx>;\
GetUncompressedLengthInternal_fp     = InternalGetUncompressedLength<with_bmi_avx>;\
SNAPPY_compress_fragment_fp          = internal::CompressFragment_crc32;

static void aocl_register_snappy_fmv(int optOff, CpuFeatures cpuFeatures) {
    // Derive dispatch tier from runtime CPU feature bits.
    // Priority: BMI2+AVX2 path > AVX path > C path.
    int tier = 0; // 0=C, 1=AVX, 2=BMI2+AVX2

#if defined(AOCL_SNAPPY_AVX2_OPT)
    if ((cpuFeatures & (FEATURE_AVX2 | FEATURE_BMI2)) == (FEATURE_AVX2 | FEATURE_BMI2)) {
        tier = 2;
    } else
#endif
#if defined(AOCL_SNAPPY_AVX_OPT)
    if ((cpuFeatures & FEATURE_AVX) == FEATURE_AVX) {
        tier = 1;
    } else
#endif
    {
        tier = 0;
    }

    if (optOff) {
        // Optimization disabled:
        // keep reference/raw setup for AOCL-specific hooks, but still allow
        // ISA-templated decode paths selected from detected CPU capability.
        SNAPPY_SAW_raw_uncompress_fp = SAW_RawUncompress_c;
        CalculateTableSize_fp = CalculateTableSize;
#ifdef AOCL_ENABLE_THREADS
        SNAPPY_SAW_raw_uncompress_direct_fp = SAW_RawUncompressDirect;
#endif

        if (tier == 0) {
            SET_FP_TO_WITH_C
        } else if (tier == 1) {
#if defined(AOCL_SNAPPY_AVX_OPT)
            SET_FP_TO_WITH_AVX
#else
            SET_FP_TO_WITH_C
#endif
        } else {
#if defined(AOCL_SNAPPY_AVX2_OPT)
            SET_FP_TO_WITH_BMI_AVX
#elif defined(AOCL_SNAPPY_AVX_OPT)
            SET_FP_TO_WITH_AVX
#else
            SET_FP_TO_WITH_C
#endif
        }
        return;
    }

    // Optimization enabled:
    // first set common AOCL table-size hook where supported.
#if defined(AOCL_SNAPPY_OPT) && !defined(AOCL_SNAPPY_HIGH_COMPRESSION)
    CalculateTableSize_fp = AOCL_CalculateTableSize;
#endif

    if (tier == 0) {
        // C/SSE-equivalent behavior
        SET_FP_TO_WITH_C
#ifdef AOCL_SNAPPY_OPT
        SNAPPY_compress_fragment_fp = internal::AOCL_CompressFragment_c;
#endif
        SNAPPY_SAW_raw_uncompress_fp = SAW_RawUncompress_c;
#ifdef AOCL_ENABLE_THREADS
        SNAPPY_SAW_raw_uncompress_direct_fp = SAW_RawUncompressDirect;
        InternalUncompressDirectArray_fp = InternalUncompressDirect<SnappyArrayWriter, with_c>;
#endif
        return;
    }

    if (tier == 1) {
        // AVX behavior
#if defined(AOCL_SNAPPY_AVX_OPT)
        SET_FP_TO_WITH_AVX
        SNAPPY_compress_fragment_fp = internal::AOCL_CompressFragment_crc32;
        SNAPPY_SAW_raw_uncompress_fp = AOCL_SAW_RawUncompress_avx;
#ifdef AOCL_ENABLE_THREADS
        SNAPPY_SAW_raw_uncompress_direct_fp = AOCL_SAW_RawUncompressDirect;
        InternalUncompressDirectAOCLArray_fp = InternalUncompressDirect<AOCL_SnappyArrayWriter_AVX, with_avx>;
        InternalUncompressDirectArray_fp = InternalUncompressDirect<SnappyArrayWriter, with_avx>;
#endif
        InternalUncompressAOCLArray_fp = InternalUncompress_avx<AOCL_SnappyArrayWriter_AVX, with_avx>;
#else
        SET_FP_TO_WITH_C
        SNAPPY_SAW_raw_uncompress_fp = SAW_RawUncompress_c;
#ifdef AOCL_ENABLE_THREADS
        SNAPPY_SAW_raw_uncompress_direct_fp = SAW_RawUncompressDirect;
        InternalUncompressDirectArray_fp = InternalUncompressDirect<SnappyArrayWriter, with_c>;
#endif
#endif
        return;
    }

    // BMI+AVX behavior (also used for AVX2/AVX512-class machines).
#if defined(AOCL_SNAPPY_AVX2_OPT)
    SET_FP_TO_WITH_BMI_AVX
    SNAPPY_SAW_raw_uncompress_fp = AOCL_SAW_RawUncompress_avx;
    SNAPPY_compress_fragment_fp = internal::AOCL_CompressFragment_crc32;
#ifdef AOCL_ENABLE_THREADS
    SNAPPY_SAW_raw_uncompress_direct_fp = AOCL_SAW_RawUncompressDirect;
    InternalUncompressDirectAOCLArray_fp = InternalUncompressDirect<AOCL_SnappyArrayWriter_AVX, with_bmi_avx>;
    InternalUncompressDirectArray_fp = InternalUncompressDirect<SnappyArrayWriter, with_bmi_avx>;
#endif
    InternalUncompressAOCLArray_fp = InternalUncompress_avx<AOCL_SnappyArrayWriter_AVX, with_bmi_avx>;
#elif defined(AOCL_SNAPPY_AVX_OPT)
    SET_FP_TO_WITH_AVX
    SNAPPY_SAW_raw_uncompress_fp = AOCL_SAW_RawUncompress_avx;
    SNAPPY_compress_fragment_fp = internal::AOCL_CompressFragment_crc32;
#ifdef AOCL_ENABLE_THREADS
    SNAPPY_SAW_raw_uncompress_direct_fp = AOCL_SAW_RawUncompressDirect;
    InternalUncompressDirectAOCLArray_fp = InternalUncompressDirect<AOCL_SnappyArrayWriter_AVX, with_avx>;
    InternalUncompressDirectArray_fp = InternalUncompressDirect<SnappyArrayWriter, with_avx>;
#endif
    InternalUncompressAOCLArray_fp = InternalUncompress_avx<AOCL_SnappyArrayWriter_AVX, with_avx>;
#else
    SET_FP_TO_WITH_C
    SNAPPY_SAW_raw_uncompress_fp = SAW_RawUncompress_c;
#ifdef AOCL_ENABLE_THREADS
    SNAPPY_SAW_raw_uncompress_direct_fp = SAW_RawUncompressDirect;
    InternalUncompressDirectArray_fp = InternalUncompressDirect<SnappyArrayWriter, with_c>;
#endif
#endif
}

char* aocl_setup_snappy(int optOff, int optLevel, size_t insize,
    size_t level, size_t windowLog) {
    AOCL_ENTER_CRITICAL(setup_snappy)
    if (!setup_ok_snappy) {
        CpuFeatures cpuFeatures = Dispatcher_GetSupportedFeaturesForLevel(Dispatcher_IntToLevel((int)optLevel));
        optOff = optOff ? 1 : get_disable_opt_flags(0);
        aocl_register_snappy_fmv(optOff, cpuFeatures);
        setup_ok_snappy = 1;
    }
    AOCL_EXIT_CRITICAL(setup_snappy)
    return NULL;
}

#ifdef AOCL_SNAPPY_OPT
static void aocl_setup_native(void) {
    AOCL_ENTER_CRITICAL(setup_snappy)
    if (!setup_ok_snappy) {
        CpuFeatures cpuFeatures = Dispatcher_GetFeaturesFromEnv();
        int optOff = get_disable_opt_flags(0);
        aocl_register_snappy_fmv(optOff, cpuFeatures);
        setup_ok_snappy = 1;
    }
    AOCL_EXIT_CRITICAL(setup_snappy)
}
#endif

void aocl_destroy_snappy(void){
    AOCL_ENTER_CRITICAL(setup_snappy)
    setup_ok_snappy = 0;
    AOCL_EXIT_CRITICAL(setup_snappy)
}

/* --------- Modifications to support dynamic ISA selection - start ---------*/
class with_bmi_avx {};
class with_avx {};
class with_c {};

/* ExtractLowBytes from reference code is split into 2 functions:
* ExtractLowBytes_bmi uses bmi instructions
* ExtractLowBytes_c does not use bmi instructions
* Selection is done by dynamic dispatcher in runtime */
static inline uint32_t ExtractLowBytes_c(const uint32_t& v, int n) {
  assert(n >= 0);
  assert(n <= 4);
  // This needs to be wider than uint32_t otherwise `mask << 32` will be
  // undefined.
  uint64_t mask = 0xffffffff;
  return v & ~(mask << (8 * n));
}

// Copies between size bytes and 64 bytes from src to dest.  size cannot exceed
// 64.  More than size bytes, but never exceeding 64, might be copied if doing
// so gives better performance.  [src, src + size) must not overlap with
// [dst, dst + size), but [src, src + 64) may overlap with [dst, dst + 64).
inline void MemCopy64_c(char* dst, const void* src, size_t size) {
  // Always copy this many bytes.  If that's below size then copy the full 64.
  constexpr int kShortMemCopy = 32;

  assert(size <= 64);
  assert(std::less_equal<const void*>()(static_cast<const char*>(src) + size,
                                        dst) ||
         std::less_equal<const void*>()(dst + size, src));

  // We know that src and dst are at least size bytes apart. However, because we
  // might copy more than size bytes the copy still might overlap past size.
  // E.g. if src and dst appear consecutively in memory (src + size >= dst).
  // TODO: Investigate wider copies on other platforms.
  std::memmove(dst, src, kShortMemCopy);
  // Profiling shows that nearly all copies are short.
  if (SNAPPY_PREDICT_FALSE(size > kShortMemCopy)) {
    std::memmove(dst + kShortMemCopy,
                 static_cast<const uint8_t*>(src) + kShortMemCopy,
                 64 - kShortMemCopy);
  }
}

inline void MemCopy64_c(ptrdiff_t dst, const void* src, size_t size) {
  // TODO: Switch to [[maybe_unused]] when we can assume C++17.
  (void)dst;
  (void)src;
  (void)size;
}

// Core decompression loop, when there is enough data available.
// Decompresses the input buffer [ip, ip_limit) into the output buffer
// [op, op_limit_min_slop). Returning when either we are too close to the end
// of the input buffer, or we exceed op_limit_min_slop or when a exceptional
// tag is encountered (literal of length > 60) or a copy-4.
// Returns {ip, op} at the points it stopped decoding.
// TODO This function probably does not need to be inlined, as it
// should decode large chunks at a time. This allows runtime dispatch to
// implementations based on CPU capability (BMI2 / perhaps 32 / 64 byte memcpy).
template <typename T>
std::pair<const uint8_t*, ptrdiff_t> DecompressBranchless_c(
    const uint8_t* ip, const uint8_t* ip_limit, ptrdiff_t op, T op_base,
    ptrdiff_t op_limit_min_slop) {
  // If deferred_src is invalid point it here.
  uint8_t safe_source[64];
  const void* deferred_src;
  size_t deferred_length;
  ClearDeferred(&deferred_src, &deferred_length, safe_source);

  // We unroll the inner loop twice so we need twice the spare room.
  op_limit_min_slop -= kSlopBytes;
  if (2 * (kSlopBytes + 1) < ip_limit - ip && op < op_limit_min_slop) {
    const uint8_t* const ip_limit_min_slop = ip_limit - 2 * kSlopBytes - 1;
    ip++;
    // ip points just past the tag and we are touching at maximum kSlopBytes
    // in an iteration.
    size_t tag = ip[-1];
#if defined(__clang__) && defined(__aarch64__)
    // Workaround for https://bugs.llvm.org/show_bug.cgi?id=51317
    // when loading 1 byte, clang for aarch64 doesn't realize that it(ldrb)
    // comes with free zero-extension, so clang generates another
    // 'and xn, xm, 0xff' before it use that as the offset. This 'and' is
    // redundant and can be removed by adding this dummy asm, which gives
    // clang a hint that we're doing the zero-extension at the load.
    asm("" ::"r"(tag));
#endif
    do {
      // The throughput is limited by instructions, unrolling the inner loop
      // twice reduces the amount of instructions checking limits and also
      // leads to reduced mov's.

      SNAPPY_PREFETCH(ip + 128);
      for (int i = 0; i < 2; i++) {
        const uint8_t* old_ip = ip;
        assert(tag == ip[-1]);
        // For literals tag_type = 0, hence we will always obtain 0 from
        // ExtractLowBytes. For literals offset will thus be kLiteralOffset.
        ptrdiff_t len_minus_offset = kLengthMinusOffset[tag];
        uint32_t next;
#if defined(__aarch64__)
        size_t tag_type = AdvanceToNextTagARMOptimized(&ip, &tag);
        // We never need more than 16 bits. Doing a Load16 allows the compiler
        // to elide the masking operation in ExtractOffset.
        next = LittleEndian::Load16(old_ip);
#else
        size_t tag_type = AdvanceToNextTagX86Optimized(&ip, &tag);
        next = LittleEndian::Load32(old_ip);
#endif
        size_t len = len_minus_offset & 0xFF;
        ptrdiff_t extracted = ExtractOffset(next, tag_type);
        ptrdiff_t len_min_offset = len_minus_offset - extracted;
        if (SNAPPY_PREDICT_FALSE(len_minus_offset > extracted)) {
          if (SNAPPY_PREDICT_FALSE(len & 0x80)) {
            // Exceptional case (long literal or copy 4).
            // Actually doing the copy here is negatively impacting the main
            // loop due to compiler incorrectly allocating a register for
            // this fallback. Hence we just break.
          break_loop:
            ip = old_ip;
            goto exit;
          }
          // Only copy-1 or copy-2 tags can get here.
          assert(tag_type == 1 || tag_type == 2);
          std::ptrdiff_t delta = (op + deferred_length) + len_min_offset - len;
          // Guard against copies before the buffer start.
          // Execute any deferred MemCopy since we write to dst here.
          MemCopy64_c(op_base + op, deferred_src, deferred_length);
          op += deferred_length;
          ClearDeferred(&deferred_src, &deferred_length, safe_source);
          if (SNAPPY_PREDICT_FALSE(delta < 0 ||
                                  !Copy64BytesWithPatternExtension(
                                      op_base + op, len - len_min_offset))) {
            goto break_loop;
          }
          // We aren't deferring this copy so add length right away.
          op += len;
          continue;
        }
        std::ptrdiff_t delta = (op + deferred_length) + len_min_offset - len;
        if (SNAPPY_PREDICT_FALSE(delta < 0)) {
          // Due to the spurious offset in literals have this will trigger
          // at the start of a block when op is still smaller than 256.
          if (tag_type != 0) goto break_loop;
          MemCopy64_c(op_base + op, deferred_src, deferred_length);
          op += deferred_length;
          DeferMemCopy(&deferred_src, &deferred_length, old_ip, len);
          continue;
        }

        // For copies we need to copy from op_base + delta, for literals
        // we need to copy from ip instead of from the stream.
        const void* from =
            tag_type ? reinterpret_cast<void*>(op_base + delta) : old_ip;
        MemCopy64_c(op_base + op, deferred_src, deferred_length);
        op += deferred_length;
        DeferMemCopy(&deferred_src, &deferred_length, from, len);
      }
    } while (ip < ip_limit_min_slop &&
             static_cast<ptrdiff_t>(op + deferred_length) < op_limit_min_slop);
  exit:
    ip--;
    assert(ip <= ip_limit);
  }
  // If we deferred a copy then we can perform.  If we are up to date then we
  // might not have enough slop bytes and could run past the end.
  if (deferred_length) {
      MemCopy64_c(op_base + op, deferred_src, deferred_length);
    op += deferred_length;
    ClearDeferred(&deferred_src, &deferred_length, safe_source);
  }
  return {ip, op};
}

// Given a table of uint16_t whose size is mask / 2 + 1, return a pointer to the
// relevant entry, if any, for the given bytes.  Any hash function will do,
// but a good hash function reduces the number of collisions and thus yields
// better compression for compressible input.
//
// REQUIRES: mask is 2 * (table_size - 1), and table_size is a power of two.
inline uint16_t* TableEntry_c(uint16_t* table, uint32_t bytes, uint32_t mask) {
  // Our choice is quicker-and-dirtier than the typical hash function;
  // empirically, that seems beneficial.  The upper bits of kMagic * bytes are a
  // higher-quality hash than the lower bits, so when using kMagic * bytes we
  // also shift right to get a higher-quality end result.  There's no similar
  // issue with a CRC because all of the output bits of a CRC are equally good
  // "hashes." So, a CPU instruction for CRC, if available, tends to be a good
  // choice.
  constexpr uint32_t kMagic = 0x1e35a7bd;
  const uint32_t hash = (kMagic * bytes) >> (31 - kMaxHashTableBits);
  return reinterpret_cast<uint16_t*>(reinterpret_cast<uintptr_t>(table) +
                                     (hash & mask));
}

namespace internal {
char* CompressFragment_c(const char* input, size_t input_size, char* op,
                       uint16_t* table, const int table_size) {
  // "ip" is the input pointer, and "op" is the output pointer.
  const char* ip = input;
  assert(input_size <= kBlockSize);
  assert((table_size & (table_size - 1)) == 0);  // table must be power of two
  const uint32_t mask = 2 * (table_size - 1);
  const char* ip_end = input + input_size;
  const char* base_ip = ip;

  const size_t kInputMarginBytes = 15;
  if (SNAPPY_PREDICT_TRUE(input_size >= kInputMarginBytes)) {
    const char* ip_limit = input + input_size - kInputMarginBytes;

    for (uint32_t preload = LittleEndian::Load32(ip + 1);;) {
      // Bytes in [next_emit, ip) will be emitted as literal bytes.  Or
      // [next_emit, ip_end) after the main loop.
      const char* next_emit = ip++;
      uint64_t data = LittleEndian::Load64(ip);
      // The body of this loop calls EmitLiteral once and then EmitCopy one or
      // more times.  (The exception is that when we're close to exhausting
      // the input we goto emit_remainder.)
      //
      // In the first iteration of this loop we're just starting, so
      // there's nothing to copy, so calling EmitLiteral once is
      // necessary.  And we only start a new iteration when the
      // current iteration has determined that a call to EmitLiteral will
      // precede the next call to EmitCopy (if any).
      //
      // Step 1: Scan forward in the input looking for a 4-byte-long match.
      // If we get close to exhausting the input then goto emit_remainder.
      //
      // Heuristic match skipping: If 32 bytes are scanned with no matches
      // found, start looking only at every other byte. If 32 more bytes are
      // scanned (or skipped), look at every third byte, etc.. When a match is
      // found, immediately go back to looking at every byte. This is a small
      // loss (~5% performance, ~0.1% density) for compressible data due to more
      // bookkeeping, but for non-compressible data (such as JPEG) it's a huge
      // win since the compressor quickly "realizes" the data is incompressible
      // and doesn't bother looking for matches everywhere.
      //
      // The "skip" variable keeps track of how many bytes there are since the
      // last match; dividing it by 32 (ie. right-shifting by five) gives the
      // number of bytes to move ahead for each iteration.
      uint32_t skip = 32;

      const char* candidate;
      if (ip_limit - ip >= 16) {
        auto delta = ip - base_ip;
        for (int j = 0; j < 4; ++j) {
          for (int k = 0; k < 4; ++k) {
            int i = 4 * j + k;
            // These for-loops are meant to be unrolled. So we can freely
            // special case the first iteration to use the value already
            // loaded in preload.
            uint32_t dword = i == 0 ? preload : static_cast<uint32_t>(data);
            assert(dword == LittleEndian::Load32(ip + i));
            uint16_t* table_entry = TableEntry_c(table, dword, mask);
            candidate = base_ip + *table_entry;
            assert(candidate >= base_ip);
            assert(candidate < ip + i);
            *table_entry = delta + i;
            if (SNAPPY_PREDICT_FALSE(LittleEndian::Load32(candidate) == dword)) {
              *op = LITERAL | (i << 2);
              UnalignedCopy128(next_emit, op + 1);
              ip += i;
              op = op + i + 2;
              goto emit_match;
            }
            data >>= 8;
          }
          data = LittleEndian::Load64(ip + 4 * j + 4);
        }
        ip += 16;
        skip += 16;
      }
      while (true) {
        assert(static_cast<uint32_t>(data) == LittleEndian::Load32(ip));
        uint16_t* table_entry = TableEntry_c(table, data, mask);
        uint32_t bytes_between_hash_lookups = skip >> 5;
        skip += bytes_between_hash_lookups;
        const char* next_ip = ip + bytes_between_hash_lookups;
        if (SNAPPY_PREDICT_FALSE(next_ip > ip_limit)) {
          ip = next_emit;
          goto emit_remainder;
        }
        candidate = base_ip + *table_entry;
        assert(candidate >= base_ip);
        assert(candidate < ip);

        *table_entry = ip - base_ip;
        if (SNAPPY_PREDICT_FALSE(static_cast<uint32_t>(data) ==
                                LittleEndian::Load32(candidate))) {
          break;
        }
        data = LittleEndian::Load32(next_ip);
        ip = next_ip;
      }

      // Step 2: A 4-byte match has been found.  We'll later see if more
      // than 4 bytes match.  But, prior to the match, input
      // bytes [next_emit, ip) are unmatched.  Emit them as "literal bytes."
      assert(next_emit + 16 <= ip_end);
      op = EmitLiteral</*allow_fast_path=*/true>(op, next_emit, ip - next_emit);

      // Step 3: Call EmitCopy, and then see if another EmitCopy could
      // be our next move.  Repeat until we find no match for the
      // input immediately after what was consumed by the last EmitCopy call.
      //
      // If we exit this loop normally then we need to call EmitLiteral next,
      // though we don't yet know how big the literal will be.  We handle that
      // by proceeding to the next iteration of the main loop.  We also can exit
      // this loop via goto if we get close to exhausting the input.
    emit_match:
      do {
        // We have a 4-byte match at ip, and no need to emit any
        // "literal bytes" prior to ip.
        const char* base = ip;
        std::pair<size_t, bool> p =
            FindMatchLength(candidate + 4, ip + 4, ip_end, &data);
        size_t matched = 4 + p.first;
        ip += matched;
        size_t offset = base - candidate;
        assert(0 == memcmp(base, candidate, matched));
        if (p.second) {
          op = EmitCopy</*len_less_than_12=*/true>(op, offset, matched);
        } else {
          op = EmitCopy</*len_less_than_12=*/false>(op, offset, matched);
        }
        if (SNAPPY_PREDICT_FALSE(ip >= ip_limit)) {
          goto emit_remainder;
        }
        // Expect 5 bytes to match
        assert((data & 0xFFFFFFFFFF) ==
               (LittleEndian::Load64(ip) & 0xFFFFFFFFFF));
        // We are now looking for a 4-byte match again.  We read
        // table[Hash(ip, mask)] for that.  To improve compression,
        // we also update table[Hash(ip - 1, mask)] and table[Hash(ip, mask)].
        *TableEntry_c(table, LittleEndian::Load32(ip - 1), mask) =
            ip - base_ip - 1;
        uint16_t* table_entry = TableEntry_c(table, data, mask);
        candidate = base_ip + *table_entry;
        *table_entry = ip - base_ip;
        // Measurements on the benchmarks have shown the following probabilities
        // for the loop to exit (ie. avg. number of iterations is reciprocal).
        // BM_Flat/6  txt1    p = 0.3-0.4
        // BM_Flat/7  txt2    p = 0.35
        // BM_Flat/8  txt3    p = 0.3-0.4
        // BM_Flat/9  txt3    p = 0.34-0.4
        // BM_Flat/10 pb      p = 0.4
        // BM_Flat/11 gaviota p = 0.1
        // BM_Flat/12 cp      p = 0.5
        // BM_Flat/13 c       p = 0.3
      } while (static_cast<uint32_t>(data) == LittleEndian::Load32(candidate));
      // Because the least significant 5 bytes matched, we can utilize data
      // for the next iteration.
      preload = data >> 8;
    }
  }

emit_remainder:
  // Emit the remaining bytes as a literal
  if (ip < ip_end) {
    op = EmitLiteral</*allow_fast_path=*/false>(op, ip, ip_end - ip);
  }

  return op;
}
}

template <typename Writer>
static bool InternalUncompress_c(Source* r, Writer* writer) {
  // Read the uncompressed length from the front of the compressed input
  SnappyDecompressor<with_c> decompressor(r);
  uint32_t uncompressed_len = 0;
  if (!decompressor.ReadUncompressedLength(&uncompressed_len)) return false;

  return InternalUncompressAllTags_c(&decompressor, writer, r->Available(),
                                   uncompressed_len);
}

template <typename Writer, typename T>
static bool InternalUncompressAllTags_c(SnappyDecompressor<T>* decompressor,
                                      Writer* writer, uint32_t compressed_len,
                                      uint32_t uncompressed_len) {
  int token = 0;
  Report(token, "snappy_uncompress", compressed_len, uncompressed_len);

  writer->SetExpectedLength(uncompressed_len);

  // Process the entire input
  decompressor->DecompressAllTags(writer);
  writer->Flush();
  return (decompressor->eof() && writer->CheckLength());
}

bool Uncompress_c(Source* compressed, Sink* uncompressed) {
  if (compressed == NULL || uncompressed == NULL) return false;
  // Read the uncompressed length from the front of the compressed input
  SnappyDecompressor<with_c> decompressor(compressed);
  uint32_t uncompressed_len = 0;
  if (!decompressor.ReadUncompressedLength(&uncompressed_len)) {
    LOG_UNFORMATTED(ERR, logCtx, "Failed to read uncompressed length");
    return false;
  }

  char c;
  size_t allocated_size;
  char* buf = uncompressed->GetAppendBufferVariable(
      1, uncompressed_len, &c, 1, &allocated_size);

  const size_t compressed_len = compressed->Available();
  // If we can get a flat buffer, then use it, otherwise do block by block
  // uncompression
  if (allocated_size >= uncompressed_len) {
    LOG_UNFORMATTED(DEBUG, logCtx, "Allocated size sufficient to store entire decompressed stream");
    SnappyArrayWriter writer(buf);
    bool result = InternalUncompressAllTags_c(&decompressor, &writer,
                                            compressed_len, uncompressed_len);
    uncompressed->Append(buf, writer.Produced());
    return result;
  } else {
    LOG_UNFORMATTED(DEBUG, logCtx, "Performing decompression block by block");
    SnappySinkAllocator allocator(uncompressed);
    SnappyScatteredWriter<SnappySinkAllocator> writer(allocator);
    return InternalUncompressAllTags_c(&decompressor, &writer, compressed_len,
                                     uncompressed_len);
  }
}

#ifdef AOCL_SNAPPY_AVX_OPT
/* Same as InternalUncompress_c, but is built with target attribute avx. */
template <typename Writer, typename T>
AOCL_SNAPPY_TARGET_AVX
static bool InternalUncompress_avx(Source* r, Writer* writer) {
  AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
  // Read the uncompressed length from the front of the compressed input
  SnappyDecompressor<T> decompressor(r);
  uint32_t uncompressed_len = 0;
  if (!decompressor.ReadUncompressedLength(&uncompressed_len)) return false;

  return InternalUncompressAllTags_avx(&decompressor, writer, r->Available(),
      uncompressed_len);
}

template <typename T>
template <class Writer>
#if defined(__GNUC__) && defined(__x86_64__)
  __attribute__((aligned(32)))
#endif
AOCL_SNAPPY_TARGET_AVX
  void
  SnappyDecompressor<T>::DecompressAllTags_avx(Writer* writer) {
    AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
    const char* ip = ip_;
    ResetLimit(ip);
    auto op = writer->GetOutputPtr();
    // We could have put this refill fragment only at the beginning of the loop.
    // However, duplicating it at the end of each branch gives the compiler more
    // scope to optimize the <ip_limit_ - ip> expression based on the local
    // context, which overall increases speed.
#define MAYBE_REFILL()                                      \
  if (SNAPPY_PREDICT_FALSE(ip >= ip_limit_min_maxtaglen_)) { \
    ip_ = ip;                                               \
    if (SNAPPY_PREDICT_FALSE(!RefillTag())) goto exit;       \
    ip = ip_;                                               \
    ResetLimit(ip);                                         \
  }                                                         \
  preload = static_cast<uint8_t>(*ip)

    // At the start of the for loop below the least significant byte of preload
    // contains the tag.
    uint32_t preload;
    MAYBE_REFILL();
    for ( ;; ) {
      {
        ptrdiff_t op_limit_min_slop;
        auto op_base = writer->GetBase(&op_limit_min_slop);
        if (op_base) {
          auto res =
              DecompressBranchless_c<decltype(op_base)>(reinterpret_cast<const uint8_t*>(ip),
                                   reinterpret_cast<const uint8_t*>(ip_limit_),
                                   op - op_base, op_base, op_limit_min_slop);
          ip = reinterpret_cast<const char*>(res.first);
          op = op_base + res.second;
          MAYBE_REFILL();
        }
      }
      const uint8_t c = static_cast<uint8_t>(preload);
      ip++;

      // Ratio of iterations that have LITERAL vs non-LITERAL for different
      // inputs.
      //
      // input          LITERAL  NON_LITERAL
      // -----------------------------------
      // html|html4|cp   23%        77%
      // urls            36%        64%
      // jpg             47%        53%
      // pdf             19%        81%
      // txt[1-4]        25%        75%
      // pb              24%        76%
      // bin             24%        76%
      if (SNAPPY_PREDICT_FALSE((c & 0x3) == LITERAL)) {
        size_t literal_length = (c >> 2) + 1u;
        if (writer->TryFastAppend(ip, ip_limit_ - ip, literal_length, &op)) {
          assert(literal_length < 61);
          ip += literal_length;
          // NOTE: There is no MAYBE_REFILL() here, as TryFastAppend()
          // will not return true unless there's already at least five spare
          // bytes in addition to the literal.
          preload = static_cast<uint8_t>(*ip);
          continue;
        }
        if (SNAPPY_PREDICT_FALSE(literal_length >= 61)) {
          // Long literal.
          const size_t literal_length_length = literal_length - 60;
          literal_length =
              ExtractLowBytes_c(LittleEndian::Load32(ip), literal_length_length) +
              1;
          ip += literal_length_length;
        }

        size_t avail = ip_limit_ - ip;
        while (avail < literal_length) {
          if (!writer->Append(ip, avail, &op)) goto exit;
          literal_length -= avail;
          reader_->Skip(peeked_);
          size_t n;
          ip = reader_->Peek(&n);
          avail = n;
          peeked_ = avail;
          if (avail == 0) goto exit;
          ip_limit_ = ip + avail;
          ResetLimit(ip);
        }
        if (!writer->Append(ip, literal_length, &op)) goto exit;
        ip += literal_length;
        MAYBE_REFILL();
      } else {
        if (SNAPPY_PREDICT_FALSE((c & 3) == COPY_4_BYTE_OFFSET)) {
          const size_t copy_offset = LittleEndian::Load32(ip);
          const size_t length = (c >> 2) + 1;
          ip += 4;

          if (!writer->AppendFromSelf(copy_offset, length, &op)) goto exit;
        } else {
          const ptrdiff_t entry = kLengthMinusOffset[c];
          preload = LittleEndian::Load32(ip);
          const uint32_t trailer = ExtractLowBytes_c(preload, c & 3);
          const uint32_t length = entry & 0xff;
          assert(length > 0);

          // copy_offset/256 is encoded in bits 8..10.  By just fetching
          // those bits, we get copy_offset (since the bit-field starts at
          // bit 8).
          const uint32_t copy_offset = trailer - entry + length;
          if (!writer->AppendFromSelf(copy_offset, length, &op)) goto exit;

          ip += (c & 3);
          // By using the result of the previous load we reduce the critical
          // dependency chain of ip to 4 cycles.
          preload >>= (c & 3) * 8;
          if (ip < ip_limit_min_maxtaglen_) continue;
        }
        MAYBE_REFILL();
      }
    }
#undef MAYBE_REFILL
  exit:
    writer->SetOutputPtr(op);
  }

template<>
template <class Writer>
#if defined(__GNUC__) && defined(__x86_64__)
__attribute__((aligned(32)))
#endif
AOCL_SNAPPY_TARGET_AVX
void SnappyDecompressor<with_avx>::DecompressAllTags(Writer* writer) {
    DecompressAllTags_avx(writer);
}

/* Same as InternalUncompressAllTags, but is built with target attribute avx. */
template <typename Writer, typename T>
AOCL_SNAPPY_TARGET_AVX
static bool InternalUncompressAllTags_avx(SnappyDecompressor<T>* decompressor,
                                      Writer* writer,
                                      uint32_t compressed_len,
                                      uint32_t uncompressed_len) {
  AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");

  int token = 0;
  Report(token, "snappy_uncompress", compressed_len, uncompressed_len);

  writer->SetExpectedLength(uncompressed_len);

  // Process the entire input
  decompressor->DecompressAllTags(writer);
  writer->Flush();
  return (decompressor->eof() && writer->CheckLength());
}

/* Same as AOCL_UncompressInternal but creates SnappyDecompressor<T>
* and calls InternalUncompressAllTags_avx */
template <typename T>
AOCL_SNAPPY_TARGET_AVX
bool Uncompress_avx(Source* compressed, Sink* uncompressed) {
  if (compressed == NULL || uncompressed == NULL) return false;
  // Read the uncompressed length from the front of the compressed input
  SnappyDecompressor<T> decompressor(compressed);
  uint32_t uncompressed_len = 0;
  if (!decompressor.ReadUncompressedLength(&uncompressed_len)) {
    return false;
  }

  char c;
  size_t allocated_size;
  char* buf = uncompressed->GetAppendBufferVariable(
      1, uncompressed_len, &c, 1, &allocated_size);

  const size_t compressed_len = compressed->Available();
  // If we can get a flat buffer, then use it, otherwise do block by block
  // uncompression
  if (allocated_size >= uncompressed_len) {
    SnappyArrayWriter writer(buf);
    bool result = InternalUncompressAllTags_avx(&decompressor, &writer,
                                            compressed_len, uncompressed_len);
    uncompressed->Append(buf, writer.Produced());
    return result;
  } else {
    SnappySinkAllocator allocator(uncompressed);
    SnappyScatteredWriter<SnappySinkAllocator> writer(allocator);
    return InternalUncompressAllTags_avx(&decompressor, &writer, compressed_len,
                                     uncompressed_len);
  }
}

AOCL_SNAPPY_TARGET_AVX
inline uint16_t* TableEntry_crc32(uint16_t* table, uint32_t bytes, uint32_t mask) {
  // Our choice is quicker-and-dirtier than the typical hash function;
  // empirically, that seems beneficial.  The upper bits of kMagic * bytes are a
  // higher-quality hash than the lower bits, so when using kMagic * bytes we
  // also shift right to get a higher-quality end result.  There's no similar
  // issue with a CRC because all of the output bits of a CRC are equally good
  // "hashes." So, a CPU instruction for CRC, if available, tends to be a good
  // choice.
#if SNAPPY_HAVE_NEON_CRC32
  // We use mask as the second arg to the CRC function, as it's about to
  // be used anyway; it'd be equally correct to use 0 or some constant.
  // Mathematically, _mm_crc32_u32 (or similar) is a function of the
  // xor of its arguments.
  const uint32_t hash = __crc32cw(bytes, mask);
#elif SNAPPY_HAVE_X86_CRC32
  const uint32_t hash = _mm_crc32_u32(bytes, mask);
#else
  constexpr uint32_t kMagic = 0x1e35a7bd;
  const uint32_t hash = (kMagic * bytes) >> (31 - kMaxHashTableBits);
#endif
  return reinterpret_cast<uint16_t*>(reinterpret_cast<uintptr_t>(table) +
                                     (hash & mask));
}

namespace internal {
//Derived from CompressFragment_c. Calls TableEntry_crc32
AOCL_SNAPPY_TARGET_AVX
char* CompressFragment_crc32(const char* input, size_t input_size, char* op,
                       uint16_t* table, const int table_size) {
  // "ip" is the input pointer, and "op" is the output pointer.
  const char* ip = input;
  assert(input_size <= kBlockSize);
  assert((table_size & (table_size - 1)) == 0);  // table must be power of two
  const uint32_t mask = 2 * (table_size - 1);
  const char* ip_end = input + input_size;
  const char* base_ip = ip;

  const size_t kInputMarginBytes = 15;
  if (SNAPPY_PREDICT_TRUE(input_size >= kInputMarginBytes)) {
    const char* ip_limit = input + input_size - kInputMarginBytes;

    for (uint32_t preload = LittleEndian::Load32(ip + 1);;) {
      // Bytes in [next_emit, ip) will be emitted as literal bytes.  Or
      // [next_emit, ip_end) after the main loop.
      const char* next_emit = ip++;
      uint64_t data = LittleEndian::Load64(ip);
      // The body of this loop calls EmitLiteral once and then EmitCopy one or
      // more times.  (The exception is that when we're close to exhausting
      // the input we goto emit_remainder.)
      //
      // In the first iteration of this loop we're just starting, so
      // there's nothing to copy, so calling EmitLiteral once is
      // necessary.  And we only start a new iteration when the
      // current iteration has determined that a call to EmitLiteral will
      // precede the next call to EmitCopy (if any).
      //
      // Step 1: Scan forward in the input looking for a 4-byte-long match.
      // If we get close to exhausting the input then goto emit_remainder.
      //
      // Heuristic match skipping: If 32 bytes are scanned with no matches
      // found, start looking only at every other byte. If 32 more bytes are
      // scanned (or skipped), look at every third byte, etc.. When a match is
      // found, immediately go back to looking at every byte. This is a small
      // loss (~5% performance, ~0.1% density) for compressible data due to more
      // bookkeeping, but for non-compressible data (such as JPEG) it's a huge
      // win since the compressor quickly "realizes" the data is incompressible
      // and doesn't bother looking for matches everywhere.
      //
      // The "skip" variable keeps track of how many bytes there are since the
      // last match; dividing it by 32 (ie. right-shifting by five) gives the
      // number of bytes to move ahead for each iteration.
      uint32_t skip = 32;

      const char* candidate;
      if (ip_limit - ip >= 16) {
        auto delta = ip - base_ip;
        for (int j = 0; j < 4; ++j) {
          for (int k = 0; k < 4; ++k) {
            int i = 4 * j + k;
            // These for-loops are meant to be unrolled. So we can freely
            // special case the first iteration to use the value already
            // loaded in preload.
            uint32_t dword = i == 0 ? preload : static_cast<uint32_t>(data);
            assert(dword == LittleEndian::Load32(ip + i));
            uint16_t* table_entry = TableEntry_crc32(table, dword, mask);
            candidate = base_ip + *table_entry;
            assert(candidate >= base_ip);
            assert(candidate < ip + i);
            *table_entry = delta + i;
            if (SNAPPY_PREDICT_FALSE(LittleEndian::Load32(candidate) == dword)) {
              *op = LITERAL | (i << 2);
              UnalignedCopy128(next_emit, op + 1);
              ip += i;
              op = op + i + 2;
              goto emit_match;
            }
            data >>= 8;
          }
          data = LittleEndian::Load64(ip + 4 * j + 4);
        }
        ip += 16;
        skip += 16;
      }
      while (true) {
        assert(static_cast<uint32_t>(data) == LittleEndian::Load32(ip));
        uint16_t* table_entry = TableEntry_crc32(table, data, mask);
        uint32_t bytes_between_hash_lookups = skip >> 5;
        skip += bytes_between_hash_lookups;
        const char* next_ip = ip + bytes_between_hash_lookups;
        if (SNAPPY_PREDICT_FALSE(next_ip > ip_limit)) {
          ip = next_emit;
          goto emit_remainder;
        }
        candidate = base_ip + *table_entry;
        assert(candidate >= base_ip);
        assert(candidate < ip);

        *table_entry = ip - base_ip;
        if (SNAPPY_PREDICT_FALSE(static_cast<uint32_t>(data) ==
                                LittleEndian::Load32(candidate))) {
          break;
        }
        data = LittleEndian::Load32(next_ip);
        ip = next_ip;
      }

      // Step 2: A 4-byte match has been found.  We'll later see if more
      // than 4 bytes match.  But, prior to the match, input
      // bytes [next_emit, ip) are unmatched.  Emit them as "literal bytes."
      assert(next_emit + 16 <= ip_end);
      op = EmitLiteral</*allow_fast_path=*/true>(op, next_emit, ip - next_emit);

      // Step 3: Call EmitCopy, and then see if another EmitCopy could
      // be our next move.  Repeat until we find no match for the
      // input immediately after what was consumed by the last EmitCopy call.
      //
      // If we exit this loop normally then we need to call EmitLiteral next,
      // though we don't yet know how big the literal will be.  We handle that
      // by proceeding to the next iteration of the main loop.  We also can exit
      // this loop via goto if we get close to exhausting the input.
    emit_match:
      do {
        // We have a 4-byte match at ip, and no need to emit any
        // "literal bytes" prior to ip.
        const char* base = ip;
        std::pair<size_t, bool> p =
            FindMatchLength(candidate + 4, ip + 4, ip_end, &data);
        size_t matched = 4 + p.first;
        ip += matched;
        size_t offset = base - candidate;
        assert(0 == memcmp(base, candidate, matched));
        if (p.second) {
          op = EmitCopy</*len_less_than_12=*/true>(op, offset, matched);
        } else {
          op = EmitCopy</*len_less_than_12=*/false>(op, offset, matched);
        }
        if (SNAPPY_PREDICT_FALSE(ip >= ip_limit)) {
          goto emit_remainder;
        }
        // Expect 5 bytes to match
        assert((data & 0xFFFFFFFFFF) ==
               (LittleEndian::Load64(ip) & 0xFFFFFFFFFF));
        // We are now looking for a 4-byte match again.  We read
        // table[Hash(ip, mask)] for that.  To improve compression,
        // we also update table[Hash(ip - 1, mask)] and table[Hash(ip, mask)].
        *TableEntry_crc32(table, LittleEndian::Load32(ip - 1), mask) =
            ip - base_ip - 1;
        uint16_t* table_entry = TableEntry_crc32(table, data, mask);
        candidate = base_ip + *table_entry;
        *table_entry = ip - base_ip;
        // Measurements on the benchmarks have shown the following probabilities
        // for the loop to exit (ie. avg. number of iterations is reciprocal).
        // BM_Flat/6  txt1    p = 0.3-0.4
        // BM_Flat/7  txt2    p = 0.35
        // BM_Flat/8  txt3    p = 0.3-0.4
        // BM_Flat/9  txt3    p = 0.34-0.4
        // BM_Flat/10 pb      p = 0.4
        // BM_Flat/11 gaviota p = 0.1
        // BM_Flat/12 cp      p = 0.5
        // BM_Flat/13 c       p = 0.3
      } while (static_cast<uint32_t>(data) == LittleEndian::Load32(candidate));
      // Because the least significant 5 bytes matched, we can utilize data
      // for the next iteration.
      preload = data >> 8;
    }
  }

emit_remainder:
  // Emit the remaining bytes as a literal
  if (ip < ip_end) {
    op = EmitLiteral</*allow_fast_path=*/false>(op, ip, ip_end - ip);
  }

  return op;
}
}
#endif /* AOCL_SNAPPY_AVX_OPT */

#ifdef AOCL_SNAPPY_AVX2_OPT
//#if SNAPPY_HAVE_BMI2
AOCL_SNAPPY_TARGET_AVX2
static inline uint32_t ExtractLowBytes_bmi(const uint32_t& v, int n) {
  AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
  assert(n >= 0);
  assert(n <= 4);
  return _bzhi_u32(v, 8 * n);
}
//#endif

AOCL_SNAPPY_TARGET_AVX2
inline void MemCopy64_avx2(char* dst, const void* src, size_t size) {
  // Always copy this many bytes.  If that's below size then copy the full 64.
  constexpr int kShortMemCopy = 32;

  assert(size <= 64);
  assert(std::less_equal<const void*>()(static_cast<const char*>(src) + size,
                                        dst) ||
         std::less_equal<const void*>()(dst + size, src));

  // We know that src and dst are at least size bytes apart. However, because we
  // might copy more than size bytes the copy still might overlap past size.
  // E.g. if src and dst appear consecutively in memory (src + size >= dst).
  // TODO: Investigate wider copies on other platforms.
  assert(kShortMemCopy <= 32);
  __m256i data = _mm256_lddqu_si256(static_cast<const __m256i *>(src));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), data);
  // Profiling shows that nearly all copies are short.
  if (SNAPPY_PREDICT_FALSE(size > kShortMemCopy)) {
    data = _mm256_lddqu_si256(static_cast<const __m256i *>(src) + 1);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst) + 1, data);
  }
}

AOCL_SNAPPY_TARGET_AVX2
inline void MemCopy64_avx2(ptrdiff_t dst, const void* src, size_t size) {
  // TODO: Switch to [[maybe_unused]] when we can assume C++17.
  (void)dst;
  (void)src;
  (void)size;
}

/* Derived from DecompressBranchless_c but calls MemCopy64_avx2 */
template <typename T>
AOCL_SNAPPY_TARGET_AVX2
std::pair<const uint8_t*, ptrdiff_t> DecompressBranchless_avx2(
    const uint8_t* ip, const uint8_t* ip_limit, ptrdiff_t op, T op_base,
    ptrdiff_t op_limit_min_slop) {
  // If deferred_src is invalid point it here.
  uint8_t safe_source[64];
  const void* deferred_src;
  size_t deferred_length;
  ClearDeferred(&deferred_src, &deferred_length, safe_source);

  // We unroll the inner loop twice so we need twice the spare room.
  op_limit_min_slop -= kSlopBytes;
  if (2 * (kSlopBytes + 1) < ip_limit - ip && op < op_limit_min_slop) {
    const uint8_t* const ip_limit_min_slop = ip_limit - 2 * kSlopBytes - 1;
    ip++;
    // ip points just past the tag and we are touching at maximum kSlopBytes
    // in an iteration.
    size_t tag = ip[-1];
#if defined(__clang__) && defined(__aarch64__)
    // Workaround for https://bugs.llvm.org/show_bug.cgi?id=51317
    // when loading 1 byte, clang for aarch64 doesn't realize that it(ldrb)
    // comes with free zero-extension, so clang generates another
    // 'and xn, xm, 0xff' before it use that as the offset. This 'and' is
    // redundant and can be removed by adding this dummy asm, which gives
    // clang a hint that we're doing the zero-extension at the load.
    asm("" ::"r"(tag));
#endif
    do {
      // The throughput is limited by instructions, unrolling the inner loop
      // twice reduces the amount of instructions checking limits and also
      // leads to reduced mov's.

      SNAPPY_PREFETCH(ip + 128);
      for (int i = 0; i < 2; i++) {
        const uint8_t* old_ip = ip;
        assert(tag == ip[-1]);
        // For literals tag_type = 0, hence we will always obtain 0 from
        // ExtractLowBytes. For literals offset will thus be kLiteralOffset.
        ptrdiff_t len_minus_offset = kLengthMinusOffset[tag];
        uint32_t next;
#if defined(__aarch64__)
        size_t tag_type = AdvanceToNextTagARMOptimized(&ip, &tag);
        // We never need more than 16 bits. Doing a Load16 allows the compiler
        // to elide the masking operation in ExtractOffset.
        next = LittleEndian::Load16(old_ip);
#else
        size_t tag_type = AdvanceToNextTagX86Optimized(&ip, &tag);
        next = LittleEndian::Load32(old_ip);
#endif
        size_t len = len_minus_offset & 0xFF;
        ptrdiff_t extracted = ExtractOffset(next, tag_type);
        ptrdiff_t len_min_offset = len_minus_offset - extracted;
        if (SNAPPY_PREDICT_FALSE(len_minus_offset > extracted)) {
          if (SNAPPY_PREDICT_FALSE(len & 0x80)) {
            // Exceptional case (long literal or copy 4).
            // Actually doing the copy here is negatively impacting the main
            // loop due to compiler incorrectly allocating a register for
            // this fallback. Hence we just break.
          break_loop:
            ip = old_ip;
            goto exit;
          }
          // Only copy-1 or copy-2 tags can get here.
          assert(tag_type == 1 || tag_type == 2);
          std::ptrdiff_t delta = (op + deferred_length) + len_min_offset - len;
          // Guard against copies before the buffer start.
          // Execute any deferred MemCopy since we write to dst here.
          MemCopy64_avx2(op_base + op, deferred_src, deferred_length);
          op += deferred_length;
          ClearDeferred(&deferred_src, &deferred_length, safe_source);
          if (SNAPPY_PREDICT_FALSE(delta < 0 ||
                                  !Copy64BytesWithPatternExtension(
                                      op_base + op, len - len_min_offset))) {
            goto break_loop;
          }
          // We aren't deferring this copy so add length right away.
          op += len;
          continue;
        }
        std::ptrdiff_t delta = (op + deferred_length) + len_min_offset - len;
        if (SNAPPY_PREDICT_FALSE(delta < 0)) {
          // Due to the spurious offset in literals have this will trigger
          // at the start of a block when op is still smaller than 256.
          if (tag_type != 0) goto break_loop;
          MemCopy64_avx2(op_base + op, deferred_src, deferred_length);
          op += deferred_length;
          DeferMemCopy(&deferred_src, &deferred_length, old_ip, len);
          continue;
        }

        // For copies we need to copy from op_base + delta, for literals
        // we need to copy from ip instead of from the stream.
        const void* from =
            tag_type ? reinterpret_cast<void*>(op_base + delta) : old_ip;
        MemCopy64_avx2(op_base + op, deferred_src, deferred_length);
        op += deferred_length;
        DeferMemCopy(&deferred_src, &deferred_length, from, len);
      }
    } while (ip < ip_limit_min_slop &&
             static_cast<ptrdiff_t>(op + deferred_length) < op_limit_min_slop);
  exit:
    ip--;
    assert(ip <= ip_limit);
  }
  // If we deferred a copy then we can perform.  If we are up to date then we
  // might not have enough slop bytes and could run past the end.
  if (deferred_length) {
      MemCopy64_avx2(op_base + op, deferred_src, deferred_length);
    op += deferred_length;
    ClearDeferred(&deferred_src, &deferred_length, safe_source);
  }
  return {ip, op};
}

template<typename T>
template <class Writer>
#if defined(__GNUC__) && defined(__x86_64__)
  __attribute__((aligned(32)))
#endif
AOCL_SNAPPY_TARGET_AVX2
  void
  SnappyDecompressor<T>::DecompressAllTags_bmi(Writer* writer) {
    AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
    const char* ip = ip_;
    ResetLimit(ip);
    auto op = writer->GetOutputPtr();
    // We could have put this refill fragment only at the beginning of the loop.
    // However, duplicating it at the end of each branch gives the compiler more
    // scope to optimize the <ip_limit_ - ip> expression based on the local
    // context, which overall increases speed.
#define MAYBE_REFILL()                                      \
  if (SNAPPY_PREDICT_FALSE(ip >= ip_limit_min_maxtaglen_)) { \
    ip_ = ip;                                               \
    if (SNAPPY_PREDICT_FALSE(!RefillTag())) goto exit;       \
    ip = ip_;                                               \
    ResetLimit(ip);                                         \
  }                                                         \
  preload = static_cast<uint8_t>(*ip)

    // At the start of the for loop below the least significant byte of preload
    // contains the tag.
    uint32_t preload;
    MAYBE_REFILL();
    for ( ;; ) {
      {
        ptrdiff_t op_limit_min_slop;
        auto op_base = writer->GetBase(&op_limit_min_slop);
        if (op_base) {
          auto res =
              DecompressBranchless_avx2<decltype(op_base)>(reinterpret_cast<const uint8_t*>(ip),
                                   reinterpret_cast<const uint8_t*>(ip_limit_),
                                   op - op_base, op_base, op_limit_min_slop);
          ip = reinterpret_cast<const char*>(res.first);
          op = op_base + res.second;
          MAYBE_REFILL();
        }
      }
      const uint8_t c = static_cast<uint8_t>(preload);
      ip++;

      // Ratio of iterations that have LITERAL vs non-LITERAL for different
      // inputs.
      //
      // input          LITERAL  NON_LITERAL
      // -----------------------------------
      // html|html4|cp   23%        77%
      // urls            36%        64%
      // jpg             47%        53%
      // pdf             19%        81%
      // txt[1-4]        25%        75%
      // pb              24%        76%
      // bin             24%        76%
      if (SNAPPY_PREDICT_FALSE((c & 0x3) == LITERAL)) {
        size_t literal_length = (c >> 2) + 1u;
        if (writer->TryFastAppend(ip, ip_limit_ - ip, literal_length, &op)) {
          assert(literal_length < 61);
          ip += literal_length;
          // NOTE: There is no MAYBE_REFILL() here, as TryFastAppend()
          // will not return true unless there's already at least five spare
          // bytes in addition to the literal.
          preload = static_cast<uint8_t>(*ip);
          continue;
        }
        if (SNAPPY_PREDICT_FALSE(literal_length >= 61)) {
          // Long literal.
          const size_t literal_length_length = literal_length - 60;
          literal_length =
              ExtractLowBytes_bmi(LittleEndian::Load32(ip), literal_length_length) +
              1;
          ip += literal_length_length;
        }

        size_t avail = ip_limit_ - ip;
        while (avail < literal_length) {
          if (!writer->Append(ip, avail, &op)) goto exit;
          literal_length -= avail;
          reader_->Skip(peeked_);
          size_t n;
          ip = reader_->Peek(&n);
          avail = n;
          peeked_ = avail;
          if (avail == 0) goto exit;
          ip_limit_ = ip + avail;
          ResetLimit(ip);
        }
        if (!writer->Append(ip, literal_length, &op)) goto exit;
        ip += literal_length;
        MAYBE_REFILL();
      } else {
        if (SNAPPY_PREDICT_FALSE((c & 3) == COPY_4_BYTE_OFFSET)) {
          const size_t copy_offset = LittleEndian::Load32(ip);
          const size_t length = (c >> 2) + 1;
          ip += 4;

          if (!writer->AppendFromSelf(copy_offset, length, &op)) goto exit;
        } else {
          const ptrdiff_t entry = kLengthMinusOffset[c];
          preload = LittleEndian::Load32(ip);
          const uint32_t trailer = ExtractLowBytes_bmi(preload, c & 3);
          const uint32_t length = entry & 0xff;
          assert(length > 0);

          // copy_offset/256 is encoded in bits 8..10.  By just fetching
          // those bits, we get copy_offset (since the bit-field starts at
          // bit 8).
          const uint32_t copy_offset = trailer - entry + length;
          if (!writer->AppendFromSelf(copy_offset, length, &op)) goto exit;

          ip += (c & 3);
          // By using the result of the previous load we reduce the critical
          // dependency chain of ip to 4 cycles.
          preload >>= (c & 3) * 8;
          if (ip < ip_limit_min_maxtaglen_) continue;
        }
        MAYBE_REFILL();
      }
    }
#undef MAYBE_REFILL
  exit:
    writer->SetOutputPtr(op);
  }

template<>
template <class Writer>
#if defined(__GNUC__) && defined(__x86_64__)
__attribute__((aligned(32)))
#endif
AOCL_SNAPPY_TARGET_AVX2
void SnappyDecompressor<with_bmi_avx>::DecompressAllTags(Writer* writer) {
    DecompressAllTags_bmi(writer);
}
#endif /* AOCL_SNAPPY_AVX2_OPT */
/* ---------- Modifications to support dynamic ISA selection - end ----------*/

/* ---------- Modifications to support AOCL optimizations - start -----------*/
size_t MaxCompressedLength_st(size_t source_bytes) {
    // Compressed data can be defined as:
    //    compressed := item* literal*
    //    item       := literal* copy
    //
    // The trailing literal sequence has a space blowup of at most 62/60
    // since a literal of length 60 needs one tag byte + one extra byte
    // for length information.
    //
    // Item blowup is trickier to measure.  Suppose the "copy" op copies
    // 4 bytes of data.  Because of a special check in the encoding code,
    // we produce a 4-byte copy only if the offset is < 65536.  Therefore
    // the copy op takes 3 bytes to encode, and this type of item leads
    // to at most the 62/60 blowup for representing literals.
    //
    // Suppose the "copy" op copies 5 bytes of data.  If the offset is big
    // enough, it will take 5 bytes to encode the copy op.  Therefore the
    // worst case here is a one-byte literal followed by a five-byte copy.
    // I.e., 6 bytes of input turn into 7 bytes of "compressed" data.
    //
    // This last factor dominates the blowup, so the final estimate is:
    return 32 + source_bytes + source_bytes / 6;
}

#ifdef AOCL_ENABLE_THREADS
size_t MaxCompressedLength_mt(size_t source_bytes) {

    size_t sz1 = MaxCompressedLength_st(source_bytes);
    size_t sz2 = 0;
    COMPRESS_BOUND_MT(source_bytes, MaxCompressedLength_st, (AOCL_INT32)kBlockSize, WINDOW_FACTOR, sz1, sz2, 0)
        return sz2;
}
#endif

#ifdef AOCL_SNAPPY_OPT
#ifndef AOCL_SNAPPY_HIGH_COMPRESSION
namespace {
uint32_t AOCL_CalculateTableSize(uint32_t input_size) {
  static_assert(
      AOCL_kMaxHashTableSize >= kMinHashTableSize,
      "AOCL_kMaxHashTableSize should be greater or equal to kMinHashTableSize.");
  if (input_size > AOCL_kMaxHashTableSize) {
    return AOCL_kMaxHashTableSize;
  }
  if (input_size < kMinHashTableSize) {
    return kMinHashTableSize;
  }
  // This is equivalent to Log2Ceiling(input_size), assuming input_size > 1.
  // 2 << Log2Floor(x - 1) is equivalent to 1 << (1 + Log2Floor(x - 1)).
  return 2u << Bits::Log2Floor(input_size - 1);
}
}  // namespace
#endif /* AOCL_SNAPPY_HIGH_COMPRESSION */

/**
 * AOCL_EmitCopyAtMost64: Same as EmitCopyAtMost64 but explicitly makes use of
 * conditional move instructions via inline assembly for GCC compiler. This is 
 * necessary because the compiler does not generate such instructions by default.
 */
template <bool len_less_than_12>
static inline char* AOCL_EmitCopyAtMost64(char* op, size_t offset, size_t len) {
  assert(len <= 64);
  assert(len >= 4);
  assert(offset < 65536);
  assert(len_less_than_12 == (len < 12));

  if (len_less_than_12) {
    uint32_t u = (len << 2) + (offset << 8);
    uint32_t copy1 = COPY_1_BYTE_OFFSET - (4 << 2) + ((offset >> 3) & 0xe0);
    uint32_t copy2 = COPY_2_BYTE_OFFSET - (1 << 2);
    // It turns out that offset < 2048 is a difficult to predict branch.
    // `perf record` shows this is the highest percentage of branch misses in
    // benchmarks. This code produces branch free code, the data dependency
    // chain that bottlenecks the throughput is so long that a few extra
    // instructions are completely free (IPC << 6 because of data deps).
#if defined(__GNUC__) && defined(__x86_64__)
    uint32_t temp;
    __asm__ __volatile__(
      "cmp $2048, %1\n\t"                    // compare offset with 2048
      "cmovb %2, %0\n\t"                     // if offset < 2048, copy copy1 to temp
      "cmovae %3, %0\n\t"                    // if offset >= 2048, copy copy2 to temp
      : "=&r"(temp)                          // output: temp
      : "r"(offset), "r"(copy1), "r"(copy2)  // input: u, offset, copy1, copy2
      : "cc"
    );
    
    u += temp;
#else
    u += offset < 2048 ? copy1 : copy2;
#endif
    LittleEndian::Store32(op, u);
    op += offset < 2048 ? 2 : 3;
  } else {
    // Write 4 bytes, though we only care about 3 of them.  The output buffer
    // is required to have some slack, so the extra byte won't overrun it.
    uint32_t u = COPY_2_BYTE_OFFSET + ((len - 1) << 2) + (offset << 8);
    LittleEndian::Store32(op, u);
    op += 3;
  }
  return op;
}

/**
 * AOCL_EmitCopy: Same as EmitCopy but calls AOCL_EmitCopyAtMost64.
 */
template <bool len_less_than_12>
static inline char* AOCL_EmitCopy(char* op, size_t offset, size_t len) {
  assert(len_less_than_12 == (len < 12));
  if (len_less_than_12) {
    return AOCL_EmitCopyAtMost64</*len_less_than_12=*/true>(op, offset, len);
  } else {
    // A special case for len <= 64 might help, but so far measurements suggest
    // it's in the noise.

    // Emit 64 byte copies but make sure to keep at least four bytes reserved.
    while (SNAPPY_PREDICT_FALSE(len >= 68)) {
      op = AOCL_EmitCopyAtMost64</*len_less_than_12=*/false>(op, offset, 64);
      len -= 64;
    }

    // One or two copies will now finish the job.
    if (len > 64) {
      op = AOCL_EmitCopyAtMost64</*len_less_than_12=*/false>(op, offset, 60);
      len -= 60;
    }

    // Emit remainder.
    if (len < 12) {
      op = AOCL_EmitCopyAtMost64</*len_less_than_12=*/true>(op, offset, len);
    } else {
      op = AOCL_EmitCopyAtMost64</*len_less_than_12=*/false>(op, offset, len);
    }
    return op;
  }
}

namespace internal {
/**
 * AOCL_CompressFragment_c calls TableEntry_c.
 * Early loads data at candidate in do-while loop under emit_match.
*/
char* AOCL_CompressFragment_c(const char* input, size_t input_size, char* op,
                       uint16_t* table, const int table_size) {
  AOCL_LOG_INIT_STATS();
  // "ip" is the input pointer, and "op" is the output pointer.
  const char* ip = input;
  assert(input_size <= kBlockSize);
  assert((table_size & (table_size - 1)) == 0);  // table must be power of two
  const uint32_t mask = 2 * (table_size - 1);
  const char* ip_end = input + input_size;
  const char* base_ip = ip;

  const size_t kInputMarginBytes = 15;
  if (input_size >= kInputMarginBytes) {
    const char* ip_limit = input + input_size - kInputMarginBytes;
    
    LOG_FORMATTED(TRACE, logCtx, "Input size = %zu, Input size until limit = %zu",
     input_size, (size_t)(ip_limit - ip));
#ifdef AOCL_SNAPPY_MATCH_SKIP_OPT
    uint32_t bbhl_prev = 0; //baseline bytes_between_hash_lookups to use
#endif
    for (uint32_t preload = LittleEndian::Load32(ip + 1);;) {
      // Bytes in [next_emit, ip) will be emitted as literal bytes.  Or
      // [next_emit, ip_end) after the main loop.
      const char* next_emit = ip++;
      uint64_t data = LittleEndian::Load64(ip);
      // The body of this loop calls EmitLiteral once and then EmitCopy one or
      // more times.  (The exception is that when we're close to exhausting
      // the input we goto emit_remainder.)
      //
      // In the first iteration of this loop we're just starting, so
      // there's nothing to copy, so calling EmitLiteral once is
      // necessary.  And we only start a new iteration when the
      // current iteration has determined that a call to EmitLiteral will
      // precede the next call to EmitCopy (if any).
      //
      // Step 1: Scan forward in the input looking for a 4-byte-long match.
      // If we get close to exhausting the input then goto emit_remainder.
      //
      // Heuristic match skipping: If 32 bytes are scanned with no matches
      // found, start looking only at every other byte. If 32 more bytes are
      // scanned (or skipped), look at every third byte, etc.. When a match is
      // found, immediately go back to looking at every byte. This is a small
      // loss (~5% performance, ~0.1% density) for compressible data due to more
      // bookkeeping, but for non-compressible data (such as JPEG) it's a huge
      // win since the compressor quickly "realizes" the data is incompressible
      // and doesn't bother looking for matches everywhere.
      //
      // The "skip" variable keeps track of how many bytes there are since the
      // last match; dividing it by 32 (ie. right-shifting by five) gives the
      // number of bytes to move ahead for each iteration.
      //
      // If AOCL match skipping optimization is enabled, the number of bytes
      // skipped while checking for a match is double that as compared to regular
      // skipping. This way, every 32 unmatched bytes, instead of skipping by
      // (skip / 32) we skip by (skip / 32) * 2

      uint32_t skip = 32;

      const char* candidate;
      if (ip_limit - ip >= 16) {
        auto delta = ip - base_ip;
        for (int j = 0; j < 4; ++j) {
          for (int k = 0; k < 4; ++k) {
            int i = 4 * j + k;
            // These for-loops are meant to be unrolled. So we can freely
            // special case the first iteration to use the value already
            // loaded in preload.
            uint32_t dword = i == 0 ? preload : static_cast<uint32_t>(data);
            assert(dword == LittleEndian::Load32(ip + i));
            uint16_t* table_entry = TableEntry_c(table, dword, mask);
            candidate = base_ip + *table_entry;
            assert(candidate >= base_ip);
            assert(candidate < ip + i);
            *table_entry = delta + i;
            if (LittleEndian::Load32(candidate) == dword) {
              *op = LITERAL | (i << 2);
              UnalignedCopy128(next_emit, op + 1);
              ip += i;
              op = op + i + 2;
              goto emit_match;
            }
            data >>= 8;
          }
          data = LittleEndian::Load64(ip + 4 * j + 4);
        }
        ip += 16;
        skip += 16;
      }
      while (true) {
        assert(static_cast<uint32_t>(data) == LittleEndian::Load32(ip));
        uint16_t* table_entry = TableEntry_c(table, data, mask);

#ifdef AOCL_SNAPPY_MATCH_SKIP_OPT
        uint32_t bytes_between_hash_lookups = bbhl_prev + ((skip >> 5) << 1);
        skip += (skip >> 5);
#else
        uint32_t bytes_between_hash_lookups = skip >> 5;
        skip += bytes_between_hash_lookups;
#endif
        LOG_FORMATTED(TRACE, logCtx, "skip = %u", skip);

        AOCL_LOG_UPDATE_SKIP(bytes_between_hash_lookups);
        const char* next_ip = ip + bytes_between_hash_lookups;
        if (next_ip > ip_limit) {
          ip = next_emit;
          goto emit_remainder;
        }
        candidate = base_ip + *table_entry;
        assert(candidate >= base_ip);
        assert(candidate < ip);

        *table_entry = ip - base_ip;
        if (static_cast<uint32_t>(data) ==
                                LittleEndian::Load32(candidate)) {
#ifdef AOCL_SNAPPY_MATCH_SKIP_OPT
            //set offset to 0 or 1/2 of current value depending on how large 
            //bytes_between_hash_lookups(bbhl) is.
            //For files that compress well, bbhl tends to be small,
            //and no additonal offset is required (bbhl_prev=0).
            //For files that do not compress well, finding matches once bbhl
            //is fairly large is unreliable. Hence, instead of starting to match with bbhl=1
            //in the next iteration, we start with an offset of bbhl/2.
            //This gives significant speed up for files that do not compress well, at the
            //expense of slight ratio degradation.
            bbhl_prev = bytes_between_hash_lookups > AOCL_SNAPPY_MATCH_SKIPPING_THRESHOLD ? (bytes_between_hash_lookups >> 1) : 0;
#endif
          break;
        }
        data = LittleEndian::Load32(next_ip);
        ip = next_ip;
      }

      // Step 2: A 4-byte match has been found.  We'll later see if more
      // than 4 bytes match.  But, prior to the match, input
      // bytes [next_emit, ip) are unmatched.  Emit them as "literal bytes."
      assert(next_emit + 16 <= ip_end);
      op = EmitLiteral</*allow_fast_path=*/true>(op, next_emit, ip - next_emit);

      // Step 3: Call EmitCopy, and then see if another EmitCopy could
      // be our next move.  Repeat until we find no match for the
      // input immediately after what was consumed by the last EmitCopy call.
      //
      // If we exit this loop normally then we need to call EmitLiteral next,
      // though we don't yet know how big the literal will be.  We handle that
      // by proceeding to the next iteration of the main loop.  We also can exit
      // this loop via goto if we get close to exhausting the input.
    emit_match:
      uint32_t candidate_data = 0;
      do {
        // We have a 4-byte match at ip, and no need to emit any
        // "literal bytes" prior to ip.
        const char* base = ip;
        std::pair<size_t, bool> p =
            FindMatchLength(candidate + 4, ip + 4, ip_end, &data);
        size_t matched = 4 + p.first;
        AOCL_LOG_UPDATE_MATCH(matched);
        ip += matched;
        size_t offset = base - candidate;
        assert(0 == memcmp(base, candidate, matched));
        if (p.second) {
          op = EmitCopy</*len_less_than_12=*/true>(op, offset, matched);
        } else {
          op = EmitCopy</*len_less_than_12=*/false>(op, offset, matched);
        }
        if (ip >= ip_limit) {
          goto emit_remainder;
        }
        // Expect 5 bytes to match
        assert((data & 0xFFFFFFFFFF) ==
               (LittleEndian::Load64(ip) & 0xFFFFFFFFFF));
        // We are now looking for a 4-byte match again.  We read
        // table[Hash(ip, mask)] for that.  To improve compression,
        // we also update table[Hash(ip - 1, mask)] and table[Hash(ip, mask)].
        *TableEntry_c(table, LittleEndian::Load32(ip - 1), mask) =
            ip - base_ip - 1;
        uint16_t* table_entry = TableEntry_c(table, data, mask);
        candidate = base_ip + *table_entry;
        candidate_data = LittleEndian::Load32(candidate);
        *table_entry = ip - base_ip;
        // Measurements on the benchmarks have shown the following probabilities
        // for the loop to exit (ie. avg. number of iterations is reciprocal).
        // BM_Flat/6  txt1    p = 0.3-0.4
        // BM_Flat/7  txt2    p = 0.35
        // BM_Flat/8  txt3    p = 0.3-0.4
        // BM_Flat/9  txt3    p = 0.34-0.4
        // BM_Flat/10 pb      p = 0.4
        // BM_Flat/11 gaviota p = 0.1
        // BM_Flat/12 cp      p = 0.5
        // BM_Flat/13 c       p = 0.3
      } while (static_cast<uint32_t>(data) == candidate_data);
      // Because the least significant 5 bytes matched, we can utilize data
      // for the next iteration.
      preload = data >> 8;
    }
  }

 emit_remainder:
  // Emit the remaining bytes as a literal
  LOG_FORMATTED(TRACE, logCtx, "Emit remaining %d bytes", (int)(ip_end - ip));
  if (ip < ip_end) {
    op = EmitLiteral</*allow_fast_path=*/false>(op, ip, ip_end - ip);
  }

  AOCL_LOG_CLEAR_STATS(input_size);
  return op;
}
}
#endif /* AOCL_SNAPPY_OPT */

#ifdef AOCL_SNAPPY_AVX_OPT
namespace {
AOCL_SNAPPY_TARGET_AVX
inline void AOCL_FastMemcopy64Bytes(char* dst, char* src) {
    AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
    // assume: kSlopBytes is 64
    // assume: there is always space to copy 64 bytes
    // assume: copy is always from a lower address to a higher address (op - offset) to (op)
    // assume: there is a likely overlap between the src and dst buffers
    //
    // The way to do this using the standard library is to use memmove.
    // In this case however, the aim is to copy exactly 64 bytes from the
    // buffer pointed to by src to the buffer pointed to by dst. To do this
    // we can utilize vector intrinsics to simply load 64 bytes from src,
    // and then copy them over to dst.
    //
    // Data is copied first and then written, which means that before we
    // write anything to the dst buffer, we copy all the data that we need
    // into registers. This way, even in cases of overlapping src and dst
    // buffers, the data is safely copied first and then safely written.
    __m256i* dst1 = (__m256i*)dst;
    __m256i* src1 = (__m256i*)src;

    // take snapshot of 64 bytes from src
    __m256i s1 = _mm256_lddqu_si256(src1);
    __m256i s2 = _mm256_lddqu_si256(src1 + 1);

    // paste the 64 byte snapshot at destination
    _mm256_storeu_si256(dst1, s1);
    _mm256_storeu_si256(dst1 + 1, s2);
}
}

class AOCL_SnappyArrayWriter_AVX {
 private:
  char* base_;
  char* op_;
  char* op_limit_;
  // If op < op_limit_min_slop_ then it's safe to unconditionally write
  // kSlopBytes starting at op.
  char* op_limit_min_slop_;

 public:
  inline explicit AOCL_SnappyArrayWriter_AVX(char* dst)
      : base_(dst),
        op_(dst),
        op_limit_(dst),
        op_limit_min_slop_(dst) {
  AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
  }  // Safe default see invariant.

  inline void SetExpectedLength(size_t len) {
    op_limit_ = op_ + len;
    // Prevent pointer from being past the buffer.
    op_limit_min_slop_ = op_limit_ - std::min<size_t>(kSlopBytes - 1, len);
  }

  inline bool CheckLength() const {
    return op_ == op_limit_;
  }

  char* GetOutputPtr() { return op_; }
  char* GetBase(ptrdiff_t* op_limit_min_slop) {
      *op_limit_min_slop = op_limit_min_slop_ - base_;
      return base_;
  }
  void SetOutputPtr(char* op) { op_ = op; }

  inline bool Append(const char* ip, size_t len, char** op_p) {
    char* op = *op_p;
    const size_t space_left = op_limit_ - op;
    if (space_left < len) return false;
    std::memcpy(op, ip, len);
    *op_p = op + len;
    return true;
  }

  inline bool TryFastAppend(const char* ip, size_t available, size_t len,
                            char** op_p) {
    char* op = *op_p;
    const size_t space_left = op_limit_ - op;
    if (len <= 16 && available >= 16 + kMaximumTagLength && space_left >= 16) {
      // Fast path, used for the majority (about 95%) of invocations.
      UnalignedCopy128(ip, op);
      *op_p = op + len;
      return true;
    } else {
      return false;
    }
  }

  AOCL_SNAPPY_TARGET_AVX
  SNAPPY_ATTRIBUTE_ALWAYS_INLINE
  inline bool AppendFromSelf(size_t offset, size_t len, char** op_p) {
    assert(len > 0);
    char* const op = *op_p;
    assert(op >= base_);
    char* const op_end = op + len;

#ifdef __clang__
    __builtin_prefetch(op - offset);
#endif

    // Check if we try to append from before the start of the buffer.
    if (SNAPPY_PREDICT_FALSE(static_cast<size_t>(op - base_) < offset))
      return false;

    if (SNAPPY_PREDICT_FALSE((kSlopBytes < 64 && len > kSlopBytes) || op >= op_limit_min_slop_ || offset < len)) {
      if (op_end > op_limit_ || offset == 0) return false;
      *op_p = IncrementalCopy(op - offset, op, op_end, op_limit_);
      return true;
    }
    AOCL_FastMemcopy64Bytes(op, op - offset);
    *op_p = op_end;
    return true;
  }

  inline size_t Produced() const {
    assert(op_ >= base_);
    return op_ - base_;
  }
  inline void Flush() {}
};

AOCL_SNAPPY_TARGET_AVX
bool AOCL_SAW_RawUncompress_avx(const char* compressed, size_t compressed_length, char* uncompressed) {
  AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
  ByteArraySource reader(compressed, compressed_length);
  AOCL_SnappyArrayWriter_AVX output(uncompressed);
  return InternalUncompressAOCLArray_fp(&reader, &output);
}

namespace internal {
AOCL_SNAPPY_TARGET_AVX
/**
 * Derived from AOCL_CompressFragment_c. Calls TableEntry_crc32.
 * Post encoding a match, updates table[Hash(ip, mask)] and then table[Hash(ip-1, mask)].
 * Calls AOCL_EmitCopy for emitting copies.
*/
char* AOCL_CompressFragment_crc32(const char* input, size_t input_size, char* op,
                       uint16_t* table, const int table_size) {
  AOCL_LOG_INIT_STATS();
  // "ip" is the input pointer, and "op" is the output pointer.
  const char* ip = input;
  assert(input_size <= kBlockSize);
  assert((table_size & (table_size - 1)) == 0);  // table must be power of two
  const uint32_t mask = 2 * (table_size - 1);
  const char* ip_end = input + input_size;
  const char* base_ip = ip;

  const size_t kInputMarginBytes = 15;
  if (SNAPPY_PREDICT_TRUE(input_size >= kInputMarginBytes)) {
#if AOCL_DECOMPRESS_FAST > 0
/**
 * In fast decompression mode, 8 bytes are read instead of the usual 4 bytes,
 * hence the input size until limit is reduced by 4 bytes.
 */
    const char* ip_limit = input + input_size - kInputMarginBytes - 4;
#else
    const char* ip_limit = input + input_size - kInputMarginBytes;
#endif
    
    LOG_FORMATTED(TRACE, logCtx, "Input size = %zu, Input size until limit = %zu",
     input_size, (size_t)(ip_limit - ip));
#ifdef AOCL_SNAPPY_MATCH_SKIP_OPT
    uint32_t bbhl_prev = 0; //baseline bytes_between_hash_lookups to use
#endif
    for (uint32_t preload = LittleEndian::Load32(ip + 1);;) {
      // Bytes in [next_emit, ip) will be emitted as literal bytes.  Or
      // [next_emit, ip_end) after the main loop.
      const char* next_emit = ip++;
      uint64_t data = LittleEndian::Load64(ip);
      // The body of this loop calls EmitLiteral once and then EmitCopy one or
      // more times.  (The exception is that when we're close to exhausting
      // the input we goto emit_remainder.)
      //
      // In the first iteration of this loop we're just starting, so
      // there's nothing to copy, so calling EmitLiteral once is
      // necessary.  And we only start a new iteration when the
      // current iteration has determined that a call to EmitLiteral will
      // precede the next call to EmitCopy (if any).
      //
      // Step 1: Scan forward in the input looking for a 4-byte-long match.
      // If we get close to exhausting the input then goto emit_remainder.
      //
      // Heuristic match skipping: If 32 bytes are scanned with no matches
      // found, start looking only at every other byte. If 32 more bytes are
      // scanned (or skipped), look at every third byte, etc.. When a match is
      // found, immediately go back to looking at every byte. This is a small
      // loss (~5% performance, ~0.1% density) for compressible data due to more
      // bookkeeping, but for non-compressible data (such as JPEG) it's a huge
      // win since the compressor quickly "realizes" the data is incompressible
      // and doesn't bother looking for matches everywhere.
      //
      // The "skip" variable keeps track of how many bytes there are since the
      // last match; dividing it by 32 (ie. right-shifting by five) gives the
      // number of bytes to move ahead for each iteration.
      //
      // If AOCL match skipping optimization is enabled, the number of bytes
      // skipped while checking for a match is double that as compared to regular
      // skipping. This way, every 32 unmatched bytes, instead of skipping by
      // (skip / 32) we skip by (skip / 32) * 2

      uint32_t skip = 32;

      const char* candidate;
      if (ip_limit - ip >= 16) {
        auto delta = ip - base_ip;
        for (int j = 0; j < 4; ++j) {
          for (int k = 0; k < 4; ++k) {
            int i = 4 * j + k;
            // These for-loops are meant to be unrolled. So we can freely
            // special case the first iteration to use the value already
            // loaded in preload.
            uint32_t dword = i == 0 ? preload : static_cast<uint32_t>(data);
            assert(dword == LittleEndian::Load32(ip + i));
            uint16_t* table_entry = TableEntry_crc32(table, dword, mask);
            candidate = base_ip + *table_entry;
            assert(candidate >= base_ip);
            assert(candidate < ip + i);
            *table_entry = delta + i;
            if (SNAPPY_PREDICT_FALSE(LittleEndian::Load32(candidate) == dword)) {
              *op = LITERAL | (i << 2);
              UnalignedCopy128(next_emit, op + 1);
              ip += i;
              op = op + i + 2;
              goto emit_match;
            }
            data >>= 8;
          }
          data = LittleEndian::Load64(ip + 4 * j + 4);
        }
        ip += 16;
        skip += 16;
      }
      while (true) {
        assert(static_cast<uint32_t>(data) == LittleEndian::Load32(ip));
        uint16_t* table_entry = TableEntry_crc32(table, data, mask);

#ifdef AOCL_SNAPPY_MATCH_SKIP_OPT
        uint32_t bytes_between_hash_lookups = bbhl_prev + ((skip >> 5) << 1);
        skip += (skip >> 5);
#else
        uint32_t bytes_between_hash_lookups = skip >> 5;
        skip += bytes_between_hash_lookups;
#endif
        LOG_FORMATTED(TRACE, logCtx, "skip = %u", skip);

        AOCL_LOG_UPDATE_SKIP(bytes_between_hash_lookups);
        const char* next_ip = ip + bytes_between_hash_lookups;
        if (SNAPPY_PREDICT_FALSE(next_ip > ip_limit)) {
          ip = next_emit;
          goto emit_remainder;
        }
        candidate = base_ip + *table_entry;
        assert(candidate >= base_ip);
        assert(candidate < ip);

        *table_entry = ip - base_ip;

#if AOCL_DECOMPRESS_FAST == 1
        if (candidate > base_ip) {
          uint64_t ip64 = LittleEndian::Load64(ip-1);
          uint64_t ca64 = LittleEndian::Load64(candidate-1);

          // Matching 5 bytes starting at ip-1 and candidate-1
          if ((ip64 & 0xFFFFFFFFFF) == (ca64 & 0xFFFFFFFFFF)) {
            ip-=1; candidate-=1;
#ifdef AOCL_SNAPPY_MATCH_SKIP_OPT
            bbhl_prev = bytes_between_hash_lookups > AOCL_SNAPPY_MATCH_SKIPPING_THRESHOLD ? (bytes_between_hash_lookups >> 1) : 0;
#endif /* AOCL_SNAPPY_MATCH_SKIP_OPT */
            break;
          }

          // Matching 5 bytes starting at ip and candidate
          if ((ip64 & 0xFFFFFFFFFF00) == (ca64 & 0xFFFFFFFFFF00)) {
#ifdef AOCL_SNAPPY_MATCH_SKIP_OPT
            bbhl_prev = bytes_between_hash_lookups > AOCL_SNAPPY_MATCH_SKIPPING_THRESHOLD ? (bytes_between_hash_lookups >> 1) : 0;
#endif /* AOCL_SNAPPY_MATCH_SKIP_OPT */
            break;
          }
        }

#elif AOCL_DECOMPRESS_FAST == 2
        // Matching 5 bytes starting at ip and candidate
        if (SNAPPY_PREDICT_FALSE((LittleEndian::Load64(ip) & 0xFFFFFFFFFF)==(LittleEndian::Load64(candidate) & 0xFFFFFFFFFF))) {
#ifdef AOCL_SNAPPY_MATCH_SKIP_OPT
          bbhl_prev = bytes_between_hash_lookups > AOCL_SNAPPY_MATCH_SKIPPING_THRESHOLD ? (bytes_between_hash_lookups >> 1) : 0;
#endif /* AOCL_SNAPPY_MATCH_SKIP_OPT */
          break;
        }

#else /* !AOCL_DECOMPRESS_FAST */
        if (SNAPPY_PREDICT_FALSE(static_cast<uint32_t>(data) ==
                                LittleEndian::Load32(candidate))) {

#ifdef AOCL_SNAPPY_MATCH_SKIP_OPT
            //set offset to 0 or 1/2 of current value depending on how large 
            //bytes_between_hash_lookups(bbhl) is.
            //For files that compress well, bbhl tends to be small,
            //and no additonal offset is required (bbhl_prev=0).
            //For files that do not compress well, finding matches once bbhl
            //is fairly large is unreliable. Hence, instead of starting to match with bbhl=1
            //in the next iteration, we start with an offset of bbhl/2.
            //This gives significant speed up for files that do not compress well, at the
            //expense of slight ratio degradation.
            bbhl_prev = bytes_between_hash_lookups > AOCL_SNAPPY_MATCH_SKIPPING_THRESHOLD ? (bytes_between_hash_lookups >> 1) : 0;
#endif
          break;
        }
#endif /* AOCL_DECOMPRESS_FAST */
        data = LittleEndian::Load32(next_ip);
        ip = next_ip;
      }

      // Step 2: A 4-byte match has been found.  We'll later see if more
      // than 4 bytes match.  But, prior to the match, input
      // bytes [next_emit, ip) are unmatched.  Emit them as "literal bytes."
      assert(next_emit + 16 <= ip_end);

#if AOCL_DECOMPRESS_FAST == 1
      // EmitLiteral if there is literal data to emit.
      if(SNAPPY_PREDICT_FALSE(ip > next_emit)) {
        op = EmitLiteral</*allow_fast_path=*/true>(op, next_emit, ip - next_emit);
      }
#else
      op = EmitLiteral</*allow_fast_path=*/true>(op, next_emit, ip - next_emit);
#endif

      // Step 3: Call EmitCopy, and then see if another EmitCopy could
      // be our next move.  Repeat until we find no match for the
      // input immediately after what was consumed by the last EmitCopy call.
      //
      // If we exit this loop normally then we need to call EmitLiteral next,
      // though we don't yet know how big the literal will be.  We handle that
      // by proceeding to the next iteration of the main loop.  We also can exit
      // this loop via goto if we get close to exhausting the input.
    emit_match:
      uint32_t candidate_data = 0;
      do {
        // We have a 4-byte match at ip, and no need to emit any
        // "literal bytes" prior to ip.
        const char* base = ip;
        std::pair<size_t, bool> p =
            FindMatchLength(candidate + 4, ip + 4, ip_end, &data);
        size_t matched = 4 + p.first;
        AOCL_LOG_UPDATE_MATCH(matched);
        ip += matched;
        size_t offset = base - candidate;
        assert(0 == memcmp(base, candidate, matched));
        if (p.second) {
          op = AOCL_EmitCopy</*len_less_than_12=*/true>(op, offset, matched);
        } else {
          op = AOCL_EmitCopy</*len_less_than_12=*/false>(op, offset, matched);
        }
        if (SNAPPY_PREDICT_FALSE(ip >= ip_limit)) {
          goto emit_remainder;
        }
        // Expect 5 bytes to match
        assert((data & 0xFFFFFFFFFF) ==
               (LittleEndian::Load64(ip) & 0xFFFFFFFFFF));
        // We are now looking for a 4-byte match again.  We read
        // table[Hash(ip, mask)] for that.  To improve compression,
        // we also update table[Hash(ip - 1, mask)] and table[Hash(ip, mask)].
        uint32_t prevIpData = LittleEndian::Load32(ip - 1);
        uint16_t* table_entry = TableEntry_crc32(table, data, mask);
        candidate = base_ip + *table_entry;
        candidate_data = LittleEndian::Load32(candidate);
        *table_entry = ip - base_ip;
        *TableEntry_crc32(table, prevIpData, mask) = ip - base_ip - 1;
        // Measurements on the benchmarks have shown the following probabilities
        // for the loop to exit (ie. avg. number of iterations is reciprocal).
        // BM_Flat/6  txt1    p = 0.3-0.4
        // BM_Flat/7  txt2    p = 0.35
        // BM_Flat/8  txt3    p = 0.3-0.4
        // BM_Flat/9  txt3    p = 0.34-0.4
        // BM_Flat/10 pb      p = 0.4
        // BM_Flat/11 gaviota p = 0.1
        // BM_Flat/12 cp      p = 0.5
        // BM_Flat/13 c       p = 0.3
      } while (static_cast<uint32_t>(data) == candidate_data);
      // Because the least significant 5 bytes matched, we can utilize data
      // for the next iteration.
      preload = data >> 8;
    }
  }

 emit_remainder:
  // Emit the remaining bytes as a literal
  LOG_FORMATTED(TRACE, logCtx, "Emit remaining %d bytes", (int)(ip_end - ip));
  if (ip < ip_end) {
    op = EmitLiteral</*allow_fast_path=*/false>(op, ip, ip_end - ip);
  }

  AOCL_LOG_CLEAR_STATS(input_size);
  return op;
}
}
#endif /* AOCL_SNAPPY_AVX_OPT */

#ifdef AOCL_SNAPPY_OPT
#ifdef AOCL_SNAPPY_ENABLE_DECOMPRESS_BRANCHLESS
#if defined(__GNUC__) && !defined(__clang__)
/**
 * The ternary operators used in the reference code cause slower code execution 
 * when compiled with GCC as the generated assembly code produces branch instructions 
 * instead of conditional move instructions. Therefore, to optimize the code,
 * it is beneficial to replace the ternary operators with inline assembly that makes
 * use of conditional move instructions.
 */
SNAPPY_ATTRIBUTE_ALWAYS_INLINE
inline size_t AOCL_AdvanceToNextTagX86Optimized_gcc(const uint8_t** ip_p, size_t* tag) {
  const uint8_t*& ip = *ip_p;
  size_t tag_type = *tag & 3;

  size_t len;
  __asm__ __volatile__(
    "cmp $0, %1\n\t"           // compare tag_type with 0
    "cmove %2, %0\n\t"         // if tag_type == 0, move lit_len (*tag >> 2)+1 to `len`
    "cmovne %3, %0\n\t"        // if tag_type != 0, move tag_type to `len`
    : "=&r"(len)               // output: len
    : "r"(tag_type), "r"((*tag >> 2) + 1), "r"(tag_type)  // input: tag_type, (*tag >> 2)+1, tag_type
    : "cc"
  );
  ip += (len + 1);
  *tag = ip[-1];
  return tag_type;
}
#endif

/**
 * The use of lookup table for extracting the offset results in better
 * performance than using memcpy.
 */
inline uint32_t AOCL_ExtractOffset(uint32_t val, size_t tag_type) {
#if defined(__aarch64__)
  constexpr uint64_t kExtractMasksCombined = 0x0000FFFF00FF0000ull;
  return val & static_cast<uint32_t>(
      (kExtractMasksCombined >> (tag_type * 16)) & 0xFFFF);
#else
  static constexpr uint32_t kExtractMasks[4] = {0, 0xFF, 0xFFFF, 0};
  return val & kExtractMasks[tag_type];
#endif
}

/**
 * Same as DecompressBranchless_avx2, but calls AOCL_AdvanceToNextTagX86Optimized
 * (when compiled with GCC) and AOCL_ExtractOffset.
 */
template <typename T>
AOCL_SNAPPY_TARGET_AVX2
std::pair<const uint8_t*, ptrdiff_t> AOCL_DecompressBranchless_avx2(
    const uint8_t* ip, const uint8_t* ip_limit, ptrdiff_t op, T op_base,
    ptrdiff_t op_limit_min_slop) {
  // If deferred_src is invalid point it here.
  uint8_t safe_source[64];
  const void* deferred_src;
  size_t deferred_length;
  ClearDeferred(&deferred_src, &deferred_length, safe_source);

  // We unroll the inner loop twice so we need twice the spare room.
  op_limit_min_slop -= kSlopBytes;
  if (2 * (kSlopBytes + 1) < ip_limit - ip && op < op_limit_min_slop) {
    const uint8_t* const ip_limit_min_slop = ip_limit - 2 * kSlopBytes - 1;
    ip++;
    // ip points just past the tag and we are touching at maximum kSlopBytes
    // in an iteration.
    size_t tag = ip[-1];
#if defined(__clang__) && defined(__aarch64__)
    // Workaround for https://bugs.llvm.org/show_bug.cgi?id=51317
    // when loading 1 byte, clang for aarch64 doesn't realize that it(ldrb)
    // comes with free zero-extension, so clang generates another
    // 'and xn, xm, 0xff' before it use that as the offset. This 'and' is
    // redundant and can be removed by adding this dummy asm, which gives
    // clang a hint that we're doing the zero-extension at the load.
    asm("" ::"r"(tag));
#endif
    do {
      // The throughput is limited by instructions, unrolling the inner loop
      // twice reduces the amount of instructions checking limits and also
      // leads to reduced mov's.

      SNAPPY_PREFETCH(ip + 128);
      for (int i = 0; i < 2; i++) {
        const uint8_t* old_ip = ip;
        assert(tag == ip[-1]);
        // For literals tag_type = 0, hence we will always obtain 0 from
        // ExtractLowBytes. For literals offset will thus be kLiteralOffset.
        ptrdiff_t len_minus_offset = kLengthMinusOffset[tag];
        uint32_t next;
#if defined(__aarch64__)
        size_t tag_type = AdvanceToNextTagARMOptimized(&ip, &tag);
        // We never need more than 16 bits. Doing a Load16 allows the compiler
        // to elide the masking operation in ExtractOffset.
        next = LittleEndian::Load16(old_ip);
#else
#if defined(__GNUC__) && !defined(__clang__)
        size_t tag_type = AOCL_AdvanceToNextTagX86Optimized_gcc(&ip, &tag);
#else
        size_t tag_type = AdvanceToNextTagX86Optimized(&ip, &tag);
#endif
        next = LittleEndian::Load32(old_ip);
#endif
        size_t len = len_minus_offset & 0xFF;
        ptrdiff_t extracted = AOCL_ExtractOffset(next, tag_type);
        ptrdiff_t len_min_offset = len_minus_offset - extracted;
        if (SNAPPY_PREDICT_FALSE(len_minus_offset > extracted)) {
          if (SNAPPY_PREDICT_FALSE(len & 0x80)) {
            // Exceptional case (long literal or copy 4).
            // Actually doing the copy here is negatively impacting the main
            // loop due to compiler incorrectly allocating a register for
            // this fallback. Hence we just break.
          break_loop:
            ip = old_ip;
            goto exit;
          }
          // Only copy-1 or copy-2 tags can get here.
          assert(tag_type == 1 || tag_type == 2);
          std::ptrdiff_t delta = (op + deferred_length) + len_min_offset - len;
          // Guard against copies before the buffer start.
          // Execute any deferred MemCopy since we write to dst here.
          MemCopy64_avx2(op_base + op, deferred_src, deferred_length);
          op += deferred_length;
          ClearDeferred(&deferred_src, &deferred_length, safe_source);
          if (SNAPPY_PREDICT_FALSE(delta < 0 ||
                                  !Copy64BytesWithPatternExtension(
                                      op_base + op, len - len_min_offset))) {
            goto break_loop;
          }
          // We aren't deferring this copy so add length right away.
          op += len;
          continue;
        }
        std::ptrdiff_t delta = (op + deferred_length) + len_min_offset - len;
        if (SNAPPY_PREDICT_FALSE(delta < 0)) {
          // Due to the spurious offset in literals have this will trigger
          // at the start of a block when op is still smaller than 256.
          if (tag_type != 0) goto break_loop;
          MemCopy64_avx2(op_base + op, deferred_src, deferred_length);
          op += deferred_length;
          DeferMemCopy(&deferred_src, &deferred_length, old_ip, len);
          continue;
        }

        // For copies we need to copy from op_base + delta, for literals
        // we need to copy from ip instead of from the stream.
        const void* from =
            tag_type ? reinterpret_cast<void*>(op_base + delta) : old_ip;
        MemCopy64_avx2(op_base + op, deferred_src, deferred_length);
        op += deferred_length;
        DeferMemCopy(&deferred_src, &deferred_length, from, len);
      }
    } while (ip < ip_limit_min_slop &&
             static_cast<ptrdiff_t>(op + deferred_length) < op_limit_min_slop);
  exit:
    ip--;
    assert(ip <= ip_limit);
  }
  // If we deferred a copy then we can perform.  If we are up to date then we
  // might not have enough slop bytes and could run past the end.
  if (deferred_length) {
      MemCopy64_avx2(op_base + op, deferred_src, deferred_length);
    op += deferred_length;
    ClearDeferred(&deferred_src, &deferred_length, safe_source);
  }
  return {ip, op};
}

#endif /* AOCL_SNAPPY_ENABLE_DECOMPRESS_BRANCHLESS */
#endif /* AOCL_SNAPPY_OPT */

#ifdef AOCL_SNAPPY_AVX2_OPT
template<>
template<>
#if defined(__GNUC__) && defined(__x86_64__)
__attribute__((aligned(32)))
#endif
/* Derived from DecompressAllTags_bmi. 
   + DecompressBranchless loop removed
   + char_table used in place of kLengthMinusOffset
   + Data access optimizations related to early loading of data (for clang).
*/
AOCL_SNAPPY_TARGET_AVX2
void SnappyDecompressor<with_bmi_avx>::DecompressAllTags_bmi<AOCL_SnappyArrayWriter_AVX>(AOCL_SnappyArrayWriter_AVX* writer) {
    AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
    const char* ip = ip_;
    ResetLimit(ip);
    auto op = writer->GetOutputPtr();
    // We could have put this refill fragment only at the beginning of the loop.
    // However, duplicating it at the end of each branch gives the compiler more
    // scope to optimize the <ip_limit_ - ip> expression based on the local
    // context, which overall increases speed.
#define MAYBE_REFILL()                                      \
  if (SNAPPY_PREDICT_FALSE(ip >= ip_limit_min_maxtaglen_)) { \
    ip_ = ip;                                               \
    if (SNAPPY_PREDICT_FALSE(!RefillTag())) goto exit;       \
    ip = ip_;                                               \
    ResetLimit(ip);                                         \
  }                                                         \
  preload = static_cast<uint8_t>(*ip)

    // At the start of the for loop below the least significant byte of preload
    // contains the tag.
    uint32_t preload;
    MAYBE_REFILL();
#if defined(__GNUC__) && !defined(__clang__) 
    __asm__(".p2align 6");
    __asm__("nop");
    __asm__(".p2align 5");
#endif
    for ( ;; ) {
#ifdef AOCL_SNAPPY_ENABLE_DECOMPRESS_BRANCHLESS
      {
        ptrdiff_t op_limit_min_slop;
        auto op_base = writer->GetBase(&op_limit_min_slop);
        if (op_base) {
          auto res =
              AOCL_DecompressBranchless_avx2<decltype(op_base)>(reinterpret_cast<const uint8_t*>(ip),
                                   reinterpret_cast<const uint8_t*>(ip_limit_),
                                   op - op_base, op_base, op_limit_min_slop);
          ip = reinterpret_cast<const char*>(res.first);
          op = op_base + res.second;
          MAYBE_REFILL();
        }
      }
#endif /* AOCL_SNAPPY_ENABLE_DECOMPRESS_BRANCHLESS */
      const uint8_t c = static_cast<uint8_t>(preload);
      ip++;

      // Ratio of iterations that have LITERAL vs non-LITERAL for different
      // inputs.
      //
      // input          LITERAL  NON_LITERAL
      // -----------------------------------
      // html|html4|cp   23%        77%
      // urls            36%        64%
      // jpg             47%        53%
      // pdf             19%        81%
      // txt[1-4]        25%        75%
      // pb              24%        76%
      // bin             24%        76%
#if defined(__clang__)
      uint32_t ipData = LittleEndian::Load32(ip);
#endif
      if (SNAPPY_PREDICT_FALSE((c & 0x3) == LITERAL)) {
        size_t literal_length = (c >> 2) + 1u;
        if (writer->TryFastAppend(ip, ip_limit_ - ip, literal_length, &op)) {
          assert(literal_length < 61);
          ip += literal_length;
          // NOTE: There is no MAYBE_REFILL() here, as TryFastAppend()
          // will not return true unless there's already at least five spare
          // bytes in addition to the literal.
          preload = static_cast<uint8_t>(*ip);
          continue;
        }
        if (SNAPPY_PREDICT_FALSE(literal_length >= 61)) {
          // Long literal.
          const size_t literal_length_length = literal_length - 60;
          literal_length =
#if defined(__clang__)
              ExtractLowBytes_bmi(ipData, literal_length_length) +
#else
              ExtractLowBytes_bmi(LittleEndian::Load32(ip), literal_length_length) +
#endif
              1;
          ip += literal_length_length;
        }

        size_t avail = ip_limit_ - ip;
        while (avail < literal_length) {
          if (!writer->Append(ip, avail, &op)) goto exit;
          literal_length -= avail;
          reader_->Skip(peeked_);
          size_t n;
          ip = reader_->Peek(&n);
          avail = n;
          peeked_ = avail;
          if (avail == 0) goto exit;
          ip_limit_ = ip + avail;
          ResetLimit(ip);
        }
        if (!writer->Append(ip, literal_length, &op)) goto exit;
        ip += literal_length;
        MAYBE_REFILL();
      } else {
        if (SNAPPY_PREDICT_FALSE((c & 3) == COPY_4_BYTE_OFFSET)) {
          const size_t copy_offset = LittleEndian::Load32(ip);
          const size_t length = (c >> 2) + 1;
          ip += 4;

          if (!writer->AppendFromSelf(copy_offset, length, &op)) goto exit;
        } else {
          const uint32_t entry = char_table[c];
#if defined(__clang__)
          const uint32_t trailer = ExtractLowBytes_bmi(ipData, c & 3);
#else
          preload = LittleEndian::Load32(ip);
          const uint32_t trailer = ExtractLowBytes_bmi(preload, c & 3);
#endif
          const uint32_t length = entry & 0xff;

          // copy_offset/256 is encoded in bits 8..10.  By just fetching
          // those bits, we get copy_offset (since the bit-field starts at
          // bit 8).
          const uint32_t copy_offset = (entry & 0x700) + trailer;
          if (!writer->AppendFromSelf(copy_offset, length, &op)) goto exit;

          ip += (c & 3);
          // By using the result of the previous load we reduce the critical
          // dependency chain of ip to 4 cycles.
#if defined(__clang__)
          preload = ipData >> ((c & 3) * 8);
#else
          preload >>= (c & 3) * 8;
#endif
          if (ip < ip_limit_min_maxtaglen_) continue;
        }
        MAYBE_REFILL();
      }
    }
#undef MAYBE_REFILL
  exit:
    writer->SetOutputPtr(op);
  }

template<>
template<>
#if defined(__GNUC__) && defined(__x86_64__)
  __attribute__((aligned(32)))
#endif
AOCL_SNAPPY_TARGET_AVX2
void SnappyDecompressor<with_bmi_avx>::DecompressAllTags<>(AOCL_SnappyArrayWriter_AVX* writer) {
      DecompressAllTags_bmi(writer);
}
#endif /* AOCL_SNAPPY_AVX2_OPT */

#ifdef AOCL_ENABLE_THREADS
void AOCL_RawCompress_mt(const char* input, size_t input_length, char* compressed,
                 size_t* compressed_length, CompressionOptions options) {
aocl_thread_group_t thread_group_handle;
  aocl_thread_info_t cur_thread_info;
  AOCL_INT32 ret_status = -1;
  AOCL_INT32 maxCompressedLength = (AOCL_INT32)MaxCompressedLength(input_length);

  ret_status = aocl_setup_parallel_compress_mt(&thread_group_handle, (char *)input, compressed,
                                              (AOCL_INT32)input_length, maxCompressedLength,
                                              (AOCL_INT32)kBlockSize, WINDOW_FACTOR);
  if (ret_status < 0)
    return;

  if (thread_group_handle.num_threads == 1) {
    ByteArraySource reader(input, input_length);
    UncheckedByteArraySink writer(compressed);
    LOG_UNFORMATTED(INFO, logCtx, "Running single threaded compress");
    Compress(&reader, &writer, options);

    // Compute how many bytes were added
    *compressed_length = (writer.CurrentDestination() - compressed);
    LOG_UNFORMATTED(TRACE, logCtx, "Exit");
    return;
  }
  else {
#ifdef AOCL_THREADS_LOG
      printf("Compress Thread [id: %d] : Before parallel region\n", omp_get_thread_num());
#endif
      LOG_FORMATTED(INFO, logCtx, "Running multi threaded compress on %u threads", thread_group_handle.num_threads);
#pragma omp parallel private(cur_thread_info) shared(thread_group_handle) num_threads(thread_group_handle.num_threads)
    {
#ifdef AOCL_THREADS_LOG
      printf("Compress Thread [id: %d] : Inside parallel region\n", omp_get_thread_num());
#endif
      AOCL_INT32 thread_max_src_size = thread_group_handle.common_part_src_size + thread_group_handle.leftover_part_src_bytes;
      uint32_t cmpr_bound_pad = (AOCL_INT32)MaxCompressedLength(thread_max_src_size) - thread_max_src_size;
      uint32_t is_error = 1;
      uint32_t thread_id = omp_get_thread_num();
      AOCL_INT32 partition_compressed_length = 0;

      if (aocl_do_partition_compress_mt(&thread_group_handle, &cur_thread_info, cmpr_bound_pad, thread_id) == 0)
      {
        ByteArraySource reader(cur_thread_info.partition_src, cur_thread_info.partition_src_size);
        UncheckedByteArraySink writer(cur_thread_info.dst_trap);
        Compress(&reader, &writer, options);

        // Compute how many bytes were added
        partition_compressed_length = (writer.CurrentDestination() - cur_thread_info.dst_trap);
        is_error = 0;
      } // aocl_do_partition_compress_mt

      thread_group_handle.threads_info_list[thread_id].partition_src = cur_thread_info.partition_src;
      thread_group_handle.threads_info_list[thread_id].dst_trap = cur_thread_info.dst_trap;
      thread_group_handle.threads_info_list[thread_id].dst_trap_size = partition_compressed_length;
      thread_group_handle.threads_info_list[thread_id].partition_src_size = cur_thread_info.partition_src_size;
      thread_group_handle.threads_info_list[thread_id].is_error = is_error;
      thread_group_handle.threads_info_list[thread_id].num_child_threads = 0;

    } // #pragma omp parallel
#ifdef AOCL_THREADS_LOG
    printf("Compress Thread [id: %d] : After parallel region\n", omp_get_thread_num());
#endif
    
    // set pointers to the desired locations in their respective buffers
    AOCL_CHAR* dst_org = thread_group_handle.dst;
    AOCL_CHAR* dst_ptr = dst_org;
    thread_group_handle.dst += ret_status /* on success, ret_status stores the RAP metadata length */;
    dst_ptr += RAP_START_OF_PARTITIONS;

    uint32_t thread_cnt = 0;
    aocl_thread_info_t *thread_info_iter;
    uint32_t combined_uncompressed_length = 0;

    // compute the sum of all uncompressed lengths and check for errors
    for (; thread_cnt < thread_group_handle.num_threads; ++thread_cnt) {
      thread_info_iter = &thread_group_handle.threads_info_list[thread_cnt];

      // check if this thread encountered any errors during compression
      if (thread_info_iter->is_error) {
        aocl_destroy_parallel_compress_mt(&thread_group_handle);

#ifdef AOCL_THREADS_LOG
        printf("Compress Thread [id: %d] : Encountered ERROR\n", thread_cnt);
#endif
        LOG_FORMATTED(ERR, logCtx, "Compress Thread [id: %d] : Encountered ERROR", thread_cnt);
        return;
      }

      // read the uncompressed length from the snappy stream produced
      uint32_t uncompressed_length_from_stream;
      AOCL_INT32 data_start_offset = AOCL_GetUncompressedLengthAndVarintByteWidth(thread_info_iter->dst_trap,
                                                                             thread_info_iter->dst_trap_size,
                                                                             &uncompressed_length_from_stream);

      // check if the uncompressed length matches the source partition's size
      if (data_start_offset == 0 || uncompressed_length_from_stream != thread_info_iter->partition_src_size) {
        aocl_destroy_parallel_compress_mt(&thread_group_handle);

#ifdef AOCL_THREADS_LOG
        printf("Compress Thread [id: %d] : Encountered ERROR\n", thread_cnt);
#endif
        LOG_FORMATTED(ERR, logCtx, "Compress Thread [id: %d] : Encountered ERROR", thread_cnt);
        return;
      }

      // add the uncompressed length to a length accumulator
      LOG_FORMATTED(DEBUG, logCtx, "Uncompressed length from Thread [id: %d] is %u", thread_cnt, uncompressed_length_from_stream);
      combined_uncompressed_length += uncompressed_length_from_stream;

      // store data start location (byte immediately after the varint) for later
      thread_info_iter->additional_state_info = (AOCL_VOID*)(thread_info_iter->dst_trap + data_start_offset);

      // modify the thread's buffer length to adjust for the varint
      thread_info_iter->dst_trap_size -= data_start_offset;
    }

    // encode length accumulator's value as a varint and write to destination buffer
    char* after_varint = Varint::Encode32(thread_group_handle.dst, combined_uncompressed_length);
    *compressed_length = (size_t)(after_varint - dst_org);
    thread_group_handle.dst = after_varint;

    thread_cnt = 0;
    // since at this point, 'compressed_length' stores the number of bytes
    // between the start of the buffer and the byte immediately after the
    // encoded varint, it also stores the offset for the starting location of
    // the data of the first compressed buffer
    uint32_t thread_dst_offset = (uint32_t)(*compressed_length);

    // For the first thread:
    // generate RAP data and write to corresponding location in destination buffer
    *(uint32_t*)dst_ptr = thread_dst_offset;
    dst_ptr += RAP_OFFSET_BYTES;
    thread_dst_offset += thread_group_handle.threads_info_list[0].dst_trap_size;
    *(AOCL_INT32*)dst_ptr = thread_group_handle.threads_info_list[0].dst_trap_size;
    dst_ptr += RAP_LEN_BYTES;
    *(AOCL_INT32*)dst_ptr = thread_group_handle.threads_info_list[0].partition_src_size;
    dst_ptr += DECOMP_LEN_BYTES;

    // compute cumulative dst_trap_size and save in unsued member partition_src_size
    thread_group_handle.threads_info_list[0].partition_src_size = 0;

    // update the compressed_length to include the current thread's compressed length
    *compressed_length += thread_group_handle.threads_info_list[0].dst_trap_size;

    for (thread_cnt = 1; thread_cnt < thread_group_handle.num_threads; ++thread_cnt) {
      thread_info_iter = &thread_group_handle.threads_info_list[thread_cnt];


      // generate RAP data and write to corresponding location in destination buffer
      *(uint32_t*)dst_ptr = thread_dst_offset;
      dst_ptr += RAP_OFFSET_BYTES;
      thread_dst_offset += thread_info_iter->dst_trap_size;
      *(AOCL_INT32*)dst_ptr = thread_info_iter->dst_trap_size;
      dst_ptr += RAP_LEN_BYTES;
      *(AOCL_INT32*)dst_ptr = thread_info_iter->partition_src_size;
      dst_ptr += DECOMP_LEN_BYTES;

      /* compute cumulative dst_trap_size and save in unsued member partition_src_size
      * This is used as offset to indicate starting points of compressed data blocks in dst */
      thread_group_handle.threads_info_list[thread_cnt].partition_src_size =
              thread_group_handle.threads_info_list[thread_cnt - 1].partition_src_size +
              thread_group_handle.threads_info_list[thread_cnt - 1].dst_trap_size; //cur_offset i.e. cumulative dst_trap_size

      // update the compressed_length to include the current thread's compressed length
      *compressed_length += thread_info_iter->dst_trap_size;
    }

    /* copy compressed data from threads to dst multi-threaded */
#pragma omp parallel private(cur_thread_info) shared(thread_group_handle) num_threads(thread_group_handle.num_threads)
    {
      uint32_t thread_cnt = omp_get_thread_num();
      cur_thread_info = thread_group_handle.threads_info_list[thread_cnt];
      memcpy(thread_group_handle.dst + cur_thread_info.partition_src_size, //cur_thread_info.partition_src_size contains cur_offset
          (AOCL_CHAR*)cur_thread_info.additional_state_info, cur_thread_info.dst_trap_size);
    }
    // free the memory allocated for the the thread_info_list and/or for each thread's dst_trap
    aocl_destroy_parallel_compress_mt(&thread_group_handle);
  }
}
bool AOCL_RawUncompress_mt(const char* compressed, size_t compressed_length, char* uncompressed) {
aocl_thread_group_t thread_group_handle;
  aocl_thread_info_t cur_thread_info;
  AOCL_INT32 ret_status = -1;

  // check 'compressed' here, since passing a null pointer to the setup function
  // will lead to a segmentation fault
  if (compressed != NULL) {
    ret_status = aocl_setup_parallel_decompress_mt(&thread_group_handle, 
                                                  (char *)compressed, uncompressed,
                                                  compressed_length, 
                                                  0 /* maxDecompressedSize (not required by snappy, hence 0) */, 
                                                  0 /* use_ST_decompressor (not required here, hence 0) */);
    if (ret_status < 0)
      return false;
  }
  else {
    // if 'compressed' is a null pointer, we can set 'ret_status' to 0 and pass
    // over the control flow to the single threaded compressor. This way we can
    // guarantee that the behavior for this special case remains the same,
    // whether we use the multithreaded compressor or the single threaded one.
    ret_status = 0;
  }

  if (ret_status == 0 /* for when compressed is NULL*/ || AOCL_MT_PARTITIONS_NOT_FOUND(thread_group_handle)) {
    LOG_UNFORMATTED(INFO, logCtx, "Running single threaded decompression");
    size_t ulength;
    const char *start_compressed = compressed + ret_status;
    size_t compressed_length_actual = compressed_length - ret_status;

    if (!GetUncompressedLength(start_compressed, compressed_length_actual, &ulength))
      return false;
    if (ulength != 0 && uncompressed == NULL)
      return false;

    return SNAPPY_SAW_raw_uncompress_fp(start_compressed, compressed_length_actual, uncompressed);
  }
  else {
    LOG_FORMATTED(INFO, logCtx, "Running multi threaded decompress on %u threads", thread_group_handle.num_threads);
#ifdef AOCL_THREADS_LOG
        printf("Decompress Thread [id: %d] : Before parallel region\n", omp_get_thread_num());
#endif

#pragma omp parallel private(cur_thread_info) shared(thread_group_handle) num_threads(thread_group_handle.num_threads)
    {
#ifdef AOCL_THREADS_LOG
      printf("Decompress Thread [id: %d] : Inside parallel region\n", omp_get_thread_num());
#endif
      uint32_t is_error = 1;
      uint32_t thread_id = omp_get_thread_num();
      bool local_result = false;
      AOCL_INT32 thread_parallel_res = 0;

      AOCL_MT_PROCESS_PARTITION_START(thread_group_handle, ti_cur, thread_id)
      thread_parallel_res = aocl_do_partition_decompress_mt(&thread_group_handle, 
          &cur_thread_info, AOCL_MT_CUR_THREAD_SERIAL_ID(ti_cur));
      if (thread_parallel_res == AOCL_MT_DECOMP_PARTITION_SUCCESS || 
          thread_parallel_res == AOCL_MT_DECOMP_PARTITION_ERR_INSUFFICIENT_DST_SPACE /* Expected error due to unspecified destination size. */)
      {
        local_result = SNAPPY_SAW_raw_uncompress_direct_fp(cur_thread_info.partition_src, 
            cur_thread_info.partition_src_size, cur_thread_info.dst_trap, cur_thread_info.dst_trap_size);
        is_error = local_result ? 0 : 1;
      } // aocl_do_partition_decompress_mt
      else if (thread_parallel_res == AOCL_MT_DECOMP_PARTITION_EMPTY_SRC)
      {
        local_result = 0;
        is_error = 0;
      }

      ti_cur->partition_src = cur_thread_info.partition_src;
      ti_cur->dst_trap = cur_thread_info.dst_trap;
      ti_cur->dst_trap_size = cur_thread_info.dst_trap_size;
      ti_cur->partition_src_size = cur_thread_info.partition_src_size;
      ti_cur->is_error = is_error;
      ti_cur->num_child_threads = 0;
      AOCL_MT_PROCESS_PARTITION_END(ti_cur)
    } // #pragma omp parallel

#ifdef AOCL_THREADS_LOG
    printf("Decompress Thread [id: %d] : After parallel region\n", omp_get_thread_num());
#endif

    /* Check all threads for errors. If any failed, cleanup and return false. */
    for (uint32_t thread_cnt = 0; thread_cnt < thread_group_handle.num_threads; thread_cnt++)
    {
      AOCL_MT_PROCESS_PARTITION_START(thread_group_handle, ti_cur, thread_cnt)
      // In case of any thread partitioning or alloc errors, exit the decompression process with error
      if (ti_cur->is_error)
      {
        aocl_destroy_parallel_decompress_mt(&thread_group_handle);
#ifdef AOCL_THREADS_LOG
        printf("Decompress Thread [id: %d] : Encountered ERROR\n", thread_cnt);
#endif
        LOG_FORMATTED(ERR, logCtx, "Decompress Thread [id: %d] : Encountered ERROR", thread_cnt);
        return false;
      }

      AOCL_MT_PROCESS_PARTITION_END(ti_cur)
    }

    // free the memory allocated for the the thread_info_list and/or for each thread's dst_trap
    aocl_destroy_parallel_decompress_mt(&thread_group_handle);
    return true;
  }
}

// since multithreaded decompression happens using externally available uncompressed
// lengths rather than reading the lengths from the stream, we need a decompression
// function that takes the uncompressed length as a parameter and which does not read
// or modify the stream pointer for the computation of the uncompressed length
template <typename Writer, typename T>
static bool InternalUncompressDirect(Source* r, Writer* writer, uint32_t uncompressed_len) {
    SnappyDecompressor<T> decompressor(r);
    return InternalUncompressAllTags_c(&decompressor, writer, r->Available(),
        uncompressed_len);
}

// By default, the snappy API doesn't return the length of the uncompressed
// file. To get that information, the caller has to make a call to
// 'GetUncompressedLength'. Files compressed using the AOCL multithreaded
// compressor have a RAP frame header with its RAP metadata which is absent in
// files compressed using the single threaded compressors. This causes a
// difference in the location of the encoded uncompressed length varint in the
// stream; where for the single theaded compressor's output it is located in the
// first few bytes of the stream and for the the multithreaded compressor's
// output it is located in the bytes immediately after the RAP frame header.
// Currently, the only way to get the width of the RAP frame header is to make a
// call to 'aocl_setup_parallel_decompress_mt' and so we need to make a call to
// this function to figure out the offset for the location of the varint.
bool GetUncompressedLengthFromMTCompressedBuffer(const char* start, size_t n, size_t* result) {
  if (start == NULL || result == NULL) return false;
  aocl_thread_group_t thread_group_handle;
  AOCL_INT32 offset = aocl_setup_parallel_decompress_mt(&thread_group_handle, 
                                                  (char *)start,
                                                  NULL, /* 'outbuf', safe, since the setup function does not do any operation on it */
                                                  n,
                                                  0 /* maxDecompressedSize (not required by snappy, hence 0) */, 
                                                  1 /* use_ST_decompressor (get only the RAP metadata length, no allocations done) */);
  const char* start_of_stream = start + offset;
  size_t size_of_stream = n - offset;
  uint32_t v = 0;
  const char* limit = start_of_stream + size_of_stream;
  if (Varint::Parse32WithLimit(start_of_stream, limit, &v) != NULL) {
    *result = v;
    return true;
  } else {
    return false;
  }
}

// similar to GetUncompressedLength; difference being that in addition to setting the 
// value encoded within the varint in `result`, it returns a non-zero value signifying
// the number of bytes occupied by the varint in the stream. In case of errors during
// parsing of the varint, the function returns 0.
AOCL_INT32 AOCL_GetUncompressedLengthAndVarintByteWidth(const char* start, size_t n, uint32_t* result) {
  if (start == NULL || result == NULL) return 0;
  *result = 0;
  uint32_t v = 0;
  const char* limit = start + n;
  const char* end_of_varint = Varint::Parse32WithLimit(start, limit, &v);
  if (end_of_varint != NULL) {
    *result = v;
    return (AOCL_INT32)(end_of_varint - start);
  } else {
    return 0;
  }
}

// for multithreaded decompression, where the uncompressed length is not available
// on the stream but rather is stored externally, we need to have a decompression
// function that takes the uncompressed length as a parameter. The following functions
// are top level functions that facilitate said behavior
bool SAW_RawUncompressDirect(const char* compressed, size_t compressed_length, char* uncompressed, uint32_t uncompressed_len) {
  ByteArraySource reader(compressed, compressed_length);
  SnappyArrayWriter output(uncompressed);
  return InternalUncompressDirectArray_fp(&reader, &output, uncompressed_len);
}

#ifdef AOCL_SNAPPY_AVX_OPT
/* Same as InternalUncompressDirect, but is built with target attribute avx. */
template <typename Writer, typename T>
AOCL_SNAPPY_TARGET_AVX
static bool InternalUncompressDirect_avx(Source* r, Writer* writer, uint32_t uncompressed_len) {
  AOCL_SIMD_UNIT_TEST(DEBUG, logCtx, "Enter");
  SnappyDecompressor<T> decompressor(r);
  return InternalUncompressAllTags_avx(&decompressor, writer, r->Available(),
      uncompressed_len);
}

bool AOCL_SAW_RawUncompressDirect(const char* compressed, size_t compressed_length, char* uncompressed, uint32_t uncompressed_len) {
  ByteArraySource reader(compressed, compressed_length);
  AOCL_SnappyArrayWriter_AVX output(uncompressed);
  return InternalUncompressDirectAOCLArray_fp(&reader, &output, uncompressed_len);
}
#endif /* AOCL_SNAPPY_AVX_OPT */
#endif /* AOCL_ENABLE_THREADS */

/* ------------ Modifications to support AOCL optimizations - end -----------*/

}  // namespace snappy
