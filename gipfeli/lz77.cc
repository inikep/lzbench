// Copyright 2011 Google Inc. All Rights Reserved.

#include "lz77.h"

#include <algorithm>

#include "find_match_length.h"
#include "enum.h"
#include "stubs-internal.h"

namespace util {
namespace compression {
namespace gipfeli {

namespace {

// Any hash function will produce a valid compressed bitstream, but a good
// hash function reduces the number of collisions and thus yields better
// compression for compressible input, and more speed for incompressible
// input. Of course, it doesn't hurt if the hash function is reasonably fast
// either, as it gets called a lot.
inline uint32 HashBytes(uint32 bytes, int shift) {
  uint32 kMul = 0x1e35a7bd;
  return (bytes * kMul) >> shift;
}

inline uint32 Hash(const char* p, int shift) {
  return HashBytes(UNALIGNED_LOAD32(p), shift);
}

// For 0 <= offset <= 4, GetUint32AtOffset(UNALIGNED_LOAD64(p), offset) will
// equal UNALIGNED_LOAD32(p + offset).  Motivation: On x86-64 hardware we have
// empirically found that overlapping loads such as
// UNALIGNED_LOAD32(p) ... UNALIGNED_LOAD32(p+1) ... UNALIGNED_LOAD32(p+2)
// are slower than UNALIGNED_LOAD64(p) followed by shifts and casts to uint32.
static inline uint32 GetUint32AtOffset(uint64 v, int offset) {
  return v >> (LittleEndian::IsLittleEndian() ? 8 * offset : 32 - 8 * offset);
}

}  // namespace

LZ77::LZ77(int input_size) {
  commands_ = new uint32[MaxCompressedCommandsSize(input_size)];
  content_ = new char[MaxCompressedContentSize(input_size)];
  table_bits_ = Bits::Log2Floor(input_size | 0x100);
  if (table_bits_ > kMaxHashTableBits) {
    table_bits_ = kMaxHashTableBits;
  }
  hash_table_ = new uint16[1 << table_bits_];
  memset(hash_table_, 0, (1 << table_bits_) * sizeof(*hash_table_));
}

LZ77::~LZ77() {
  delete[] hash_table_;
  delete[] content_;
  delete[] commands_;
}

inline char* EmitLiteral(const char* literal,
                         int len,
                         const bool allow_fast_path,
                         char* op,
                         uint32* command) {
  *command = len;
  if (allow_fast_path && len <= 16) {
    UNALIGNED_STORE64(op, UNALIGNED_LOAD64(literal));
    UNALIGNED_STORE64(op + 8, UNALIGNED_LOAD64(literal + 8));
    return op + len;
  } else {
    char* ret = op + len;
    memcpy(op, literal, len);
    return ret;
  }
}

// REQUIRES length to be at least 4 and offset at least 1
inline void EmitCopy(const uint32 offset,
                     uint32 length,
                     uint32** commands) {
  // In 80% cases length <= 7, and in 95% cases length <= 15.
  // If the length is big we will split the copying operation to several
  // smaller ones, such that each length, which is saved to commands,
  // must be between 4 and 67, inclusive. We use later exactly 6-bits
  // to store its value minus 4.
  uint32 copy_command = COPY | offset;
  while (length > 67) {
    length -= 64;
    **commands = copy_command | (64 << 24);
    ++(*commands);
  }
  **commands = copy_command | (length << 24);
  ++(*commands);
}

void EmitCopyForTesting(const uint32 offset,
                        uint32 length,
                        uint32* commands,
                        uint32* commands_size) {
  uint32 *pt = commands + *commands_size;
  EmitCopy(offset, length, &pt);
  *commands_size = pt - commands;
}

// Compresses "input" string to the "content" and "commands" buffers.
void LZ77::CompressFragment(const char* input,
                            const size_t input_size,
                            const char* prev_block,
                            char** content,
                            uint32* content_size,
                            uint32** commands,
                            uint32* commands_size) {
  uint32 *commands_current = commands_;
  uint16* hash_table = hash_table_;
  // "ip" is the input pointer, and "op" is the output pointer.
  char* op = content_;
  const char* ip = input;
  const char* ip_end = input + input_size;
  const char* base_ip = ip;
  // Bytes in [next_emit, ip) will be emitted as literal bytes.  Or
  // [next_emit, ip_end) after the main loop.
  const char* next_emit = ip;

  const int shift = 32 - table_bits_;
  const size_t kInputMarginBytes = 15;
  if (PREDICT_TRUE(input_size >= kInputMarginBytes)) {
    const char* ip_limit = input + input_size - kInputMarginBytes;

    for (uint32 next_hash = Hash(++ip, shift); ; ) {
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
      // scanned, look at every third byte, etc.. When a match is found,
      // immediately go back to looking at every byte. This is a small loss
      // (~5% performance, ~0.1% density) for compressible data due to more
      // bookkeeping, but for non-compressible data (such as JPEG) it's a huge
      // win since the compressor quickly "realizes" the data is incompressible
      // and doesn't bother looking for matches everywhere.
      //
      // The "skip" variable keeps track of how many bytes there are since the
      // last match; dividing it by 32 (ie. right-shifting by five) gives the
      // number of bytes to move ahead for each iteration.
      uint32 skip = 32;

      const char* next_ip = ip;
      const char* candidate = NULL;
      bool in_prev_block = false;
      do {
        ip = next_ip;
        const uint32 hash = next_hash;
        const int bytes_between_hash_lookups = skip++ >> 5;
        next_ip = ip + bytes_between_hash_lookups;
        if (PREDICT_FALSE(next_ip > ip_limit)) {
          goto emit_remainder;
        }
        next_hash = Hash(next_ip, shift);
        size_t offset = hash_table[hash];
        candidate = base_ip + offset;
        if (candidate >= ip) {
          candidate = prev_block + offset;
          in_prev_block = true;
        } else {
          in_prev_block = false;
        }
        hash_table[hash] = ip - base_ip;
      } while (UNALIGNED_LOAD32(ip) != UNALIGNED_LOAD32(candidate));

      // Step 2: A 4-byte match has been found.  We'll later see if more
      // than 4 bytes match.  But, prior to the match, input
      // bytes [next_emit, ip) are unmatched.  Emit them as "literal bytes."
      op = EmitLiteral(next_emit, ip - next_emit, true, op, commands_current);
      commands_current++;

      // Step 3: Call EmitCopy, and then see if another EmitCopy could
      // be our next move.  Repeat until we find no match for the
      // input immediately after what was consumed by the last EmitCopy call.
      //
      // If we exit this loop normally then we need to call EmitLiteral next,
      // though we don't yet know how big the literal will be.  We handle that
      // by proceeding to the next iteration of the main loop.  We also can exit
      // this loop via goto if we get close to exhausting the input.
      uint64 input_bytes = 0;

      do {
        // We have a 4-byte match at ip, and no need to emit any
        // "literal bytes" prior to ip.
        const char* base = ip;

        int matched = 4;
        if (!in_prev_block || ((prev_block + kBlockSize) == base_ip)
            || (candidate + 4 >= (prev_block + kBlockSize))) {
          matched += util::compression::FindMatchLength(
              reinterpret_cast<const uint8*>(candidate) + 4,
              reinterpret_cast<const uint8*>(ip) + 4,
              reinterpret_cast<const uint8*>(ip_end));
        } else {
          matched += MultiBlockFindMatchLength(
              reinterpret_cast<const uint8*>(candidate) + 4,
              reinterpret_cast<const uint8*>(prev_block + kBlockSize),
              reinterpret_cast<const uint8*>(base_ip),
              reinterpret_cast<const uint8*>(ip) + 4,
              reinterpret_cast<const uint8*>(ip_end));
        }

        ip += matched;
        int offset;
        if (in_prev_block) {
          offset = base - base_ip + prev_block + kBlockSize - candidate;
        } else {
          offset = base - candidate;
        }

        EmitCopy(offset, matched, &commands_current);

        // We could immediately start working at ip now, but to improve
        // compression we first update hash_table[Hash(ip - 1, ...)].
        const char* insert_tail = ip - 3;
        next_emit = ip;
        if (PREDICT_FALSE(ip >= ip_limit)) {
          goto emit_remainder;
        }
        input_bytes = UNALIGNED_LOAD64(insert_tail);

        int n = insert_tail - base_ip;
        uint32 prev_hash = HashBytes(GetUint32AtOffset(input_bytes, 0), shift);
        hash_table[prev_hash] = n++;
        prev_hash = HashBytes(GetUint32AtOffset(input_bytes, 1), shift);
        hash_table[prev_hash] = n++;
        prev_hash = HashBytes(GetUint32AtOffset(input_bytes, 2), shift);
        hash_table[prev_hash] = n++;

        uint32 cur_hash = HashBytes(GetUint32AtOffset(input_bytes, 3), shift);
        int candidate_offset = hash_table[cur_hash];
        candidate = base_ip + candidate_offset;
        if (candidate >= ip) {
          candidate = prev_block + candidate_offset;
          in_prev_block = true;
        } else {
          in_prev_block = false;
        }
        hash_table[cur_hash] = n;
      } while (GetUint32AtOffset(input_bytes, 3)
               == UNALIGNED_LOAD32(candidate));

      next_hash = HashBytes(GetUint32AtOffset(input_bytes, 4), shift);
      ++ip;
    }
  }

 emit_remainder:
  // Emit the remaining bytes as a literal
  if (next_emit < ip_end) {
    op = EmitLiteral(next_emit, ip_end - next_emit, false, op,
                     commands_current);
    ++commands_current;
  }
  *content = content_;
  *content_size = op - content_;
  *commands = commands_;
  *commands_size = commands_current - commands_;
}

void LZ77::ResetHashTable(int input_size) {
  table_bits_ = Bits::Log2Floor(input_size | 0x100);
  if (table_bits_ > kMaxHashTableBits) {
    table_bits_ = kMaxHashTableBits;
  }
  memset(hash_table_, 0, (1 << table_bits_) * sizeof(*hash_table_));
}

// The value of content is always a subsequence of the input.
size_t LZ77::MaxCompressedContentSize(size_t input_size) {
  return input_size;
}

// The worst case scenario is if these two commands are repeated all the time:
// - emit literals of length 1 (use 1 command for it)
// - copy of length 4 (use 1 command for it).
// So we get at most 2 commands for each 5 symbols of the input.
// The exception when we need more commands than the input size is
// the input of size exactly 1, when we need 1 command.
//
// We need one extra command reserved because CompressCommands pre-fetches the
// next one before the comparison on array size.
size_t LZ77::MaxCompressedCommandsSize(size_t input_size) {
  return 2 * input_size / 5 + 2;
}

}  // namespace gipfeli
}  // namespace compression
}  // namespace util
