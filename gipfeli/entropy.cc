// Copyright 2011 Google Inc. All Rights Reserved.

#include "entropy.h"

#include <string.h>
#include <string>

#include "entropy_code_builder.h"
#include "enum.h"
#include "stubs-internal.h"

namespace util {
namespace compression {
namespace gipfeli {

// Gives order of importance for masks (combination of samplers that found
// the symbol).
static const uint8 g_order[32] = {
  0, 3, 2, 7, 2, 7, 5, 11,
  2, 7, 5, 11, 5, 11, 9, 14,
  1, 6, 4, 10, 4, 10, 8, 13,
  4, 10, 8, 13, 8, 13, 12, 15
};

namespace testing {

int ExposeOrder(int x) {
  return g_order[x];
}

}  // namespace testing

uint64* Entropy::GetWriteBitsOutput() const {
  return write_bits_output_;
}

uint32 Entropy::GetWriteBitsOutputSize() const {
  return write_bits_pos_ - write_bits_output_;
}

inline void Entropy::SampleContent(const uint8* __restrict input,
                                   const uint32 input_size,
                                   uint8* mask) {
  const uint8 *ip = input;
  static const int kSamplePeriod = 43;

  int samples = input_size / kSamplePeriod + 1;
  int i = 0;
  // We sample the input to 5 bits of the mask.
  // Three samplers sample twice per kSamplePeriod,
  // one sample 3 times and one 1 time per kSamplePeriod.
  while (--samples) {
    mask[ip[i]] |= 1;
    i += 2;
    mask[ip[i]] |= 2;
    i += 5;
    mask[ip[i]] |= 4;
    i += 6;
    mask[ip[i]] |= 8;
    i += 3;
    mask[ip[i]] |= 16;
    i += 7;
    mask[ip[i]] |= 1;
    i += 2;
    mask[ip[i]] |= 2;
    i += 5;
    mask[ip[i]] |= 4;
    i += 6;
    mask[ip[i]] |= 8;
    i += 3;
    mask[ip[i]] |= 1;
    i += 4;
  }
}

inline bool Entropy::CountSamples(uint8* __restrict mask) {
  // The value chars_count[i] is number of symbols being categorized to the i-th
  // order of histogram.
  uint8 chars_count[16] = { 0 };
  for (int i = 255; i != 0; --i) {
    chars_count[g_order[mask[i]]]++;
  }
  if (chars_count[g_order[mask[0]]] != 255) {
    chars_count[g_order[mask[0]]]++;
  }

  // Gives relative estimate on number of symbols in the input given that
  // combination of samplers corresponding to i-th order occurred.
  int histogram[16] = {
    33, 81, 86, 93, 145, 157, 161, 177,
    239, 266, 270, 312, 387, 492, 717, 1000
  };
  int histogram_15 = 43000;
  for (int i = 0; i < 15; i++) {
    histogram_15 -= histogram[i] * chars_count[i];
  }
  if (chars_count[15] > 0) {
    histogram_15 /= chars_count[15];
    histogram[15] = std::max(histogram_15, 1000);
  }

  uint32 proportion_first_32 = 0;
  int to_add = 32;
  int i = 15;
  while (to_add > 0) {
    proportion_first_32 += histogram[i] * chars_count[i];
    to_add -= chars_count[i];
    i--;
  }
  proportion_first_32 += histogram[i + 1] * to_add;

  uint32 proportion_from_96 = 0;
  to_add = 160;
  i = 0;
  while (to_add > 0) {
    proportion_from_96 += histogram[i] * chars_count[i];
    to_add -= chars_count[i];
    i++;
  }
  proportion_from_96 += histogram[i - 1] * to_add;

  // For first 32 symbols we save 2-bits per symbol if we decide to apply
  // entropy encoding. On the other hand from symbols 96 onwards, we lose
  // 2 bits per symbol. We decide whether to apply compression depending
  // on their proportion (where the estimation of proportion is biased towards
  // not doing compression). 352 bits is a penalty for overhead needed to
  // encode entropy-code table.
  return proportion_first_32 * 6 + 352 > proportion_from_96 * 10;
}

// Return number of bytes used to communicate the mask
int Entropy::BuildEntropyCodeBitmask(const int* assign_length,
                                     uint8* mask) {
  // First save mask of 6 and 8 bit long symbols in two levels. We split 256
  // symbols to 32 segments of length 8. First 32 bits will encode if the
  // segment is not empty. Then for each non-empty segment we use 8-bits
  // to encode which symbols are present.
  uint8* mask_start = mask;
  int non_empty = 0;
  for (int i = 0; i < 4; i++) {
    *mask = 0;
    for (int j = 0; j < 8; j++) {
      int present = 0;
      uint8 value = 0;
      for (int k = 0; k < 8; k++) {
        if (assign_length[64 * i + 8 * j + k] <= 8) {
          present = 1;
          value |= 1 << (7 - k);
        }
      }
      *mask |= present << (7 - j);

      if (present) {
        mask_start[4 + non_empty] = value;
        non_empty++;
      }
    }
    mask++;
  }
  mask += non_empty;

  // In the second phase we encode which symbols have length 6 as a subset
  // bit-mask from count bits, where 96 is number of symbols having 6 or 8
  // bit code. So we will use 12 bytes.
  int used_bits = 0;
  *mask = 0;
  for (int i = 0; i < 256; i++) {
    if (assign_length[i] <= 8) {
      *mask <<= 1;
      *mask += (assign_length[i] == 6);
      used_bits++;
      if (used_bits == 8) {
        used_bits = 0;
        mask++;
        if (mask - mask_start >= 48)
          break;
        *mask = 0;
      }
    }
  }

  return mask - mask_start;
}

void Entropy::StartWriteBits(uint32* bits,
                             uint64* bit_buffer_64,
                             char* output) {
  *bit_buffer_64 = 0;
  *bits = 0;
  write_bits_output_ = reinterpret_cast<uint64 *>(output);
  write_bits_pos_ = write_bits_output_;
}

void Entropy::FlushBits(uint32 bits, uint64* bit_buffer_64) {
  // Output the rest of bit_buffer
  if (bits != 0) {
    *bit_buffer_64 <<= 64 - bits;
    *write_bits_pos_++ = *bit_buffer_64;
  }
}

// See unit tests for how these constants are computed.
static const uint8 kLengthBits[80] = {
  12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 15, 15, 15, 18, 18, 18,
  13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 19, 19, 19, 19, 19, 19,
  22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
  22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,
  22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22
};

static const uint8 kOffsetBits[80] = {
  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 13, 13, 16, 16, 16,
  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 16, 16, 16, 16, 16, 16,
  16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
  16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
  16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16
};

static const uint8 kBitsType[80] = {
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
};

static const uint8 kLengthPlus[] = {
  4, 4, 4, 8, 4, 4, 4, 4
};

namespace testing {

uint8 ExposeKLengthBits(int x) {
  return kLengthBits[x];
}

uint8 ExposeKOffsetBits(int x) {
  return kOffsetBits[x];
}

uint8 ExposeKBitsType(int x) {
  return kBitsType[x];
}

}  // namespace testing

// Format of Compressed commands:
// LITERALS starts with 00 followed
//   by _length_ (if _length_<48) which is in the next 6bits.
// Otherwise:
//   _length_-48 is number of following bits
//   where length is stored
// COPY starts with 010, 011, 100, 101, 110, 111
//   each of them means different number of bits to be written
void Entropy::CompressCommands(const uint32* __restrict commands,
                               const uint32 commands_size,
                               char* __restrict output) {
  // Commands size should always be at least one.
  uint64 bit_buffer_64 = 0;
  uint32 bits = 0;
  StartWriteBits(&bits, &bit_buffer_64, output);
  int cmd = commands[0];
  for (int i = 0; static_cast<uint32>(i) < commands_size; ) {
    if (!(cmd & COPY)) {
      // Type of command is emit literals => the next value is length.
      const int length = cmd - 1;
      ++i;
      cmd = commands[i];
      if (PREDICT_TRUE(length < 53)) {
        // output 00length (using 8 bits total)
        WriteBits(8, length, &bits, &bit_buffer_64);
      } else {
        const int bit_length = Bits::Log2FloorNonZero(length) + 1;
        // else output 47 + bitlength of (length) and then length
        WriteBits(8 + bit_length, ((47 + bit_length) << bit_length) | length,
                  &bits, &bit_buffer_64);
      }
    } else {
      // Backward reference, i.e. the next values are length, offset.
      const int length = cmd >> 24;
      const int offset = (cmd & 0x1ffff) - 1;
      const int bits_length = Bits::Log2FloorNonZero(length);
      const int index = (bits_length - 2) * 16 +
          (Bits::Log2FloorNonZero(offset | 1));
      ++i;
      cmd = commands[i];
      WriteBits(kLengthBits[index] + 3,
                (kBitsType[index] << kLengthBits[index]) |
                (length - kLengthPlus[bits_length]) << kOffsetBits[index] |
                offset,
                &bits,
                &bit_buffer_64);
    }
  }
  FlushBits(bits, &bit_buffer_64);
}

char* Entropy::Compress(const uint8* __restrict content,
                        const uint32 content_size,
                        const uint32* __restrict commands,
                        const uint32 commands_size,
                        char* output) {
  uint8 mask[512] = { 0 };
  uint32 bits = 0;
  uint64 bit_buffer_64 = 0;

  bool use_entropy_code = false;
  // We do not use entropy coding for literals if the size of the content is
  // small. Using entropy coding brings overhead (up to 44 bytes) and can save
  // at most 25% of the size of the content.
  if (content_size > 200) {
    SampleContent(content, content_size, mask);
    use_entropy_code = CountSamples(mask);
  }

  if (use_entropy_code) {
    // We decided to compress the data.
    UNALIGNED_STORE16(output, commands_size);
    output += 2;

    // Compress Backward references and write them to the output.
    CompressCommands(commands, commands_size, output);
    output += GetWriteBitsOutputSize() * sizeof(uint64);

    // Build the conversion table for Literals from the sampled data.
    int assign_value[256];
    int assign_length[256];
    for (int i = 0; i < 256; i++) mask[i] = g_order[mask[i]];
    EntropyCodeBuilder builder;
    builder.FindLimits(mask);
    builder.ProduceSymbolOrder(mask, assign_value, assign_length);

    // Maximum length of the mask can be 4 + 32 + 12 bytes
    uint8 mask[48];
    int mask_length = BuildEntropyCodeBitmask(assign_length, mask);
    memcpy(output, mask, mask_length);
    output += mask_length;

    // Compress Literals (content).
    StartWriteBits(&bits, &bit_buffer_64, output);

    // Every literal can be represented by at most 10 bits, so we can always
    // safely pack 6 of them at the same time to 64-bit integer.
    const uint8 *pt = content;
    int n = content_size / 6 + 1;
    while (--n) {
      int c = *pt++;
      uint64 value = assign_value[c];
      int length = assign_length[c];
      // macro U is used to repeat the code 5 times
      // its performance is better than for loop
      #define U {\
            c = *pt++;\
            value <<= assign_length[c];\
            length += assign_length[c];\
            value |= assign_value[c];\
            }
      U U U U U
      #undef U
      WriteBits(length, value, &bits, &bit_buffer_64);
    }
    n = &content[content_size] - pt + 1;
    while (--n) {
      const int c = *pt++;
      uint64 value = assign_value[c];
      int length = assign_length[c];
      WriteBits(length, value, &bits, &bit_buffer_64);
    }
    FlushBits(bits, &bit_buffer_64);
    output += GetWriteBitsOutputSize() * sizeof(uint64);
  } else {
    // We decided not to compress the content (according to sample it
    // seems not compressible).

    UNALIGNED_STORE16(output, commands_size);
    output += 2;

    // Compress Backward references (guess: there are not many of them).
    CompressCommands(commands, commands_size, output);
    output += GetWriteBitsOutputSize() * sizeof(uint64);

    // Stores 0 as the bitmask for entropy encoding, which means that
    // there is no entropy encoding.
    UNALIGNED_STORE32(output, 0);
    output += 4;

    // Copy content directly to the output.
    memcpy(output, content, content_size);
    output += content_size;
  }
  return output;
}

}  // namespace gipfeli
}  // namespace compression
}  // namespace util
