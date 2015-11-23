// Copyright 2011 Google Inc. All Rights Reserved.

#ifndef UTIL_COMPRESSION_GIPFELI_INTERNAL_ENTROPY_H_
#define UTIL_COMPRESSION_GIPFELI_INTERNAL_ENTROPY_H_

#include <string>

#include "stubs-internal.h"

namespace util {
namespace compression {
namespace gipfeli {

// Applies entropy coding compression to the output of LZ77 encoding,
// i.e. to the literals in "content" and to the emit literals and emit
// copies commands in "commands". It requires output to be at least
// MaxCompressedSize().
//
// Sample usage:
//  util::compression::gipfeli::Entropy e;
//  uint16 commands[] = {0, 5};
//  uint32 commands_size = 2;
//  string content= "abc12";
//  char output[60];
//  e.Compress(content.data(), content.size(), commands,
//             commands_size, output);
class Entropy {
 public:
  // Compresses the "content" and "commands" which are output of LZ77 encoding
  // using entropy encoding.
  // Writes output to "output" and returns the pointer where the next output
  // can be written.
  char* Compress(const uint8* __restrict content,
                 const uint32 content_size,
                 const uint32* __restrict commands,
                 const uint32 commands_size,
                 char* __restrict output);

  static int MaxCompressedSize(uint32 content_size, uint32 commands_size) {
    // copied from lz77.h.
    static const int kBlockLog = 16;

    // Max number of bits used to encode a length;
    static const int kLengthLog = 6;

    // Bytes to encode the number of commands in a block.
    static const int kCommandsLenSize = 2;

    // Max overhead of a literal-encoding table.
    static const int kMaxLitLenTableSize = 48;

    // Maximum bits to encode a "literal" command.
    static const int kMaxLitSize = 8 + kBlockLog;

    // Maximum bits to encode a "copy" command.
    static const int kMaxCopySize = 3 + kLengthLog + kBlockLog;

    // Max bits to encode any command.
    static const int kMaxCommandSize =
        kMaxLitSize > kMaxCopySize ? kMaxLitSize : kMaxCopySize;

    // Max bits per byte of content.
    static const int kMaxContentBits = 10;

    // 64-bits words to encode the commands.
    const int command_words = (commands_size * kMaxCommandSize + 63) / 64;

    // 64-bit words needed to encode the content.
    const int content_words = (content_size * kMaxContentBits + 63) / 64;

    return kCommandsLenSize + kMaxLitLenTableSize +
        (command_words + content_words) * sizeof(uint64);
  }

  // All methods below are not private only to allow easier testing

  // Samples content to detemine the approximation of its histogram.
  static inline void SampleContent(const uint8* __restrict input,
                                   const uint32 input_size,
                                   uint8* mask);

  // Builds histogram and determines whether to do entropy coding of content.
  static inline bool CountSamples(uint8* mask);

  // Builds entropy encoding bitmask to communicate encoding of symbols.
  // Returns number of bytes used to communicate the mask.
  static int BuildEntropyCodeBitmask(const int* assign_length,
                                     uint8* mask);

  // Write commands to the output compressed as variable-long bit values.
  // It adds about 10 to 15% compression improvement to represent commands,
  // which is important, because after applying LZ77 to text / html,
  // around half of the data can be commands.
  void CompressCommands(const uint32* __restrict commands,
                        const uint32 commands_size,
                        char* output);

  // Methods which write bits to the output. They have to be always used in
  // the following order:
  // StartWriteBits(..);
  // WriteBits(..);
  // WriteBits(..);
  // ...
  // FlushBits(..);
  void StartWriteBits(uint32* bits,
                      uint64* bit_buffer_64,
                      char* output);

  inline void WriteBits(int bit_length,
                        uint64 value,
                        uint32* bits,
                        uint64* bit_buffer_64);

  void FlushBits(uint32 bits,
                 uint64* bit_buffer_64);

  // Allows access to written bits
  uint64* GetWriteBitsOutput() const;

  uint32 GetWriteBitsOutputSize() const;

 private:
  // TODO(user) consider merging write_bits_pos_ with bit_buffer_64,
  // since they point to the same uint64.
  uint64* write_bits_output_;
  uint64* write_bits_pos_;

  static const int kSamplers = 5;
};

void Entropy::WriteBits(int length,
                        uint64 value,
                        uint32* __restrict bits,
                        uint64* __restrict bit_buffer_64) {
  int v = 64 - length;
  if (*bits < static_cast<uint32>(v)) {
    // No split needed.
    *bit_buffer_64 <<= length;
    *bit_buffer_64 |= value;
    *bits += length;
  } else {
    // Split needed.
    *bit_buffer_64 <<= 64 - *bits;
    *bits -= v;
    *bit_buffer_64 |= value >> *bits;
    *write_bits_pos_++ = *bit_buffer_64;
    // We can write the whole value in the bit_buffer_64.
    // Consequent WriteBits or FlushBits calls will erase
    // the extra high bits.
    *bit_buffer_64 = value;
  }
}

}  // namespace gipfeli
}  // namespace compression
}  // namespace util

#endif  // UTIL_COMPRESSION_GIPFELI_INTERNAL_ENTROPY_H_
