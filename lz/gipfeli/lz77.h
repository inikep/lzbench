// Copyright 2011 Google Inc. All Rights Reserved.
//
// This LZ77 part of Gipfeli has been influenced by and reuses parts of Snappy.

#ifndef UTIL_COMPRESSION_GIPFELI_INTERNAL_LZ77_H_
#define UTIL_COMPRESSION_GIPFELI_INTERNAL_LZ77_H_

#include "stubs-internal.h"

namespace util {
namespace compression {
namespace gipfeli {

// This class is used to do compression by using backward references.
// The example usage:
//   string input = "1234567890";
//   uint32 content_size = 0;
//   uint32 commands_size = 0;
//   char* content = NULL;
//   uint32* commands = NULL;
//   util::compression::gipfeli::LZ77 lz77(input.size());
//   lz77.CompressFragment(input.data(), input.size(), &content, &content_size,
//                         &commands, &commands_size);
class LZ77 {
 public:
  // It is constructed with the maximal size of the input_size with which
  // Compress Fragment can be called later.
  //
  // Constructs arrays "content" and "commands" that are sufficiently long,
  // and allocates hash table "table_" which size is the power of two
  explicit LZ77(int input_size);

  ~LZ77();

  // Compresses "input" string.
  // On return, the pointers to the resulting buffers containing content and
  // commands will be set to "content" and "commands". They do not need to be
  // set in advance.
  // REQUIRES: "input" is at most "kBlockSize" bytes long.
  void CompressFragment(const char* input,
                        const size_t input_size,
                        const char* prev_block,
                        char** content,
                        uint32* content_size,
                        uint32** commands,
                        uint32* commands_size);

  // Reset all values in hash table to 0.
  void ResetHashTable(int input_size);

  // The value of content is always a subsequence of the input.
  size_t MaxCompressedContentSize(size_t input_size);

  // The worst case scenario is if these two commands are repeated all the time:
  // - emit literals of length 1 (use 2 commands for it)
  // - copy of length 4 (use 3 commands for it)
  size_t MaxCompressedCommandsSize(size_t input_size);

 private:
  char* content_;
  uint32* commands_;
  uint16 *hash_table_;
  int table_bits_;

  DISALLOW_COPY_AND_ASSIGN(LZ77);
};

// Compression code chops up the input into blocks of at most kBlockSize.
// At the same time we allow back-references up to kBlockSize, i.e.
// we have the sliding window approach which allow us to back-reference
// also to previous blocks.
static const int kBlockLog = 16;
static const int kBlockSize = 1 << kBlockLog;

static const int kMaxHashTableBits = 15;
static const int kMaxHashTableSize = 1 << kMaxHashTableBits;

void EmitCopyForTesting(const uint32 offset,
                        uint32 length,
                        uint32* commands,
                        uint32* commands_size);

}  // namespace gipfeli
}  // namespace compression
}  // namespace util

#endif  // UTIL_COMPRESSION_GIPFELI_INTERNAL_LZ77_H_
