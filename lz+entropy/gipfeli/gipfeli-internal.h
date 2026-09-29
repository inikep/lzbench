// Copyright 2011 Google Inc. All Rights Reserved.
//         lenhardt@google.com (Rasto Lenhardt)

#ifndef UTIL_COMPRESSION_GIPFELI_INTERNAL_GIPFELI_H_
#define UTIL_COMPRESSION_GIPFELI_INTERNAL_GIPFELI_H_

#include "gipfeli.h"
#include "compression.h"

namespace util {
namespace compression {
namespace gipfeli {

class LZ77;

class Gipfeli : public util::compression::Compressor {
 public:
  Gipfeli() : compressor_state_(NULL), decompressor_state_(NULL) {}
  virtual ~Gipfeli() {
    delete compressor_state_;
    delete decompressor_state_;
  }

  virtual size_t Compress(const string& input, string* output);
  virtual size_t MaxCompressedLength(size_t source_bytes);
  virtual bool Uncompress(
      const string& compressed, string* uncompressed);
  virtual bool GetUncompressedLength(
      const string& compressed, size_t* result);
  virtual size_t CompressStream(
      Source* source, Sink* sink);
  virtual bool GetUncompressedLengthStream(
      Source* compressed, size_t* result);
  virtual bool UncompressStream(
      Source* source, Sink* sink);

 private:
  class CompressorState {
   public:
    CompressorState() : lz77_(NULL), lz77_input_size_(0) {}
    ~CompressorState();
    LZ77* ReallocateLZ77(int size);
   private:
    LZ77* lz77_;
    size_t lz77_input_size_;
  };

  class DecompressorState {
   public:
    DecompressorState() : commands_(NULL), commands_size_(0) {}
    ~DecompressorState() { delete[] commands_; }
    uint32* ReallocateCommands(int size);
   private:
    uint32* commands_;
    int commands_size_;
  };

  size_t RawCompress(const string& input, char* compressed);
  bool RawUncompress(const char* compressed, size_t compressed_length,
                     char* uncompressed, size_t uncompressed_length);
  bool InternalRawUncompress(
      const char* compressed, size_t compressed_length,
      char* uncompressed, size_t uncompressed_length);
  bool InternalRawUncompressStream(
      Source* source,
      Sink* uncompressed, size_t uncompressed_length);

  CompressorState* compressor_state_;
  DecompressorState* decompressor_state_;

  DISALLOW_COPY_AND_ASSIGN(Gipfeli);
};

class GipfeliAdaptor : public util::compression::Compressor {
 public:
  GipfeliAdaptor() {}
  virtual ~GipfeliAdaptor() {}
  virtual inline size_t Compress(const string& input, string* output) {
    Gipfeli gipfeli;
    return gipfeli.Compress(input, output);
  }
  virtual inline size_t MaxCompressedLength(size_t source_bytes) {
    Gipfeli gipfeli;
    return gipfeli.MaxCompressedLength(source_bytes);
  }
  virtual inline bool Uncompress(
      const string& compressed, string* uncompressed) {
    Gipfeli gipfeli;
    return gipfeli.Uncompress(compressed, uncompressed);
  }
  virtual inline bool GetUncompressedLength(
      const string& compressed, size_t* result) {
    Gipfeli gipfeli;
    return gipfeli.GetUncompressedLength(compressed, result);
  }
  virtual inline size_t CompressStream(
      Source* source, Sink* sink) {
    Gipfeli gipfeli;
    return gipfeli.CompressStream(source, sink);
  }
  virtual inline bool GetUncompressedLengthStream(
      Source* compressed, size_t* result) {
    Gipfeli gipfeli;
    return gipfeli.GetUncompressedLengthStream(compressed, result);
  }
  virtual inline bool UncompressStream(
      Source* source, Sink* sink) {
    Gipfeli gipfeli;
    return gipfeli.UncompressStream(source, sink);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(GipfeliAdaptor);
};

static const int kMaxIncrementCopyOverflow = 10;

// Copy "len" bytes from "src" to "op", one byte at a time.  Used for
// handling COPY operations where the input and output regions may
// overlap.  For example, suppose:
//    src    == "ab"
//    op     == src + 2
//    len    == 20
// After IncrementalCopy(src, op, len), the result will have
// eleven copies of "ab"
//    ababababababababababab
// Note that this does not match the semantics of either memcpy()
// or memmove().
inline void IncrementalCopy(const char* src, char* op, int len) {
  do {
    *op++ = *src++;
  } while (--len > 0);
}

// Equivalent to IncrementalCopy except that it can write up to ten extra
// bytes after the end of the copy, and that it is faster.
//
// The main part of this loop is a simple copy of eight bytes at a time until
// we've copied (at least) the requested amount of bytes.  However, if op and
// src are less than eight bytes apart (indicating a repeating pattern of
// length < 8), we first need to expand the pattern in order to get the correct
// results. For instance, if the buffer looks like this, with the eight-byte
// <src> and <op> patterns marked as intervals:
//
//    abxxxxxxxxxxxx
//    [------]           src
//      [------]         op
//
// a single eight-byte copy from <src> to <op> will repeat the pattern once,
// after which we can move <op> two bytes without moving <src>:
//
//    ababxxxxxxxxxx
//    [------]           src
//        [------]       op
//
// and repeat the exercise until the two no longer overlap.
//
// This allows us to do very well in the special case of one single byte
// repeated many times, without taking a big hit for more general cases.
//
// The worst case of extra writing past the end of the match occurs when
// op - src == 1 and len == 1; the last copy will read from byte positions
// [0..7] and write to [4..11], whereas it was only supposed to write to
// position 1. Thus, ten excess bytes.

inline void IncrementalCopyFastPath(const char* src, char* op, int len) {
  while (op - src < 8) {
    UNALIGNED_STORE64(op, UNALIGNED_LOAD64(src));
    len -= op - src;
    op += op - src;
  }
  while (len > 0) {
    UNALIGNED_STORE64(op, UNALIGNED_LOAD64(src));
    src += 8;
    op += 8;
    len -= 8;
  }
}

}  // namespace gipfeli
}  // namespace compression
}  // namespace util

#endif  // UTIL_COMPRESSION_GIPFELI_INTERNAL_GIPFELI_H_
