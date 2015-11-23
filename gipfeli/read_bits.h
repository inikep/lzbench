// Copyright 2011 Google Inc. All Rights Reserved.

#ifndef UTIL_COMPRESSION_GIPFELI_INTERNAL_READ_BITS_H_
#define UTIL_COMPRESSION_GIPFELI_INTERNAL_READ_BITS_H_

#include "stubs-internal.h"

namespace util {
namespace compression {
namespace gipfeli {

// Reads bits from the "input" char stream and returns pointer to the stream
// after reading of the input finished. Sample usage:
// char* input;
// char* input_end
// ReadBits bits;
// bits.Start(input, input_end);
// x = bits.Read(3);
// ...
// y = bits.Read(7);
// input = bits.Stop();
// if (input > input_end) error has occurred
class ReadBits {
 public:
  void Start(char* input, const char* input_end) {
    bits_left_ = 64;
    current_ = UNALIGNED_LOAD64(input);
    bits_input_ = input + sizeof(uint64);
    input_end_ = input_end;
    error_ = false;
  }

  uint32 Read(int length) {
    uint32 ret;
    if (PREDICT_TRUE(length <= bits_left_)) {
      ret = current_ >> (64 - length);
      current_ <<= length;
      bits_left_ -= length;
    } else {
      ret = (current_ >> (64 - bits_left_)) << (length - bits_left_);
      length -= bits_left_;
      if (PREDICT_FALSE((bits_input_ + sizeof(uint64)) > input_end_)) {
        error_ = true;
        bits_left_ = 0;
        return 0;
      }
      current_ = UNALIGNED_LOAD64(bits_input_);
      bits_input_ += sizeof(uint64);
      ret += current_ >> (64 - length);
      current_ <<= length;
      bits_left_ = 64 - length;
    }
    return ret;
  }

  char* Stop() {
    if (error_) {
      // Pointer set to input_end + 1 signals that an error has occurred.
      return const_cast<char*>(input_end_ + 1);
    }
    return bits_input_;
  }

 private:
  int bits_left_;
  uint64 current_;
  char* bits_input_;
  const char* input_end_;
  bool error_;
};

}  // namespace gipfeli
}  // namespace compression
}  // namespace util
#endif  // UTIL_COMPRESSION_GIPFELI_INTERNAL_READ_BITS_H_
