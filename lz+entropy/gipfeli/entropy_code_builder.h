// Copyright 2011 Google Inc. All Rights Reserved.

#ifndef UTIL_COMPRESSION_GIPFELI_INTERNAL_ENTROPY_CODE_BUILDER_H_
#define UTIL_COMPRESSION_GIPFELI_INTERNAL_ENTROPY_CODE_BUILDER_H_

#include "stubs-internal.h"

namespace util {
namespace compression {
namespace gipfeli {

// Orders symbols according to sampled data.
// It assigns the most frequent 32 values 6-bit code (with prefix 0),
// then next 64 values 8-bit code (with prefix 10) and all other values have
// 10-bit code (with prefix 11). Returns for each symbol i, its new value
// in assign_value and its bit-length in assign_length.
class EntropyCodeBuilder {
 public:
  void FindLimits(uint8* symbols);

  void ProduceSymbolOrder(uint8* symbols,
                          int* assign_value,
                          int* assign_length);
  int limit_32_;
  int limit_96_;
  int choose_at_limit_32_;
  int choose_at_limit_96_;
};

}  // namespace gipfeli
}  // namespace compression
}  // namespace util

#endif  // UTIL_COMPRESSION_GIPFELI_INTERNAL_ENTROPY_CODE_BUILDER_H_
