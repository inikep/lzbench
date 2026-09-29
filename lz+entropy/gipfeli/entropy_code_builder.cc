// Copyright 2011 Google Inc. All Rights Reserved.

#include "entropy_code_builder.h"

#include <string.h>

#include "stubs-internal.h"

namespace util {
namespace compression {
namespace gipfeli {

// Builds histogram and finds positions (limits), where to split symbols
// to 32 most often used, 64 less used and other even less used.
void EntropyCodeBuilder::FindLimits(uint8* symbols) {
  uint16 histogram[32];
  memset(&histogram[0], 0, sizeof(histogram));
  for (int i = 0; i < 256; i++) {
    histogram[symbols[i]]++;
  }
  limit_32_ = -1;
  limit_96_ = -1;
  choose_at_limit_32_ = 0;
  choose_at_limit_96_ = 0;
  int count = 0;
  for (int i = 31; i >= 0; i--) {
    count += histogram[i];
    if (count >= 32 && limit_32_ == -1) {
      limit_32_ = i;
      choose_at_limit_32_ = histogram[i] - (count - 32);
    }
    if (count >=96 && limit_96_ == -1) {
      limit_96_ = i;
      if (limit_32_ != limit_96_) {
        choose_at_limit_96_ = histogram[i] - (count - 96);
      } else {
        choose_at_limit_96_ = 64;
      }
    }
  }
}

// Builds symbol order and the entropy code using the limits computed
// by FindLimits method.
void EntropyCodeBuilder::ProduceSymbolOrder(uint8* symbols,
                        int* assign_value,
                        int* assign_length) {
  int best_index = 0;
  int next_best_index = 0;
  for (int i = 0; i < 256; i++) {
    if (symbols[i] >= limit_32_) {
      if (symbols[i] == limit_32_) {
        if (--choose_at_limit_32_ == 0) {
          limit_32_++;
        }
      }
      assign_value[i] = best_index;
      assign_length[i] = 6;
      ++best_index;
      continue;
    }
    if (symbols[i] >= limit_96_) {
      if (symbols[i] == limit_96_) {
        if (--choose_at_limit_96_ == 0) {
          limit_96_++;
        }
      }
      assign_value[i] = 0x80 | next_best_index;
      assign_length[i] = 8;
      ++next_best_index;
      continue;
    }
    assign_value[i] = 0x300 | i;
    assign_length[i] = 10;
  }
}

}  // namespace gipfeli
}  // namespace compression
}  // namespace util
