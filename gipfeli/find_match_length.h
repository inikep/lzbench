// Copyright 2011 Google Inc. All Rights Reserved.
//
// Modified from flate/snappy code from jyrki/sesse. The key
// difference between this code and that is the support for
// compare over two disjoint blocks.

#ifndef UTIL_COMPRESSION_GIPFELI_INTERNAL_FIND_MATCH_LENGTH_H_
#define UTIL_COMPRESSION_GIPFELI_INTERNAL_FIND_MATCH_LENGTH_H_

#include "stubs-internal.h"

namespace util {
namespace compression {

// Return the largest match of n bytes starting at s1 and s2.
//
// Assumptions
//
// a) s1 and s2 are in two separate blocks next to each other.
//    i.e. once you hit the end of s1, we continue matching from
//    start of s2.
//
// b) For peformance, we copy the first 8 bytes of s2 also at the end of
//    s1. This allows us to just do UNALIGNED_LOAD64 at the end of the
//    block.
//
// Separate implementation for x86_64, for speed.
//

#if defined(__GNUC__) && defined(ARCH_K8)

static inline int MultiBlockFindMatchLength(const uint8* s1,
                                            const uint8* s1_limit,
                                            const uint8* s2_base,
                                            const uint8* s2,
                                            const uint8* s2_limit) {
  const uint8* s2_start = s2;

  // Find out how long the match is. We loop over the data 64 bits at a
  // time until we find a 64-bit block that doesn't match; then we find
  // the first non-matching bit and use that to calculate the total
  // length of the match.
  while (PREDICT_TRUE(s2 <= s2_limit - 8)) {
    if (PREDICT_FALSE(UNALIGNED_LOAD64(s2) == UNALIGNED_LOAD64(s1))) {
      s2 += 8;
      s1 += 8;
    } else {
      int matched = s2 - s2_start;
      // On current (mid-2008) Opteron models there is a 3% more
      // efficient code sequence to find the first non-matching byte.
      // However, what follows is ~10% better on Intel Core 2 and newer,
      // and we expect AMD's bsf instruction to improve.
      uint64 x = UNALIGNED_LOAD64(s2) ^ UNALIGNED_LOAD64(s1);
      int matching_bits = Bits::FindLSBSetNonZero64(x);
      matched += matching_bits >> 3;
      return matched;
    }
    // If we hit end of s1 block, switch to start of s2. Note that we
    // assume first 8 bytes of s2 are copied at end of s1 for ease of
    // comparison.
    if (PREDICT_FALSE(s1 >= s1_limit)) {
      s1 = s2_base + (s1 - s1_limit);
      s1_limit = s2_limit;
    }
  }
  while (PREDICT_TRUE((s2 < s2_limit) && (*s1 == *s2))) {
    s1++;
    s2++;
  }
  return s2 - s2_start;
}

#else

static inline int MultiBlockFindMatchLength(const uint8* s1,
                                            const uint8* s1_limit,
                                            const uint8* s2_base,
                                            const uint8* s2,
                                            const uint8* s2_limit) {
  const uint8 *s2_start = s2;

  // Find out how long the match is. We loop over the data 32 bits at a
  // time until we find a 32-bit block that doesn't match; then we find
  // the first non-matching bit and use that to calculate the total
  // length of the match.
  while (s2 <= s2_limit - 4 &&
         UNALIGNED_LOAD32(s2) == UNALIGNED_LOAD32(s1)) {
    if (PREDICT_FALSE(s1 >= s1_limit)) {
      s1 = s2_base + (s1 - s1_limit);
      s1_limit = s2_limit;
    }
    s2 += 4;
    s1 += 4;
  }

  while ((s2 < s2_limit) && (*s1 == *s2)) {
    s1++;
    s2++;
  }
  return s2 - s2_start;
}

#endif

}  // namespace compression
}  // namespace util

#endif  // UTIL_COMPRESSION_GIPFELI_INTERNAL_FIND_MATCH_LENGTH_H_
