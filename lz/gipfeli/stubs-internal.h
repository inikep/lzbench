// Various stubs for the open-source version of Gipfeli.

#ifndef UTIL_COMPRESSION_GIPFELI_OPENSOURCE_STUBS_INTERNAL_H_
#define UTIL_COMPRESSION_GIPFELI_OPENSOURCE_STUBS_INTERNAL_H_

#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string>

#include "config.h"

#if defined(__x86_64__)
// Enable 64-bit optimized versions of some routines.
#define ARCH_K8 1
#endif

// Static prediction hints.
#ifdef HAVE_BUILTIN_EXPECT
#define PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#else
#define PREDICT_FALSE(x) x
#define PREDICT_TRUE(x) x
#endif

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class.
// (Note: this is not required when building for C++11, but is still needed
// for any portable, non-C++11 code.)
//
// For disallowing only assign or copy, delete the relevant operator or
// constructor, for example:
// void operator=(const TypeName&) = delete;
// Note, that most uses of DISALLOW_ASSIGN and DISALLOW_COPY are broken
// semantically, one should either use disallow both or neither. Try to
// avoid these in new code.
//
// When building with C++11 toolchains, just use the language support
// for explicitly deleted methods. This doesn't delete the move constructor,
// so objects can still be stored in move-aware containers, like std::vector.
#ifdef LANG_CXX11
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  void operator=(const TypeName&) = delete
#else
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&);               \
  void operator=(const TypeName&)
#endif

namespace util {
namespace compression {

typedef int8_t int8;
typedef uint8_t uint8;
typedef int16_t int16;
typedef uint16_t uint16;
typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;

static const uint32 kuint32max = static_cast<uint32>(0xFFFFFFFF);
static const int64 kint64max = static_cast<int64>(0x7FFFFFFFFFFFFFFFLL);

// Potentially unaligned loads and stores.

// x86 and PowerPC can simply do these loads and stores native.

#if !defined(GIPFELI_NO_UNALIGNED) && (defined(__i386__) || defined(__x86_64__) || defined(__powerpc__))

#define UNALIGNED_LOAD16(_p) (*reinterpret_cast<const uint16 *>(_p))
#define UNALIGNED_LOAD32(_p) (*reinterpret_cast<const uint32 *>(_p))
#define UNALIGNED_LOAD64(_p) (*reinterpret_cast<const uint64 *>(_p))

#define UNALIGNED_STORE16(_p, _val) (*reinterpret_cast<uint16 *>(_p) = (_val))
#define UNALIGNED_STORE32(_p, _val) (*reinterpret_cast<uint32 *>(_p) = (_val))
#define UNALIGNED_STORE64(_p, _val) (*reinterpret_cast<uint64 *>(_p) = (_val))

// ARMv7 and newer support native unaligned accesses, but only of 16-bit
// and 32-bit values (not 64-bit); older versions either raise a fatal signal,
// do an unaligned read and rotate the words around a bit, or do the reads very
// slowly (trip through kernel mode). There's no simple #define that says just
// “ARMv7 or higher”, so we have to filter away all ARMv5 and ARMv6
// sub-architectures.
//
// This is a mess, but there's not much we can do about it.

#elif !defined(GIPFELI_NO_UNALIGNED) && \
      (defined(__arm__) && \
       !defined(__ARM_ARCH_4__) && \
       !defined(__ARM_ARCH_4T__) && \
       !defined(__ARM_ARCH_5__) && \
       !defined(__ARM_ARCH_5T__) && \
       !defined(__ARM_ARCH_5TE__) && \
       !defined(__ARM_ARCH_5TEJ__) && \
       !defined(__ARM_ARCH_6__) && \
       !defined(__ARM_ARCH_6J__) && \
       !defined(__ARM_ARCH_6K__) && \
       !defined(__ARM_ARCH_6Z__) && \
       !defined(__ARM_ARCH_6ZK__) && \
       !defined(__ARM_ARCH_6T2__))

#define UNALIGNED_LOAD16(_p) (*reinterpret_cast<const uint16 *>(_p))
#define UNALIGNED_LOAD32(_p) (*reinterpret_cast<const uint32 *>(_p))

#define UNALIGNED_STORE16(_p, _val) (*reinterpret_cast<uint16 *>(_p) = (_val))
#define UNALIGNED_STORE32(_p, _val) (*reinterpret_cast<uint32 *>(_p) = (_val))

inline uint64 UNALIGNED_LOAD64(const void *p) {
  uint64 t;
  memcpy(&t, p, sizeof t);
  return t;
}

inline void UNALIGNED_STORE64(void *p, uint64 v) {
  memcpy(p, &v, sizeof v);
}

#else

// These functions are provided for architectures that don't support
// unaligned loads and stores.

inline uint16 UNALIGNED_LOAD16(const void *p) {
  uint16 t;
  memcpy(&t, p, sizeof t);
  return t;
}

inline uint32 UNALIGNED_LOAD32(const void *p) {
  uint32 t;
  memcpy(&t, p, sizeof t);
  return t;
}

inline uint64 UNALIGNED_LOAD64(const void *p) {
  uint64 t;
  memcpy(&t, p, sizeof t);
  return t;
}

inline void UNALIGNED_STORE16(void *p, uint16 v) {
  memcpy(p, &v, sizeof v);
}

inline void UNALIGNED_STORE32(void *p, uint32 v) {
  memcpy(p, &v, sizeof v);
}

inline void UNALIGNED_STORE64(void *p, uint64 v) {
  memcpy(p, &v, sizeof v);
}

#endif

class LittleEndian {
 public:
#ifdef WORDS_BIGENDIAN
  static bool IsLittleEndian() { return false; }
#else  // !defined(WORDS_BIGENDIAN)
  static bool IsLittleEndian() { return true; }
#endif  // !defined(WORDS_BIGENDIAN)
};

// Some bit-manipulation functions.
class Bits {
 public:
  // Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
  static int Log2Floor(uint32 n);

  // Potentially faster version of Log2Floor() that returns an
  // undefined value if n == 0
  static int Log2FloorNonZero(uint32 n);

  // Return the first set least / most significant bit, 0-indexed.  Returns an
  // undefined value if n == 0.  FindLSBSetNonZero() is similar to ffs() except
  // that it's 0-indexed.
  static int FindLSBSetNonZero(uint32 n);
  static int FindLSBSetNonZero64(uint64 n);

  // Return number of bits set to 1
  static int CountOnes(uint32 n);

 private:
  DISALLOW_COPY_AND_ASSIGN(Bits);
};

#ifdef HAVE_BUILTIN_CTZ

inline int Bits::Log2Floor(uint32 n) {
  return n == 0 ? -1 : (31 ^ __builtin_clz(n));
}

inline int Bits::Log2FloorNonZero(uint32 n) {
  return 31 ^ __builtin_clz(n);
}

inline int Bits::FindLSBSetNonZero(uint32 n) {
  return __builtin_ctz(n);
}

inline int Bits::FindLSBSetNonZero64(uint64 n) {
  return __builtin_ctzll(n);
}

#else  // Portable versions.

inline int Bits::Log2Floor(uint32 n) {
  if (n == 0)
    return -1;
  int log = 0;
  uint32 value = n;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32 x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  assert(value == 1);
  return log;
}

inline int Bits::Log2FloorNonZero(uint32 n) {
  return Bits::Log2Floor(n);
}

inline int Bits::FindLSBSetNonZero(uint32 n) {
  int rc = 31;
  for (int i = 4, shift = 1 << 4; i >= 0; --i) {
    const uint32 x = n << shift;
    if (x != 0) {
      n = x;
      rc -= shift;
    }
    shift >>= 1;
  }
  return rc;
}

// FindLSBSetNonZero64() is defined in terms of FindLSBSetNonZero().
inline int Bits::FindLSBSetNonZero64(uint64 n) {
  const uint32 bottombits = static_cast<uint32>(n);
  if (bottombits == 0) {
    // Bottom bits are zero, so scan in top bits
    return 32 + FindLSBSetNonZero(static_cast<uint32>(n >> 32));
  } else {
    return FindLSBSetNonZero(bottombits);
  }
}
#endif  // End portable versions.

inline int Bits::CountOnes(uint32 n) {
  n -= ((n >> 1) & 0x55555555);
  n = ((n >> 2) & 0x33333333) + (n & 0x33333333);
  return (((n + (n >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
}

// If you know the internal layout of the std::string in use, you can
// replace this function with one that resizes the string without
// filling the new space with zeros (if applicable) --
// it will be non-portable but faster.
inline void STLStringResizeUninitialized(std::string* s, size_t new_size) {
  s->resize(new_size);
}

// Return a mutable char* pointing to a string's internal buffer,
// which may not be null-terminated. Writing through this pointer will
// modify the string.
//
// string_as_array(&str)[i] is valid for 0 <= i < str.size() until the
// next call to a string method that invalidates iterators.
//
// In C++11 you may simply use &str[0] to get a mutable char*.
//
// Prior to C++11, there was no standard-blessed way of getting a mutable
// reference to a string's internal buffer. The requirement that string be
// contiguous is officially part of the C++11 standard [string.require]/5.
inline char* string_as_array(std::string* str) {
  return str->empty() ? NULL : &*str->begin();
}

// Return the largest n such that
//   s1[0,n-1] == s2[0,n-1]
//   and n <= (s2_limit - s2).
//
// Separate implementation for x86_64 and little endian PPC, for speed.
#if (defined(__GNUC__) && (defined(ARCH_K8) \
                           || (defined(__ppc64__) && defined(_LITTLE_ENDIAN))))
static inline int FindMatchLength(const uint8* s1,
                                  const uint8* s2,
                                  const uint8* s2_limit) {
  int matched = 0;
  // Find out how long the match is. We loop over the data 64 bits at a
  // time until we find a 64-bit block that doesn't match; then we find
  // the first non-matching bit and use that to calculate the total
  // length of the match.
  while (PREDICT_TRUE(s2 <= s2_limit - 8)) {
    if (PREDICT_FALSE(UNALIGNED_LOAD64(s2) == UNALIGNED_LOAD64(s1 + matched))) {
      s2 += 8;
      matched += 8;
    } else {
      // On current (mid-2008) Opteron models there is a 3% more
      // efficient code sequence to find the first non-matching byte.
      // However, what follows is ~10% better on Intel Core 2 and newer,
      // and we expect AMD's bsf instruction to improve.
      uint64 x = UNALIGNED_LOAD64(s2) ^ UNALIGNED_LOAD64(s1 + matched);
      int matching_bits = Bits::FindLSBSetNonZero64(x);
      matched += matching_bits >> 3;
      return matched;
    }
  }
  while (PREDICT_TRUE(s2 < s2_limit)) {
    if (PREDICT_TRUE(s1[matched] == *s2)) {
      ++s2;
      ++matched;
    } else {
      return matched;
    }
  }
  return matched;
}
// Same, but max length limited by an integer. This is a better fit for
// the use in util/compression/flate.
static inline int FindMatchLengthWithLimit(const uint8 *s1, const uint8 *s2,
                                           size_t limit) {
  int matched = 0;
  size_t limit2 = (limit >> 3) + 1;  // + 1 is for pre-decrement in while
  while (PREDICT_TRUE(--limit2)) {
    if (PREDICT_FALSE(UNALIGNED_LOAD64(s2) == UNALIGNED_LOAD64(s1 + matched))) {
      s2 += 8;
      matched += 8;
    } else {
      uint64 x = UNALIGNED_LOAD64(s2) ^ UNALIGNED_LOAD64(s1 + matched);
      int matching_bits = Bits::FindLSBSetNonZero64(x);
      matched += matching_bits >> 3;
      return matched;
    }
  }
  limit = (limit & 7) + 1;  // + 1 is for pre-decrement in while
  while (--limit) {
    if (PREDICT_TRUE(s1[matched] == *s2)) {
      ++s2;
      ++matched;
    } else {
      return matched;
    }
  }
  return matched;
}
#else
static inline int FindMatchLength(const uint8 *s1, const uint8 *s2,
                                  const uint8 *s2_limit) {
  int matched = 0;
  const uint8 *s2_ptr = s2;
  // Find out how long the match is. We loop over the data 32 bits at a
  // time until we find a 32-bit block that doesn't match; then we find
  // the first non-matching bit and use that to calculate the total
  // length of the match.
  while (s2_ptr <= s2_limit - 4 &&
         UNALIGNED_LOAD32(s2_ptr) == UNALIGNED_LOAD32(s1 + matched)) {
    s2_ptr += 4;
    matched += 4;
  }
  while ((s2_ptr < s2_limit) && (s1[matched] == *s2_ptr)) {
    ++s2_ptr;
    ++matched;
  }
  return matched;
}

static inline int FindMatchLengthWithLimit(const uint8 *s1, const uint8 *s2,
                                           size_t limit) {
  return FindMatchLength(s1, s2, s2 + limit);
}
#endif

}  // namespace compression
}  // namespace util

#endif  // UTIL_COMPRESSION_GIPFELI_OPENSOURCE_STUBS_INTERNAL_H_
