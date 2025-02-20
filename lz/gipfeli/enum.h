// Copyright 2011 Google Inc. All Rights Reserved.

#ifndef UTIL_COMPRESSION_GIPFELI_INTERNAL_ENUM_H_
#define UTIL_COMPRESSION_GIPFELI_INTERNAL_ENUM_H_

namespace util {
namespace compression {
namespace gipfeli {

static const uint32 LITERAL = 0;
static const uint32 COPY = 0x800000;

inline bool CommandIsCopy(uint32 val) {
  return val & COPY;
}

inline int CommandCopyLength(uint32 val) {
  return val >> 24;
}

inline int CommandCopyOffset(uint32 val) {
  return val & 0x1ffff;
}

inline int CommandLiteralLength(uint32 val) {
  return val;
}

inline uint32 CommandCopy(uint32 length, uint32 offset) {
  return COPY | (length << 24) | offset;
}

}  // namespace gipfeli
}  // namespace compression
}  // namespace util
#endif  // UTIL_COMPRESSION_GIPFELI_INTERNAL_ENUM_H_
