// Copyright 2012 Google Inc. All Rights Reserved.
//
// Classes to efficiently implement stream based compression/decompression.
// The functions in this file are called from a tight decompression loop
// and are performance critical. Having them entirely defined in the
// header file allows the compiler to inline and optimize better.

#ifndef UTIL_COMPRESSION_GIPFELI_INTERNAL_STREAM_H_
#define UTIL_COMPRESSION_GIPFELI_INTERNAL_STREAM_H_

#include <algorithm>
#include <vector>

#include "gipfeli-internal.h"
#include "lz77.h"
#include "stubs-internal.h"
#include "sinksource.h"

namespace util {
namespace compression {
namespace gipfeli {

//-----------------------------------------------------------------------------

// Wraps around a sink and provides convenient Append functions.
class Writer {
 public:
  Writer()
      : sink_(NULL),
        num_bytes_left_(0),
        current_memblock_(NULL),
        previous_memblock_(NULL),
        op_base_(NULL),
        op_ptr_(NULL),
        op_limit_(NULL),
        flat_(false) {
  }

  // Set the expected length and allocate the first block
  void Initialize(Sink* sink, char* output, size_t len) {
    num_bytes_left_ = len;
    sink_ = sink;
    if (output == NULL) {
      AllocateBlock();
    } else {
      flat_ = true;
      op_base_ = op_ptr_ = output;
      op_limit_ = op_base_ + len;
    }
  }

  // If the previous block is used up, add a new block
  bool MaybeFinishBlock() {
    if (flat_ || op_limit_ > op_ptr_)
      return false;
    AllocateBlock();
    return true;
  }

  // Check if the length of what was written to the sink
  // matches the expected length
  inline bool CheckLength() const {
    return num_bytes_left_ == 0;
  }

  // Check if space is left in current block
  inline bool CheckAvailable(size_t len) const {
    return ((op_limit_ - op_ptr_) >= len);
  }

  // Append byte assuming space exists in block
  inline void AppendByte(const char c) {
    *op_ptr_++ = c;
  }

  // Append from a source buffer
  inline void Append(const char* ip, size_t available, size_t len) {
    const size_t space_left = op_limit_ - op_ptr_;
    if (len <= 16 && available >= 16 && space_left >= 16) {
      UNALIGNED_STORE64(op_ptr_, UNALIGNED_LOAD64(ip));
      UNALIGNED_STORE64(op_ptr_ + 8, UNALIGNED_LOAD64(ip + 8));
    } else {
      memcpy(op_ptr_, ip, len);
    }
    op_ptr_ += len;
  }

  // Append from ourselves using offset bytes from current position
  // as source
  bool AppendFromSelf(size_t offset, size_t len) {
    if (PREDICT_FALSE(offset == 0)) return false;

    const size_t space_left = op_limit_ - op_ptr_;
    if (offset <= op_ptr_ - op_base_) {
      return AppendFromSameBlock(offset, len, space_left);
    }

    // If we get here, we need to copy from the previous block
    // Overflow checks
    const size_t prev_available = offset - (op_ptr_ - op_base_);
    if (PREDICT_FALSE(previous_memblock_ == NULL ||
                      kBlockSize < prev_available)) {
      return false;
    }
    if (PREDICT_FALSE(len > space_left)) return false;

    const char* prev_ptr = static_cast<char*> (previous_memblock_->data())
        + kBlockSize - prev_available;

    if (len <= prev_available) {
      Append(prev_ptr, prev_available, len);
      return true;
    }

    memcpy(op_ptr_, prev_ptr, prev_available);
    op_ptr_ += prev_available;
    return AppendFromSameBlock(
      offset, len - prev_available, space_left - prev_available);
  }

  bool OnError() {
    // For the last block, if there was an error, then only
    // write as much as we were able to decompress correctly.
    if (!flat_) {
      MemBlock* last_memblock = current_memblock_;
      size_t discard_size = last_memblock->length() - (op_ptr_ - op_base_);
      if (discard_size > 0) {
        last_memblock->DiscardSuffix(discard_size);
      }
    }
    Flush();
    return false;
  }

  // Called at the end of the decompress. We ask the allocator
  // write all blocks to the sink.
  void Flush() {
    num_bytes_left_ -= op_ptr_ - op_base_;
    if (!flat_) {
      if (previous_memblock_ != NULL) {
        sink_->AppendMemBlock(previous_memblock_);
      }
      sink_->AppendMemBlock(current_memblock_);
    } else {
      sink_->Append(op_base_, op_ptr_ - op_base_);
    }
  }

 private:
  Sink* sink_;
  size_t num_bytes_left_;
  NewedMemBlock* current_memblock_;
  NewedMemBlock* previous_memblock_;

  // Pointer into current output block
  char* op_base_;       // Base of output block
  char* op_ptr_;        // Pointer to next unfilled byte in block
  char* op_limit_;      // Pointer just past block

  bool flat_;

  size_t AllocateBlock() {
    num_bytes_left_ -= (op_ptr_ - op_base_);
    if (previous_memblock_ != NULL) {
      sink_->AppendMemBlock(previous_memblock_);
    }
    previous_memblock_ = current_memblock_;

    const size_t size = std::min<size_t>(kBlockSize, num_bytes_left_);
    op_ptr_ = op_base_ = new char[size];
    op_limit_ = op_base_ + size;
    current_memblock_ = new NewedMemBlock(op_base_, size);
    return size;
  }

  bool AppendFromSameBlock(
      size_t offset, size_t len, size_t space_left) {
    if (len <= 16 && offset >= 8 && space_left >= 16) {
      UNALIGNED_STORE64(op_ptr_, UNALIGNED_LOAD64(op_ptr_ - offset));
      UNALIGNED_STORE64(op_ptr_ + 8, UNALIGNED_LOAD64(op_ptr_ - offset + 8));
    } else if (space_left >= len + kMaxIncrementCopyOverflow) {
      IncrementalCopyFastPath(op_ptr_ - offset, op_ptr_, len);
    } else {
      if (PREDICT_FALSE(len > space_left)) return false;
      IncrementalCopy(op_ptr_ - offset, op_ptr_, len);
    }
    op_ptr_ += len;
    return true;
  }

  DISALLOW_COPY_AND_ASSIGN(Writer);
};

//-----------------------------------------------------------------------------

// A class that wraps a byte source and supports
// convenient read functions.
class Reader {
 public:
  explicit Reader(Source* source) : source_(source) {
    ip_ = source_->Peek(&peeked_size_);
    ip_end_ = ip_ + peeked_size_;
  }

  virtual ~Reader() {
    source_->Skip(peeked_size_ - (ip_end_ - ip_));
  }

  bool Read16(uint32* result) {
    size_t avail = ip_end_ - ip_;
    if (PREDICT_TRUE(avail >= 2)) {
      *result = UNALIGNED_LOAD16(ip_);
      ip_ += 2;
      return true;
    }
    if (PREDICT_TRUE(ReadSlow(scratch_, 2, avail))) {
      *result = UNALIGNED_LOAD16(scratch_);
      return true;
    }
    return false;
  }

  const char* Read(char* scratch, size_t size) {
    size_t avail = ip_end_ - ip_;
    if (PREDICT_TRUE(avail >= size)) {
      ip_ += size;
      return (ip_ - size);
    }
    if (PREDICT_TRUE(ReadSlow(scratch, size, avail))) {
      return scratch;
    }
    return NULL;
  }

  inline bool Read64(uint64* result) {
    if (PREDICT_TRUE((ip_end_ - ip_) >= 8)) {
      *result = UNALIGNED_LOAD64(ip_);
      ip_ += 8;
      return true;
    }
    if (PREDICT_TRUE(ReadSlow(scratch_, 8, ip_end_ - ip_))) {
      *result = UNALIGNED_LOAD64(scratch_);
      return true;
    }
    return false;
  }

  bool AppendTo(Writer* writer, size_t size) {
    size_t avail = ip_end_ - ip_;
    do {
      if (PREDICT_TRUE(avail >= size)) {
        writer->Append(ip_, avail, size);
        ip_ += size;
        return true;
      } else {
        writer->Append(ip_, avail, avail);
        size -= avail;
        if (PREDICT_FALSE(((avail = Reload()) < 1)))
          return false;
      }
    } while (1);
    return false;
  }

  inline bool Eof() {
    return ((ip_end_ <= ip_) &&
            !(source_->Available() - peeked_size_));
  }

 private:
  // If this function is called, we assume, size > avail
  bool ReadSlow(char* buf, size_t size, size_t avail) {
    do {
      memcpy(buf, ip_, avail);
      buf += avail;
      size -= avail;
      if (PREDICT_TRUE((avail = Reload()) >= size)) {
        memcpy(buf, ip_, size);
        ip_ += size;
        return true;
      }
    } while (avail);
    return false;
  }

  int Reload() {
    source_->Skip(peeked_size_);
    ip_ = source_->Peek(&peeked_size_);
    ip_end_ = ip_ + peeked_size_;
    return peeked_size_;
  }

  Source* source_;
  const char* ip_;
  const char* ip_end_;
  size_t peeked_size_;
  char scratch_[8];

  DISALLOW_COPY_AND_ASSIGN(Reader);
};

//-----------------------------------------------------------------------------

// A class that reads bits efficiently from
// a reader by chunking it up 64 bits at a
// time.
//
// TODO(user): Try optimization of breaking up
// into inline and non-inline function.
class BitStreamReader {
 public:
  explicit BitStreamReader(Reader* reader) :
      reader_(reader), error_(false), bits_left_(64), current_(0) {
    if (PREDICT_FALSE(!reader->Read64(&current_))) {
      error_ = true;
      bits_left_ = 0;
    }
  }

  // length has max value of 16 - so uint32 is safe.
  uint32 Read(int length) {
    uint32 ret;
    if (PREDICT_TRUE(length <= bits_left_)) {
      ret = current_ >> (64 - length);
      current_ <<= length;
      bits_left_ -= length;
    } else {
      ret = (current_ >> (64 - bits_left_)) << (length - bits_left_);
      length -= bits_left_;
      if (PREDICT_FALSE(!reader_->Read64(&current_))) {
        error_ = true;
        bits_left_ = 0;
        return 0;
      }
      ret += current_ >> (64 - length);
      current_ <<= length;
      bits_left_ = 64 - length;
    }
    return ret;
  }

  bool error() {
    return error_;
  }

 private:
  Reader* reader_;
  bool error_;
  int bits_left_;
  uint64 current_;

  DISALLOW_COPY_AND_ASSIGN(BitStreamReader);
};

}  // namespace gipfeli
}  // namespace compression
}  // namespace util

#endif  // UTIL_COMPRESSION_GIPFELI_INTERNAL_STREAM_H_
