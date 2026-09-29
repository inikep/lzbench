#ifndef UTIL_COMPRESSION_GIPFELI_OPENSOURCE_SINKSOURCE_H_
#define UTIL_COMPRESSION_GIPFELI_OPENSOURCE_SINKSOURCE_H_

// Based on Snappy sink source implementation.

#include <algorithm>
#include <stddef.h>
#include "stubs-internal.h"

using std::min;

namespace util {
namespace compression {

// MemBlock needed for Sink
class MemBlock {
 public:
  virtual ~MemBlock() { }

  void* data() { return data_; }
  const void* data() const { return data_; }
  size_t length() const { return length_; }

  // Original pointer/size passed to constructor (before adjusts)
  void*  orig_data() { return orig_data_; }
  const void* orig_data() const { return orig_data_; }
  size_t orig_length() const { return orig_length_; }

  // Helper routines to reduce the extent of the visible block.  These
  // do not affect orig_data/orig_length, so those values can be used
  // for appropriate cleanup in the destructors.
  //
  // It is an error to call these routines with "n > length()".
  // Also note, multiple discard operations are cumulative.
  // E.g., two calls to "DiscardPrefix(10)" are equivalent to
  // one call to "DiscardPrefix(20)".
  void DiscardPrefix(size_t n) {
    data_ = reinterpret_cast<char*>(data_) + n;
    length_ -= n;
  }
  void DiscardSuffix(size_t n) {
    length_ -= n;
  }

 protected:
  // For use by subclasses
  MemBlock(void* mem, size_t len) {
    data_ = orig_data_ = mem;
    length_ = orig_length_ = len;
  }

 private:
  void*  data_;
  size_t length_;
  void*  orig_data_;
  size_t orig_length_;

  DISALLOW_COPY_AND_ASSIGN(MemBlock);
};

// Data block 'space' of specified 'length' must have been created by 'new
// char[]', and it will be delete[]'ed when this object is destroyed.
class NewedMemBlock: public MemBlock {
 public:
  NewedMemBlock(char* space, size_t len)
      : MemBlock(space, len) {
      assert(orig_data() != NULL);
  }
  virtual ~NewedMemBlock() {
    delete[] reinterpret_cast<char*>(orig_data());
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(NewedMemBlock);
};

// A Sink is an interface that consumes a sequence of bytes.
class Sink {
 public:
  Sink() { }
  virtual ~Sink() { }

  // Append "bytes[0,n-1]" to this.
  virtual void Append(const char* bytes, size_t n) = 0;

  // Appends the data from the given MemBlock (see //strings/memblock.h).  The
  // ByteSink takes ownership of the block, so the given MemBlock must be heap
  // allocated.  The default implementation calls through to Append(), then
  // deletes the given block.
  virtual void AppendMemBlock(MemBlock* block) {
    Append(reinterpret_cast<char*>(block->data()), block->length());
    delete block;
  }

  // Returns a writable buffer for appending and writes the buffer's capacity to
  // *allocated_size. Guarantees *allocated_size >= min_size.
  // May return a pointer to the caller-owned scratch buffer which must have
  // scratch_size >= min_size.
  //
  // The returned buffer is only valid until the next operation
  // on this ByteSink.
  //
  // After writing at most *allocated_size bytes, call Append() with the
  // pointer returned from this function and the number of bytes written.
  // Many Append() implementations will avoid copying bytes if this function
  // returned an internal buffer.
  //
  // If the sink implementation allocates or reallocates an internal buffer,
  // it should use the desired_size_hint if appropriate. If a caller cannot
  // provide a reasonable guess at the desired capacity, it should set
  // desired_size_hint = 0.
  //
  // If a non-scratch buffer is returned, the caller may only pass
  // a prefix to it to Append(). That is, it is not correct to pass an
  // interior pointer to Append().
  //
  // The default implementation always returns the scratch buffer.
  virtual char* GetAppendBuffer(
      size_t min_size, size_t desired_size_hint, char* scratch,
      size_t scratch_size, size_t* allocated_size) {
    *allocated_size = scratch_size;
    return scratch;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(Sink);
};

// A Source is an interface that yields a sequence of bytes
class Source {
 public:
  Source() { }
  virtual ~Source() { }

  // Return the number of bytes left to read from the source
  virtual size_t Available() const = 0;

  // Peek at the next flat region of the source.  Does not reposition
  // the source.  The returned region is empty iff Available()==0.
  //
  // Returns a pointer to the beginning of the region and store its
  // length in *len.
  //
  // The returned region is valid until the next call to Skip() or
  // until this object is destroyed, whichever occurs first.
  //
  // The returned region may be larger than Available() (for example
  // if this ByteSource is a view on a substring of a larger source).
  // The caller is responsible for ensuring that it only reads the
  // Available() bytes.
  virtual const char* Peek(size_t* len) = 0;

  // Skip the next n bytes.  Invalidates any buffer returned by
  // a previous call to Peek().
  // REQUIRES: Available() >= n
  virtual void Skip(size_t n) = 0;

  void CopyTo(Sink* sink, size_t n) {
    while (n > 0) {
      size_t fragment_size;
      const char* fragment_data = Peek(&fragment_size);
      if (fragment_size == 0) {
        break;
      }
      fragment_size = min<size_t>(n, fragment_size);
      sink->Append(fragment_data, fragment_size);
      Skip(fragment_size);
      n -= fragment_size;
    }
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(Source);
};

// A Source implementation that yields the contents of a flat array
class ByteArraySource : public Source {
 public:
  ByteArraySource(const char* p, size_t n) : ptr_(p), left_(n) { }
  virtual ~ByteArraySource() { }

  virtual size_t Available() const { return left_; }

  virtual const char* Peek(size_t* len) {
    *len = left_;
    return ptr_;
  }

  virtual void Skip(size_t n) {
    left_ -= n;
    ptr_ += n;
  }

 private:
  const char* ptr_;
  size_t left_;
};

// A Sink implementation that writes to a flat array without any bound checks.
class UncheckedByteArraySink : public Sink {
 public:
  explicit UncheckedByteArraySink(char* dest) : dest_(dest) {}
  virtual ~UncheckedByteArraySink() {}
  virtual void Append(const char* data, size_t n) {
    // Do no copying if the caller filled in the result of GetAppendBuffer()
    if (data != dest_) {
      memcpy(dest_, data, n);
    }
    dest_ += n;
  }
  char* GetAppendBufferVariable(size_t min_size, size_t desired_size_hint,
                                char* scratch, size_t scratch_size,
                                size_t* allocated_size) {
    *allocated_size = desired_size_hint;
    return dest_;
  }

 private:
  char* dest_;
};

}  // namespace compression
}  // namespace util

#endif  // UTIL_COMPRESSION_GIPFELI_OPENSOURCE_SINKSOURCE_H_
