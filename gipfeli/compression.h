// Base API to be supported by all compression algorithms.
// Based on Snappy API.

#ifndef UTIL_COMPRESSION_PUBLIC_COMPRESSION_H_
#define UTIL_COMPRESSION_PUBLIC_COMPRESSION_H_

#include <stdlib.h>  // for size_t
#include <string>

#include "sinksource.h"
#include "stubs-internal.h"

using std::string;

namespace util {
namespace compression {

class Compressor {
 public:
  virtual ~Compressor() { }

  // ------------------------------------------------------------------------
  // Higher-level string based routines (should be sufficient for most users)
  // ------------------------------------------------------------------------

  // Sets "*output" to the compressed version of input. Original contents
  // of *output are lost.
  //
  // REQUIRES: "input[]" is not an alias of "*output".
  //
  // Returns size of the compressed string.
  virtual size_t Compress(const string& input, string* output) = 0;

  // Decompresses "compressed" to "*uncompressed".
  // Original contents of "*uncompressed" are lost.
  //
  // REQUIRES: "compressed" is not an alias of "*uncompressed".
  //
  // Returns false if the message is corrupted and could not be decompressed.
  virtual bool Uncompress(const string& compressed, string* uncompressed)
      = 0;

  // ------------------------------------------------------------------------
  // Generic compression/decompression routines based on ByteSource/ByteSink.
  //
  // To compress/decompress into Cords, flat arrays etc., use the
  // corresponding ByteSource/Sink (e.g. CordByteSink, CheckedArrayByteSink,
  // DataBuffer, StringByteSink etc.)
  // ------------------------------------------------------------------------

  // Compress the bytes read from "*source" and append to "*sink". Return the
  // number of bytes written.
  virtual size_t CompressStream(
      Source* source, Sink* sink) = 0;

  // Decompresses the bytes from "*source" and append them to "*sink".
  //
  // Returns false in case of an error during decompression and writes to
  // the sink only the data successfully decompressed.
  virtual bool UncompressStream(
      Source* source, Sink* sink) = 0;

  // ------------------------------------------------------------------------
  // To compress/decompress into flat arrays, we need to know the max size
  // of the compressed/uncompressed data. The following routines provide
  // that.
  //
  // Example usage of compressing a string into a raw flat buffer
  //
  //    char* buf = new char[MaxCompressedLength(str.size()];
  //    UncheckedArrayByteSink sink(buf);
  //    ArrayByteSource source(&str);
  //    compressor->CompressStream(source, sink);
  // ------------------------------------------------------------------------

  // Returns the maximal size of the compressed representation of
  // input data that is "source_bytes" bytes in length;
  virtual size_t MaxCompressedLength(size_t source_bytes) = 0;

  // Stores the length of the uncompressed data in *result. This operation
  // is expected to take O(1) time.
  //
  // REQUIRES: "compressed" was produced by Compress().
  //
  // Returns false on parsing error.
  virtual bool GetUncompressedLength(
      const string& compressed, size_t* result) = 0;

  // Stores the length of the uncompressed data in *result. Note that
  // the bytesource may not point to the start of the data any more
  // and hence may need to be re-initialized before passing to Uncompress.
  //
  // REQUIRES: "compressed" was produced by Compress().
  //
  // Returns false on parsing error.
  virtual bool GetUncompressedLengthStream(
      Source* compressed, size_t* result) = 0;

 protected:
  Compressor() { }

 private:
  DISALLOW_COPY_AND_ASSIGN(Compressor);
};

}  // namespace compression
}  // namespace util

#endif  // UTIL_COMPRESSION_PUBLIC_COMPRESSION_H_
