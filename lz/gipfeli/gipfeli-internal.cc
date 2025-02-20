// Copyright 2011 Google Inc. All Rights Reserved.

#include "gipfeli-internal.h"

#include <algorithm>
#include <stdlib.h>
#include <string>

#include "entropy.h"
#include "lz77.h"
#include "sinksource.h"
#include "stubs-internal.h"

namespace util {
namespace compression {

Compressor* NewGipfeliCompressor() {
  return new gipfeli::Gipfeli();
}

namespace gipfeli {

Gipfeli::CompressorState::~CompressorState() {
  delete lz77_;
}

LZ77* Gipfeli::CompressorState::ReallocateLZ77(int size) {
  if (lz77_ == NULL || lz77_input_size_ < static_cast<uint32>(size)) {
    lz77_input_size_ = size;
    delete lz77_;
    lz77_ = new LZ77(size);
  } else {
    lz77_->ResetHashTable(size);
  }
  return lz77_;
}

size_t Gipfeli::Compress(const string& input, string* output) {
  STLStringResizeUninitialized(output, MaxCompressedLength(input.size()));
  size_t compressed_length = RawCompress(input, string_as_array(output));
  STLStringResizeUninitialized(output, compressed_length);
  return compressed_length;
}

// This method gives the upper bound for the compressed size of the input.
// The result is based on the following calculation.
// Every emitted literal can bring 2 bits of overhead.
// Every copy command saves more "worst case" bits than it costs:
//    -- min savings is 4 literals of 10 bits, saving 4 * 10 bits
//    -- cost is 8 bits for an extra length command + 18 bits for the copy
// For a savings of 40 - (18 + 8) == 14 bits.
// However both the commands and literals are padded to 64 bits so the
// switch could cost 64 bits due to worst case rounding (1 word extra for
// commands, none saved for literals). For large blocks this isn't an issue,
// but it can be for smaller ones.

static const int kMaxLengthSize = 9;
size_t Gipfeli::MaxCompressedLength(size_t nbytes) {
  size_t times = nbytes / kBlockSize;
  size_t full_blocks = times * Entropy::MaxCompressedSize(kBlockSize, 1);
  nbytes -= times * kBlockSize;
  size_t partial_block = Entropy::MaxCompressedSize(nbytes, 1) + sizeof(uint64);
  return kMaxLengthSize + full_blocks + partial_block;
}

size_t Gipfeli::RawCompress(const string& input, char* compressed) {
  char* output_position = compressed;

  // Store length of the input. First number of bytes of the length and then
  // the length.
  int bytes_used = 0;
  size_t len = input.size();
  output_position++;
  while (len > 0) {
    *output_position++ = len & 0xff;
    len >>= 8;
    bytes_used++;
  }

  // TODO(user): to store bytes_used, 3 bits are sufficient. We could
  // use the next 5 bits to store additional information.
  *(compressed) = bytes_used;

  if (compressor_state_ == NULL)
    compressor_state_ = new CompressorState();

  size_t lz77_init_size = std::min<size_t>(kBlockSize, input.size());
  LZ77* lz77 = compressor_state_->ReallocateLZ77(lz77_init_size);
  Entropy e;

  char* content = NULL;
  uint32* commands = NULL;
  uint32 content_size = 0;
  uint32 commands_size = 0;

  // Split input to kBlock sized blocks
  for (size_t position = 0; position < input.size();) {
    const char* prev_block = (position) ? input.data() + position - kBlockSize :
        NULL;
    const size_t length = std::min<size_t>(kBlockSize, input.size() - position);
    lz77->CompressFragment(input.data() + position, length, prev_block,
                           &content, &content_size, &commands, &commands_size);
    output_position = e.Compress(reinterpret_cast<uint8*>(content),
                                 content_size, commands, commands_size,
                                 output_position);
    position += length;
  }

  return output_position - compressed;
}

// A reader that wraps around a byte source and returns one contigious
// block at a time of block_size_ bytes. It will try to avoid copying
// if possible.
//
// Since the compressor needs the previous block for backward references
// we use a double buffer scheme and guarantee the previous block is
// always valid.

class BlockReader {
 public:
  BlockReader(Source* source, size_t block_size)
      : source_(source), block_size_(block_size), block_(NULL),
        prev_block_(NULL), pending_skip_size_(0), flat_(NULL) {
    bytes_left_ = source_->Available();

    // Check if we can get a flat buffer to hold all the input
    size_t fragment_size;
    const char* fragment_data = source_->Peek(&fragment_size);
    if (fragment_size >= bytes_left_) {
      flat_ = fragment_data;
      pending_skip_size_ = bytes_left_;
    }
  }

  virtual ~BlockReader() {
    if (flat_) {
      source_->Skip(pending_skip_size_);
    } else {
      delete[] block_;
      delete[] prev_block_;
    }
  }

  // Get a pointer to the next block of input. We guarantee that
  // the block we previously returned is still valid.
  string GetNextBlock();

 private:
  Source* source_;
  const size_t block_size_;
  char* block_;
  char* prev_block_;
  size_t pending_skip_size_;
  size_t bytes_left_;
  const char* flat_;

  DISALLOW_COPY_AND_ASSIGN(BlockReader);
};

string BlockReader::GetNextBlock() {
  if (bytes_left_ <= 0) return string();

  // We will return at least block_size_ except for the last
  // block.
  const size_t block_size = std::min<size_t>(bytes_left_, block_size_);
  bytes_left_ -= block_size;

  if (flat_) {
    flat_ += block_size;
    return string(flat_ - block_size, block_size);
  }

  std::swap(prev_block_, block_);

  // We need to allocate an extra 8 bytes at the end for an
  // optimization explained below.
  if (block_ == NULL) {
    block_ = new char[block_size + 8];
  }

  UncheckedByteArraySink sink(block_);
  source_->CopyTo(&sink, block_size);

  // For optimization, we copy the first 8 bytes of the current
  // block to the end of the previous block. That way, while doing
  // matches we don't need to special case the last few bytes of
  // previous block.
  if (prev_block_) {
    *reinterpret_cast<uint64 *>(prev_block_ + kBlockSize) =
        *reinterpret_cast<uint64 *>(block_);
  }

  return string(block_, block_size);
}

// Encode the length of the input into the first few bytes
// of the output. The first byte is the number of bytes used
// for encoding. The subsequent bytes have the actual length.
//
// Returns the number of bytes written to the sink.

size_t EncodeLength(Source* reader, Sink* writer) {
  size_t len = reader->Available();
  char ulength[kMaxLengthSize];

  int i = 1;
  while (len > 0) {
    ulength[i] = len & 0xff;
    len >>= 8;
    ++i;
  }

  ulength[0] = i - 1;
  writer->Append(ulength, i);
  return i;
}

size_t Gipfeli::CompressStream(
    Source* reader, Sink* writer) {
  size_t bytes_written = EncodeLength(reader, writer);

  if (compressor_state_ == NULL) {
    compressor_state_ = new CompressorState();
  }

  BlockReader block_reader(reader, kBlockSize);
  string input_block = block_reader.GetNextBlock();
  LZ77* lz77 = compressor_state_->ReallocateLZ77(input_block.size());

  Entropy e;
  char* scratch_output = NULL;
  int scratch_output_size = 0;
  char* content = NULL;
  uint32* commands = NULL;
  uint32 content_size = 0;
  uint32 commands_size = 0;

  string prev_block;
  while (input_block.size()) {
    lz77->CompressFragment(input_block.data(), input_block.size(),
                           prev_block.empty() ? NULL : prev_block.data(),
                           &content, &content_size, &commands, &commands_size);

    // We add 64 bytes for potential entropy overhead.
    const size_t max_output_size =
        Entropy::MaxCompressedSize(content_size, commands_size);

    // Allocate a scratch buffer in case the writer does not have space for us.
    if (static_cast<uint32>(scratch_output_size) < max_output_size) {
      // Round up to an allocation quata to avoid reallocating too often.
      static const int kOutputQuanta = 4096;
      scratch_output_size = max_output_size + kOutputQuanta -
          (max_output_size % kOutputQuanta);
      delete[] scratch_output;
      scratch_output = new char[scratch_output_size];
    }

    size_t dummy;
    char* dest = writer->GetAppendBuffer(max_output_size, max_output_size,
                                         scratch_output, max_output_size,
                                         &dummy);
    char* end = e.Compress(reinterpret_cast<uint8*>(content), content_size,
                           commands, commands_size, dest);
    size_t output_size = end - dest;

    // This should never happen and if it does the memory is already corrupted.
    if (output_size > max_output_size) {
    }

    if (scratch_output == dest) {
      NewedMemBlock* new_block = new NewedMemBlock(scratch_output, output_size);
      try {
        writer->AppendMemBlock(new_block);
      } catch (...) {
        delete new_block;
        throw;
      }
      scratch_output = NULL;
      scratch_output_size = 0;
    } else {
      writer->Append(dest, output_size);
    }

    bytes_written += output_size;
    std::swap(prev_block, input_block);
    input_block = block_reader.GetNextBlock();
  }

  delete[] scratch_output;
  return bytes_written;
}

}  // namespace gipfeli
}  // namespace compression
}  // namespace util
