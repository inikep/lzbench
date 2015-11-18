// Copyright 2011 Google Inc. All Rights Reserved.

#include "gipfeli-internal.h"

#include <stdlib.h>
#include <string>

#include "enum.h"
#include "lz77.h"
#include "read_bits.h"
#include "stream.h"

#include "stubs-internal.h"

namespace util {
namespace compression {
namespace gipfeli {

// This method reads bitmask containing the infromation about which
// entropy code is used. It builds the conversion tables "convert_6bit"
// and "convert_8bit", which assign to each compressed symbol of length 6 or 8
// bits, the original symbol. They are used later to convert the literals.
// REQUIRES: convert_6bit to have size 32
// REQUIRES: convert_8bit to have size 64
char* DecompressMask(const uint32 upper,
                     char* input,
                     uint8* convert_6bit,
                     uint8* convert_8bit) {
  uint8* ip = reinterpret_cast<uint8*>(input);

  // First phase: read 96 symbols that will have 6 or 8-bit long codes.
  uint8 to_be_converted[96];
  int count = 0;

  uint32 and_value = 1U << 31;
  for (int i = 0; i < 32; i++) {
    if ((and_value & upper) != 0) {
      for (int j = 0; j < 8; j++) {
        if (((*ip) & (1 << (7 - j))) != 0) {
          if (PREDICT_FALSE(count >= 96)) {  // Corrupted input
            return NULL;
          }
          to_be_converted[count] = 8 * i + j;
          count++;
        }
      }
      ip++;
    }
    and_value >>= 1;
  }

  if (PREDICT_FALSE(count != 96))  // Corrupted input
    return NULL;

  // Second phase: Split 96 symbols to ones with 6-bit and ones with 8-bit
  // long codes and assign the symbol to every code.
  int count_6bit = 0;
  int count_8bit = 0;
  for (int i = 0; i < (count + 7) / 8; i++) {
    for (int j = 0; j < 8; j++) if (8 * i + j < count) {
      if (((*ip) & (1 << (7 - j))) != 0) {
        if (PREDICT_FALSE(count_6bit >= 32)) {  // Corrupted input
          return NULL;
        }
        convert_6bit[count_6bit] = to_be_converted[8 * i + j];
        count_6bit++;
      } else {
        if (PREDICT_FALSE(count_8bit >= 64)) {  // Corrupted input
          return NULL;
        }
        convert_8bit[count_8bit] = to_be_converted[8 * i + j];
        count_8bit++;
      }
    }
    ip++;
  }

  if (PREDICT_FALSE((count_6bit != 32) || (count_8bit != 64)))
    return NULL;

  return reinterpret_cast<char*>(ip);
}

static const uint16 length_length[] = {0, 0, 2, 2, 2, 3, 3, 6};
static const uint16 offset_length[] = {0, 0, 10, 13, 16, 10, 16, 16};
static const uint16 length_change[] = {0, 0, 4, 4, 4, 8, 8, 4};

char* DecompressCommands(char* input,
                         const char* input_end,
                         uint32 commands_real_size,
                         uint32* commands) {
  ReadBits bits;
  bits.Start(input, input_end);
  uint32 commands_size = 0;
  while (commands_size < commands_real_size) {
    uint32 value = bits.Read(3);
    if (value < 2) {
      value = (value << 5) + bits.Read(5);
      if (value < 53) {
        commands[commands_size++] = value + 1;
      } else {
        commands[commands_size++] = bits.Read(value - 47) + 1;
      }
    } else {
      size_t length = bits.Read(length_length[value]) + length_change[value];
      size_t offset = bits.Read(offset_length[value]) + 1;
      commands[commands_size++] = CommandCopy(length, offset);
    }
  }
  return bits.Stop();
}

bool DecompressCommandsStream(Reader* reader, uint32 commands_real_size,
                              uint32* commands) {
  BitStreamReader bits(reader);
  uint32 commands_size = 0;
  while (commands_size < commands_real_size) {
    uint32 value = bits.Read(3);
    if (value < 2) {
      value = (value << 5) + bits.Read(5);
      if (value < 53) {
        commands[commands_size++] = value + 1;
      } else {
        commands[commands_size++] = bits.Read(value - 47) + 1;
      }
    } else {
      size_t length = bits.Read(length_length[value]) + length_change[value];
      size_t offset = bits.Read(offset_length[value]) + 1;
      commands[commands_size++] = CommandCopy(length, offset);
    }
  }
  return !bits.error();
}

uint32* Gipfeli::DecompressorState::ReallocateCommands(int size) {
  if (commands_ == NULL || commands_size_ < size) {
    delete[] commands_;
    commands_ = new uint32[size];
    commands_size_ = size;
  }
  return commands_;
}

// Returns the size of the uncompressed string.
bool Gipfeli::GetUncompressedLength(const string& compressed,
                                    size_t* uncompressed_length) {
  if (compressed.size() == 0) {
    return false;
  }

  size_t bytes_used = *(compressed.data());
  if (bytes_used > 4 || (compressed.size() < 1 + bytes_used)) {
    return false;
  }

  *uncompressed_length = 0;
  for (int i = bytes_used - 1; i >= 0; i--) {
    *uncompressed_length <<= 8;
    *uncompressed_length |= static_cast<unsigned char>(compressed[i + 1]);
  }

  // Uncompressed string can have at most (2 GB - 1).
  if (*uncompressed_length >= 1ULL << 31) {
    return false;
  }

  return true;
}

bool Gipfeli::Uncompress(const string& input, string* output) {
  size_t ulength;
  if (!GetUncompressedLength(input, &ulength)) {
    return false;
  }
  if ((static_cast<uint64>(ulength) ) > output->max_size()) {
    return false;
  }
  STLStringResizeUninitialized(output, ulength);
  return RawUncompress(input.data(), input.size(),
                       string_as_array(output), ulength);
}

bool Gipfeli::RawUncompress(
    const char* compressed, size_t compressed_length,
    char* uncompressed, size_t uncompressed_length) {
  char* ip = const_cast<char*>(compressed);
  size_t bytes_used = *ip;
  ip += (1 + bytes_used);
  size_t ip_size = compressed_length - (bytes_used + 1);
  return InternalRawUncompress(
      ip, ip_size, uncompressed, uncompressed_length);
}

bool Gipfeli::InternalRawUncompress(
    const char* compressed, size_t compressed_length,
    char* uncompressed, size_t uncompressed_length) {
  char* ip = const_cast<char*>(compressed);
  char* ip_end = ip + compressed_length;
  char* op = uncompressed;
  char* op_start = op;
  char* op_end = op + uncompressed_length;

  if (decompressor_state_ == NULL)
    decompressor_state_ = new DecompressorState();

  while (ip - compressed < compressed_length) {
    if (PREDICT_FALSE(ip_end - ip < 10)) {
      // 2 bytes for commands size and at least 8 bytes for the first commands.
      return false;
    }
    uint32 commands_size = UNALIGNED_LOAD16(ip);
    ip += 2;

    uint32* commands = decompressor_state_->ReallocateCommands(commands_size);
    ip = DecompressCommands(ip, ip_end, commands_size, commands);
    if (PREDICT_FALSE(ip + 4 > ip_end)) {
      return false;
    }
    uint32 upper = 0;
    for (int i = 0; i < 4; i++) {
      uint8 val = (uint8)(*ip);
      ip++;
      upper <<= 8;
      upper += val;
    }
    uint8 convert_6bit[32] = { 0 };
    uint8 convert_8bit[64] = { 0 };
    if (upper == 0) {
      // No Entropy Coding.
      for (int i = 0; i < commands_size; i++) {
        if (!CommandIsCopy(commands[i])) {
          if (PREDICT_FALSE(op + commands[i] > op_end) ||
              ip + commands[i] > ip_end ) {
            return false;
          }
          memcpy(op, ip, commands[i]);
          ip += commands[i];
          op += commands[i];
        } else {
          size_t len = CommandCopyLength(commands[i]);
          size_t offset = CommandCopyOffset(commands[i]);
          size_t space_left = op_end - op;

          // -1u below prevents infinite loop if offset == 0
          if (PREDICT_FALSE(op - op_start <= offset - 1u)) {
            return false;
          }

          if (len <= 16 && offset >= 8 && space_left >= 16) {
            // Fast path, used for the majority (70-80%) of dynamic invocations.
            UNALIGNED_STORE64(op, UNALIGNED_LOAD64(op - offset));
            UNALIGNED_STORE64(op + 8, UNALIGNED_LOAD64(op - offset + 8));
          } else if (space_left >= len + kMaxIncrementCopyOverflow) {
            IncrementalCopyFastPath(op - offset, op, len);
          } else {
            if (PREDICT_FALSE(space_left < len)) {
              return false;
            }
            IncrementalCopy(op - offset, op, len);
          }
          op += len;
        }
      }
    } else {
      // Entropy Coding.
      if (ip + Bits::CountOnes(upper) + 12 > ip_end) {
        return false;
      }
      ip = DecompressMask(upper, ip, convert_6bit, convert_8bit);
      if (PREDICT_FALSE(ip == NULL)) {
        return false;
      }
      ReadBits bits;
      bits.Start(ip, ip_end);
      if (PREDICT_FALSE(ip_end - ip < 8)) {
        return false;
      }
      for (int i = 0; i < commands_size; i++) {
        if (!CommandIsCopy(commands[i])) {
          if (PREDICT_FALSE(op + commands[i] > op_end)) {
            return false;
          }
          for (int j = 0; j < commands[i]; j++) {
            uint32 val = bits.Read(6);
            if (PREDICT_TRUE(val < 32)) {
              *op++ = convert_6bit[val];
            } else if (val >= 48) {
              val = ((val - 48) << 4) + bits.Read(4);
              *op++ = val;
            } else {
              val = ((val - 32) << 2) + bits.Read(2);
              *op++ = convert_8bit[val];
            }
          }
        } else {
          size_t len = CommandCopyLength(commands[i]);
          size_t offset = CommandCopyOffset(commands[i]);
          size_t space_left = op_end - op;

          // -1u catches offset == 0 case
          if (PREDICT_FALSE(op - op_start <= offset - 1u)) {
            return false;
          }

          if (len <= 16 && offset >= 8 && space_left >= 16) {
            // Fast path, used for the majority (70-80%) of dynamic invocations.
            UNALIGNED_STORE64(op, UNALIGNED_LOAD64(op - offset));
            UNALIGNED_STORE64(op + 8, UNALIGNED_LOAD64(op - offset + 8));
          } else if (space_left >= len + kMaxIncrementCopyOverflow) {
            IncrementalCopyFastPath(op - offset, op, len);
          } else {
            if (PREDICT_FALSE(space_left < len)) {
              return false;
            }
            IncrementalCopy(op - offset, op, len);
          }
          op += len;
        }
      }
      ip = bits.Stop();
      if (PREDICT_FALSE(ip > ip_end)) {
        return false;
      }
    }
  }

  if (PREDICT_FALSE(ip != (compressed + compressed_length)))
    return false;
  return (op == op_end);
}

bool Gipfeli::GetUncompressedLengthStream(
    Source* source, size_t* result) {
  Reader reader(source);
  char scratch[5];
  const char* ip;

  if ((ip = reader.Read(scratch, 1)) == NULL)
    return false;

  // First byte is the number of bytes used to encode the length
  size_t bytes_used = *ip;
  if (bytes_used > 4) return false;

  if ((ip = reader.Read(scratch, bytes_used)) == NULL)
    return false;

  // Remaining bytes encode the length
  *result = 0;
  for (int i = bytes_used - 1; i >= 0; --i) {
    *result <<= 8;
    *result |= static_cast<unsigned char>(*(ip + i));
  }

  // Uncompressed string can have at most (2 GB - 1).
  if (*result >= 1ULL << 31) {
    return false;
  }

  return true;
}

bool Gipfeli::UncompressStream(
    Source* compressed, Sink* uncompressed) {
  size_t uncompressed_len;
  if (!GetUncompressedLengthStream(compressed, &uncompressed_len)) {
    return false;
  }
  return InternalRawUncompressStream(
      compressed, uncompressed, uncompressed_len);
}

bool Gipfeli::InternalRawUncompressStream(
    Source* source, Sink* sink,
    size_t uncompressed_length) {
  char scratch[64];
  const char* buf;
  char* ip;

  // See if we can get a contigous output buffer
  char c;
  size_t output_fragment_size;
  char* output_fragment = sink->GetAppendBuffer(
      1, uncompressed_length, &c, 1, &output_fragment_size);

  Writer writer;
  if (output_fragment_size >= uncompressed_length) {
    writer.Initialize(sink, output_fragment, uncompressed_length);
  } else {
    writer.Initialize(sink, NULL, uncompressed_length);
  }

  Reader reader(source);

  if (decompressor_state_ == NULL)
    decompressor_state_ = new DecompressorState();

  while (!reader.Eof()) {
    uint32 commands_size;
    if (PREDICT_FALSE(!reader.Read16(&commands_size))) {
      return writer.OnError();
    }

    uint32* commands = decompressor_state_->ReallocateCommands(commands_size);
    if (PREDICT_FALSE(!DecompressCommandsStream(
            &reader, commands_size, commands))) {
      return writer.OnError();
    }

    if (PREDICT_FALSE((buf = reader.Read(scratch, 4)) == NULL)) {
      return writer.OnError();
    }

    uint32 upper = 0;
    for (int i = 0; i < 4; ++i) {
      uint8 val = (uint8)(buf[i]);
      upper <<= 8;
      upper += val;
    }

    uint8 convert_6bit[32] = { 0 };
    uint8 convert_8bit[64] = { 0 };

    // We assume Gipfeli compression is one block at a time and each
    // block starts with a command set. So this is a good point to grow
    // the block.
    if (commands_size) {
      writer.MaybeFinishBlock();
    }

    if (upper == 0) {
      // No Entropy Coding.
      for (int i = 0; i < commands_size; i++) {
        if (!CommandIsCopy(commands[i])) {
          if (PREDICT_FALSE(!writer.CheckAvailable(commands[i]))) {
            return writer.OnError();
          }
          if (PREDICT_FALSE(!reader.AppendTo(&writer, commands[i]))) {
            return writer.OnError();
          }
        } else {
          size_t len = CommandCopyLength(commands[i]);
          size_t offset = CommandCopyOffset(commands[i]);
          if (PREDICT_FALSE(!writer.AppendFromSelf(offset, len))) {
            return writer.OnError();
          }
        }
      }
    } else {
      // Entropy Coding.
      if (PREDICT_FALSE((ip = const_cast<char*>(reader.Read(
          scratch, Bits::CountOnes(upper) + 12))) == NULL)) {
        return writer.OnError();
      }
      if (DecompressMask(upper, ip, convert_6bit, convert_8bit)
          == NULL) {
        return writer.OnError();
      }
      BitStreamReader bits(&reader);
      for (int i = 0; i < commands_size; i++) {
        if (!CommandIsCopy(commands[i])) {
          if (PREDICT_FALSE(!writer.CheckAvailable(commands[i]))) {
            return writer.OnError();
          }
          for (int j = 0; j < commands[i]; j++) {
            uint32 val = bits.Read(6);
            if (PREDICT_TRUE(val < 32)) {
              writer.AppendByte(convert_6bit[val]);
            } else if (val >= 48) {
              writer.AppendByte(((val - 48) << 4) + bits.Read(4));
            } else {
                val = ((val - 32) << 2) + bits.Read(2);
                writer.AppendByte(convert_8bit[val]);
            }
          }
        } else {
          size_t len = CommandCopyLength(commands[i]);
          size_t offset = CommandCopyOffset(commands[i]);
          if (PREDICT_FALSE(!writer.AppendFromSelf(offset, len))) {
            return writer.OnError();
          }
        }
      }
      if (PREDICT_FALSE(bits.error())) {
        return writer.OnError();
      }
    }
  }

  writer.Flush();
  return writer.CheckLength();
}

}  // namespace gipfeli
}  // namespace compression
}  // namespace util
