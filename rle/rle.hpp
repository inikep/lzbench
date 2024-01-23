#ifndef RLE_H
#define RLE_H

#include <cstddef>
#include <cstdint>

namespace rle {

// #define RLE_WORD_SIZE_16
// 16-bits run-length word allows for very long sequences,
// but is also very inefficient if the run-lengths are generally
// short. Byte-size words are used if this is not defined.
//
#ifdef RLE_WORD_SIZE_16
using RleWord = std::uint16_t;
constexpr RleWord max_run_length = RleWord(0xFFFF); // Max run length: 65535 => 4 bytes.
#else                                               // !RLE_WORD_SIZE_16
using RleWord = std::uint8_t;
constexpr RleWord max_run_length = RleWord(0xFF); // Max run length: 255 => 2 bytes.
#endif                                              // RLE_WORD_SIZE_16

} // namespace rle

std::size_t rle_compress(const std::uint8_t *input, const std::size_t in_size_bytes, std::uint8_t *output,
                                const std::size_t out_size_bytes);


std::size_t rle_decompress(const std::uint8_t *input, const std::size_t in_size_bytes, std::uint8_t *output,
                                const std::size_t out_size_bytes);


#endif // RLE_H
