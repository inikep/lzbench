/*
 * indexed_bzip2 codec for lzbench.
 *
 * One lzbench entry covers both modes: -I1 selects the sequential BZ2Reader
 * and serial compressor, while -I N > 1 selects ParallelBZ2Reader and the
 * parallel chunk compressor just like the ibzip2 CLI shipped with indexed_bzip2.
 */

#include "codecs.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include <filereader/BufferView.hpp>
#include <indexed_bzip2/BZ2Reader.hpp>
#include <indexed_bzip2/ParallelBZ2Reader.hpp>

#ifdef LIBRAPIDARCHIVE_WITH_BZIP2_COMPRESSION
    #include <indexed_bzip2/Bzip2Compressor.hpp>
#endif

#ifndef BENCH_REMOVE_INDEXED_BZIP2

namespace
{
#ifdef LIBRAPIDARCHIVE_WITH_BZIP2_COMPRESSION
int64_t
compressWithIndexedBzip2( const void* input,
                          size_t      inputSize,
                          void*       output,
                          size_t      outputSize,
                          int         level,
                          int         parallelism )
{
    const auto* const bytes = static_cast<const uint8_t*>( input );
    auto* const outputBytes = static_cast<uint8_t*>( output );

    try {
        /* Keep -I1 as one stream so that its compressed size is byte-identical to bzip2. */
        if ( parallelism <= 1 ) {
            const auto compressed = indexed_bzip2::compressBzip2Stream( bytes, inputSize, level );
            if ( compressed.size() > outputSize ) {
                return -1;
            }
            std::memcpy( outputBytes, compressed.data(), compressed.size() );
            return static_cast<int64_t>( compressed.size() );
        }

        rapidgzip::BufferViewFileReader inputFile( bytes, inputSize );
        size_t totalSize = 0;
        const auto writeData =
            [outputBytes, outputSize, &totalSize] ( const std::vector<uint8_t>& data ) {
                if ( totalSize + data.size() > outputSize ) {
                    throw std::runtime_error( "Compressed output does not fit into the lzbench buffer." );
                }
                std::memcpy( outputBytes + totalSize, data.data(), data.size() );
                totalSize += data.size();
            };
        const auto result =
            indexed_bzip2::compressBzip2Parallel( inputFile, writeData, level, static_cast<size_t>( parallelism ) );
        return static_cast<int64_t>( result.compressedBytes );
    } catch ( ... ) {
        return -1;
    }
}
#endif

int64_t
decompressWithIndexedBzip2( const void* compressedData,
                            size_t      compressedSize,
                            void*       outputBuffer,
                            size_t      outputSize,
                            int         parallelism )
{
    if ( ( compressedData == nullptr ) || ( compressedSize == 0 )
         || ( ( outputBuffer == nullptr ) && ( outputSize > 0 ) ) ) {
        return -1;
    }

    try {
        std::unique_ptr<indexed_bzip2::BZ2ReaderInterface> reader;
        if ( parallelism > 1 ) {
            reader = std::make_unique<indexed_bzip2::ParallelBZ2Reader>(
                std::make_unique<rapidgzip::BufferViewFileReader>( compressedData, compressedSize ),
                static_cast<size_t>( parallelism ) );
        } else {
            reader = std::make_unique<indexed_bzip2::BZ2Reader>(
                std::make_unique<rapidgzip::BufferViewFileReader>( compressedData, compressedSize ) );
        }

        auto* const output = static_cast<char*>( outputBuffer );
        size_t totalDecoded = 0;
        while ( totalDecoded < outputSize ) {
            const auto nBytesRead = reader->read( output + totalDecoded, outputSize - totalDecoded );
            if ( nBytesRead == 0 ) {
                break;
            }
            totalDecoded += nBytesRead;
        }

        /* The extra read detects a too-small output buffer and makes the sequential reader parse the
         * end-of-stream block, which also verifies the stream CRC. */
        if ( totalDecoded == outputSize ) {
            char extraByte{};
            if ( reader->read( &extraByte, 1 ) != 0 ) {
                return -1;
            }
        }

        return static_cast<int64_t>( totalDecoded );
    } catch ( ... ) {
        return -1;
    }
}
}

int64_t
lzbench_indexed_bzip2_compress( char*            inbuf,
                                size_t           insize,
                                char*            outbuf,
                                size_t           outsize,
                                codec_options_t* codec_options )
{
#ifdef LIBRAPIDARCHIVE_WITH_BZIP2_COMPRESSION
    const int parallelism = codec_options->threads > 0 ? codec_options->threads : 1;
    return compressWithIndexedBzip2( inbuf, insize, outbuf, outsize, codec_options->level, parallelism );
#else
    (void)inbuf;
    (void)insize;
    (void)outbuf;
    (void)outsize;
    (void)codec_options;
    return -1;
#endif
}

int64_t
lzbench_indexed_bzip2_decompress( char*            inbuf,
                                  size_t           insize,
                                  char*            outbuf,
                                  size_t           outsize,
                                  codec_options_t* codec_options )
{
    const int parallelism = codec_options->threads > 0 ? codec_options->threads : 1;
    return decompressWithIndexedBzip2( inbuf, insize, outbuf, outsize, parallelism );
}

#endif  // BENCH_REMOVE_INDEXED_BZIP2
