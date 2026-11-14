#pragma once

#ifdef LIBRAPIDARCHIVE_WITH_BZIP2_COMPRESSION

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <bzlib.h>

#include <core/FileUtils.hpp>
#include <core/ThreadPool.hpp>
#include <filereader/FileReader.hpp>
#include <indexed_bzip2/Bzip2Format.hpp>


namespace indexed_bzip2
{
struct Bzip2CompressionResult
{
    uint64_t uncompressedBytes{ 0 };
    uint64_t compressedBytes{ 0 };
};


/**
 * Compresses one independent bzip2 stream. The result is a complete stream with its own header and footer,
 * so complete streams from multiple workers can be concatenated.
 */
[[nodiscard]] inline std::vector<uint8_t>
compressBzip2Stream( const uint8_t* const input,
                     const size_t         size,
                     const int            level )
{
    if ( ( level < 1 ) || ( level > 9 ) ) {
        throw std::invalid_argument( "bzip2 compression level must be in the range 1..9." );
    }

    std::vector<uint8_t> compressed( size + size / 100U + 600U );
    auto compressedSize = static_cast<unsigned int>( compressed.size() );
    /* libbz2 rejects a null source pointer even for an empty input, so provide a valid dummy address. */
    static constexpr char EMPTY_INPUT = 0;
    const auto* const source = size > 0 ? reinterpret_cast<const char*>( input ) : &EMPTY_INPUT;
    const auto result = BZ2_bzBuffToBuffCompress(
        reinterpret_cast<char*>( compressed.data() ), &compressedSize,
        const_cast<char*>( source ), static_cast<unsigned int>( size ),
        level, /* verbosity */ 0, /* workFactor */ 0 );
    if ( result != BZ_OK ) {
        throw std::runtime_error( "bzip2 compression failed with error code " + std::to_string( result ) );
    }
    compressed.resize( compressedSize );
    return compressed;
}


/**
 * @return the @p bitCount bits at @p bitOffset as the most significant bits of the result.
 */
[[nodiscard]] inline uint64_t
readBzip2Bits( const std::vector<uint8_t>& compressed,
               const size_t                bitOffset,
               const size_t                bitCount )
{
    uint64_t value = 0;
    for ( size_t i = 0; i < bitCount; ++i ) {
        const auto bit = bitOffset + i;
        value = ( value << 1U ) | ( ( compressed[bit / 8U] >> ( 7U - ( bit % 8U ) ) ) & 1U );
    }
    return value;
}


/**
 * @return The bit offset of the stream's EOS block magic, or nullopt if the stream does not end with an
 *         EOS block followed by a 32-bit stream CRC and at most seven padding bits.
 *
 * An EOS anywhere else is either a false positive inside block data or the end of an earlier concatenated
 * stream. Searching only the last 39 bits avoids mistaking those for the final EOS.
 */
[[nodiscard]] inline std::optional<size_t>
findBzip2EndOfStreamBitOffset( const std::vector<uint8_t>& compressed )
{
    constexpr auto TRAILER_BITS = bzip2::MAGIC_BITS_SIZE + 32U;
    const auto bitCount = compressed.size() * 8U;
    if ( bitCount < 32U + TRAILER_BITS ) {
        return std::nullopt;
    }

    const auto latest = bitCount - TRAILER_BITS;
    const auto earliest = latest >= 7U ? latest - 7U : 0U;
    for ( auto bit = latest + 1U; bit > earliest; ) {
        --bit;
        if ( readBzip2Bits( compressed, bit, bzip2::MAGIC_BITS_SIZE ) != bzip2::MAGIC_BITS_EOS ) {
            continue;
        }

        bool paddingIsZero = true;
        for ( auto paddingBit = bit + TRAILER_BITS; paddingBit < bitCount; ++paddingBit ) {
            if ( readBzip2Bits( compressed, paddingBit, 1 ) != 0 ) {
                paddingIsZero = false;
                break;
            }
        }
        if ( paddingIsZero ) {
            return bit;
        }
    }
    return std::nullopt;
}


/**
 * @return Whether a complete bzip2 stream header starts after the EOS block and its 32-bit CRC at @p eosBit.
 *
 * After an EOS block, the 32-bit combined stream CRC is followed by zero padding to the next byte boundary.
 * A real concatenated stream starts there with the "BZh" bytes. This makes stream boundaries much less likely
 * to be confused with the EOS magic occurring randomly inside block data.
 */
[[nodiscard]] inline bool
isBzip2StreamBoundary( const std::vector<uint8_t>& compressed,
                       const size_t                eosBit )
{
    const auto bitCount = compressed.size() * 8U;
    const auto crcEnd = eosBit + bzip2::MAGIC_BITS_SIZE + 32U;
    const auto nextStreamBit = ( crcEnd + 7U ) / 8U * 8U;
    if ( nextStreamBit + 8U * bzip2::MAGIC_BYTES_BZ2.size() > bitCount ) {
        return false;
    }

    const auto header = readBzip2Bits( compressed, nextStreamBit, 8U * bzip2::MAGIC_BYTES_BZ2.size() );
    constexpr auto HEADER_MASK = 0xFFFFFFULL;
    return ( header >> 8U ) == HEADER_MASK
           && ( ( header & 0xFFU ) >= static_cast<uint64_t>( '1' ) )
           && ( ( header & 0xFFU ) <= static_cast<uint64_t>( '9' ) );
}


struct Bzip2ChunkScanResult
{
    /** Compressed bit offset of the first data block, if the chunk is not an empty stream. */
    std::optional<size_t> firstBlockBit;
    /** Compressed bit offset of the final EOS block. */
    std::optional<size_t> endOfStreamBit;
    /** Additional block or stream boundaries whose decoded offset cannot be derived from the compressed data. */
    std::vector<size_t> ambiguousBoundaryBits;
};


/**
 * Scans one complete in-memory bzip2 stream for block boundaries.
 *
 * The parallel compressor knows each worker chunk's decoded start and size, so the first data block and the
 * final EOS block can be mapped to exact decoded offsets. Additional blocks inside the chunk are visible as
 * magic sequences, but their decoded sizes are only known after decoding. They are reported as ambiguous
 * instead of being guessed so that the resulting index never contains a wrong decoded offset.
 *
 * A concatenated stream boundary is only accepted when the EOS is followed by a full "BZh<level>" header,
 * which keeps false positives from EOS magic inside block data out of the warning.
 */
[[nodiscard]] inline Bzip2ChunkScanResult
scanBzip2ChunkOffsets( const std::vector<uint8_t>& compressed )
{
    /* The stream header plus the first 48-bit magic must fit before readBzip2Bits may access the bytes. */
    constexpr auto HEADER_AND_BLOCK_MAGIC_BYTES = ( 32U + bzip2::MAGIC_BITS_SIZE + CHAR_BIT - 1U ) / CHAR_BIT;
    if ( ( compressed.size() < HEADER_AND_BLOCK_MAGIC_BYTES )
         || ( std::memcmp( compressed.data(), bzip2::MAGIC_BYTES_BZ2.data(),
                           bzip2::MAGIC_BYTES_BZ2.size() ) != 0 ) ) {
        throw std::logic_error( "Compressed bzip2 chunk does not start with a stream header." );
    }

    Bzip2ChunkScanResult result;
    const auto firstMagic = readBzip2Bits( compressed, 32U, bzip2::MAGIC_BITS_SIZE );
    if ( firstMagic == bzip2::MAGIC_BITS_EOS ) {
        result.endOfStreamBit = 32U;
        return result;
    }
    if ( firstMagic != bzip2::MAGIC_BITS_BLOCK ) {
        throw std::logic_error( "Compressed bzip2 chunk does not start with a block magic." );
    }
    result.firstBlockBit = 32U;

    result.endOfStreamBit = findBzip2EndOfStreamBitOffset( compressed );
    if ( !result.endOfStreamBit ) {
        throw std::logic_error( "Compressed bzip2 chunk does not end with an EOS block." );
    }

    constexpr auto MAGIC_MASK = ( 1ULL << bzip2::MAGIC_BITS_SIZE ) - 1U;
    const auto bitCount = compressed.size() * 8U;
    auto window = readBzip2Bits( compressed, 0U, bzip2::MAGIC_BITS_SIZE );
    for ( size_t bit = 0; bit + bzip2::MAGIC_BITS_SIZE <= bitCount; ++bit ) {
        if ( bit > 0 ) {
            const auto nextBit = bit + bzip2::MAGIC_BITS_SIZE - 1U;
            window = ( ( window << 1U ) & MAGIC_MASK )
                     | ( ( compressed[nextBit / 8U] >> ( 7U - ( nextBit % 8U ) ) ) & 1U );
        }

        if ( ( window == bzip2::MAGIC_BITS_BLOCK )
             && ( bit != *result.firstBlockBit )
             && ( bit < *result.endOfStreamBit ) ) {
            result.ambiguousBoundaryBits.push_back( bit );
        } else if ( ( window == bzip2::MAGIC_BITS_EOS )
                    && ( bit != *result.endOfStreamBit )
                    && isBzip2StreamBoundary( compressed, bit ) ) {
            result.ambiguousBoundaryBits.push_back( bit );
        }
    }

    return result;
}


/**
 * Compresses @p inputFile into independent bzip2 streams in parallel and writes them in input order.
 *
 * The bzip2 level's block-size limit applies to the RLE1-encoded data, not to the raw input. RLE1's
 * worst case is a run of exactly four identical bytes:
 *
 *     raw:  A A A A
 *     RLE1: A A A A <count=0>
 *
 * which expands four bytes to five, i.e., at most 5/4. libbz2 flushes at level * 100000 - 19 encoded
 * bytes, so keep the raw chunk at 4/5 of that limit. This avoids a full block followed by a tiny tail
 * block when the expansion crosses the limit.
 *
 * @param onBlockOffset Called with each known (compressed bit offset, decoded byte offset) pair in output
 *        order while the streams are written. The compressed input is scanned in memory, so exporting the
 *        index works for non-seekable outputs such as stdout and never requires a second I/O pass. Without
 *        this callback no scan is done. The scan relies on the 80% cap for exact per-block offsets; chunks
 *        that still contain additional blocks are reported and those inner offsets are omitted.
 */
[[nodiscard]] inline Bzip2CompressionResult
compressBzip2Parallel( rapidgzip::FileReader&                                    inputFile,
                       const std::function<void( const std::vector<uint8_t>& )>& writeData,
                       const int                                                 level,
                       const size_t                                              parallelism,
                       const std::function<void( size_t, size_t )>&              onBlockOffset = {} )
{
    if ( ( level < 1 ) || ( level > 9 ) ) {
        throw std::invalid_argument( "bzip2 compression level must be in the range 1..9." );
    }

    const auto blockSize = ( static_cast<size_t>( level ) * 100000U - 19U ) * 4U / 5U;
    rapidgzip::ThreadPool pool( std::max<size_t>( 1, parallelism ) );
    const auto maxInFlight = std::max<size_t>( 1, pool.capacity() * 2 );
    struct PendingStream
    {
        std::future<std::vector<uint8_t> > future;
        size_t decodedOffset{ 0 };
        size_t uncompressedSize{ 0 };
    };
    std::deque<PendingStream> pending;
    Bzip2CompressionResult result;

    /* Scan the compressed chunk while it is still in memory. This records the first data block and the final
     * EOS block, whose decoded offsets are known from the chunk boundaries, without reopening the output or
     * relying on libbz2's block flushing.
     *
     * ParallelBZ2Reader cannot recover the decoded offset of an inner block from a coarser imported entry, so
     * a partial index would fail during parallel decoding. Reject such chunks instead of writing a misleading
     * index. */
    const auto writePending =
        [&writeData, &result, &onBlockOffset] ( PendingStream& pendingStream )
        {
            const auto streamStartBits = result.compressedBytes * 8U;
            const auto data = pendingStream.future.get();
            if ( onBlockOffset ) {
                const auto scan = scanBzip2ChunkOffsets( data );
                if ( !scan.ambiguousBoundaryBits.empty() ) {
                    std::stringstream msg;
                    msg << "Compressed bzip2 chunk at byte offset " << ( streamStartBits / 8U )
                        << " contains an additional block boundary at compressed bit offset "
                        << ( streamStartBits + scan.ambiguousBoundaryBits.front() )
                        << ". An index streamed from this chunk would omit inner block offsets "
                        << "and cannot be used for parallel decompression. Compress with a "
                        << "smaller chunk size or export the index after compression.";
                    throw std::runtime_error( std::move( msg ).str() );
                }
                if ( scan.firstBlockBit ) {
                    onBlockOffset( streamStartBits + *scan.firstBlockBit, pendingStream.decodedOffset );
                }
                if ( scan.endOfStreamBit ) {
                    onBlockOffset( streamStartBits + *scan.endOfStreamBit,
                                   pendingStream.decodedOffset + pendingStream.uncompressedSize );
                }
            }
            writeData( data );
            result.compressedBytes += data.size();
        };

    std::vector<uint8_t> block( blockSize );
    bool wroteStream{ false };
    while ( true ) {
        size_t bytesRead{ 0 };
        while ( bytesRead < block.size() ) {
            const auto nBytesRead = inputFile.read(
                reinterpret_cast<char*>( block.data() + bytesRead ), block.size() - bytesRead );
            if ( nBytesRead == 0 ) {
                if ( inputFile.fail() ) {
                    throw std::runtime_error( "Failed to read input for bzip2 compression!" );
                }
                break;
            }
            bytesRead += nBytesRead;
        }
        if ( bytesRead == 0 ) {
            break;
        }

        const auto decodedOffset = result.uncompressedBytes;
        result.uncompressedBytes += bytesRead;
        std::vector<uint8_t> data( block.begin(), block.begin() + bytesRead );
        pending.emplace_back(
            PendingStream{
                pool.submit(
                    [data = std::move( data ), level] () {
                        return compressBzip2Stream( data.data(), data.size(), level );
                    } ),
                decodedOffset,
                bytesRead,
            } );
        wroteStream = true;

        while ( pending.size() > maxInFlight ) {
            writePending( pending.front() );
            pending.pop_front();
        }

        if ( bytesRead < block.size() ) {
            break;
        }
    }

    while ( !pending.empty() ) {
        writePending( pending.front() );
        pending.pop_front();
    }

    /* An empty input still needs one valid bzip2 stream so that the output is a valid empty archive. */
    if ( !wroteStream ) {
        const auto emptyStream = compressBzip2Stream( nullptr, 0, level );
        if ( onBlockOffset ) {
            /* An empty stream has no data block; its first and only block is the EOS at bit 32. */
            const auto scan = scanBzip2ChunkOffsets( emptyStream );
            if ( scan.endOfStreamBit ) {
                onBlockOffset( *scan.endOfStreamBit, 0 );
            }
        }
        writeData( emptyStream );
        result.compressedBytes += emptyStream.size();
    }

    /* A scanned index ends with the physical file end, which bounds the last block. */
    if ( onBlockOffset ) {
        onBlockOffset( result.compressedBytes * 8U, result.uncompressedBytes );
    }

    return result;
}


/**
 * Writes the compressed stream to a file descriptor.
 */
[[nodiscard]] inline Bzip2CompressionResult
compressBzip2Parallel( rapidgzip::FileReader&                       inputFile,
                       const int                                    outputFileDescriptor,
                       const int                                    level,
                       const size_t                                 parallelism,
                       const std::function<void( size_t, size_t )>& onBlockOffset = {} )
{
    if ( outputFileDescriptor < 0 ) {
        throw std::invalid_argument( "A valid output file descriptor is required for bzip2 compression." );
    }

    return compressBzip2Parallel(
        inputFile,
        [outputFileDescriptor] ( const std::vector<uint8_t>& data ) {
            const auto errorCode = rapidgzip::writeAllToFd( outputFileDescriptor, data.data(), data.size() );
            if ( errorCode == EPIPE ) {
                throw rapidgzip::BrokenPipeException();
            }
            if ( errorCode != 0 ) {
                throw std::runtime_error( "Failed to write compressed data because of: "
                                          + std::string( strerror( errorCode ) )
                                          + " (" + std::to_string( errorCode ) + ")" );
            }
        },
        level,
        parallelism,
        onBlockOffset );
}
}  // namespace indexed_bzip2

#endif  // LIBRAPIDARCHIVE_WITH_BZIP2_COMPRESSION
