#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string_view>

namespace bzip2
{
constexpr auto MAGIC_BITS_BLOCK = 0x314159265359ULL;  /* bcd(pi) */
constexpr auto MAGIC_BITS_EOS = 0x177245385090ULL;    /* bcd(sqrt(pi)) */
constexpr auto MAGIC_BITS_SIZE = 48;
constexpr std::string_view MAGIC_BYTES_BZ2 = "BZh";

/* Constants for huffman coding */
static constexpr uint32_t MAX_GROUPS = 6;
static constexpr int GROUP_SIZE = 50;       /* 64 would have been more efficient */
static constexpr int MAX_HUFCODE_BITS = 20; /* Longest huffman code allowed */
static constexpr size_t MAX_SYMBOLS = 258;  /* 256 literals + RUNA + RUNB */
static constexpr uint16_t SYMBOL_RUNA = 0;
static constexpr uint16_t SYMBOL_RUNB = 1;

/* bzip2 levels 1-9 use blocks of at most 100000 to 900000 bytes before the BWT. */
static constexpr size_t MAX_BLOCK_SIZE = 900000;

/**
 * @return Whether @p offsets is the canonical single-entry index of an empty bzip2 stream.
 *
 * An empty stream has no data block, so its only block entry is the EOS block. Callers may use either the
 * EOS bit offset or zero, which means "look after the stream header". The decoded offset must be zero;
 * otherwise this is a truncated index that starts at a later EOS, which is not an empty stream.
 */
template<typename BitReader>
[[nodiscard]] inline bool
isEmptyBzip2BlockMap( const std::map<std::size_t, std::size_t>& offsets,
                      BitReader&                                bitReader )
{
    if ( offsets.size() != 1 ) {
        return false;
    }
    const auto& [compressedOffset, decodedOffset] = *offsets.begin();
    if ( ( decodedOffset != 0 ) || !bitReader.seekable() ) {
        return false;
    }

    const auto compressedBits = bitReader.size();
    if ( !compressedBits ) {
        return false;
    }
    if ( *compressedBits == 0 ) {
        return compressedOffset == 0;
    }

    const auto magicOffset = compressedOffset == 0 ? std::size_t{ 32 } : compressedOffset;
    if ( magicOffset + MAGIC_BITS_SIZE > *compressedBits ) {
        return false;
    }

    const auto oldPosition = bitReader.tell();
    bitReader.seek( static_cast<long long int>( magicOffset ) );
    const auto firstMagic = bitReader.read( MAGIC_BITS_SIZE );
    bitReader.seek( static_cast<long long int>( oldPosition ) );
    return firstMagic == MAGIC_BITS_EOS;
}


/**
 * Validate block offsets before importing them.
 *
 * The last entry is the end sentinel and must be preceded by an EOS block. Otherwise, the index is incomplete,
 * for example because it was exported for only a decoded range. Accepting it would silently treat the last data
 * block as EOS and truncate the output. The first entry must start at decoded offset zero so that the beginning
 * of the stream remains accessible.
 */
template<typename BitReader>
inline void
validateBzip2BlockOffsets( const std::map<std::size_t, std::size_t>& offsets,
                           BitReader&                                bitReader )
{
    const auto emptyStream = isEmptyBzip2BlockMap( offsets, bitReader );
    if ( ( offsets.size() < 2 ) && !emptyStream ) {
        throw std::invalid_argument( "Block offset map must contain at least one valid block and one EOS block!" );
    }
    if ( offsets.empty() || ( offsets.begin()->second != 0 ) ) {
        throw std::invalid_argument( "The first block offset must map compressed offset 0 to decoded offset 0!" );
    }
    for ( auto it = std::next( offsets.begin() ); it != offsets.end(); ++it ) {
        if ( it->second < std::prev( it )->second ) {
            throw std::invalid_argument( "Decoded block offsets must be non-decreasing!" );
        }
    }

    if ( bitReader.seekable() && ( offsets.size() >= 2 ) ) {
        const auto eosOffsetInBits = std::prev( offsets.end(), 2 )->first;
        const auto oldPosition = bitReader.tell();
        bool eosMagic = false;
        try {
            bitReader.seek( static_cast<long long int>( eosOffsetInBits ) );
            eosMagic = bitReader.read( MAGIC_BITS_SIZE ) == MAGIC_BITS_EOS;
        } catch ( const std::exception& ) {
            eosMagic = false;
        }
        bitReader.seek( static_cast<long long int>( oldPosition ) );
        if ( !eosMagic ) {
            throw std::invalid_argument( "The block offset index is incomplete: the last entry is not "
                                         "preceded by an EOS block!" );
        }

        /* An index for only the first of several concatenated streams also ends with a valid EOS. Check whether
         * a further stream starts at the sentinel so that such an index is not accepted as a complete file. */
        const auto sentinelOffsetInBits = std::prev( offsets.end() )->first;
        const auto sizeInBits = bitReader.size();
        bool followsAnotherStream = false;
        if ( sizeInBits && ( sentinelOffsetInBits <= *sizeInBits )
             && ( *sizeInBits - sentinelOffsetInBits >= 32 ) ) {
            try {
                bitReader.seek( static_cast<long long int>( sentinelOffsetInBits ) );
                constexpr auto BZH = ( uint32_t( 'B' ) << 16U ) | ( uint32_t( 'Z' ) << 8U ) | uint32_t( 'h' );
                const auto header = bitReader.read( 32 );
                const auto level = header & 0xFFU;
                followsAnotherStream = ( ( header >> 8U ) == BZH )
                                       && ( level >= static_cast<uint32_t>( '1' ) )
                                       && ( level <= static_cast<uint32_t>( '9' ) );
            } catch ( const std::exception& ) {
                followsAnotherStream = false;
            }
        }
        bitReader.seek( static_cast<long long int>( oldPosition ) );
        if ( followsAnotherStream ) {
            throw std::invalid_argument( "The block offset index is incomplete: another bzip2 stream "
                                         "follows the last indexed one!" );
        }
    }
}
}  // namespace bzip2
