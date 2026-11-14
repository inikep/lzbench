#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <core/common.hpp>


namespace rapidgzip
{
/**
 * Should get block offsets and decoded sizes and will do conversions between decoded and encoded offsets!
 *
 * The idea is that at first any forward seeking should be done using read calls and the read call will
 * push all block information to the BlockMap.
 *
 * This class expects @ref push to be called with monotonically increasing arguments.
 */
class BlockMap
{
public:
    struct BlockInfo
    {
    public:
        [[nodiscard]] bool
        contains( size_t dataOffset ) const
        {
            return ( decodedOffsetInBytes <= dataOffset ) && ( dataOffset < decodedOffsetInBytes + decodedSizeInBytes );
        }

        friend std::ostream&
        operator<<( std::ostream&    out,
                    const BlockInfo& blockInfo )
        {
            out << "BlockInfo{ blockIndex: " << blockInfo.blockIndex
                << ", encodedOffsetInBits: " << formatBits( blockInfo.encodedOffsetInBits )
                << ", encodedSizeInBits: " << formatBits( blockInfo.encodedSizeInBits )
                << ", decodedOffsetInBytes: " << formatBytes( blockInfo.decodedOffsetInBytes )
                << ", decodedSizeInBytes: " << formatBytes( blockInfo.decodedSizeInBytes )
                << " }";
            return out;
        };

    public:
        /** Each block in the stream will be given an increasing index number. */
        size_t blockIndex{ 0 };
        size_t encodedOffsetInBits{ 0 };
        size_t encodedSizeInBits{ 0 };
        size_t decodedOffsetInBytes{ 0 };
        size_t decodedSizeInBytes{ 0 };
    };

    using BlockOffsets = std::vector<std::pair</* encoded offset in bits */ size_t,
                                               /* decoded offset in bytes */ size_t> >;
    using BlockOffsetCallback = std::function<void ( size_t encodedOffset, size_t decodedOffset )>;

public:
    BlockMap() = default;

    /**
     * @return decoded offset in bytes, i.e., the sum of all previously decoded block data.
     */
    size_t
    push( size_t encodedBlockOffset,
          size_t encodedSize,
          size_t decodedSize )
    {
        std::optional<std::pair<size_t, size_t> > inserted;
        BlockOffsetCallback callback;
        {
            const std::scoped_lock lock( m_mutex );

            if ( m_finalized ) {
                throw std::invalid_argument( "May not insert into finalized block map!" );
            }

            std::optional<size_t> decodedOffset;
            if ( m_blockToDataOffsets.empty() ) {
                decodedOffset = 0;
            } else if ( encodedBlockOffset > m_blockToDataOffsets.back().first ) {
                decodedOffset = m_blockToDataOffsets.back().second + m_lastBlockDecodedSize;
            }

            /* If successive value or empty, then simply append */
            if ( decodedOffset ) {
                m_blockToDataOffsets.emplace_back( encodedBlockOffset, *decodedOffset );
                if ( decodedSize == 0 ) {
                    m_eosBlocks.emplace_back( encodedBlockOffset );
                }
                m_lastBlockDecodedSize = decodedSize;
                m_lastBlockEncodedSize = encodedSize;
                inserted = std::make_pair( encodedBlockOffset, *decodedOffset );
                callback = m_blockOffsetCallback;
            } else {
                /* Generally, block inserted offsets should always be increasing!
                 * But do ignore duplicates after confirming that there is no data inconsistency. */
                const auto match = std::lower_bound(
                    m_blockToDataOffsets.begin(), m_blockToDataOffsets.end(), std::make_pair( encodedBlockOffset, 0 ),
                    [] ( const auto& a, const auto& b ) { return a.first < b.first; } );

                if ( ( match == m_blockToDataOffsets.end() ) || ( match->first != encodedBlockOffset ) ) {
                    throw std::invalid_argument( "Inserted block offsets should be strictly increasing!" );
                }

                if ( std::next( match ) == m_blockToDataOffsets.end() ) {
                    throw std::logic_error( "In this case, the new block should already have been appended above!" );
                }

                const auto impliedDecodedSize = std::next( match )->second - match->second;
                if ( impliedDecodedSize != decodedSize ) {
                    throw std::invalid_argument( "Got duplicate block offset with inconsistent size!" );
                }

                /* Quietly ignore duplicate insertions. Note that match->first == encodedBlockOffset. */
                return match->second;
            }
        }

        if ( inserted && callback ) {
            callback( inserted->first, inserted->second );
        }
        return inserted->second;
    }

    /**
     * Returns the block containing the given data offset. May return a block which does not contain the given
     * offset. In that case it will be the last block.
     */
    [[nodiscard]] BlockInfo
    findDataOffset( size_t dataOffset ) const
    {
        const std::scoped_lock lock( m_mutex );

        /* find offset from map (key and values should be sorted in ascending order, so we can bisect!) */
        const auto blockOffset = std::lower_bound(
            m_blockToDataOffsets.rbegin(), m_blockToDataOffsets.rend(), std::make_pair( 0, dataOffset ),
            [] ( std::pair<size_t, size_t> a, std::pair<size_t, size_t> b ) { return a.second > b.second; } );

        if ( blockOffset == m_blockToDataOffsets.rend() ) {
            return {};
        }

        if ( dataOffset < blockOffset->second ) {
            throw std::logic_error( "Algorithm for finding the block to an offset is faulty!" );
        }

        return get( blockOffset );
    }

    [[nodiscard]] std::optional<BlockInfo>
    getEncodedOffset( size_t encodedOffsetInBits ) const
    {
        const std::scoped_lock lock( m_mutex );

        /* find offset from map (key and values should be sorted in ascending order, so we can bisect!) */
        const auto blockOffset = std::lower_bound(
            m_blockToDataOffsets.rbegin(), m_blockToDataOffsets.rend(), std::make_pair( encodedOffsetInBits, 0 ),
            [] ( std::pair<size_t, size_t> a, std::pair<size_t, size_t> b ) { return a.first > b.first; } );

        if ( ( blockOffset == m_blockToDataOffsets.rend() ) || ( blockOffset->first != encodedOffsetInBits ) ) {
            return std::nullopt;
        }

        return get( blockOffset );
    }

    void
    setBlockOffsetCallback( BlockOffsetCallback callback )
    {
        const std::scoped_lock lock( m_mutex );
        m_blockOffsetCallback = std::move( callback );
    }

    /**
     * Returns number of non-EOS blocks. This is necessary to have a number in sync with BlockFinder,
     * which does not find EOS blocks!
     */
    [[nodiscard]] size_t
    dataBlockCount() const
    {
        const std::scoped_lock lock( m_mutex );
        return m_blockToDataOffsets.size() - m_eosBlocks.size();
    }

    void
    finalize()
    {
        std::optional<std::pair<size_t, size_t> > added;
        BlockOffsetCallback callback;
        {
            const std::scoped_lock lock( m_mutex );

            if ( m_finalized ) {
                return;
            }

            /* Add last empty block if it does not already exist in order to imply the size of the last real block. */
            if ( m_blockToDataOffsets.empty() ) {
                assert( m_lastBlockEncodedSize == 0 );
                assert( m_lastBlockDecodedSize == 0 );
                m_blockToDataOffsets.emplace_back( m_lastBlockEncodedSize, m_lastBlockDecodedSize );
                added = m_blockToDataOffsets.back();
            } else if ( ( m_lastBlockEncodedSize != 0 ) || ( m_lastBlockDecodedSize != 0 ) ) {
                const auto& [lastEncodedOffset, lastDecodedOffset] = m_blockToDataOffsets.back();
                m_blockToDataOffsets.emplace_back( lastEncodedOffset + m_lastBlockEncodedSize,
                                                   lastDecodedOffset + m_lastBlockDecodedSize );
                added = m_blockToDataOffsets.back();
            }

            m_lastBlockEncodedSize = 0;
            m_lastBlockDecodedSize = 0;

            m_finalized = true;
            callback = m_blockOffsetCallback;
        }

        if ( added && callback ) {
            callback( added->first, added->second );
        }
    }

    void
    clear()
    {
        const std::scoped_lock lock( m_mutex );
        m_blockToDataOffsets.clear();
        m_eosBlocks.clear();
        m_lastBlockEncodedSize = 0;
        m_lastBlockDecodedSize = 0;
        m_finalized = false;
    }

    [[nodiscard]] bool
    finalized() const
    {
        const std::scoped_lock lock( m_mutex );
        return m_finalized;
    }

    void
    setBlockOffsets( std::map<size_t, size_t> const& blockOffsets )
    {
        const std::scoped_lock lock( m_mutex );

        m_blockToDataOffsets.assign( blockOffsets.begin(), blockOffsets.end() );
        m_lastBlockEncodedSize = 0;
        m_lastBlockDecodedSize = 0;

        /* Find EOS blocks in map. */
        m_eosBlocks.clear();
        for ( auto it = m_blockToDataOffsets.begin(), nit = std::next( m_blockToDataOffsets.begin() );
              nit != m_blockToDataOffsets.end(); ++it, ++nit )
        {
            /* Only push blocks with no data, i.e., EOS blocks. */
            if ( it->second == nit->second ) {
                m_eosBlocks.push_back( it->first );
            }
        }
        /* Last block is assumed to be EOS. */
        m_eosBlocks.push_back( m_blockToDataOffsets.back().first );

        m_finalized = true;
    }

    [[nodiscard]] std::map<size_t, size_t>
    blockOffsets() const
    {
        const std::scoped_lock lock( m_mutex );

        return { m_blockToDataOffsets.begin(), m_blockToDataOffsets.end() };
    }

    [[nodiscard]] std::pair<size_t, size_t>
    back() const
    {
        const std::scoped_lock lock( m_mutex );

        if ( m_blockToDataOffsets.empty() ) {
            throw std::out_of_range( "Can not return last element of empty block map!" );
        }
        return m_blockToDataOffsets.back();
    }

    [[nodiscard]] bool
    empty() const
    {
        return m_blockToDataOffsets.empty();
    }

private:
    [[nodiscard]] BlockInfo
    get( const typename BlockOffsets::const_reverse_iterator& blockOffset ) const
    {
        BlockInfo result;
        if ( blockOffset == m_blockToDataOffsets.rend() ) {
            return result;
        }

        result.encodedOffsetInBits = blockOffset->first;
        result.decodedOffsetInBytes = blockOffset->second;
        result.blockIndex = std::distance( blockOffset, m_blockToDataOffsets.rend() ) - 1;

        if ( blockOffset == m_blockToDataOffsets.rbegin() ) {
            result.decodedSizeInBytes = m_lastBlockDecodedSize;
            result.encodedSizeInBits = m_lastBlockEncodedSize;
        } else {
            const auto higherBlock = std::prev( /* reverse! */ blockOffset );
            if ( higherBlock->second < blockOffset->second ) {
                throw std::logic_error( "Data offsets are not monotonically increasing!" );
            }
            result.decodedSizeInBytes = higherBlock->second - blockOffset->second;
            result.encodedSizeInBits = higherBlock->first - blockOffset->first;
        }

        return result;
    }

private:
    mutable std::mutex m_mutex;

    /** If complete, the last block will be of size 0 and indicate the end of stream! */
    BlockOffsets m_blockToDataOffsets;
    std::vector<size_t> m_eosBlocks;
    BlockOffsetCallback m_blockOffsetCallback;
    bool m_finalized{ false };

    size_t m_lastBlockEncodedSize{ 0 };  /**< Encoded block size of m_blockToDataOffsets.rbegin() */
    size_t m_lastBlockDecodedSize{ 0 };  /**< Decoded block size of m_blockToDataOffsets.rbegin() */
};
}  // namespace rapidgzip
