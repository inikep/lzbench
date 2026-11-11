#pragma once

#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <core/AffinityHelpers.hpp>
#include <core/BlockFinder.hpp>
#include <core/BlockMap.hpp>
#include <core/common.hpp>
#include <core/ParallelBitStringFinder.hpp>
#include <filereader/FileReader.hpp>
#include <filereader/Shared.hpp>

#include "BZ2BlockFetcher.hpp"
#include "BZ2ReaderInterface.hpp"
#include "StreamCRCVerifier.hpp"
#include "bzip2.hpp"

#ifdef WITH_PYTHON_SUPPORT
    #include <filereader/Python.hpp>
    #include <filereader/Standard.hpp>
#endif


namespace indexed_bzip2
{
/**
 * @note Calls to this class are not thread-safe! Even though they use threads to evaluate them in parallel.
 */
class ParallelBZ2Reader final :
    public BZ2ReaderInterface
{
public:
    using BlockFetcher = BZ2BlockFetcher<rapidgzip::FetchingStrategy::FetchNextAdaptive>;
    using BlockFinder = typename BlockFetcher::BlockFinder;
    using BitReader = bzip2::BitReader;

public:
    /* Constructors */

    explicit
    ParallelBZ2Reader( rapidgzip::UniqueFileReader fileReader,
                       size_t                      parallelization = 0 ) :
        m_sharedFileReader( ensureSharedFileReader( std::move( fileReader ) ) ),
        m_fetcherParallelization( parallelization == 0 ? rapidgzip::availablePhysicalCores() : parallelization ),
        m_startBlockFinder(
            [&] ()
            {
                return std::make_shared<BlockFinder>(
                    std::make_unique<rapidgzip::ParallelBitStringFinder<bzip2::MAGIC_BITS_SIZE> >(
                        m_sharedFileReader->clone(),
                        bzip2::MAGIC_BITS_BLOCK,
                        m_finderParallelization
                ) );
            } )
    {
        if ( !m_bitReader.seekable() ) {
            throw std::invalid_argument( "Parallel BZ2 Reader will not work on non-seekable input like stdin (yet)!" );
        }
    }

#ifdef WITH_PYTHON_SUPPORT
    explicit
    ParallelBZ2Reader( int    fileDescriptor,
                       size_t parallelization = 0 ) :
        ParallelBZ2Reader( std::make_unique<rapidgzip::StandardFileReader>( fileDescriptor ), parallelization )
    {}

    explicit
    ParallelBZ2Reader( const std::string& filePath,
                       size_t             parallelization = 0 ) :
        ParallelBZ2Reader( std::make_unique<rapidgzip::StandardFileReader>( filePath ), parallelization )
    {}

    explicit
    ParallelBZ2Reader( PyObject* pythonObject,
                       size_t    parallelization = 0 ) :
        ParallelBZ2Reader( std::make_unique<rapidgzip::PythonFileReader>( pythonObject ), parallelization )
    {}
#endif

    /* FileReader overrides */

    [[nodiscard]] int
    fileno() const override
    {
        throw std::logic_error( "This is a virtual file object, which has no corresponding file descriptor!" );
    }

    [[nodiscard]] bool
    seekable() const override
    {
        return m_bitReader.seekable();
    }

    void
    close() override
    {
        m_blockFetcher.reset();
        m_blockFinder.reset();
        m_bitReader.close();
        m_sharedFileReader.reset();
    }

    [[nodiscard]] bool
    closed() const override
    {
        return m_bitReader.closed();
    }

    [[nodiscard]] bool
    eof() const override
    {
        return m_atEndOfFile;
    }

    [[nodiscard]] bool
    fail() const override
    {
        throw std::logic_error( "Not implemented!" );
    }

    [[nodiscard]] size_t
    tell() const override
    {
        if ( m_atEndOfFile ) {
            const auto fileSize = size();
            if ( !fileSize ) {
                throw std::logic_error( "When the file end has been reached, the block map should have been finalized "
                                        "and the file size should be available!" );
            }
            return *fileSize;
        }
        return m_currentPosition;
    }

    void
    clearerr() override
    {
        m_bitReader.clearerr();
        m_atEndOfFile = false;
        throw std::invalid_argument( "Not fully tested!" );
    }

    /* BZ2ReaderInterface overrides */

    using BZ2ReaderInterface::read;

    size_t
    read( const WriteFunctor& writeFunctor,
          const size_t        nBytesToRead = std::numeric_limits<size_t>::max() ) override
    {
        if ( closed() ) {
            throw std::invalid_argument( "You may not call read on closed ParallelBZ2Reader!" );
        }

        if ( eof() || ( nBytesToRead == 0 ) ) {
            return 0;
        }

        size_t nBytesDecoded = 0;
        while ( ( nBytesDecoded < nBytesToRead ) && !eof() ) {
            std::shared_ptr<BlockFetcher::BlockData> blockData;

        #ifdef WITH_PYTHON_SUPPORT
            rapidgzip::checkPythonSignalHandlers();
            const rapidgzip::ScopedGILUnlock unlockedGIL;
        #endif

            auto blockInfo = m_blockMap.findDataOffset( m_currentPosition );
            if ( !blockInfo.contains( m_currentPosition ) ) {
                /* Fetch new block for the first time and add information to block map. */
                const auto dataBlockIndex = m_blockMap.dataBlockCount();
                const auto encodedOffsetInBits = blockFinder().get( dataBlockIndex );
                if ( !encodedOffsetInBits ) {
                    /* The block finder only looks for data-block magic, so a file with a valid header and
                     * only an EOS block has no blocks. Validate that case instead of reporting any input
                     * without blocks, including a truncated header, as a successfully decoded empty stream. */
                    if ( m_blockMap.dataBlockCount() == 0 ) {
                        validateStreamWithoutBlocks();
                    }
                    m_blockMap.finalize();
                    m_atEndOfFile = true;
                    break;
                }

                blockData = blockFetcher().get( *encodedOffsetInBits, dataBlockIndex );
                const auto decodedOffsetInBytes = m_blockMap.push( blockData->encodedOffsetInBits,
                                                                   blockData->encodedSizeInBits,
                                                                   blockData->data.size() );
                m_streamCRCVerifier.appendBlock( decodedOffsetInBytes, blockData->data.size(),
                                                 blockData->calculatedCRC );

                /* Check whether the next block is an EOS block, which has a different magic byte string
                 * and therefore will not be found by the block finder! Such a block will span 48 + 32 + (0..7) bits.
                 * However, the last 0 to 7 bits are only padding and not needed! */
                if ( !blockData->isEndOfFile && blockData->nextBlockHeader ) {
                    const auto& nextBlockHeaderData = *blockData->nextBlockHeader;
                    if ( nextBlockHeaderData.isEndOfStreamBlock ) {
                        const auto existingEOS =
                            m_blockMap.getEncodedOffset( nextBlockHeaderData.encodedOffsetInBits );
                        const auto eosDecodedOffsetInBytes =
                            existingEOS
                            ? existingEOS->decodedOffsetInBytes
                            : m_blockMap.push( nextBlockHeaderData.encodedOffsetInBits,
                                               nextBlockHeaderData.encodedSizeInBits,
                                               0 );
                        m_streamCRCVerifier.finishStream( nextBlockHeaderData.encodedOffsetInBits,
                                                          eosDecodedOffsetInBytes,
                                                          nextBlockHeaderData.expectedCRC );

                        const auto nextStreamOffsetInBits = nextBlockHeaderData.encodedOffsetInBits +
                                                            nextBlockHeaderData.encodedSizeInBits;
                        if ( nextStreamOffsetInBits < m_bitReader.size() ) {
                            try
                            {
                                BitReader nextBzip2StreamBitReader( m_bitReader );
                                nextBzip2StreamBitReader.seekTo( nextStreamOffsetInBits );
                                bzip2::readBzip2Header( nextBzip2StreamBitReader );
                            } catch ( const std::exception& ) {
                                std::cerr << "[Warning] Trailing garbage after EOF ignored!\n";
                                /**
                                 * Stop reading the file here! 'bzip2-tests' comes with a test in which there actually
                                 * comes further valid bzip2 data after a gap. But that data, which the block finder will
                                 * find without problems, should not be read anymore!
                                 * The block finder already might have prefetched further values, so we need to truncate it!
                                 * @todo Maybe add an --ignore-invalid option like with tarfile's --ignore-zeros.
                                 */
                                m_blockFinder->finalize( m_blockMap.dataBlockCount() );
                            }
                        }
                    }
                }

                /* We could also directly continue but then the block would be refetched, and fetching quite complex. */
                blockInfo = m_blockMap.findDataOffset( m_currentPosition );
                if ( !blockInfo.contains( m_currentPosition ) ) {
                    continue;
                }
            } else {
                blockData = blockFetcher().get( blockInfo.encodedOffsetInBits );
                /* Imported offsets only provide the difference of two decoded offsets. If that does not match
                 * the actual block, copying from the block would silently shift all following output while the
                 * stream CRC is accumulated with the wrong offsets and still passes. */
                if ( blockInfo.decodedSizeInBytes != blockData->data.size() ) {
                    throw std::invalid_argument( "Imported block offsets do not match the decoded block size!" );
                }
                m_streamCRCVerifier.appendBlock( blockInfo.decodedOffsetInBytes, blockInfo.decodedSizeInBytes,
                                                 blockData->calculatedCRC );

                /* Imported block offsets already contain the EOS blocks, which findDataOffset does not visit.
                 * Verify an EOS directly following this block, including intermediate concatenated streams. */
                const auto nextBlockOffsetInBits = blockInfo.encodedOffsetInBits + blockInfo.encodedSizeInBits;
                const auto nextBlock = m_blockMap.getEncodedOffset( nextBlockOffsetInBits );
                if ( nextBlock && ( nextBlock->decodedSizeInBytes == 0 ) ) {
                    const auto eosHeader = blockFetcher().readBlockHeader( nextBlockOffsetInBits );
                    if ( eosHeader.isEndOfStreamBlock ) {
                        m_streamCRCVerifier.finishStream( nextBlockOffsetInBits,
                                                          nextBlock->decodedOffsetInBytes,
                                                          eosHeader.expectedCRC );
                    }
                }
            }

            /* Copy data from fetched block to output. */

            const auto offsetInBlock = m_currentPosition - blockInfo.decodedOffsetInBytes;

            if ( offsetInBlock >= blockData->data.size() ) {
                throw std::logic_error( "Block does not contain the requested offset even though it "
                                        "shouldn't be according to block map!" );
            }

            const auto nBytesToDecode = std::min( blockData->data.size() - offsetInBlock,
                                                  nBytesToRead - nBytesDecoded );
            if ( writeFunctor ) {
                writeFunctor( blockData->data.data() + offsetInBlock, nBytesToDecode );
            }

            nBytesDecoded += nBytesToDecode;
            m_currentPosition += nBytesToDecode;
        }

        return nBytesDecoded;
    }

    size_t
    seek( long long int offset,
          int           origin = SEEK_SET ) override
    {
        if ( closed() ) {
            throw std::invalid_argument( "You may not call seek on closed ParallelBZ2Reader!" );
        }

        if ( origin == SEEK_END ) {
            /* size() requires the block offsets to be available! */
            if ( !m_blockMap.finalized() ) {
                read();
            }
        }
        const auto positiveOffset = effectiveOffset( offset, origin );

        if ( positiveOffset == tell() ) {
            return positiveOffset;
        }

        /* Backward seeking is no problem at all! 'tell' may only return <= size()
         * as value meaning we are now < size() and therefore EOF can be cleared! */
        if ( positiveOffset < tell() ) {
            m_atEndOfFile = false;
            m_currentPosition = positiveOffset;
            return positiveOffset;
        }

        /* m_blockMap is only accessed by read and seek, which are not to be called from different threads,
         * so we do not have to lock it. */
        const auto blockInfo = m_blockMap.findDataOffset( positiveOffset );
        if ( positiveOffset < blockInfo.decodedOffsetInBytes ) {
            throw std::logic_error( "Block map returned unwanted block!" );
        }

        if ( blockInfo.contains( positiveOffset ) ) {
            m_atEndOfFile = false;
            m_currentPosition = positiveOffset;
            return tell();
        }

        if ( m_blockMap.finalized() ) {
            m_atEndOfFile = true;
            m_currentPosition = m_blockMap.back().second;
            return tell();
        }

        /* Jump to furthest known point as performance optimization. Note that even if that is right after
         * the last byte, i.e., offset == size(), then no eofbit is set even in ifstream! In ifstream you
         * can even seek to after the file end with no fail bits being set in my tests! */
        m_atEndOfFile = false;
        m_currentPosition = blockInfo.decodedOffsetInBytes + blockInfo.decodedSizeInBytes;
        read( -1, nullptr, positiveOffset - tell() );
        return tell();
    }

    /* BZip2 specific methods */

    /**
     * @return vectors of block data: offset in file, offset in decoded data
     *         (cumulative size of all prior decoded blocks).
     */
    [[nodiscard]] std::map<size_t, size_t>
    blockOffsets() override
    {
        if ( !m_blockMap.finalized() ) {
            read();
            if ( !m_blockMap.finalized() || !blockFinder().finalized() ) {
                throw std::logic_error( "Reading everything should have finalized the block map!" );
            }
        }

        return m_blockMap.blockOffsets();
    }

    // NOLINTBEGIN(misc-no-recursion)
    void
    setBlockOffsets( std::map<size_t, size_t> offsets ) override
    {
        if ( offsets.empty() ) {
            throw std::invalid_argument( "May not clear offsets. Construct a new ParallelBZ2Reader instead!" );
        }

        bzip2::validateBzip2BlockOffsets( offsets, m_bitReader );

        setBlockFinderOffsets( offsets );
        m_blockMap.setBlockOffsets( offsets );
    }

    /**
     * @return number of processed bits of compressed bzip2 input file stream
     * @note Bzip2 is block based and blocks are currently read fully, meaning that the granularity
     *       of the returned position is ~100-900kB. It's only useful for a rough estimate.
     */
    [[nodiscard]] size_t
    tellCompressed() const override
    {
        if ( m_blockMap.empty() ) {
            return 0;
        }
        const auto blockInfo = m_blockMap.findDataOffset( m_currentPosition );
        if ( blockInfo.contains( m_currentPosition ) ) {
            return blockInfo.encodedOffsetInBits;
        }
        return m_blockMap.back().first;
    }

    /**
     * Closes all threads and saves the work. They will be restarted when needed again, e.g., on seek or read.
     * This is intended for use with fusepy. You can start a ParallelBZ2Reader use it to create the block map
     * and print out user output and then you join all threads before FUSE forks the process. FUSE requires
     * threads to be created after it forks, it seems:
     * @see https://github.com/libfuse/libfuse/wiki/FAQ#how-should-threads-be-started
     * Personally, the only problem I observed was background process not finishing even after unmounting,
     * however, contrary to the FAQ it seems that threads were not joined because the file system seemed to work.
     */
    void
    joinThreads()
    {
        if ( m_blockFetcher ) {
            m_blockFetcher->joinThreads();
        }
        if ( m_blockFinder ) {
            m_blockFinder->stopThreads();
        }
    }

private:
    BlockFinder&
    blockFinder()
    {
        if ( m_blockFinder ) {
            return *m_blockFinder;
        }

        if ( !m_startBlockFinder ) {
            throw std::logic_error( "Block finder creator was not initialized correctly!" );
        }

        m_blockFinder = m_startBlockFinder();
        if ( !m_blockFinder ) {
            throw std::logic_error( "Block finder creator failed to create new block finder!" );
        }

        if ( m_blockMap.finalized() ) {
            setBlockFinderOffsets( m_blockMap.blockOffsets() );
        }

        return *m_blockFinder;
    }

    BlockFetcher&
    blockFetcher()
    {
        if ( m_blockFetcher ) {
            return *m_blockFetcher;
        }

        /* As a side effect, blockFinder() creates m_blockFinder if not already initialized! */
        if ( !blockFinder().finalized() ) {
            blockFinder().startThreads();
        }

        m_blockFetcher = std::make_unique<BlockFetcher>( m_bitReader, m_blockFinder, m_fetcherParallelization );
        m_blockFetcher->setShowProfileOnDestruction( m_showProfileOnDestruction );

        if ( !m_blockFetcher ) {
            throw std::logic_error( "Block fetcher should have been initialized!" );
        }

        return *m_blockFetcher;
    }

    /**
     * Called when the block finder found no data blocks. An empty input is an empty archive, but any other
     * input must at least consist of a valid stream header followed by an EOS block. Otherwise a truncated
     * header would be reported as a successfully decoded empty stream.
     */
    void
    validateStreamWithoutBlocks()
    {
        const auto fileSize = m_bitReader.size();
        if ( !fileSize ) {
            throw std::logic_error( "The size of a seekable input must be known!" );
        }
        if ( *fileSize == 0 ) {
            return;
        }

        BitReader bitReader( m_bitReader );
        bitReader.seekTo( 0 );
        bzip2::readBzip2Header( bitReader );
        const auto eosOffsetInBits = static_cast<size_t>( bitReader.tell() );
        const auto magicBytes = bitReader.read( bzip2::MAGIC_BITS_SIZE );
        if ( magicBytes != bzip2::MAGIC_BITS_EOS ) {
            throw std::domain_error( "Truncated bzip2 stream: input ended before the EOS block." );
        }
        /* A stream without data blocks has the initial stream CRC of zero. */
        const auto expectedCRC = static_cast<uint32_t>( bitReader.read( 32 ) );
        m_streamCRCVerifier.finishStream( eosOffsetInBits, 0, expectedCRC );
    }

    void
    setBlockFinderOffsets( const std::map<size_t, size_t>& offsets )
    {
        if ( offsets.empty() ) {
            throw std::invalid_argument( "A non-empty list of block offsets is required!" );
        }

        BlockFinder::BlockOffsets encodedBlockOffsets;
        for ( auto it = offsets.begin(), nit = std::next( offsets.begin() ); nit != offsets.end(); ++it, ++nit )
        {
            /* Ignore blocks with no data, i.e., EOS blocks. */
            if ( it->second != nit->second ) {
                encodedBlockOffsets.push_back( it->first );
            }
        }
        /* The last block is not pushed because "std::next( it )" is end but last block must be EOS anyways
         * or else BlockMap will not work correctly because the implied size of that last block is 0! */

        blockFinder().setBlockOffsets( std::move( encodedBlockOffsets ) );
    }
    // NOLINTEND(misc-no-recursion)

private:
    std::unique_ptr<rapidgzip::SharedFileReader> m_sharedFileReader;
    BitReader m_bitReader{ m_sharedFileReader->clone() };

    size_t m_currentPosition = 0;  /**< the current position as can only be modified with read or seek calls. */
    bool m_atEndOfFile = false;

    StreamCRCVerifier m_streamCRCVerifier;

private:
    size_t const m_fetcherParallelization;
    /** The block finder is much faster than the fetcher and therefore does not require es much parallelization! */
    size_t const m_finderParallelization{ rapidgzip::ceilDiv( m_fetcherParallelization, 64U ) };

    std::function<std::shared_ptr<BlockFinder>( void )> const m_startBlockFinder;

    /* These are the three larger "sub modules" of ParallelBZ2Reader */

    /** Necessary for prefetching decoded blocks in parallel. */
    std::shared_ptr<BlockFinder>               m_blockFinder;
    std::unique_ptr<BlockFetcher>              m_blockFetcher;
};
}  // namespace indexed_bzip2
