#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>

#include <core/BlockFetcher.hpp>
#include <core/BlockFinder.hpp>
#include <core/ParallelBitStringFinder.hpp>
#include <core/ReusableVector.hpp>

#include "bzip2.hpp"

namespace indexed_bzip2
{
struct BlockHeaderData
{
    size_t encodedOffsetInBits{ std::numeric_limits<size_t>::max() };
    size_t encodedSizeInBits{ 0 };  /**< When calling readBlockheader, only contains valid data if EOS block. */

    uint32_t expectedCRC{ 0 };  /**< if isEndOfStreamBlock == true, then this is the stream CRC. */
    bool isEndOfStreamBlock{ false };
    bool isEndOfFile{ false };
};


struct BlockData :
    public BlockHeaderData
{
    explicit
    BlockData( std::shared_ptr<rapidgzip::ReusableBufferPool<uint8_t> > bufferPool = {} ) :
        data( std::move( bufferPool ) )
    {}

public:
    rapidgzip::ReusableVector<uint8_t> data;
    uint32_t calculatedCRC{ 0xFFFFFFFFL };

    /** Header of the block that follows this one, parsed by the worker to keep it off the consumer thread. */
    std::optional<BlockHeaderData> nextBlockHeader;
};


template<typename FetchingStrategy = rapidgzip::FetchingStrategy::FetchNextAdaptive>
class BZ2BlockFetcher :
    public rapidgzip::BlockFetcher<
        rapidgzip::BlockFinder<rapidgzip::ParallelBitStringFinder<bzip2::MAGIC_BITS_SIZE> >,
        BlockData,
        FetchingStrategy>
{
public:
    using BaseType = rapidgzip::BlockFetcher<rapidgzip::BlockFinder<rapidgzip::ParallelBitStringFinder<bzip2::MAGIC_BITS_SIZE>>,
                                             BlockData,
                                             FetchingStrategy>;
    using BitReader = bzip2::BitReader;

public:
    BZ2BlockFetcher( BitReader                                       bitReader,
                     std::shared_ptr<typename BaseType::BlockFinder> blockFinder,
                     size_t                                          parallelization ) :
        BaseType( std::move( blockFinder ), parallelization ),
        m_bitReader( bitReader )
    {
        /* Validate and consume the stream header so that the block offsets found later are meaningful. */
        bzip2::readBzip2Header( bitReader );
    }

    ~BZ2BlockFetcher() override
    {
        this->stopThreadPool();
    }

    void
    setStatisticsEnabled( bool enabled )
    {
        BaseType::setStatisticsEnabled( enabled );
        m_bufferPool->setStatisticsEnabled( enabled );
    }

    [[nodiscard]] bool
    statisticsEnabled() const
    {
        return BaseType::statisticsEnabled();
    }

    void
    setShowProfileOnDestruction( bool show )
    {
        BaseType::setShowProfileOnDestruction( show );
        m_bufferPool->setShowProfileOnDestruction( show );
        this->setStatisticsEnabled( show );
    }

    /**
     * Reads only the fixed-size part of a block header. The block magic already identifies an EOS block and
     * for a data block the caller only needs to know that this is not one, so parsing the Huffman trees here
     * would be wasted work on the consumer thread.
     */
    [[nodiscard]] static BlockHeaderData
    readBlockHeader( BitReader& bitReader )
    {
        BlockHeaderData result;
        result.encodedOffsetInBits = bitReader.tell();

        const auto magic = ( static_cast<uint64_t>( bitReader.read<24>() ) << 24U )
                           | static_cast<uint64_t>( bitReader.read<24>() );
        result.isEndOfStreamBlock = magic == bzip2::MAGIC_BITS_EOS;
        if ( !result.isEndOfStreamBlock ) {
            if ( magic != bzip2::MAGIC_BITS_BLOCK ) {
                std::stringstream msg;
                msg << "[BZip2 block header] invalid compressed magic 0x" << std::hex << magic
                    << " at offset " << rapidgzip::formatBits( result.encodedOffsetInBits );
                throw std::domain_error( std::move( msg ).str() );
            }
            result.isEndOfFile = bitReader.eof();
            return result;
        }

        result.expectedCRC = bitReader.read( 32 );
        const auto nBitsInByte = static_cast<uint8_t>( bitReader.tell() & 7LLU );
        if ( nBitsInByte > 0 ) {
            bitReader.read( static_cast<uint8_t>( 8 ) - nBitsInByte );
        }
        result.encodedSizeInBits = bitReader.tell() - result.encodedOffsetInBits;
        result.isEndOfFile = bitReader.eof();
        return result;
    }

    [[nodiscard]] BlockHeaderData
    readBlockHeader( size_t blockOffset ) const
    {
        BitReader bitReader( m_bitReader );
        bitReader.seekTo( blockOffset );
        return readBlockHeader( bitReader );
    }

private:
    [[nodiscard]] BlockData
    decodeBlock( size_t                  blockOffset,
                 [[maybe_unused]] size_t nextBlockOffset ) const override
    {
        BitReader bitReader( m_bitReader );
        bitReader.seekTo( blockOffset );
        /* Keep the 135 KiB decoder state off the worker stack. Its address there depends on the caller and
         * interacts with the hot buffers in the Huffman/MTF loop, which made parallel decoding several times
         * slower with the -O3 build. Reusing it per thread also retains the multi-megabyte BWT buffers.
         * This is function-local scratch, not a decoder instance: decodeBlock must not be re-entered on the
         * same thread because the Block keeps a pointer to the local BitReader below. */
        static thread_local std::unique_ptr<bzip2::Block> blockHolder;
        if ( !blockHolder ) {
            blockHolder = std::make_unique<bzip2::Block>();
        }
        auto& block = *blockHolder;
        block.reset( bitReader );

        BlockData result( m_bufferPool );
        result.encodedOffsetInBits = blockOffset;
        result.isEndOfStreamBlock = block.eos();
        result.isEndOfFile = block.eof();
        result.expectedCRC = block.headerCRC();

        /* Actually, this should never happen with the current implementation because only blocks found by the
         * block finder will be handled here and the block finder does not search for EOS magic bits. */
        if ( block.eos() ) {
            result.encodedSizeInBits = block.encodedSizeInBits;
            return result;
        }

        block.readBlockData();

        /* A block may expand to tens of MiB (a run of 4 + count bytes can represent up to 255 input
         * bytes), so the size cannot be derived from the block size in the stream header. Scanning the
         * RLE1 output once is much cheaper than repeatedly growing the output buffer, which copied and
         * value-initialized several times the final size. */
        const auto decodedSize = block.decodedSize();
        result.data.resize( decodedSize );

        size_t decodedDataSize = 0;
        do
        {
            decodedDataSize += block.read(
                result.data.size() - decodedDataSize,
                reinterpret_cast<char*>( result.data.data() ) + decodedDataSize
            );
        }
        while ( !block.eob() && ( decodedDataSize < result.data.size() ) );

        if ( !block.eob() ) {
            throw std::logic_error( "[BZip2 block data] Decoded size does not match the RLE1 scan!" );
        }

        result.data.resize( decodedDataSize );
        result.encodedSizeInBits = block.encodedSizeInBits;
        result.calculatedCRC = block.dataCRC();

        /* The bit reader stops after the compressed block, so the next block header can be inspected here
         * instead of on the consumer thread. */
        result.nextBlockHeader = readBlockHeader( bitReader );

        /* nextBlockOffset only is a candidate found by the block finder. It may be a false positive inside this
         * block, so the decoded encoded size is authoritative and the block does not have to end before it. */

        return result;
    }

private:
    /* Variables required by decodeBlock and which therefore should be either const or locked. */
    const BitReader m_bitReader;
    const std::shared_ptr<rapidgzip::ReusableBufferPool<uint8_t> > m_bufferPool{
        std::make_shared<rapidgzip::ReusableBufferPool<uint8_t> >()
    };
};
}  // namespace indexed_bzip2
