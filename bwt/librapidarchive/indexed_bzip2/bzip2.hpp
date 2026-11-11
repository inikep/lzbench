/**
 * Modified version of bzcat.c part of toybox commit 7bf68329eb3b
 * by Rob Landley released under SPDX-0BSD license.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <core/BitManipulation.hpp>
#include <filereader/BitReader.hpp>

#include "Bzip2Crc32.hpp"
#include "Bzip2Format.hpp"
#include "Bzip2InverseBWT.hpp"
#include "Bzip2MoveToFront.hpp"
#include "Bzip2SymbolDecoder.hpp"

#if LIBRAPIDARCHIVE_BZIP2_CRC_PCLMUL
    /* The AVX-512 RLE scan uses the same compiler/architecture gate as the PCLMUL CRC header. */
    #define LIBRAPIDARCHIVE_BZIP2_AVX512_SCAN 1
#else
    #define LIBRAPIDARCHIVE_BZIP2_AVX512_SCAN 0
#endif


/**
 * @see https://github.com/dsnet/compress/blob/master/doc/bzip2-format.pdf
 * @verbatim
 * Symbol               Expression
 * ------------------------------------------
 * BZipFile             := BZipStream (one or more)
 * └──BZipStream        := StreamHeader StreamBlock* StreamFooter
 *    ├──StreamHeader   := HeaderMagic Version Level                          -> readBzip2Header
 *    ├──StreamBlock    := BlockHeader BlockTrees BlockData(Huffman encoded)
 *    │ ├──BlockHeader  := BlockMagic BlockCRC Randomized OrigPtr             -> readBlockHeader
 *    │ ├──BlockTrees   := SymMap NumTrees NumSels Selectors Trees            -> readTrees
 *    | |  ├──SymMap    := MapL1 MapL2{1,16}
 *    │ |  ├──Selectors := Selector{NumSels}
 *    │ |  └──Trees     := (BitLen Delta{NumSyms}){NumTrees}
 *    | └──BlockData    := Huffman-Encoded data
 *    └──StreamFooter   := FooterMagic StreamCRC Padding                      -> Block::eos
 * @endverbatim

 * 1. Run-length encoding (RLE) of initial data
 * 2. Burrows–Wheeler transform (BWT), or block sorting
 * 3. Move-to-front (MTF) transform
 * 4. Run-length encoding (RLE) of MTF result
 * 5. Huffman coding
 * 6. Selection between multiple Huffman tables
 * 7. Unary base-1 encoding of Huffman table selection
 * 8. Delta encoding (Δ) of Huffman-code bit lengths
 * 9. Sparse bit array showing which symbols are used
 */
namespace bzip2
{
#if LIBRAPIDARCHIVE_BZIP2_AVX512_SCAN
#define LIBRAPIDARCHIVE_BZIP2_AVX512_TARGET __attribute__( ( target( "avx512f,avx512bw" ) ) )

[[nodiscard]] inline bool
hasAvx512RLEScan() noexcept
{
    static const bool supported =
        ( __builtin_cpu_supports( "avx512f" ) != 0 ) && ( __builtin_cpu_supports( "avx512bw" ) != 0 );
    return supported;
}


/**
 * Expand the whole RLE1 prefix of a block 64 positions at a time. A position starts a run when it and the next
 * three bytes are equal. The byte after those four is the additional repeat count. The first set bit of the
 * equality mask is therefore the first run start, and everything before it is a literal.
 *
 * Stops when the caller's buffer is full, the source is exhausted, or fewer than 68 source bytes remain, so
 * the scalar and SSE2 paths in decodeBlock can handle the tail.
 */
LIBRAPIDARCHIVE_BZIP2_AVX512_TARGET
inline void
expandRLEAvx512( const uint8_t* source,
                 const size_t   sourceSize,
                 size_t&        position,
                 char* const    output,
                 size_t&        nBytesDecoded,
                 const size_t   nMaxBytesToDecode,
                 uint8_t&       repeatByte,
                 size_t&        repeatCount )
{
    while ( nBytesDecoded < nMaxBytesToDecode ) {
        if ( repeatCount > 0 ) {
            const auto nBytesToRepeat = std::min( repeatCount, nMaxBytesToDecode - nBytesDecoded );
            for ( size_t i = 0; i < nBytesToRepeat; ++i ) {
                output[nBytesDecoded + i] = static_cast<char>( repeatByte );
            }
            nBytesDecoded += nBytesToRepeat;
            repeatCount -= nBytesToRepeat;
            continue;
        }

        if ( ( position >= sourceSize )
             || ( nBytesDecoded + sizeof( __m512i ) > nMaxBytesToDecode )
             || ( position + sizeof( __m512i ) + 4 > sourceSize ) ) {
            return;
        }

        const auto* const window = source + position;
        const auto v0 = _mm512_loadu_si512( window );
        const auto v1 = _mm512_loadu_si512( window + 1 );
        const auto v2 = _mm512_loadu_si512( window + 2 );
        const auto v3 = _mm512_loadu_si512( window + 3 );
        const auto runStarts = _mm512_cmpeq_epi8_mask( v0, v1 )
                               & _mm512_cmpeq_epi8_mask( v0, v2 )
                               & _mm512_cmpeq_epi8_mask( v0, v3 );
        if ( runStarts == 0 ) {
            _mm512_storeu_si512( output + nBytesDecoded, v0 );
            nBytesDecoded += sizeof( __m512i );
            position += sizeof( __m512i );
            continue;
        }

        const auto literalCount = static_cast<size_t>( __builtin_ctzll( runStarts ) );
        if ( literalCount > 0 ) {
            const auto literalMask = ( __mmask64( 1 ) << literalCount ) - 1;
            _mm512_mask_storeu_epi8( output + nBytesDecoded, literalMask, v0 );
        }
        nBytesDecoded += literalCount;
        position += literalCount;
        repeatByte = source[position];
        repeatCount = 4 + source[position + 4];
        position += 5;
    }
}
#undef LIBRAPIDARCHIVE_BZIP2_AVX512_TARGET
#endif


using BitReader = rapidgzip::BitReader<true, uint64_t>;


#if defined( __ARM_NEON )
/**
 * @return four mask bits per byte, set for equal bytes, so that ctz(mask) / 4 is the first equal byte.
 */
[[nodiscard]] inline uint64_t
equalByteMask( const uint8x16_t equal ) noexcept
{
    const auto narrowed = vshrn_n_u16( vreinterpretq_u16_u8( equal ), 4 );
    return vget_lane_u64( vreinterpret_u64_u8( narrowed ), 0 );
}
#endif


/**
 * @return 1..9 representing the bzip2 block size of 100k to 900k
 */
inline uint8_t
readBzip2Header( BitReader& bitReader )
{
    for ( const auto magicByte : MAGIC_BYTES_BZ2 ) {
        const auto readByte = static_cast<char>( static_cast<uint8_t>( bitReader.read<8>() ) );
        if ( readByte != magicByte ) {
            std::stringstream msg;
            msg << "Input header is not BZip2 magic string 'BZh' (0x"
                << std::hex << int('B') << int('Z') << int('h') << std::dec << "). Mismatch at bit position "
                << bitReader.tell() - 8 << " with " << readByte << " (0x" << std::hex
                /* first cast to unsigned of same length so that the hex representation makes sense. */
                << static_cast<int>( static_cast<uint8_t>( readByte ) )
                << ") should be " << magicByte;
            throw std::domain_error( std::move( msg ).str() );
        }
    }

    // Next byte ASCII '1'-'9', indicates block size in units of 100k of
    // uncompressed data. Allocate intermediate buffer for block.
    const auto i = static_cast<char>( static_cast<uint8_t>( bitReader.read<8>() ) );
    if ( ( i < '1' ) || ( i > '9' ) ) {
        std::stringstream msg;
        msg << "Blocksize must be one of '0' (" << std::hex << static_cast<int>( '0' ) << ") ... '9' ("
            << static_cast<int>( '9' ) << ") but is " << i << " (" << static_cast<int>( i ) << ")";
        throw std::domain_error( std::move( msg ).str() );
    }

    return static_cast<uint8_t>( i - '0' );
}


struct Block  // NOLINT(clang-analyzer-optin.performance.Padding)
{
public:
    /**
     * m ibzip2 && src/tools/ibzip2 -P 1 -f -L offsets -i scorep-pragzip-bgzf-16MiB.tar.bz2
     * @verbatim
     * [BZ2Reader] Time spent:
     * decodeBlock                   : 2.99293s
     * readBlockHeader               : 17.9826s
     *     readSymbolMaps            : 0.000444198s
     *     readSelectors             : 0.0294166s
     *     readTrees                 : 0.0168973s
     *     createHuffmanTable        : 16.8445s
     *     burrowsWheelerPreparation : 1.08934s
     * @endverbatim
     */
    struct Statistics
    {
    public:
        void
        merge( const Statistics& other )
        {
            durations.merge( other.durations );
        }

    public:
        struct Durations
        {
        public:
            void
            merge( const Durations& other )
            {
                readBlockHeader           += other.readBlockHeader;
                decodeBlock               += other.decodeBlock;
                readSymbolMaps            += other.readSymbolMaps;
                readSelectors             += other.readSelectors;
                readTrees                 += other.readTrees;
                createHuffmanTable        += other.createHuffmanTable;
                burrowsWheelerPreparation += other.burrowsWheelerPreparation;
            }

        public:
            double readBlockHeader{ 0 };
            double decodeBlock{ 0 };

            /* Parts of readBlockHeader. */
            double readSymbolMaps{ 0 };
            double readSelectors{ 0 };
            double readTrees{ 0 };
            double createHuffmanTable{ 0 };
            double burrowsWheelerPreparation{ 0 };
        };

        Durations durations;
    };

public:
    Block() = default;

    ~Block() = default;

    /** Better don't allow copies because the bitreader would be shared, which might be problematic */
    Block( const Block& ) = delete;

    Block&
    operator=( const Block& ) = delete;

    Block( Block&& ) = delete;

    Block&
    operator=( Block&& ) = delete;

    explicit
    Block( BitReader& bitReader )
    {
        reset( bitReader );
    }

    /**
     * Reinitialize this block for a new block header while keeping the allocated BWT buffers. This avoids
     * allocating and freeing the multi-megabyte buffers for every bzip2 block.
     */
    void
    reset( BitReader& bitReader )
    {
        m_bitReader = &bitReader;
        readBlockHeader();
    }

    /**
     * First pass, read block's symbols into bwtData[bwtDataCount].
     *
     * This undoes three types of compression: huffman coding, run length encoding,
     * and move to front encoding.  We have to undo all those to know when we've
     * read enough input.
     *
     * It is not automatically called by the constructor and must be called manually for non-EOS blocks.
     * The interface is like this because ParallelBZ2Reader slows down when calling it automatically
     * because it would not be called on the worker threads but on the main thread!
     */
    void
    readBlockData();

    /**
     * @return True if this is a special enf-of-stream bzip2 block, which contains no data.
     */
    [[nodiscard]] constexpr bool
    eos() const noexcept
    {
        return m_atEndOfStream;
    }

    /**
     * @return True if all data has been read from this block.
     */
    [[nodiscard]] constexpr bool
    eob() const noexcept
    {
        return eos() || !bwdata.hasData();
    }

    [[nodiscard]] constexpr bool
    eof() const noexcept
    {
        return m_atEndOfFile;
    }

    [[nodiscard]] BitReader&
    bitReader()
    {
        if ( m_bitReader != nullptr ) {
            return *m_bitReader;
        }
        throw std::invalid_argument( "Block has not been initialized yet!" );
    }

    /**
     * Currently, the logic is limited and might write up to nMaxBytesToDecode + 255 characters
     * to the output buffer! Currently, the caller has to ensure that the output buffer is large enough.
     */
    [[nodiscard]] size_t
    read( const size_t nMaxBytesToDecode,
          char*        outputBuffer )
    {
        const auto t0 = rapidgzip::now();
        const auto result = bwdata.decodeBlock( nMaxBytesToDecode, outputBuffer );
        statistics.durations.decodeBlock += rapidgzip::duration( t0 );
        return result;
    }

    /**
     * @return The current CRC32 checksum of the decoded data. If all the data of this block has been decoded,
     *         this should match header CRC.
     */
    [[nodiscard]] constexpr uint32_t
    dataCRC() const noexcept
    {
        return bwdata.dataCRC;
    }

    /**
     * @return The CRC32 checksum as stored in the bzip2 block header.
     */
    [[nodiscard]] constexpr uint32_t
    headerCRC() const noexcept
    {
        return bwdata.headerCRC;
    }

    /**
     * Scan the RLE1-encoded BWT output once to determine the exact decoded size of this block.
     * This is only valid after readBlockData(), which produces the RLE1-encoded output.
     */
    [[nodiscard]] size_t
    decodedSize() const noexcept
    {
        return bwdata.decodedSize();
    }

private:
    template<uint8_t nBits>
    [[nodiscard]] uint32_t
    getBits()
    {
        return static_cast<uint32_t>( bitReader().read<nBits>() );
    }

    [[nodiscard]] uint32_t
    getBits( uint8_t nBits )
    {
        return static_cast<uint32_t>( bitReader().read( nBits ) );
    }

    void
    readBlockHeader();

    void
    readBlockTrees()
    {
        const auto tReadSymbolMaps = rapidgzip::now();
        readSymbolMaps();
        /* A data block needs at least one used byte. With an empty symbol map there is no front byte for a
         * run and no literal symbol, but the decoder would silently produce zero bytes. */
        if ( symbolCount == 0 ) {
            throw std::domain_error( "[BZip2 block header] The block's symbol map is empty" );
        }
        const auto tReadSelectors = rapidgzip::now();
        readSelectors();
        const auto tReadTrees = rapidgzip::now();
        readTrees();

        statistics.durations.readSymbolMaps += rapidgzip::duration( tReadSymbolMaps, tReadSelectors );
        statistics.durations.readSelectors += rapidgzip::duration( tReadSelectors, tReadTrees );
        statistics.durations.readTrees += rapidgzip::duration( tReadTrees );
    }

    void
    readSymbolMaps();

    void
    readSelectors();

    void
    readTrees();

public:
    struct BurrowsWheelerTransformData
    {
        BurrowsWheelerTransformData() = default;
        BurrowsWheelerTransformData( const BurrowsWheelerTransformData& ) = delete;
        BurrowsWheelerTransformData& operator=( const BurrowsWheelerTransformData& ) = delete;

        friend Block;

    public:
        /**
         * Currently, the logic is limited and might write up to nMaxBytesToDecode + 255 characters
         * to the output buffer! Currently, the caller has to ensure that the output buffer is large enough.
         */
        [[nodiscard]] size_t
        decodeBlock( size_t nMaxBytesToDecode,
                     char*  outputBuffer );

        [[nodiscard]] constexpr bool
        hasData() const noexcept
        {
            return inverseBWT && ( ( inverseBWT->position < writeCount ) || ( inverseBWT->repeatCount > 0 ) );
        }

        /**
         * @return The number of bytes that decodeBlock will produce, i.e., the result of expanding the
         *         RLE1 runs in bwtData. A run consists of four identical bytes followed by a count byte
         *         and decodes to 4 + count bytes. Runs shorter than five bytes at the end are literals.
         */
        [[nodiscard]] size_t
        decodedSize() const noexcept
        {
            size_t decodedSize = 0;
            uint32_t position = 0;

        #if LIBRAPIDARCHIVE_BZIP2_HAS_SSE2
            /* Test four shifted 16-byte windows at once to find the next run start. Positions before it
             * are literals, which avoids one branch and comparison per byte for literal-heavy blocks. */
            while ( position + 20 <= writeCount ) {
                const auto* const source = bwtData.get() + position;
                const auto v0 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( source ) );
                const auto v1 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( source + 1 ) );
                const auto v2 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( source + 2 ) );
                const auto v3 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( source + 3 ) );
                const auto equal = _mm_and_si128( _mm_and_si128( _mm_cmpeq_epi8( v0, v1 ),
                                                                 _mm_cmpeq_epi8( v0, v2 ) ),
                                                  _mm_cmpeq_epi8( v0, v3 ) );
                const auto runStartMask = static_cast<uint32_t>( _mm_movemask_epi8( equal ) );
                if ( runStartMask == 0 ) {
                    decodedSize += 16;
                    position += 16;
                    continue;
                }

                const auto literalCount = static_cast<uint32_t>( rapidgzip::trailingZeroCount( runStartMask ) );
                decodedSize += literalCount;
                position += literalCount;
                decodedSize += 4 + bwtData[position + 4];
                position += 5;
            }
        #endif

            for ( ; position < writeCount; ) {
                const auto byte = bwtData[position];
                if ( ( position + 4 < writeCount )
                     && ( bwtData[position + 1] == byte )
                     && ( bwtData[position + 2] == byte )
                     && ( bwtData[position + 3] == byte ) ) {
                    decodedSize += 4 + bwtData[position + 4];
                    position += 5;
                } else {
                    ++decodedSize;
                    ++position;
                }
            }
            return decodedSize;
        }

    private:
        /**
         * Creates the inverse BWT of the block from the run records of the symbol decoder.
         * Must only be called after initializing origPtr and the records.
         */
        void
        prepare( size_t recordCount );

        /**
         * Reset the per-block state while retaining the allocated buffers for reuse.
         * The header and symbol pass of the next block overwrite everything that is read.
         */
        void
        resetState()
        {
            origPtr = 0;
            writeCount = 0;
            inverseBWT.reset();
            dataCRC = 0xFFFFFFFFL;
            headerCRC = 0;
        }

    private:
        uint32_t origPtr = 0;
        /** Size of the BWT string, i.e., of the block before the initial run-length encoding is undone. */
        uint32_t writeCount = 0;

        /* The output of the inverse BWT is still run-length encoded, so one run may span several decodeBlock
         * calls. This state only exists after prepare and is needed for resuming the expansion. */
        struct InverseBWTData
        {
            size_t position{ 0 };
            size_t repeatCount{ 0 };
            uint8_t repeatByte{ 0 };
        };

        std::optional<InverseBWTData> inverseBWT;

        uint32_t dataCRC = 0xFFFFFFFFL;  /* CRC of block as calculated by us */
        uint32_t headerCRC = 0;  /* what the block data CRC should be */

        /* Keeps its bookkeeping buffers so that they are only allocated once per reader or worker. */
        InverseBWT inverter;

        /**
         * Simply allocate for the maximum block size so that there is no problem when the block size changes
         * (e.g. in pbzip2 files). None of these buffers is zero-initialized because everything that is read
         * has been written for the same block before. They are allocated once per reader or worker.
         */
        std::unique_ptr<uint32_t[]> lastToFirstMapping{ new uint32_t[LAST_TO_FIRST_MAPPING_SIZE] };

        /* The run records of the symbol decoder. They are not needed anymore when the BWT gets inverted, so
         * the buffer doubles as its scratch space. */
        std::unique_ptr<uint8_t[]> records{ new uint8_t[RUN_RECORD_BUFFER_SIZE] };
        static_assert( RUN_RECORD_BUFFER_SIZE >= InverseBWT::ARENA_SIZE, "The buffer is reused as arena." );

        /* The inverse BWT output, which is the block with its runs still encoded. The eight tail bytes are
         * slack for vectorized readers. */
        std::unique_ptr<uint8_t[]> bwtData{ new uint8_t[MAX_BLOCK_SIZE + sizeof( uint64_t )] };
    };

public:
    Statistics statistics;

    size_t encodedOffsetInBits = 0;
    size_t encodedSizeInBits = 0;

private:
    uint64_t magicBytes{ 0 };
    bool isRandomized{ false };

    /* First pass decompression data (Huffman and MTF decoding) */

    /**
     * mapping table: if some byte values are never used (encoding things
     * like ascii text), the compression code removes the gaps to have fewer
     * symbols to deal with, and writes a sparse bitfield indicating which
     * values were present.  We make a translation table to convert the symbols
     * back to the corresponding bytes.
     */
    std::array<uint8_t, 256> symbolToByte{};
    /** Used for the selectors, which are decoded before the block data. */
    std::array<uint8_t, 256> mtfSymbol{};
    unsigned int symbolCount{ 0 };
    /**
     * Every GROUP_SIZE many symbols we switch huffman coding tables.
     * Each group has a selector, which is an index into the huffman coding table arrays.
     *
     * Read in the group selector array, which is stored as MTF encoded
     * bit runs.  (MTF = Move To Front.  Every time a symbol occurs it's moved
     * to the front of the table, so it has a shorter encoding next time.)
     */
    uint16_t selectorsCount{ 0 };

    std::array<uint8_t, 32768> selectors{};  // nSelectors=15 bits
    std::array<SymbolTable, MAX_GROUPS> symbolTables{};
    /** For each coding table, the result of @ref computeNearTailWeight. Used to choose the symbol kernel. */
    std::array<uint32_t, MAX_GROUPS> nearTailWeights{};
    uint32_t groupCount = 0;

    /* Second pass decompression data (burrows-wheeler transform) */
    BurrowsWheelerTransformData bwdata;

    BitReader* m_bitReader = nullptr;
    bool m_atEndOfStream = false;
    bool m_atEndOfFile = false;
};


/* Read block header at start of a new compressed data block.  Consists of:
 *
 * 48 bits : Block signature, either pi (data block) or e (EOF block).
 * 32 bits : bw->headerCRC
 * 1  bit  : obsolete feature flag.
 * 24 bits : origPtr (Burrows-wheeler unwind index, only 20 bits ever used)
 * 16 bits : Mapping table index.
 *[16 bits]: symToByte[symTotal] (Mapping table.  For each bit set in mapping
 *           table index above, read another 16 bits of mapping table data.
 *           If corresponding bit is unset, all bits in that mapping table
 *           section are 0.)
 *  3 bits : groupCount (how many huffman tables used to encode, anywhere
 *           from 2 to MAX_GROUPS)
 * variable: hufGroup[groupCount] (MTF encoded huffman table data.)
 */
inline void
Block::readBlockHeader()
{
    const auto tReadBlockHeader = rapidgzip::now();

    /* A Block may be reused for many headers. Reset the per-block state and timing while keeping the
     * expensive buffers allocated. */
    statistics = {};
    bwdata.resetState();
    m_atEndOfFile = false;

    encodedOffsetInBits = bitReader().tell();
    encodedSizeInBits = 0;

    magicBytes = ( (uint64_t)getBits<24>() << 24U ) | (uint64_t)getBits<24>();
    bwdata.headerCRC = getBits( 32 );
    m_atEndOfStream = magicBytes == MAGIC_BITS_EOS;
    if ( m_atEndOfStream ) {
        /* read byte padding bits */
        const auto nBitsInByte = static_cast<uint8_t>( bitReader().tell() & 7LLU );
        if ( nBitsInByte > 0 ) {
            bitReader().read( uint8_t( 8 ) - nBitsInByte );
        }

        encodedSizeInBits = bitReader().tell() - encodedOffsetInBits;
        m_atEndOfFile = bitReader().eof();
        return;
    }

    if ( magicBytes != MAGIC_BITS_BLOCK ) {
        std::stringstream msg;
        msg << "[BZip2 block header] invalid compressed magic 0x" << std::hex << magicBytes
            << " at offset " << rapidgzip::formatBits( encodedOffsetInBits );
        throw std::domain_error( std::move( msg ).str() );
    }

    isRandomized = getBits<1>() != 0;
    if ( isRandomized ) {
        throw std::domain_error( "[BZip2 block header] deprecated isRandomized bit is not supported" );
    }

    bwdata.origPtr = getBits<24>();
    if ( bwdata.origPtr > MAX_BLOCK_SIZE ) {
        std::stringstream msg;
        msg << "[BZip2 block header] origPtr " << bwdata.origPtr << " is larger than the maximum block size: "
            << MAX_BLOCK_SIZE;
        throw std::logic_error( std::move( msg ).str() );
    }

    readBlockTrees();
    statistics.durations.readBlockHeader += rapidgzip::duration( tReadBlockHeader );
}


inline void
Block::readSymbolMaps()
{
    /**
     * The Mapping table itself is compressed in two parts:
     * huffmanUsedMap: each bit indicates whether the corresponding range [0...15], [16...31] is present.
     * huffman_used_bitmaps: 0-16 16-bit bitmaps
     * The Huffman map gives 0, 10, 11, 100, 101, ... (8-bit) symbols
     * Instead of storing 2 * 256 bytes ( 0b : A, 10b : B, .. ) for the table, the first part is left out.
     * And for short maps, only the first n-th are actually stored.
     * The second half is also assumed to be ordered, so that we only need to store which symbols are actually
     * present.
     * This however means that the Huffmann table can't be frequency sorted, therefore this is done in a
     * second step / table, the mtfSymbol (move to front) map.
     * This would need 256 bits to store the table in huffman_used_bitmaps.
     * These bits are split in groups of 16 and the presence of each group is encoded in huffmanUsedMap
     * to save even more bytes.
     * @verbatim
     *  10001000 00000000     # huffmanUsedMap (bit map)
     *  ^   ^
     *  |   [64,95]
     *  [0...15]
     *  00000000 00100000     # huffman_used_bitmaps[0]
     *  ^          ^    ^
     *  0          10   15
     *          (newline)
     *  00000100 10001001     # huffman_used_bitmaps[1]
     *  ^    ^   ^   ^  ^
     *  64   69  72  76 95
     *       E   H   L  O
     * @endverbatim
     */
    const uint16_t huffmanUsedMap = getBits<16>();
    /* Can at most grow up to 256 symbols, i.e., MAX_SYMBOLS - 2 (RUNA, RUNB). */
    symbolCount = 0;
    for ( int i = 0; i < 16; i++ ) {
        if ( ( huffmanUsedMap & ( 1U << ( 15U - i ) ) ) != 0 ) {
            const auto bitmap = getBits<16>();
            for ( int j = 0; j < 16; j++ ) {
                if ( ( bitmap & ( 1U << ( 15U - j ) ) ) != 0 ) {
                    symbolToByte[symbolCount++] = ( 16 * i ) + j;
                }
            }
        }
    }
}


inline void
Block::readSelectors()
{
    // How many different huffman coding groups does this block use?
    groupCount = getBits<3>();
    if ( ( groupCount < 2 ) || ( groupCount > MAX_GROUPS ) ) {
        std::stringstream msg;
        msg << "[BZip2 block header] Invalid Huffman coding group count " << groupCount;
        throw std::logic_error( std::move( msg ).str() );
    }

    // nSelectors: Every GROUP_SIZE many symbols we switch huffman coding
    // tables.  Each group has a selector, which is an index into the huffman
    // coding table arrays.
    //
    // Read in the group selector array, which is stored as MTF encoded
    // bit runs.  (MTF = Move To Front.  Every time a symbol occurs it's moved
    // to the front of the table, so it has a shorter encoding next time.)
    selectorsCount = getBits<15>();
    if ( selectorsCount == 0 ) {
        std::stringstream msg;
        msg << "[BZip2 block header] The number of selectors " << selectorsCount << " is invalid";
        throw std::logic_error( std::move( msg ).str() );
    }

    /* The "selectors", referring to the Huffman trees to decode a group of 50 symbols, can only range from 0-5,
     * because MAX_GROUPS = 6, and are encoded as zero-terminated 1-bits, where the number of 1-bits represents
     * the Huffman tree ID. */
    static constexpr std::array<uint8_t, ( 1U << MAX_GROUPS )> BITS_TO_SELECTOR =
        [] () {
            std::array<uint8_t, ( 1U << MAX_GROUPS )> result{};
            uint8_t lowestBitsSet{ 0 };
            for ( uint8_t selector = 0; selector < MAX_GROUPS; ++selector ) {
                const auto paddingBitsCount = MAX_GROUPS - selector;
                const auto maxPaddingBits = static_cast<uint8_t>( 1U << paddingBitsCount );
                const auto highestBitsSet = static_cast<uint8_t>( lowestBitsSet << ( MAX_GROUPS - selector ) );
                for ( uint8_t paddingBits = 0; paddingBits < maxPaddingBits; ++paddingBits ) {
                    result[highestBitsSet | paddingBits] = selector;
                }

                lowestBitsSet <<= 1U;
                lowestBitsSet |= 1U;
            }
        /* A full line of 1-bits is not allowed because it would indicate a selector index larger than MAX_GROUPS. */
            result.back() = MAX_GROUPS;
            return result;
        } ();

    std::iota( mtfSymbol.begin(), mtfSymbol.begin() + groupCount, 0 );
    for ( size_t i = 0; i < selectorsCount; i++ ) {
        const auto j = BITS_TO_SELECTOR.at( m_bitReader->peek<MAX_GROUPS>() );
        if ( j >= groupCount ) {
            std::stringstream msg;
            msg << "[BZip2 block header] Could not find zero termination after " << groupCount << " bits";
            throw std::domain_error( std::move( msg ).str() );
        }
        /* Validate before consuming because a full line of MAX_GROUPS 1-bits has no zero terminator to seek past. */
        m_bitReader->seekAfterPeek( j + 1 );

        // Decode MTF to get the next selector, and move it to the front.
        const auto uc = mtfSymbol[j];
        memmove( mtfSymbol.data() + 1, mtfSymbol.data(), j );
        mtfSymbol[0] = uc;
        selectors[i] = uc;
    }
}


/**
 * bzip2 blocks are many times larger than usual gzip blocks. That's why multiple Huffman tree per block are
 * supported and necessary. Similarly to deflate, the trees are stores as code lengths per symbol.
 */
inline void
Block::readTrees()
{
    // Read the huffman coding tables for each group, which code for symTotal
    // literal symbols, plus two run symbols (RUNA, RUNB)
    const auto symCount = symbolCount + 2;
    for ( size_t j = 0; j < groupCount; j++ ) {
        // Read lengths
        std::array<uint8_t, MAX_SYMBOLS> lengths{};
        unsigned int hh = getBits<5>();
        for ( unsigned int symbol = 0; symbol < symCount; symbol++ ) {
            while ( true ) {
                // !hh || hh > MAX_HUFCODE_BITS in one test.
                if ( MAX_HUFCODE_BITS - 1 < hh - 1 ) {
                    std::stringstream msg;
                    msg << "[BZip2 block header] start_huffman_length " << hh
                        << " is larger than " << MAX_HUFCODE_BITS << " or zero\n";
                    throw std::logic_error( std::move( msg ).str() );
                }

                // Stop if first bit is 0, otherwise second bit says whether to
                // increment or decrement.
                if ( getBits<1>() != 0 ) {
                    hh += 1 - ( getBits<1>() << 1U );
                } else {
                    break;
                }
            }
            if ( hh > std::numeric_limits<uint8_t>::max() ) {
                std::stringstream msg;
                msg << "[BZip2 block header] The read code length is unexpectedly large: " << hh;
                throw std::logic_error( std::move( msg ).str() );
            }
            lengths[symbol] = static_cast<uint8_t>( hh );
        }

        const auto error = symbolTables[j].initializeFromLengths( lengths.data(), symCount );
        if ( error != rapidgzip::Error::NONE ) {
            throw std::domain_error( toString( error ) );
        }
        /* Only valid code lengths reach this. */
        nearTailWeights[j] = computeNearTailWeight( lengths.data(), symCount );
    }
}


/**
 * This undoes the Huffman coding, the zero run-length encoding, and the Move-to-Front transform, and then
 * inverts the Burrows-Wheeler transform.
 */
inline void
Block::readBlockData()
{
    const auto t0 = rapidgzip::now();
    auto& reader = *m_bitReader;

    SymbolDecoderState state;
    state.mtf = createPackedMTF( symbolToByte.data(), symbolCount );
    state.record = bwdata.records.get();
    state.recordLimit = state.record + RUN_RECORD_SIZE * MAX_RUN_RECORDS;
    /* The first record belongs to the byte that initially is at the front of the Move-to-Front list.
     * It stays empty if the block does not start with a run of that byte. */
    state.record[0] = symbolCount > 0 ? symbolToByte[0] : uint8_t( 0 );
    state.record[1] = 0;
    state.symbolCount = symbolCount;
    state.selector = selectors.data();
    state.selectorEnd = selectors.data() + selectorsCount;
    state.tables = symbolTables.data();

    /* See prefersPortableKernel for why some blocks are faster without SIMD. */
    auto kernel = symbolKernel();
    if ( ( kernel != SymbolKernel::PORTABLE ) && !isSymbolKernelOverridden() ) {
        uint64_t nearTailWeight = 0;
        for ( size_t i = 0; i < selectorsCount; ++i ) {
            nearTailWeight += nearTailWeights[selectors[i]];
        }
        if ( prefersPortableKernel( nearTailWeight, selectorsCount ) ) {
            kernel = SymbolKernel::PORTABLE;
        }
    }

    /** @note The loops for this are too short to be profiled individually. */
    auto status = SymbolDecoderStatus::NEED_INPUT;
    while ( status == SymbolDecoderStatus::NEED_INPUT ) {
        /* Decode directly from the bytes that the bit reader has buffered as long as whole groups of
         * symbols are available. This is where almost all symbols are decoded. */
        if ( const auto bufferOffset = reader.bufferBitPosition(); bufferOffset.has_value() ) {
            const auto& buffer = reader.buffer();
            auto offset = *bufferOffset;
            status = decodeSymbols( state, buffer.data(), buffer.size(), offset, kernel );
            if ( offset != *bufferOffset ) {
                reader.seekInsideBuffer( offset );
            }
            if ( status != SymbolDecoderStatus::NEED_INPUT ) {
                break;
            }
        }

        /* Decode one group with the bit reader, which refills its buffer when necessary and also works
         * for the last bytes of the file. */
        status = decodeSymbolGroup( state, reader );
    }

    switch ( status )
    {
    case SymbolDecoderStatus::NEED_INPUT:  // Not possible after the loop, but avoids a compiler warning.
    case SymbolDecoderStatus::END_OF_BLOCK:
        break;
    case SymbolDecoderStatus::INVALID_CODE:
        throw std::domain_error( "[BZip2 block data] Invalid Huffman code" );
    case SymbolDecoderStatus::OUT_OF_SELECTORS:
    {
        /* All selectors have been used up without the end-of-block symbol. */
        std::stringstream msg;
        msg << "[BZip2 block data] selector " << selectorsCount << " out of maximum range " << selectorsCount;
        throw std::domain_error( std::move( msg ).str() );
    }
    case SymbolDecoderStatus::BLOCK_TOO_LARGE:
    {
        std::stringstream msg;
        msg << "[BZip2 block data] The run length exceeds the maximum block size " << MAX_BLOCK_SIZE;
        throw std::domain_error( std::move( msg ).str() );
    }
    }

    /* state.record points to the last record, not behind it. */
    const auto recordCount = static_cast<size_t>( state.record - bwdata.records.get() ) / RUN_RECORD_SIZE + 1;
    statistics.durations.createHuffmanTable += rapidgzip::duration( t0 );

    const auto tPrepareStart = rapidgzip::now();
    bwdata.prepare( recordCount );
    statistics.durations.burrowsWheelerPreparation += rapidgzip::duration( tPrepareStart );

    encodedSizeInBits = bitReader().tell() - encodedOffsetInBits;
}


inline void
Block::BurrowsWheelerTransformData::prepare( const size_t recordCount )
{
    dataCRC = 0xFFFFFFFFL;
    inverseBWT.reset();

    std::array<uint32_t, 256> byteCounts{};
    const auto size = createLastToFirstMapping( records.get(), recordCount, lastToFirstMapping.get(), byteCounts );
    if ( size == LAST_TO_FIRST_MAPPING_TOO_LARGE ) {
        std::stringstream msg;
        msg << "[BZip2 block data] The block exceeds the maximum block size " << MAX_BLOCK_SIZE;
        throw std::domain_error( std::move( msg ).str() );
    }

    // Now we know the size of the block, do a better sanity check on origPtr.
    writeCount = size;
    if ( origPtr >= size ) {
        std::stringstream msg;
        msg << "[BZip2 block data] origPtr error " << origPtr;
        throw std::domain_error( std::move( msg ).str() );
    }

    /* Turn the byte counts into the positions of the first occurrence of each byte in the sorted string. */
    std::array<uint32_t, 256> cumulativeCounts{};
    uint32_t sum = 0;
    for ( size_t i = 0; i < byteCounts.size(); ++i ) {
        cumulativeCounts[i] = sum;
        sum += byteCounts[i];
    }

    inverter.decode( lastToFirstMapping.get(), cumulativeCounts, size, origPtr, records.get(), bwtData.get() );
    inverseBWT = InverseBWTData{};
}


inline size_t
Block::BurrowsWheelerTransformData::decodeBlock( const size_t nMaxBytesToDecode,
                                                 char*        outputBuffer )
{
    if ( ( outputBuffer == nullptr ) || !hasData() ) {
        return 0;
    }

    size_t nBytesDecoded = 0;
    uint32_t crc = dataCRC;

    if ( inverseBWT ) {
        auto& state = *inverseBWT;
        while ( nBytesDecoded < nMaxBytesToDecode ) {
            if ( state.repeatCount > 0 ) {
                const auto nBytesToRepeat = std::min( state.repeatCount, nMaxBytesToDecode - nBytesDecoded );
                for ( size_t i = 0; i < nBytesToRepeat; ++i ) {
                    outputBuffer[nBytesDecoded++] = static_cast<char>( state.repeatByte );
                }
                state.repeatCount -= nBytesToRepeat;
                continue;
            }

            if ( state.position >= writeCount ) {
                break;
            }

        #if LIBRAPIDARCHIVE_BZIP2_AVX512_SCAN
            if ( hasAvx512RLEScan() ) {
                expandRLEAvx512( bwtData.get(), writeCount, state.position, outputBuffer,
                                 nBytesDecoded, nMaxBytesToDecode, state.repeatByte, state.repeatCount );
                if ( ( nBytesDecoded >= nMaxBytesToDecode ) || ( state.position >= writeCount ) ) {
                    break;
                }
            }
        #endif

        #if LIBRAPIDARCHIVE_BZIP2_HAS_SSE2 || defined( __ARM_NEON )
            /* Scan 16 positions at once. The scalar loop short-circuits after the first mismatch, but it still costs
             * one load and one compare per literal. A 128-bit/16-byte compare finds the first run start and lets whole
             * blocks of literals be copied with one load and one store. */
            bool handledRun = false;
            while ( ( nBytesDecoded + 16 <= nMaxBytesToDecode )
                    && ( state.position + 20 <= writeCount ) ) {
                const auto* const source = bwtData.get() + state.position;
                /* Simply load from byte-shifted offsets instead of load + shift to avoid instruction dependencies
                 * and to effectively load 16 + 3 bytes as we need to compare 4 subsequent bytes for each position. */
            #if LIBRAPIDARCHIVE_BZIP2_HAS_SSE2
                const auto v0 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( source ) );
                const auto v1 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( source + 1 ) );
                const auto v2 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( source + 2 ) );
                const auto v3 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( source + 3 ) );
                /* all four neighbors equal = ( ( v0 == v1 ) & ( v0 == v2 ) ) & ( v0 == v3 ) */
                const auto equal = _mm_and_si128( _mm_and_si128( _mm_cmpeq_epi8( v0, v1 ),
                                                                 _mm_cmpeq_epi8( v0, v2 ) ),
                                                  _mm_cmpeq_epi8( v0, v3 ) );
                /* Fast-path for only literals, no run-lengths. */
                const auto runStartMask = static_cast<uint32_t>( _mm_movemask_epi8( equal ) );
                if ( runStartMask == 0 ) {
                    _mm_storeu_si128( reinterpret_cast<__m128i*>( outputBuffer + nBytesDecoded ), v0 );
                    nBytesDecoded += sizeof( __m128i );
                    state.position += sizeof( __m128i );
                    continue;
                }

                const auto literalCount = static_cast<size_t>( rapidgzip::trailingZeroCount( runStartMask ) );
            #else
                const auto v0 = vld1q_u8( source );
                const auto v1 = vld1q_u8( source + 1 );
                const auto v2 = vld1q_u8( source + 2 );
                const auto v3 = vld1q_u8( source + 3 );
                const auto equal = vandq_u8( vandq_u8( vceqq_u8( v0, v1 ), vceqq_u8( v0, v2 ) ),
                                             vceqq_u8( v0, v3 ) );
                const auto runStartMask = equalByteMask( equal );
                if ( runStartMask == 0 ) {
                    vst1q_u8( reinterpret_cast<uint8_t*>( outputBuffer + nBytesDecoded ), v0 );
                    nBytesDecoded += 16;
                    state.position += 16;
                    continue;
                }

                const auto literalCount = static_cast<size_t>( __builtin_ctzll( runStartMask ) / 4 );
            #endif
                for ( size_t i = 0; i < literalCount; ++i ) {
                    outputBuffer[nBytesDecoded + i] = static_cast<char>( source[i] );
                }
                nBytesDecoded += literalCount;
                state.position += literalCount;
                state.repeatByte = source[literalCount];
                state.repeatCount = 4 + bwtData[state.position + 4];
                state.position += 5;
                handledRun = true;
                break;
            }
            if ( handledRun ) {
                continue;
            }
            /* The inner loop can fill the caller's buffer exactly. Recheck before the scalar path,
             * which would otherwise write one byte past nMaxBytesToDecode. */
            if ( ( nBytesDecoded >= nMaxBytesToDecode ) || ( state.position >= writeCount ) ) {
                break;
            }
        #endif

            const auto byte = bwtData[state.position];
            /* Four identical bytes are followed by a count of additional copies. */
            if ( ( state.position + 4 < writeCount )
                 && ( bwtData[state.position + 1] == byte )
                 && ( bwtData[state.position + 2] == byte )
                 && ( bwtData[state.position + 3] == byte ) ) {
                state.repeatByte = byte;
                state.repeatCount = 4 + bwtData[state.position + 4];
                state.position += 5;
            } else {
                outputBuffer[nBytesDecoded++] = static_cast<char>( byte );
                ++state.position;
            }
        }
    }

    /* The bytes of this call are contiguous in outputBuffer, so the CRC can be computed in one slicing
     * pass after the run expansion instead of interleaved with it. */
    crc = updateCRC32( crc, outputBuffer, nBytesDecoded );
    dataCRC = crc;

    /* decompression of this block completed successfully */
    if ( inverseBWT && ( inverseBWT->position >= writeCount ) && ( inverseBWT->repeatCount == 0 ) ) {
        dataCRC = ~dataCRC;
        if ( dataCRC != headerCRC ) {
            std::stringstream msg;
            msg << "Calculated CRC " << std::hex << dataCRC << " for block mismatches " << headerCRC;
            throw std::runtime_error( std::move( msg ).str() );
        }
    }

    return nBytesDecoded;
}
}  // namespace bzip2
