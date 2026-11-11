#pragma once

#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <sys/stat.h>

#include <core/BitManipulation.hpp>
#include <core/common.hpp>
#include <filereader/FileReader.hpp>
#include <filereader/Shared.hpp>


namespace rapidgzip
{
/**
 * @todo Make BitReader access with copying and such work for input stream.
 *       This might need another abstraction layer which keeps chunks of data of the file until it has been read once!
 *       Normally, the BZ2 reader should indeed read everything once, meaning nothing should "leak".
 *       That abstraction layer would also open the file only a single time and then hold a mutex locked shared_ptr
 *       to it.
 * This bit reader, returns bits in the order appropriate for bzip2, i.e., it goes over all bytes in order
 * and then returns the bits of each byte >starting from the most significant<. This is contrary to the usual
 * bit numbering starting from the least-significant one and also contrary to to DEFLATE (RFC 1951)!
 * @see https://github.com/dsnet/compress/blob/master/doc/bzip2-format.pdf
 * @see https://tools.ietf.org/html/rfc1951
 */
template<bool MOST_SIGNIFICANT_BITS_FIRST,
         typename BitBuffer>
class BitReader final :
    public FileReader
{
public:
    static_assert( std::is_unsigned_v<BitBuffer>, "Bit buffer type must be unsigned!" );

    using bit_count_t = uint32_t;

    /**
     * If it is too large, then the use case of only reading one Bzip2 block per opened BitReader
     * will load much more data than necessary because of the too large buffer.
     * The size should also be a multiple of the block size of the underlying device.
     * Any power of 2 larger than 4096 (4k blocks) should be safe bet.
     * 4K is too few, and will lead to a 2x slowdown in some test because of the frequent buffer refills.
     */
    static constexpr size_t DEFAULT_BUFFER_REFILL_SIZE = 128_Ki;
    static constexpr int NO_FILE = -1;
    static constexpr auto MAX_BIT_BUFFER_SIZE = std::numeric_limits<BitBuffer>::digits;

    class BitReaderException :
        public std::exception
    {};

    /**
     * This exception is thrown by refillBitBuffer when the I/O buffer has been emptied.
     * Because the I/O buffer is rather large compared to the bit buffer, it only empties infrequently.
     * Using an exception instead of conditionals **doubled** performance in a synthetic BitReader benchmark
     * and improved the gzip decoder performance by ~20%!
     */
    class BufferNeedsToBeRefilled :
        public BitReaderException
    {};

    struct Statistics
    {
        size_t byteBufferRefillCount{ 0 };
        size_t bitBufferRefillCount{ 0 };
    };

public:
    explicit
    BitReader( UniqueFileReader fileReader,
               const size_t     bufferRefillSize = DEFAULT_BUFFER_REFILL_SIZE ) :
        /* The UniqueFileReader input argument sufficiently conveys and ensures that the file ownership is taken.
         * But, because BitReader has a fileno getter returning the underlying fileno, it is possible that the
         * file position is changed from the outside. To still keep correct behavior in that case, we have to
         * to make it a SharedFileReader, which keeps track of the intended file position. */
        m_file( ensureSharedFileReader( std::move( fileReader ) ) ),
        m_bufferRefillSize( bufferRefillSize )
    {
        if ( m_bufferRefillSize == 0 ) {
            throw std::invalid_argument( "The buffer size must be larger than zero!" );
        }
    }

    ~BitReader() override = default;

    BitReader( BitReader&& other ) noexcept = default;

    BitReader&
    operator=( BitReader&& other ) noexcept = default;

    BitReader&
    operator=( const BitReader& other ) = delete;

    BitReader( const BitReader& other ) :
        m_file( other.m_file ? other.m_file->clone() : UniqueFileReader() ),
        m_bufferRefillSize( other.m_bufferRefillSize ),
        m_inputBuffer( other.m_inputBuffer )
    {
        if ( dynamic_cast<const SharedFileReader*>( other.m_file.get() ) == nullptr ) {
            throw std::invalid_argument( "Cannot copy BitReader if does not contain a SharedFileReader!" );
        }

        assert( static_cast<bool>( m_file ) == static_cast<bool>( other.m_file ) );
        if ( UNLIKELY( m_file && !m_file->seekable() ) ) [[unlikely]] {
            throw std::invalid_argument( "Copying BitReader to unseekable file not supported yet!" );
        }
        seek( other.tell() );
    }

    /* File Reader Interface Implementation */

protected:
    [[nodiscard]] UniqueFileReader
    cloneRaw() const override
    {
        return std::make_unique<BitReader>( *this );
    }

public:
    [[nodiscard]] bool
    fail() const override
    {
        throw std::logic_error( "Not implemented" );
    }

    /**
     * @note This function is not as cheap as one might thing because it dynamically tests for EOF using the
     *       position and size instead of checking an internal (cached) flag!
     */
    [[nodiscard]] bool
    eof() const override
    {
        if ( const auto fileSize = size(); seekable() && fileSize.has_value() ) {
            return tell() >= *fileSize;
        }
        return ( m_inputBufferPosition >= m_inputBuffer.size() ) && ( !m_file || m_file->eof() );
    }

    [[nodiscard]] bool
    seekable() const override
    {
        return !m_file || m_file->seekable();
    }

    void
    close() override
    {
        m_file.reset();
        m_inputBuffer.clear();
        m_inputBufferPosition = 0;
        clearBitBuffer();
    }

    [[nodiscard]] bool
    closed() const override
    {
        return !m_file && m_inputBuffer.empty();
    }

    /* Bit Reading Methods */

    /**
     * Forcing to inline this function is superimportant because it depends even between gcc versions,
     * whether it is actually inlined or not but inlining can save 30%!
     * Note that splitting part of this method off into readSafe made the compiler actually
     * inline this now small function and thereby sped up runtimes significantly!
     */
    forceinline BitBuffer
    read( bit_count_t bitsWanted )
    {
        /* Handling bitsWanted == 0 here would incur a 75% slowdown for the benchmark reading single bits!
         * Just let the caller handle that case. Performance comes first, especially at this steep price for safety. */
        assert( bitsWanted > 0 );
        /* Reading the whole buffer is not allowed because it would require another expensive rarely used branch! */
        assert( bitsWanted < MAX_BIT_BUFFER_SIZE );

        if ( LIKELY( bitsWanted <= bitBufferSize() ) ) [[likely]] {
            const auto result = peekUnsafe( bitsWanted );
            seekAfterPeek( bitsWanted );
            return result;
        }

        return read2( bitsWanted );
    }

private:
    BitBuffer
    read2( bit_count_t bitsWanted )
    {
        const auto bitsInResult = bitBufferSize();
        assert( bitsWanted >= bitsInResult );
        const auto bitsNeeded = bitsWanted - bitsInResult;
        BitBuffer bits{ 0 };

        /* Read out the remaining bits from m_bitBuffer to the lowest bits in @ref bits.
         * This is copy-pasted from @ref peekUnsafe because for MSB, we don't have to do the expensive
         * bitBufferSize() == 0 branching! */
        if constexpr ( MOST_SIGNIFICANT_BITS_FIRST ) {
            bits = m_bitBuffer & N_LOWEST_BITS_SET_LUT<BitBuffer>[bitBufferSize()];
        } else {
            bits = bitBufferSize() == 0
                   ? BitBuffer( 0 )
                   : ( m_bitBuffer >> m_bitBufferFree )
                     & N_LOWEST_BITS_SET_LUT<BitBuffer>[bitBufferSize()];
        }

        /* If the system endianness matches the BitReader endianness and the byte buffer contains enough bytes
         * for the requested number of bits, then refill the whole bit buffer with one unaligned memory read.
         * This makes the assumption that read2 is only ever called when all the current bit buffer bits are
         * not enough. */
        if constexpr ( !MOST_SIGNIFICANT_BITS_FIRST && ( ENDIAN != Endian::UNKNOWN ) ) {
            constexpr bit_count_t BYTES_WANTED = sizeof( BitBuffer );
            constexpr bit_count_t BITS_WANTED = sizeof( BitBuffer ) * CHAR_BIT;

            if ( LIKELY( m_inputBufferPosition + BYTES_WANTED < m_inputBuffer.size() ) ) [[likely]] {
                m_originalBitBufferSize = BITS_WANTED;
                m_bitBufferFree = MAX_BIT_BUFFER_SIZE - BITS_WANTED;
                m_bitBuffer = loadUnaligned<BitBuffer>( &m_inputBuffer[m_inputBufferPosition] );

                m_inputBufferPosition += BYTES_WANTED;

                bits |= peekUnsafe( bitsNeeded ) << bitsInResult;
                seekAfterPeek( bitsNeeded );

                m_statistics.bitBufferRefillCount++;
                return bits;
            }
        }

        clearBitBuffer();
        try {
            fillBitBuffer();
        } catch ( const BufferNeedsToBeRefilled& ) {
            refillBuffer();
            try {
                refillBitBuffer();
            } catch ( const BufferNeedsToBeRefilled& ) {
                /* When fillBitBuffer does not throw, then it has been filled almost completely and it is ensured
                 * that we have enough bits as long as fewer than the bit buffer size were requested.
                 * Removing this if from the non-throwing frequent path, improves performance measurably! */
                if ( UNLIKELY( bitsNeeded > bitBufferSize() ) ) [[unlikely]] {
                    throw EndOfFileReached();
                }
            }
        }

        /* Append remaining requested bits. */
        if constexpr ( MOST_SIGNIFICANT_BITS_FIRST ) {
            bits = ( bits << bitsNeeded ) | peekUnsafe( bitsNeeded );
        } else {
            bits |= peekUnsafe( bitsNeeded ) << bitsInResult;
        }
        seekAfterPeek( bitsNeeded );

        return bits;
    }

public:
    /**
     * This is a performant unchecked helper to seek forward the same amount that has already been peeked.
     * Calling this function without calling peek beforehand with the same number of bits may corrupt the BitReader!
     */
    forceinline void
    seekAfterPeek( bit_count_t bitsWanted )
    {
        assert( bitsWanted <= bitBufferSize() );
        m_bitBufferFree += bitsWanted;
    }

    /**
     * Forcing to inline this function is superimportant because it depends even between gcc versions,
     * whether it is actually inlined or not but inlining can save 30%!
     */
    template<uint8_t bitsWanted>
    forceinline BitBuffer
    read()
    {
        if constexpr ( bitsWanted == 0 ) {
            return 0;
        } else {
            static_assert( bitsWanted <= MAX_BIT_BUFFER_SIZE, "Requested bits must fit in buffer!" );
            return read( bitsWanted );
        }
    }

    /**
     * @return Number of bytes read.
     */
    [[nodiscard]] size_t
    read( char*  outputBuffer,
          size_t nBytesToRead ) override
    {
        const auto oldTell = tell();

        if ( UNLIKELY( outputBuffer == nullptr ) ) [[unlikely]] {
            /* FileReader::read(nullptr, n) skips n bytes. seek() works in bits here, so convert and avoid
             * wrapping the signed multiplication for very large requests. */
            const auto maxSkippableBytes = static_cast<size_t>( std::numeric_limits<long long int>::max() / CHAR_BIT );
            const auto bytesToSkip = std::min( nBytesToRead, maxSkippableBytes );
            seek( static_cast<long long int>( bytesToSkip ) * CHAR_BIT, SEEK_CUR );
        } else if ( UNLIKELY( oldTell % CHAR_BIT != 0 ) ) [[unlikely]] {
            for ( size_t i = 0; i < nBytesToRead; ++i ) {
                outputBuffer[i] = static_cast<char>( read( CHAR_BIT ) );
            }
        } else {
            size_t nBytesRead{ 0 };
            /* Should be true because of oldTell % BYTE_SIZE == 0! */
            assert( bitBufferSize() % CHAR_BIT == 0 );

            /* 1. Empty bit buffer */
            for ( ; ( nBytesRead < nBytesToRead ) && ( bitBufferSize() >= CHAR_BIT ); ++nBytesRead ) {
                outputBuffer[nBytesRead] = static_cast<char>( peekUnsafe( CHAR_BIT ) );
                seekAfterPeek( CHAR_BIT );
            }

            /* 2. Empty byte buffer */
            nBytesRead += readFromBuffer( outputBuffer + nBytesRead, nBytesToRead - nBytesRead );

            /* 3. a) Read directly from underlying file or
             *    b) Refill byte buffer and read from it to avoid many small calls to FileReader::read. */
            const auto nBytesToReadFromFile = nBytesToRead - nBytesRead;
            if ( UNLIKELY( ( nBytesToReadFromFile > 0 ) && m_file ) ) [[unlikely]] {
                assert( m_inputBufferPosition == m_inputBuffer.size() );
                if ( nBytesToRead < std::min<size_t>( 1_Ki, m_bufferRefillSize ) ) {
                    /* Because nBytesToRead < m_bufferRefillSize, refilling the buffer once will suffice to read the
                     * requested amount of bytes or else we have reached EOF. */
                    refillBuffer();
                    readFromBuffer( outputBuffer + nBytesRead, nBytesToRead - nBytesRead );
                } else {
                    if ( ( nBytesToReadFromFile > 0 ) && m_file ) {
                        /* We don't need the return value because we are using tell! */
                        [[maybe_unused]] const auto nBytesReadFromFile =
                            m_file->read( outputBuffer + nBytesRead, nBytesToReadFromFile );
                        /* Clear the byte buffer because the assumed invariant that the byte buffer contains the
                         * data of m_file->tell() - m_inputBuffer.size() is wrong after reading from the file and
                         * will result in a bug when seeking back and when the supposed offset is thought to be
                         * inside the buffer. This can be triggered now after adding sparse windows because
                         * InflateWrapper will refill the buffer with this method and getUsedWindowSymbols will
                         * seek back. Before sparse windows, it might have been virtually impossible to trigger this.
                         * @todo This understanding gives two optimization ideas:
                         *       1. For InflateWrapper + getUsedWindowSymbols, reduce the byte buffer size to something
                         *          <= 32 KiB instead of 128 KiB to avoid unnecessarily large refills.
                         *       2. Check that InflateWrapper only calls this function with byte-aligned offsets
                         *          so that it doesn't trigger the possibly slower other path that has to shift
                         *          everything by a constant amount of bits. */
                        m_inputBufferPosition = 0;
                        m_inputBuffer.clear();
                    }
                }
            }
        }

        const auto nBitsRead = tell() - oldTell;
        if ( UNLIKELY( nBitsRead % CHAR_BIT != 0 ) ) [[unlikely]] {
            throw std::runtime_error( "Read not a multiple of CHAR_BIT, probably because EOF was encountered!" );
        }
        return nBitsRead / CHAR_BIT;
    }

    template<uint8_t bitsWanted>
    forceinline BitBuffer
    peek()
    {
        if constexpr ( bitsWanted == 0 ) {
            return 0;
        } else {
            static_assert( bitsWanted <= MAX_BIT_BUFFER_SIZE, "Requested bits must fit in buffer!" );
            return peek( bitsWanted );
        }
    }

private:
    BitBuffer
    peek2( bit_count_t bitsWanted )
    {
        assert( ( bitsWanted <= MAX_BIT_BUFFER_SIZE - ( CHAR_BIT - 1 ) )
                && "The last 7 bits of the buffer may not be readable because we can only refill 8-bits at a time." );
        assert( bitsWanted > 0 );

        if ( UNLIKELY( bitsWanted > bitBufferSize() ) ) [[unlikely]] {
            if constexpr ( !MOST_SIGNIFICANT_BITS_FIRST && ( ENDIAN != Endian::UNKNOWN ) ) {
                if ( LIKELY( m_inputBufferPosition + sizeof( BitBuffer ) < m_inputBuffer.size() ) ) [[likely]] {
                    /* There is no way around this special case because of the damn undefined behavior when shifting! */
                    if ( bitBufferSize() == 0 ) {
                        m_originalBitBufferSize = sizeof( BitBuffer ) * CHAR_BIT;
                        m_bitBufferFree = MAX_BIT_BUFFER_SIZE - sizeof( BitBuffer ) * CHAR_BIT;
                        m_bitBuffer = loadUnaligned<BitBuffer>( &m_inputBuffer[m_inputBufferPosition] );

                        m_inputBufferPosition += sizeof( BitBuffer );
                        return peekUnsafe( bitsWanted );
                    }

                    const auto shrinkedBitBufferSize = ceilDiv( bitBufferSize(), CHAR_BIT ) * CHAR_BIT;
                    const auto bitsToLoad  = MAX_BIT_BUFFER_SIZE - shrinkedBitBufferSize;
                    const auto bytesToLoad = bitsToLoad / CHAR_BIT;

                    /* Load new bytes directly to the left of the (virtually) shrunk bit buffer.
                     * This is possibly because read but still "loaded" (m_originalBitBufferSize) bits are to the
                     * right. */
                    const auto bytesToAppend = loadUnaligned<BitBuffer>( &m_inputBuffer[m_inputBufferPosition] );
                    m_bitBuffer = ( m_bitBuffer >> bitsToLoad )
                                  | ( bytesToAppend << ( MAX_BIT_BUFFER_SIZE - bitsToLoad ) );

                    m_originalBitBufferSize = MAX_BIT_BUFFER_SIZE;
                    m_bitBufferFree -= bitsToLoad;
                    m_inputBufferPosition += bytesToLoad;

                    return peekUnsafe( bitsWanted );
                }
            }

            try {
                /* In the case of the shortcut for filling the bit buffer by reading 64-bit, don't inline
                 * the very rarely used fallback to keep this function rather small for inlining. */
                if constexpr ( !MOST_SIGNIFICANT_BITS_FIRST && ( ENDIAN != Endian::UNKNOWN ) ) {
                    /* This point should only happen rarely, e.g., when the byte buffer needs to be refilled. */
                    refillBitBuffer();
                } else {
                    if ( bitBufferSize() == 0 ) {
                        m_bitBuffer = 0;
                        m_originalBitBufferSize = 0;
                    } else {
                        shrinkBitBuffer();

                        if constexpr ( !MOST_SIGNIFICANT_BITS_FIRST ) {
                            m_bitBuffer >>= static_cast<uint8_t>( MAX_BIT_BUFFER_SIZE - m_originalBitBufferSize );
                        }
                    }

                    fillBitBuffer();
                }
            } catch ( const BufferNeedsToBeRefilled& ) {
                refillBuffer();
                try {
                    refillBitBuffer();
                } catch ( const BufferNeedsToBeRefilled& ) {
                    /* When fillBitBuffer does not throw, then it has been filled almost completely and it is ensured
                     * that we have enough bits as long as fewer than the bit buffer size were requested.
                     * Removing this if from the non-throwing frequent path, improves performance measurably! */
                    if ( UNLIKELY( bitsWanted > bitBufferSize() ) ) [[unlikely]] {
                        throw EndOfFileReached();
                    }
                }
            }
        }

        return peekUnsafe( bitsWanted );
    }

public:
    forceinline BitBuffer
    peek( bit_count_t bitsWanted )
    {
        if ( UNLIKELY( bitsWanted > bitBufferSize() ) ) [[unlikely]] {
            return peek2( bitsWanted );
        }
        return peekUnsafe( bitsWanted );
    }

    [[nodiscard]] std::pair<BitBuffer, size_t>
    peekAvailable() const
    {
        const auto availableBits = bitBufferSize();
        if ( availableBits == 0 ) {
            return { 0, 0 };
        }
        return { peekUnsafe( availableBits ), availableBits };
    }

    /**
     * @return current position / number of bits already read.
     */
    [[nodiscard]] size_t
    tell() const override
    {
        /* Initialize with the byte buffer position converted to bits. */
        size_t position = m_inputBufferPosition * CHAR_BIT;

        /* Add the file offset from which the byte buffer was read. */
        if ( m_file ) {
            const auto filePosition = m_file->tell();
            if ( UNLIKELY( static_cast<size_t>( filePosition ) < m_inputBuffer.size() ) ) [[unlikely]] {
                throw std::logic_error( "The byte buffer should not contain more data than the file position!" );
            }
            position += ( filePosition - m_inputBuffer.size() ) * CHAR_BIT;
        }

        /* Subtract the unread bits that have already been "moved" from the byte buffer to the bit buffer. */
        if ( UNLIKELY( position < bitBufferSize() ) ) [[unlikely]] {
            throw std::logic_error( "The bit buffer should not contain more data than have been read from the file!" );
        }

        return position - bitBufferSize();
    }

    void
    clearerr() override
    {
        if ( m_file ) {
            m_file->clearerr();
        }
    }

public:
    [[nodiscard]] int
    fileno() const override
    {
        if ( UNLIKELY( !m_file ) ) [[unlikely]] {
            throw std::invalid_argument( "The file is not open!" );
        }
        return m_file->fileno();
    }

    size_t
    seek( long long int offsetBits,
          int           origin = SEEK_SET ) override;

    [[nodiscard]] std::optional<size_t>
    size() const override
    {
        auto sizeInBytes = m_inputBuffer.size();
        if ( m_file ) {
            const auto fileSize = m_file->size();
            if ( !fileSize ) {
                return std::nullopt;
            }
            sizeInBytes = *fileSize;
        }
        return sizeInBytes * CHAR_BIT;
    }

    [[nodiscard]] const std::vector<std::uint8_t>&
    buffer() const
    {
        return m_inputBuffer;
    }

    /**
     * Together with @ref buffer and @ref seekInsideBuffer, this lets decoders extract the bits from the
     * buffered bytes with their own specialized loop.
     *
     * @return The offset of the next unread bit inside @ref buffer in bits, or nothing if the bit buffer
     *         still holds bits of the previous buffer contents, which is the case right after a refill.
     */
    [[nodiscard]] std::optional<size_t>
    bufferBitPosition() const noexcept
    {
        const auto bitsMovedToBitBuffer = m_inputBufferPosition * CHAR_BIT;
        const auto unreadBits = static_cast<size_t>( bitBufferSize() );
        if ( bitsMovedToBitBuffer < unreadBits ) {
            return std::nullopt;
        }
        return bitsMovedToBitBuffer - unreadBits;
    }

    /**
     * Continues reading at the given bit offset inside @ref buffer.
     */
    void
    seekInsideBuffer( const size_t offsetInBits )
    {
        if ( offsetInBits > m_inputBuffer.size() * CHAR_BIT ) {
            throw std::out_of_range( "The bit offset is outside of the buffered data!" );
        }

        clearBitBuffer();
        m_inputBufferPosition = offsetInBits / CHAR_BIT;
        if ( const auto bitsToSkip = offsetInBits % CHAR_BIT; bitsToSkip > 0 ) {
            read( static_cast<bit_count_t>( bitsToSkip ) );
        }
    }

    [[nodiscard]] constexpr uint64_t
    bufferRefillSize() const
    {
        return m_bufferRefillSize;
    }

    [[nodiscard]] constexpr Statistics
    statistics() const
    {
        return m_statistics;
    }

private:
    size_t
    fullSeek( size_t offsetBits );

    void
    refillBuffer()
    {
        if ( UNLIKELY( !m_file ) ) [[unlikely]] {
            throw std::logic_error( "Can not refill buffer with data from non-existing file!" );
        }

        const auto oldBufferSize = m_inputBuffer.size();
        m_inputBuffer.resize( m_bufferRefillSize );
        const auto nBytesRead = m_file->read( reinterpret_cast<char*>( m_inputBuffer.data() ), m_inputBuffer.size() );
        if ( nBytesRead == 0 ) {
            m_inputBuffer.resize( oldBufferSize );
            return;
        }

        m_inputBuffer.resize( nBytesRead );
        m_inputBufferPosition = 0;

        m_statistics.byteBufferRefillCount++;
    }

    /**
     * Decreases m_originalBitBufferSize by CHAR_BIT until it is as close to bitBufferSize() as possible
     * and clears all bits outside of m_originalBitBufferSize.
     */
    void
    shrinkBitBuffer()
    {
        if ( m_originalBitBufferSize == bitBufferSize() ) {
            return;
        }

        assert( ( m_originalBitBufferSize % CHAR_BIT == 0 ) &&
                "Not necessary but should be true because we only load byte-wise and only shrink byte-wise!" );
        assert( m_originalBitBufferSize >= bitBufferSize() );
        assert( m_originalBitBufferSize >= ceilDiv( bitBufferSize(), CHAR_BIT ) * CHAR_BIT );

        m_originalBitBufferSize = ceilDiv( bitBufferSize(), CHAR_BIT ) * CHAR_BIT;

        if constexpr ( MOST_SIGNIFICANT_BITS_FIRST ) {
            m_bitBuffer &= N_LOWEST_BITS_SET_LUT<BitBuffer>[m_originalBitBufferSize];
        } else {
            m_bitBuffer &= N_HIGHEST_BITS_SET_LUT<BitBuffer>[m_originalBitBufferSize];
        }
    }

    size_t
    readFromBuffer( void* const  outputBuffer,
                    size_t const nBytesToRead )
    {
        const auto nBytesReadFromBuffer = std::min( nBytesToRead, m_inputBuffer.size() - m_inputBufferPosition );
        if ( nBytesReadFromBuffer > 0 ) {
            std::memcpy( outputBuffer, m_inputBuffer.data() + m_inputBufferPosition, nBytesReadFromBuffer );
            m_inputBufferPosition += nBytesReadFromBuffer;
        }
        return nBytesReadFromBuffer;
    }

    void
    refillBitBuffer()
    {
        /* Skip refill if it already is full (except for up to 7 empty bits) */
        if ( bitBufferSize() + CHAR_BIT > MAX_BIT_BUFFER_SIZE ) {
            return;
        }

        if ( bitBufferSize() == 0 ) {
            m_bitBuffer = 0;
            m_originalBitBufferSize = 0;
        } else {
            shrinkBitBuffer();

            if constexpr ( !MOST_SIGNIFICANT_BITS_FIRST ) {
                assert( m_originalBitBufferSize > 0 );
                /* Always checking for m_originalBitBufferSize for this damn bit shift would be too cost-prohibitive.
                 * It should never happen because in this branch we know that bitBufferSize() > 0 and at any point
                 * in time m_originalBitBufferSize >= bitBufferSize() should be true!
                 * Run unit tests in debug mode to ensure that the assert won't be triggered. */
                // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
                m_bitBuffer >>= static_cast<uint8_t>( MAX_BIT_BUFFER_SIZE - m_originalBitBufferSize );
            }
        }

        fillBitBuffer();
    }

    void
    fillBitBuffer()
    {
        /* Using an ad-hoc destructor to emulate a 'finally' does not seem to hurt performance
         * when comparing it to manually copy-pasting the shift in all outer catch clauses. */
        struct ShiftBackOnReturn
        {
            ShiftBackOnReturn( BitBuffer&     bitBuffer,
                               const uint8_t& originalBitBufferSize ) noexcept :
                m_bitBuffer( bitBuffer ),
                m_originalBitBufferSize( originalBitBufferSize )
            {}

            ~ShiftBackOnReturn() noexcept
            {
                /* Move LSB bits (which are filled left-to-right) to the left if so necessary
                 * so that the format is the same as for MSB bits! */
                if constexpr ( !MOST_SIGNIFICANT_BITS_FIRST ) {
                    if ( m_originalBitBufferSize > 0 ) {
                        m_bitBuffer <<= static_cast<uint8_t>( MAX_BIT_BUFFER_SIZE - m_originalBitBufferSize );
                    }
                }
            }

        private:
            BitBuffer& m_bitBuffer;
            const uint8_t& m_originalBitBufferSize;
        } const shiftBackOnExit( m_bitBuffer, m_originalBitBufferSize );

        /* The MSB path always appends a whole number of bytes, so the remaining space can be filled with
         * one big-endian 64-bit load instead of one load and shift per byte. The load needs eight readable
         * bytes; near the end of the input buffer keep the byte-wise fallback. */
        if constexpr ( MOST_SIGNIFICANT_BITS_FIRST ) {
            const auto bytesToAppend = static_cast<size_t>(
                ( MAX_BIT_BUFFER_SIZE - m_originalBitBufferSize ) / CHAR_BIT );
            if ( ( bytesToAppend > 0 )
                 && ( m_inputBufferPosition + sizeof( uint64_t ) <= m_inputBuffer.size() ) ) {
                auto value = byteSwap( loadUnaligned<uint64_t>( m_inputBuffer.data() + m_inputBufferPosition ) );
                if ( bytesToAppend == sizeof( uint64_t ) ) {
                    m_bitBuffer = value;
                } else {
                    value >>= static_cast<uint8_t>( CHAR_BIT * ( sizeof( uint64_t ) - bytesToAppend ) );
                    m_bitBuffer = ( m_bitBuffer << static_cast<uint8_t>( CHAR_BIT * bytesToAppend ) ) | value;
                }
                m_inputBufferPosition += bytesToAppend;
                m_originalBitBufferSize += static_cast<uint8_t>( CHAR_BIT * bytesToAppend );
                m_bitBufferFree -= static_cast<uint32_t>( CHAR_BIT * bytesToAppend );
                m_statistics.bitBufferRefillCount++;
                return;
            }
        }

        /* Refill buffer one byte at a time to enforce endianness and avoid unaligned access. */
        for ( ; m_originalBitBufferSize + CHAR_BIT <= MAX_BIT_BUFFER_SIZE;
              m_bitBufferFree -= CHAR_BIT, m_originalBitBufferSize += CHAR_BIT )
        {
            if ( UNLIKELY( m_inputBufferPosition >= m_inputBuffer.size() ) ) [[unlikely]] {
                throw BufferNeedsToBeRefilled();
            }

            if constexpr ( MOST_SIGNIFICANT_BITS_FIRST ) {
                m_bitBuffer <<= static_cast<uint8_t>( CHAR_BIT );
                m_bitBuffer |= static_cast<BitBuffer>( m_inputBuffer[m_inputBufferPosition++] );
            } else {
                m_bitBuffer |= ( static_cast<BitBuffer>( m_inputBuffer[m_inputBufferPosition++] )
                                 << m_originalBitBufferSize );
                /**
                 * Avoiding the single shift before and after the loop for LSB by modifying how the bits are
                 * appended slows it down by ~10%, probably because one additional shift per loop iteration
                 * is necessary:
                 * @verbatim
                 * m_bitBuffer <<= CHAR_BIT;
                 * m_bitBuffer |= ( static_cast<BitBuffer>( m_inputBuffer[m_inputBufferPosition++] )
                 *                << ( MAX_BIT_BUFFER_SIZE - CHAR_BIT ) );
                 * @endverbatim
                 */
            }
        }

        m_statistics.bitBufferRefillCount++;
    }

    [[nodiscard]] forceinline BitBuffer
    peekUnsafe( bit_count_t bitsWanted ) const
    {
        assert( bitsWanted <= bitBufferSize() );
        assert( bitsWanted > 0 );

        if constexpr ( MOST_SIGNIFICANT_BITS_FIRST ) {
            return ( m_bitBuffer >> ( bitBufferSize() - bitsWanted ) )
                   & N_LOWEST_BITS_SET_LUT<BitBuffer>[bitsWanted];
        } else {
            assert( bitBufferSize() > 0 );
            /* Always checking for bitBufferSize() for this damn bit shift would be too cost-prohibitive.
             * It should only happen when the caller tries to read, e.g., 0 bits, in which case undefined behavior
             * for the shift result value does not matter. Run unit tests in debug mode to ensure that the assert
             * won't be triggered. */
            // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
            return ( m_bitBuffer >> m_bitBufferFree )
                   & N_LOWEST_BITS_SET_LUT<BitBuffer>[bitsWanted];
        }
    }

    forceinline void
    clearBitBuffer()
    {
        m_originalBitBufferSize = 0;
        m_bitBufferFree = MAX_BIT_BUFFER_SIZE;
        m_bitBuffer = 0;
    }

private:
    [[nodiscard]] forceinline constexpr auto
    bitBufferSize() const noexcept
    {
        return MAX_BIT_BUFFER_SIZE - m_bitBufferFree;
    }

private:
    UniqueFileReader m_file;

    size_t m_bufferRefillSize{ DEFAULT_BUFFER_REFILL_SIZE };
    std::vector<uint8_t> m_inputBuffer;
    size_t m_inputBufferPosition = 0;  /** stores current position of first valid byte in buffer */

    /* Performance profiling metrics. */
    Statistics m_statistics;

public:
    /**
     * For MOST_SIGNIFICANT_BITS_FIRST == true (bzip2):
     *
     * m_bitBuffer stores the last read bits from m_inputBuffer on the >right side< (if not fully filled).
     * The bits are to be read from >left to right< up to a maximum of m_bitBufferSize.
     * This means that not the least significant n bits are to be returned on read but the most significant.
     * E.g., for the following example requesting 3 bits should return 011, i.e., start from m_bitBufferSize
     * and return the requested amount of bits to the right of it. After that, decrement m_bitBufferSize.
     *
     * @verbatim
     *        result = 0b011
     *        bitsWanted = 3
     *            <->
     * +-------------------+
     * |    | 101|0111|0011|
     * +-------------------+
     *        ^   ^  ^
     *        |   |  m_bitBufferSize - bitsWanted = 5 = m_bitBufferSize after reading bitsWanted
     *        |   m_bitBufferSize = 8
     *        m_originalBitBufferSize = 11
     * @endverbatim
     *
     * For MOST_SIGNIFICANT_BITS_FIRST == false (gzip):
     *
     * Bit buffer stores the last read bits from m_inputBuffer on the >left side< of m_bitBuffer (if not fully filled).
     * Basically, this works in a mirrored way of MOST_SIGNIFICANT_BITS_FIRST == true in order to be able to only
     * require one size and not an additional offset position for the bit buffer.
     * Because the bits are to be read from >right to left<, the left-alignment makes the left-most bits always
     * those valid the longest. I.e., we only have to decrement m_bitBufferSize.
     *
     * @verbatim
     *   result = 0b111
     *   bitsWanted = 3
     *        <->
     * +-------------------+
     * |0101|0111|001 |    |
     * +-------------------+
     *       ^  ^   ^
     *       |  |   m_originalBitBufferSize = 11
     *       |  m_bitBufferSize = 8
     *       m_bitBufferSize - bitsWanted = 5
     * @endverbatim
     *
     * In both cases, the amount of bits wanted are extracted by shifting to the right and and'ing with a bit mask.
     */
    BitBuffer m_bitBuffer = 0;

    /**
     * Performance consideration for the used type were done with taskset because else the timings did vary more.
     * @verbatim
     * m rapidgzip && for i in $( seq 30 ); do
     *     taskset --cpu-list 5 src/tools/rapidgzip -P 1 -d -o /dev/null 10xSRR22403185_2.fastq.gz
     * done 2>&1 | tee output &&
     * uncertainValue $( sed -nr 's|.* ([0-9.]+) MB/s|\1|p' output )
     * @endverbatim
     *
     * Results:
     * @verbatim
     * Type     | Min    | Mean +- StdDev | Max
     * ---------+--------+----------------+-------
     * uint8_t  | 401.78 | 406.42 +- 0.05 | 408.59
     * uint16_t | 398.88 | 402.83 +- 0.05 | 405.33
     * uint32_t | 414.00 | 419.25 +- 0.05 | 422.02
     * uint64_t | 412.44 | 418.06 +- 0.07 | 420.72
     * int32_t  | 411.85 | 416.47 +- 0.05 | 419.51
     * @endverbatim
     *
     * Note that the range of bit buffer size should [0,64], i.e., whether it is 8-bit or 16-bit or 8-bit signed
     * or anything else, it does not matter. None of them represent the exact allowed range and are all larger.
     * Therefore, using anything else for performance reasons makes sense to me.
     * Changing the width of m_originalBitBufferSize to 32-bit does not help, it even worsens the performance
     * back to ~415 MB/s. Well, this is probably VERY compiler-dependent. No idea what it is thinking.
     */
    //uint32_t m_bitBufferSize = 0;  // size of bitbuffer in bits
    /**
     * Same time measurements as for @ref m_bitBufferSize. Storing m_bitBufferFree instead safes one subtractions
     * on every peekUnsafe! It was always: m_bitBuffer >> ( MAX_BIT_BUFFER_SIZE - m_bitBufferSize ) and not simply is:
     * m_bitBuffer >> m_bitBufferFree. This another ~1 %:
     * @verbatim
     * m_bitBufferFree: 419.05 | 424.46 +- 0.07 | 427.36
     * m_bitBufferSize: 414.00 | 419.25 +- 0.05 | 422.02
     * @endverbatim
     * Note that this might slow down the BZ2 decoder, but I guess at this point it isn't the main goal anymore
     * and it is much slower anyway, so probably the BitReader isn't the bottleneck. If it turns out to be, then
     * it will be hard to include both variants in the same class via constexpr ifs, it might be necessary to
     * split the BitReader class into two versions.
     */
    uint32_t m_bitBufferFree{ MAX_BIT_BUFFER_SIZE };

    uint8_t m_originalBitBufferSize = 0;  // size of valid bitbuffer bits including already read ones
};


template<bool MOST_SIGNIFICANT_BITS_FIRST, typename BitBuffer>
inline size_t
BitReader<MOST_SIGNIFICANT_BITS_FIRST, BitBuffer>::seek(
    long long int offsetBits,
    int           origin )
{
    if ( origin == SEEK_END ) {
        const auto fileSize = size();
        if ( !fileSize.has_value() ) {
            if ( !m_file ) {
                throw std::logic_error( "File has already been closed!" );
            }

            if ( !m_file->seekable() ) {
                throw std::logic_error( "File is not seekable!" );
            }

            const auto realFileSize = static_cast<long long int>( m_file->seek( 0, SEEK_END ) );
            /* Because we have seeked to get the full size, we have to force a full seek
             * to restore a valid file position for the next refillBuffer call. */
            const auto absoluteOffset = saturatingAddition( std::min( offsetBits, 0LL ), realFileSize );
            return fullSeek( static_cast<size_t>( std::max( absoluteOffset, 0LL ) ) );
        }
    }

    const auto positiveOffsetBits = effectiveOffset( offsetBits, origin );

    if ( positiveOffsetBits == tell() ) {
        return positiveOffsetBits;
    }

    if ( !seekable() && ( positiveOffsetBits < tell() ) ) {
        std::stringstream message;
        message << "File is not seekable! Requested to seek to " << formatBits( positiveOffsetBits )
                << ". Currently at: " << formatBits( tell() );
        throw std::invalid_argument( std::move( message ).str() );
    }

    /* Currently, buffer-only is not supported, use BufferedFileReader as a memory-only file reader! */
    if ( !m_file ) {
        throw std::logic_error( "File has already been closed!" );
    }

    /* Performance optimizations for faster seeking inside the buffer to avoid expensive refillBuffer calls. */
    if ( positiveOffsetBits >= tell() ) {
        const auto relativeOffsets = positiveOffsetBits - tell();
        /* Seek forward inside bit buffer. */
        if ( static_cast<size_t>( relativeOffsets ) <= bitBufferSize() ) {
            seekAfterPeek( static_cast<decltype( bitBufferSize() )>( relativeOffsets ) );
            return positiveOffsetBits;
        }

        /* Seek forward inside byte buffer. */
        const auto stillToSeek = relativeOffsets - bitBufferSize();
        const auto newInputBufferPosition = m_inputBufferPosition + stillToSeek / CHAR_BIT;
        if ( newInputBufferPosition <= m_inputBuffer.size() ) {
            clearBitBuffer();

            m_inputBufferPosition = newInputBufferPosition;
            if ( stillToSeek % CHAR_BIT > 0 ) {
                read( stillToSeek % CHAR_BIT );
            }

            return positiveOffsetBits;
        }
    } else {  /* Seek back. */
        const auto relativeOffsets = tell() - positiveOffsetBits;
        /* Seek back inside bit buffer. */
        if ( relativeOffsets + bitBufferSize() <= m_originalBitBufferSize ) {
            m_bitBufferFree -= static_cast<decltype( bitBufferSize() )>( relativeOffsets );
            return positiveOffsetBits;
        }

        const auto seekBackWithBuffer = relativeOffsets + bitBufferSize();
        const auto bytesToSeekBack = static_cast<size_t>( ceilDiv( seekBackWithBuffer, CHAR_BIT ) );
        /* Seek back inside byte buffer. */
        if ( bytesToSeekBack <= m_inputBufferPosition ) {
            m_inputBufferPosition -= static_cast<decltype( m_inputBufferPosition )>( bytesToSeekBack );
            clearBitBuffer();

            const auto bitsToSeekForward = bytesToSeekBack * CHAR_BIT - seekBackWithBuffer;
            if ( bitsToSeekForward > 0 ) {
                read( static_cast<uint8_t>( bitsToSeekForward ) );
            }

            return positiveOffsetBits;
        }
    }

    return fullSeek( positiveOffsetBits );
}


template<bool MOST_SIGNIFICANT_BITS_FIRST, typename BitBuffer>
inline size_t
BitReader<MOST_SIGNIFICANT_BITS_FIRST, BitBuffer>::fullSeek( size_t offsetBits )
{
    if ( !m_file ) {
        throw std::logic_error( "File has already been closed!" );
    }

    const auto bytesToSeek = offsetBits >> 3U;
    const auto subBitsToSeek = static_cast<bit_count_t>( offsetBits & 7U );

    clearBitBuffer();

    m_inputBuffer.clear();
    m_inputBufferPosition = 0;

    if ( seekable() ) {
        const auto newPosition = m_file->seek( static_cast<long long int>( bytesToSeek ), SEEK_SET );
        /* Note that the performance optimizations above allow to seek at exactly the file end without
         * an exception. Therefore, also allow that here for consistency. */
        if ( ( m_file->eof() && ( !m_file->seekable() || ( m_file->tell() > m_file->size() ) ) ) || m_file->fail() ) {
            std::stringstream msg;
            msg << "[BitReader] Could not seek to specified byte " << bytesToSeek
                << " subbit " << static_cast<int>( subBitsToSeek )
                << ", SharedFileReader: " << ( dynamic_cast<SharedFileReader*>( m_file.get() ) != nullptr )
                << ", SinglePassFileReader: " << ( dynamic_cast<SinglePassFileReader*>( m_file.get() ) != nullptr )
                << ", tell: " << m_file->tell()
                << ", size: " << m_file->size().value_or( 0 )
                << ", feof: " << m_file->eof()
                << ", ferror: " << m_file->fail()
                << ", newPosition: " << newPosition;
            throw std::invalid_argument( std::move( msg ).str() );
        }
    } else if ( offsetBits < tell() ) {
        throw std::logic_error( "Can not emulate backward seeking on non-seekable file!" );
    } else {
        /* Emulate forward seeking on non-seekable file by reading. */
        throw std::logic_error( "Seeking forward on non-seekable input is an unfinished feature!" );
    }

    if ( subBitsToSeek > 0 ) {
        read( subBitsToSeek );
    }

    return offsetBits;
}
}  // namespace rapidgzip
