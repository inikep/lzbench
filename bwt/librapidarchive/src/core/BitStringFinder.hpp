#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <core/common.hpp>
#include <core/BitManipulation.hpp>
#include <core/simd.hpp>
#include <filereader/FileReader.hpp>


namespace rapidgzip
{
namespace simd
{
namespace bitStringFinder
{
[[nodiscard]] inline bool
isSupported( const SimdKernel kernel ) noexcept
{
    switch ( kernel ) {
    case SimdKernel::AUTOMATIC:
    case SimdKernel::PORTABLE:
        return true;
    #if LIBRAPIDARCHIVE_BIT_STRING_FINDER_X86_KERNELS
    case SimdKernel::SSE2:
        return __builtin_cpu_supports( "sse2" ) != 0;
    case SimdKernel::AVX2:
        return __builtin_cpu_supports( "avx2" ) != 0;
    case SimdKernel::AVX512:
        return ( __builtin_cpu_supports( "avx512f" ) != 0 ) && ( __builtin_cpu_supports( "avx512bw" ) != 0 );
    #else
    case SimdKernel::SSE2:
    case SimdKernel::AVX2:
    case SimdKernel::AVX512:
        return false;
    #endif
    case SimdKernel::SSE41:
    case SimdKernel::NEON:
        return false;
    }
    return false;
}


[[nodiscard]] inline SimdKernel
automaticKernel() noexcept
{
#if LIBRAPIDARCHIVE_BIT_STRING_FINDER_X86_KERNELS
    if ( isSupported( SimdKernel::AVX512 ) ) {
        return SimdKernel::AVX512;
    }
    if ( isSupported( SimdKernel::AVX2 ) ) {
        return SimdKernel::AVX2;
    }
    if ( isSupported( SimdKernel::SSE2 ) ) {
        return SimdKernel::SSE2;
    }
#endif
    /* The portable word scan is not faster than the string_view reference on hosts without x86 SIMD,
     * so leave that decision to the caller by reporting that no fast kernel was selected. */
    return SimdKernel::AUTOMATIC;
}


/**
 * Allows selecting a kernel for A/B measurements and for testing every compiled path. This is a
 * debug-only facility and is disabled unless LIBRAPIDARCHIVE_ENABLE_KERNEL_OVERRIDE is defined.
 * Production always uses the automatic selection.
 */
[[nodiscard]] inline SimdKernel
selectedKernel() noexcept
{
#ifdef LIBRAPIDARCHIVE_ENABLE_KERNEL_OVERRIDE
    static const auto kernel =
        [] () noexcept
        {
            if ( const auto candidate = kernelFromEnvironment(
                     "RAPIDGZIP_BIT_STRING_FINDER_SIMD",
                     { SimdKernel::PORTABLE, SimdKernel::SSE2, SimdKernel::AVX2, SimdKernel::AVX512 } ) ) {
                if ( isSupported( *candidate ) ) {
                    return *candidate;
                }
            }
            return automaticKernel();
        }();
    return kernel;
#else
    return automaticKernel();
#endif
}
}  // namespace bitStringFinder
}  // namespace simd


namespace detail
{
/**
 * The bzip2 block magic is 48 bit long, but block boundaries are bit-aligned, so a byte-aligned search
 * has to try all eight bit shifts. Instead of running eight independent std::string_view::find passes,
 * the kernels below scan the buffer once for the first byte of the byte-aligned middle of the magic for
 * each shift and verify a candidate with one unaligned 64-bit load.
 *
 * For a shift r, the middle of the S-bit pattern is the low S - 8 bits of bitString >> r, which starts
 * at a byte boundary. Once its first byte matches, the whole S-bit window starting at bit r of the byte
 * before the middle is checked at once.
 */

using BitStringFinderKernel = simd::SimdKernel;


[[nodiscard]] inline unsigned int
trailingZeroCount64( const uint64_t value ) noexcept
{
#if defined( __GNUC__ ) || defined( __clang__ )
    return static_cast<unsigned int>( __builtin_ctzll( value ) );
#else
    const auto lower = static_cast<uint32_t>( value );
    if ( lower != 0 ) {
        return trailingZeroCount( lower );
    }
    return 32U + trailingZeroCount( static_cast<uint32_t>( value >> 32U ) );
#endif
}


struct BitStringSearchPattern
{
    uint64_t bitString{ 0 };
    uint64_t mask{ 0 };
    uint8_t bitStringSize{ 0 };
    uint8_t needleSize{ 0 };
    std::array<uint8_t, CHAR_BIT> firstByteOfShift{};
    std::array<uint8_t, 256> shiftsByFirstByte{};
};


[[nodiscard]] inline BitStringSearchPattern
makeBitStringSearchPattern( const uint64_t bitString,
                            const uint8_t  bitStringSize ) noexcept
{
    BitStringSearchPattern pattern;
    pattern.bitString = bitString & nLowestBitsSet<uint64_t>( bitStringSize );
    pattern.mask = nLowestBitsSet<uint64_t>( bitStringSize );
    pattern.bitStringSize = bitStringSize;
    pattern.needleSize = static_cast<uint8_t>( bitStringSize / CHAR_BIT - 1U );
    pattern.shiftsByFirstByte.fill( 0 );

    for ( uint8_t shift = 0; shift < CHAR_BIT; ++shift ) {
        const auto firstByte = static_cast<uint8_t>(
            ( pattern.bitString >> shift ) >> static_cast<uint8_t>( bitStringSize - 16U ) );
        pattern.firstByteOfShift[shift] = firstByte;
        pattern.shiftsByFirstByte[firstByte] |= static_cast<uint8_t>( 1U << shift );
    }

    return pattern;
}


/**
 * @param needleOffset byte offset of the byte-aligned middle of the pattern.
 * @param shift bit shift r with which the pattern starts inside byte @p needleOffset - 1.
 * @return whether the S-bit window starting at bit @p shift of byte @p needleOffset - 1 equals the pattern.
 */
[[nodiscard]] inline bool
matchesBitStringAt( const uint8_t*                    buffer,
                    const size_t                      bufferSize,
                    const size_t                      needleOffset,
                    const uint8_t                     shift,
                    const BitStringSearchPattern&     pattern ) noexcept
{
    assert( needleOffset > 0 );

    /* A nonzero shift reads the byte after the middle as the tail. A zero shift does not, which is why
     * the scan range ends at bufferSize - needleSize instead of one byte earlier. */
    if ( ( shift > 0 ) && ( needleOffset + pattern.needleSize >= bufferSize ) ) {
        return false;
    }

    const auto startByte = needleOffset - 1U;
    const auto available = std::min<size_t>( sizeof( uint64_t ), bufferSize - startByte );
    uint64_t window = 0;
    if ( available == sizeof( uint64_t ) ) {
        window = byteSwap( loadUnaligned<uint64_t>( buffer + startByte ) );
    } else {
        for ( size_t i = 0; i < available; ++i ) {
            window |= static_cast<uint64_t>( buffer[startByte + i] ) << ( 56U - 8U * i );
        }
    }

    const auto windowShift = static_cast<uint8_t>( sizeof( uint64_t ) * CHAR_BIT - pattern.bitStringSize - shift );
    return ( ( window >> windowShift ) & pattern.mask ) == pattern.bitString;
}


inline void
appendMatchingShiftsAt( const uint8_t*                buffer,
                        const size_t                  bufferSize,
                        const size_t                  needleOffset,
                        uint8_t                       shifts,
                        const BitStringSearchPattern& pattern,
                        std::vector<size_t>&          offsets )
{
    while ( shifts != 0 ) {
        const auto shift = static_cast<uint8_t>( trailingZeroCount( shifts ) );
        shifts &= static_cast<uint8_t>( shifts - 1U );
        if ( matchesBitStringAt( buffer, bufferSize, needleOffset, shift, pattern ) ) {
            offsets.push_back( ( needleOffset - 1U ) * CHAR_BIT + shift );
        }
    }
}


inline void
appendMatchingCandidates( const uint8_t*                buffer,
                          const size_t                  bufferSize,
                          const size_t                  firstNeedleOffset,
                          const uint64_t                candidateMask,
                          const BitStringSearchPattern& pattern,
                          std::vector<size_t>&          offsets )
{
    auto remaining = candidateMask;
    while ( remaining != 0 ) {
        const auto bit = static_cast<size_t>( trailingZeroCount64( remaining ) );
        remaining &= remaining - 1U;

        const auto needleOffset = firstNeedleOffset + bit;
        appendMatchingShiftsAt( buffer, bufferSize, needleOffset,
                                pattern.shiftsByFirstByte[buffer[needleOffset]], pattern, offsets );
    }
}


inline void
findBitStringsScalarRange( const uint8_t*                buffer,
                           const size_t                  bufferSize,
                           const size_t                  firstNeedleOffset,
                           const size_t                  lastNeedleOffset,
                           const BitStringSearchPattern& pattern,
                           std::vector<size_t>&          offsets )
{
    auto needleOffset = firstNeedleOffset;
    const auto end = lastNeedleOffset + 1U;

    /* Most bytes are not a candidate, so check eight of them and branch only once. This keeps the
     * portable path close to the byte-scan throughput of the previous string_view implementation. */
    constexpr size_t UNROLL = 8;
    for ( ; needleOffset + UNROLL <= end; needleOffset += UNROLL ) {
        const auto* const bytes = buffer + needleOffset;
        const auto shifts0 = pattern.shiftsByFirstByte[bytes[0]];
        const auto shifts1 = pattern.shiftsByFirstByte[bytes[1]];
        const auto shifts2 = pattern.shiftsByFirstByte[bytes[2]];
        const auto shifts3 = pattern.shiftsByFirstByte[bytes[3]];
        const auto shifts4 = pattern.shiftsByFirstByte[bytes[4]];
        const auto shifts5 = pattern.shiftsByFirstByte[bytes[5]];
        const auto shifts6 = pattern.shiftsByFirstByte[bytes[6]];
        const auto shifts7 = pattern.shiftsByFirstByte[bytes[7]];
        if ( ( shifts0 | shifts1 | shifts2 | shifts3 | shifts4 | shifts5 | shifts6 | shifts7 ) == 0 ) {
            continue;
        }
        if ( shifts0 != 0 ) {
            appendMatchingShiftsAt( buffer, bufferSize, needleOffset + 0, shifts0, pattern, offsets );
        }
        if ( shifts1 != 0 ) {
            appendMatchingShiftsAt( buffer, bufferSize, needleOffset + 1, shifts1, pattern, offsets );
        }
        if ( shifts2 != 0 ) {
            appendMatchingShiftsAt( buffer, bufferSize, needleOffset + 2, shifts2, pattern, offsets );
        }
        if ( shifts3 != 0 ) {
            appendMatchingShiftsAt( buffer, bufferSize, needleOffset + 3, shifts3, pattern, offsets );
        }
        if ( shifts4 != 0 ) {
            appendMatchingShiftsAt( buffer, bufferSize, needleOffset + 4, shifts4, pattern, offsets );
        }
        if ( shifts5 != 0 ) {
            appendMatchingShiftsAt( buffer, bufferSize, needleOffset + 5, shifts5, pattern, offsets );
        }
        if ( shifts6 != 0 ) {
            appendMatchingShiftsAt( buffer, bufferSize, needleOffset + 6, shifts6, pattern, offsets );
        }
        if ( shifts7 != 0 ) {
            appendMatchingShiftsAt( buffer, bufferSize, needleOffset + 7, shifts7, pattern, offsets );
        }
    }

    for ( ; needleOffset <= lastNeedleOffset; ++needleOffset ) {
        appendMatchingShiftsAt( buffer, bufferSize, needleOffset,
                                pattern.shiftsByFirstByte[buffer[needleOffset]], pattern, offsets );
    }
}


inline void
findBitStringsPortable( const uint8_t*                buffer,
                        const size_t                  bufferSize,
                        const BitStringSearchPattern& pattern,
                        std::vector<size_t>&          offsets )
{
    if ( bufferSize < static_cast<size_t>( pattern.needleSize ) + 1U ) {
        return;
    }
    findBitStringsScalarRange( buffer, bufferSize, 1, bufferSize - pattern.needleSize, pattern, offsets );
}


#if LIBRAPIDARCHIVE_COMPILE_X86_SIMD_KERNELS
__attribute__( ( target( "sse2" ) ) ) inline void
findBitStringsSSE2( const uint8_t*                buffer,
                    const size_t                  bufferSize,
                    const BitStringSearchPattern& pattern,
                    std::vector<size_t>&          offsets )
{
    if ( bufferSize < static_cast<size_t>( pattern.needleSize ) + 1U ) {
        return;
    }
    const auto lastNeedleOffset = bufferSize - pattern.needleSize;

    const auto firstByte0 = _mm_set1_epi8( static_cast<char>( pattern.firstByteOfShift[0] ) );
    const auto firstByte1 = _mm_set1_epi8( static_cast<char>( pattern.firstByteOfShift[1] ) );
    const auto firstByte2 = _mm_set1_epi8( static_cast<char>( pattern.firstByteOfShift[2] ) );
    const auto firstByte3 = _mm_set1_epi8( static_cast<char>( pattern.firstByteOfShift[3] ) );
    const auto firstByte4 = _mm_set1_epi8( static_cast<char>( pattern.firstByteOfShift[4] ) );
    const auto firstByte5 = _mm_set1_epi8( static_cast<char>( pattern.firstByteOfShift[5] ) );
    const auto firstByte6 = _mm_set1_epi8( static_cast<char>( pattern.firstByteOfShift[6] ) );
    const auto firstByte7 = _mm_set1_epi8( static_cast<char>( pattern.firstByteOfShift[7] ) );

    constexpr size_t VECTOR_SIZE = 16;
    size_t needleOffset = 1;
    for ( ; needleOffset + VECTOR_SIZE <= lastNeedleOffset + 1U; needleOffset += VECTOR_SIZE ) {
        const auto data = _mm_loadu_si128( reinterpret_cast<const __m128i*>( buffer + needleOffset ) );
        auto matches = _mm_cmpeq_epi8( data, firstByte0 );
        matches = _mm_or_si128( matches, _mm_cmpeq_epi8( data, firstByte1 ) );
        matches = _mm_or_si128( matches, _mm_cmpeq_epi8( data, firstByte2 ) );
        matches = _mm_or_si128( matches, _mm_cmpeq_epi8( data, firstByte3 ) );
        matches = _mm_or_si128( matches, _mm_cmpeq_epi8( data, firstByte4 ) );
        matches = _mm_or_si128( matches, _mm_cmpeq_epi8( data, firstByte5 ) );
        matches = _mm_or_si128( matches, _mm_cmpeq_epi8( data, firstByte6 ) );
        matches = _mm_or_si128( matches, _mm_cmpeq_epi8( data, firstByte7 ) );
        const auto candidateMask = static_cast<uint32_t>( _mm_movemask_epi8( matches ) );
        if ( candidateMask != 0 ) {
            appendMatchingCandidates( buffer, bufferSize, needleOffset, candidateMask, pattern, offsets );
        }
    }

    findBitStringsScalarRange( buffer, bufferSize, needleOffset, lastNeedleOffset, pattern, offsets );
}


__attribute__( ( target( "avx2" ) ) ) inline void
findBitStringsAVX2( const uint8_t*                buffer,
                    const size_t                  bufferSize,
                    const BitStringSearchPattern& pattern,
                    std::vector<size_t>&          offsets )
{
    if ( bufferSize < static_cast<size_t>( pattern.needleSize ) + 1U ) {
        return;
    }
    const auto lastNeedleOffset = bufferSize - pattern.needleSize;

    const auto firstByte0 = _mm256_set1_epi8( static_cast<char>( pattern.firstByteOfShift[0] ) );
    const auto firstByte1 = _mm256_set1_epi8( static_cast<char>( pattern.firstByteOfShift[1] ) );
    const auto firstByte2 = _mm256_set1_epi8( static_cast<char>( pattern.firstByteOfShift[2] ) );
    const auto firstByte3 = _mm256_set1_epi8( static_cast<char>( pattern.firstByteOfShift[3] ) );
    const auto firstByte4 = _mm256_set1_epi8( static_cast<char>( pattern.firstByteOfShift[4] ) );
    const auto firstByte5 = _mm256_set1_epi8( static_cast<char>( pattern.firstByteOfShift[5] ) );
    const auto firstByte6 = _mm256_set1_epi8( static_cast<char>( pattern.firstByteOfShift[6] ) );
    const auto firstByte7 = _mm256_set1_epi8( static_cast<char>( pattern.firstByteOfShift[7] ) );

    constexpr size_t VECTOR_SIZE = 32;
    size_t needleOffset = 1;
    for ( ; needleOffset + VECTOR_SIZE <= lastNeedleOffset + 1U; needleOffset += VECTOR_SIZE ) {
        const auto data = _mm256_loadu_si256( reinterpret_cast<const __m256i*>( buffer + needleOffset ) );
        auto matches = _mm256_cmpeq_epi8( data, firstByte0 );
        matches = _mm256_or_si256( matches, _mm256_cmpeq_epi8( data, firstByte1 ) );
        matches = _mm256_or_si256( matches, _mm256_cmpeq_epi8( data, firstByte2 ) );
        matches = _mm256_or_si256( matches, _mm256_cmpeq_epi8( data, firstByte3 ) );
        matches = _mm256_or_si256( matches, _mm256_cmpeq_epi8( data, firstByte4 ) );
        matches = _mm256_or_si256( matches, _mm256_cmpeq_epi8( data, firstByte5 ) );
        matches = _mm256_or_si256( matches, _mm256_cmpeq_epi8( data, firstByte6 ) );
        matches = _mm256_or_si256( matches, _mm256_cmpeq_epi8( data, firstByte7 ) );
        const auto candidateMask = static_cast<uint32_t>( _mm256_movemask_epi8( matches ) );
        if ( candidateMask != 0 ) {
            appendMatchingCandidates( buffer, bufferSize, needleOffset, candidateMask, pattern, offsets );
        }
    }

    findBitStringsScalarRange( buffer, bufferSize, needleOffset, lastNeedleOffset, pattern, offsets );
}


__attribute__( ( target( "avx512f,avx512bw" ) ) ) inline void
findBitStringsAVX512( const uint8_t*                buffer,
                      const size_t                  bufferSize,
                      const BitStringSearchPattern& pattern,
                      std::vector<size_t>&          offsets )
{
    if ( bufferSize < static_cast<size_t>( pattern.needleSize ) + 1U ) {
        return;
    }
    const auto lastNeedleOffset = bufferSize - pattern.needleSize;

    const auto firstByte0 = _mm512_set1_epi8( static_cast<char>( pattern.firstByteOfShift[0] ) );
    const auto firstByte1 = _mm512_set1_epi8( static_cast<char>( pattern.firstByteOfShift[1] ) );
    const auto firstByte2 = _mm512_set1_epi8( static_cast<char>( pattern.firstByteOfShift[2] ) );
    const auto firstByte3 = _mm512_set1_epi8( static_cast<char>( pattern.firstByteOfShift[3] ) );
    const auto firstByte4 = _mm512_set1_epi8( static_cast<char>( pattern.firstByteOfShift[4] ) );
    const auto firstByte5 = _mm512_set1_epi8( static_cast<char>( pattern.firstByteOfShift[5] ) );
    const auto firstByte6 = _mm512_set1_epi8( static_cast<char>( pattern.firstByteOfShift[6] ) );
    const auto firstByte7 = _mm512_set1_epi8( static_cast<char>( pattern.firstByteOfShift[7] ) );

    constexpr size_t VECTOR_SIZE = 64;
    size_t needleOffset = 1;
    for ( ; needleOffset + VECTOR_SIZE <= lastNeedleOffset + 1U; needleOffset += VECTOR_SIZE ) {
        const auto data = _mm512_loadu_si512( buffer + needleOffset );
        const auto matches =
            _mm512_cmpeq_epi8_mask( data, firstByte0 )
            | _mm512_cmpeq_epi8_mask( data, firstByte1 )
            | _mm512_cmpeq_epi8_mask( data, firstByte2 )
            | _mm512_cmpeq_epi8_mask( data, firstByte3 )
            | _mm512_cmpeq_epi8_mask( data, firstByte4 )
            | _mm512_cmpeq_epi8_mask( data, firstByte5 )
            | _mm512_cmpeq_epi8_mask( data, firstByte6 )
            | _mm512_cmpeq_epi8_mask( data, firstByte7 );
        if ( matches != 0 ) {
            appendMatchingCandidates( buffer, bufferSize, needleOffset,
                                      static_cast<uint64_t>( matches ), pattern, offsets );
        }
    }

    findBitStringsScalarRange( buffer, bufferSize, needleOffset, lastNeedleOffset, pattern, offsets );
}
#endif  // LIBRAPIDARCHIVE_COMPILE_X86_SIMD_KERNELS


template<uint8_t bitStringSize>
[[nodiscard]] std::vector<size_t>
findBitStringsFast( std::string_view const&   buffer,
                    const uint64_t            bitString,
                    BitStringFinderKernel     kernel )
{
    static_assert( ( bitStringSize >= 2 * CHAR_BIT ) && ( bitStringSize <= 56 ),
                   "The fast finder needs a pattern that fits into a 64-bit window." );

    if ( ( kernel == BitStringFinderKernel::AUTOMATIC ) || !simd::bitStringFinder::isSupported( kernel ) ) {
        kernel = simd::bitStringFinder::selectedKernel();
    }
    if ( ( kernel == BitStringFinderKernel::AUTOMATIC ) || !simd::bitStringFinder::isSupported( kernel ) ) {
        kernel = BitStringFinderKernel::PORTABLE;
    }

    const auto pattern = makeBitStringSearchPattern( bitString, bitStringSize );
    const auto* const data = reinterpret_cast<const uint8_t*>( buffer.data() );
    const auto size = buffer.size();

    std::vector<size_t> offsets;
    switch ( kernel ) {
#if LIBRAPIDARCHIVE_COMPILE_X86_SIMD_KERNELS
    case BitStringFinderKernel::AVX512:
        findBitStringsAVX512( data, size, pattern, offsets );
        break;
    case BitStringFinderKernel::AVX2:
        findBitStringsAVX2( data, size, pattern, offsets );
        break;
    case BitStringFinderKernel::SSE2:
        findBitStringsSSE2( data, size, pattern, offsets );
        break;
#else
    case BitStringFinderKernel::AVX512:
    case BitStringFinderKernel::AVX2:
    case BitStringFinderKernel::SSE2:
        break;
    #endif
    case BitStringFinderKernel::AUTOMATIC:
    case BitStringFinderKernel::PORTABLE:
    case BitStringFinderKernel::SSE41:
    case BitStringFinderKernel::NEON:
        findBitStringsPortable( data, size, pattern, offsets );
        break;
    }

    /* The reference implementation visited the eight bit shifts in order. Keep that sequence so that
     * direct callers see exactly the same offsets in the same order. Equal shifts are kept stable. */
    if ( offsets.size() > 1 ) {
        std::stable_sort( offsets.begin(), offsets.end(),
                          [] ( const size_t a, const size_t b ) { return ( a % CHAR_BIT ) < ( b % CHAR_BIT ); } );
    }
    return offsets;
}


template<uint8_t bitStringSize>
[[nodiscard]] std::vector<size_t>
findBitStringsGeneric( std::string_view const& buffer,
                       uint64_t                bitString );
}  // namespace detail

/**
 * No matter the input, the data is read from an input buffer.
 * If a file is given, then that input buffer will be refilled when the input buffer empties.
 * It is less a file object and acts more like an iterator.
 * It offers a @ref find method returning the next match or std::nullopt if the end was reached.
 */
template<uint8_t bitStringSize>
class BitStringFinder
{
public:
    using ShiftedLUTTable = std::vector<std::pair</* shifted value to compare to */ uint64_t, /* mask */ uint64_t> >;

public:
    BitStringFinder( BitStringFinder&& ) noexcept = default;

    BitStringFinder( const BitStringFinder& other ) = delete;

    BitStringFinder&
    operator=( const BitStringFinder& ) = delete;

    BitStringFinder&
    operator=( BitStringFinder&& ) noexcept = delete;

    BitStringFinder( UniqueFileReader fileReader,
                     uint64_t         bitStringToFind,
                     size_t           fileBufferSizeBytes = 1_Mi ) :
        m_bitStringToFind( bitStringToFind & nLowestBitsSet<uint64_t>( bitStringSize ) ),
        m_movingBitsToKeep( bitStringSize > 0 ? bitStringSize - 1U : 0U ),
        m_movingBytesToKeep( ceilDiv( m_movingBitsToKeep, CHAR_BIT ) ),
        m_fileReader( std::move( fileReader ) ),
        m_fileChunksInBytes( std::max( fileBufferSizeBytes,
                                       static_cast<size_t>( ceilDiv( bitStringSize, CHAR_BIT ) ) ) )
    {
        if ( m_movingBytesToKeep >= m_fileChunksInBytes ) {
            std::stringstream msg;
            msg << "The file buffer size of " << m_fileChunksInBytes << "B is too small to look for strings with "
                << bitStringSize << " bits!";
            throw std::invalid_argument( std::move( msg ).str() );
        }
    }

    /** @note This overload is used for the tests but can also be useful for other things. */
    /* False positive because of delegating constructor. */
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    BitStringFinder( const char* buffer,
                     size_t      size,
                     uint64_t    bitStringToFind ) :
        BitStringFinder( UniqueFileReader(), bitStringToFind )
    {
        m_buffer.assign( buffer, buffer + size );
    }

    virtual
    ~BitStringFinder() = default;

    /** Stops internal worker threads. The sequential base finder has none. */
    virtual void
    stopThreads()
    {}

    [[nodiscard]] bool
    seekable() const
    {
        /* If m_fileReader is not set, then we are working on an in-memory buffer, which can be seeked. */
        return !m_fileReader || m_fileReader->seekable();
    }

    [[nodiscard]] bool
    eof() const
    {
        if ( m_fileReader ) {
            return bufferEof() && m_fileReader->eof();
        }
        return m_buffer.empty();
    }

    /**
     * @return the next match or std::nullopt if the end was reached.
     */
    [[nodiscard]] virtual std::optional<uint64_t>
    find();

protected:
    [[nodiscard]] bool
    bufferEof() const
    {
        return m_bufferBitsRead >= m_buffer.size() * CHAR_BIT;
    }

    size_t
    refillBuffer();

public:
    /**
     * @param bitString the lowest bitStringSize bits will be looked for in the buffer
     */
    [[nodiscard]] static std::vector<size_t>
    findBitStrings( std::string_view const& buffer,
                    uint64_t                bitString );

protected:
    const uint64_t m_bitStringToFind;

    /**
     * If the bit string is only one bit long, then we don't need to keep bits from the current buffer.
     * However, for a 2 bit string, one bit might be at the end of the current and the other at the beginning
     * of the next chunk, so we need to keep the last byte of that buffer but then mark the first 7 bits read
     * to avoid returning duplicate offsets!
     * For 8 bits, at worst 7 bits are in the current buffer and 1 in the next, so we need to keep 1 byte
     * and mark 1 bit read in the new buffer.
     * For 9 bits, keep 8 bits (1B) and mark 0 bits read.
     * This gives the formula for the bytes to keep as: ceil( ( bitStringSize - 1 ) / 8 ).
     */
    const uint8_t m_movingBitsToKeep;
    const uint8_t m_movingBytesToKeep;

    std::vector<char> m_buffer;
    std::vector<size_t> m_offsetsInBuffer;
    /**
     * How many bits from m_buffer bits are already read. The first bit string comparison will be done
     * after m_nTotalBytesRead * CHAR_BIT + m_bufferBitsRead >= bitStringSize
     */
    size_t m_bufferBitsRead = 0;

    UniqueFileReader m_fileReader;

    /** This is not the current size of @ref m_buffer but the number of bytes to read from @ref m_file if it is empty */
    const size_t m_fileChunksInBytes;
    /**
     * This value is incremented whenever the buffer is refilled. It basically acts like an overflow counter
     * for @ref m_bufferBitsRead and is required to return the absolute bit pos.
     */
    uint64_t m_nTotalBytesRead = 0;
};


template<uint8_t bitStringSize>
std::vector<size_t>
BitStringFinder<bitStringSize>::findBitStrings( std::string_view const& buffer,
                                                uint64_t                bitString )
{
    if constexpr ( ( bitStringSize >= 2 * CHAR_BIT ) && ( bitStringSize <= 56 ) ) {
        const auto kernel = simd::bitStringFinder::selectedKernel();
        if ( kernel != detail::BitStringFinderKernel::AUTOMATIC ) {
            return detail::findBitStringsFast<bitStringSize>( buffer, bitString, kernel );
        }
    }
    return detail::findBitStringsGeneric<bitStringSize>( buffer, bitString );
}


template<uint8_t bitStringSize>
std::vector<size_t>
detail::findBitStringsGeneric( std::string_view const& buffer,
                               uint64_t                bitString )
{
    static_assert( ( bitStringSize >= CHAR_BIT ) && ( bitStringSize % CHAR_BIT == 0 ),
                   "This is a highly optimized bit string finder for bzip2 magic bytes." );

    const auto findStrings =
        [] ( std::string_view const& data,
             std::string_view const& stringToFind )
        {
            std::vector<size_t> blockOffsets;
            for ( auto position = data.find( stringToFind, 0 );
                  position != std::string_view::npos;
                  position = data.find( stringToFind, position + 1U ) )
            {
                blockOffsets.push_back( position );
            }
            return blockOffsets;
        };

    const auto msbToString =
        [] ( uint64_t const bitStringToConvert,
             uint8_t  const bitStringtoConvertSize )
        {
            std::vector<char> result( bitStringtoConvertSize / CHAR_BIT );
            auto remainingSize = bitStringtoConvertSize;
            for ( auto& value : result ) {
                remainingSize -= CHAR_BIT;
                value = static_cast<char>( ( bitStringToConvert >> remainingSize ) & 0xFFU );
            }
            return result;
        };

    const auto headMatches =
        [&] ( size_t   offset,
              uint32_t nBitsBefore )
        {
            if ( nBitsBefore == 0 ) {
                return true;
            }
            if ( ( offset == 0 ) || ( ( offset - 1 ) >= buffer.size() ) ) {
                return false;
            }
            return ( static_cast<uint8_t>( buffer[offset - 1] ) & nLowestBitsSet<uint64_t>( nBitsBefore ) )
                   == ( ( bitString >> ( bitStringSize - nBitsBefore ) ) & nLowestBitsSet<uint64_t>( nBitsBefore ) );
        };

    const auto tailMatches =
        [&] ( size_t offset,
              uint32_t nBitsAfter )
        {
            if ( nBitsAfter == 0 ) {
                return true;
            }
            if ( offset >= buffer.size() ) {
                return false;
            }
            return ( static_cast<uint64_t>( static_cast<uint8_t>( buffer[offset] ) ) >> ( CHAR_BIT - nBitsAfter ) )
                   == ( bitString & nLowestBitsSet<uint64_t>( nBitsAfter ) );
        };

    std::vector<size_t> blockOffsets;
    for ( uint32_t shift = 0U; shift < CHAR_BIT; ++shift ) {
        const auto stringToFind = msbToString( bitString >> shift, bitStringSize - CHAR_BIT );
        const auto newBlockOffsets = findStrings( buffer, { stringToFind.data(), stringToFind.size() } );

        const auto nBitsAfter = shift;
        const auto nBitsBefore = CHAR_BIT - shift;

        assert( stringToFind.size() == bitStringSize / CHAR_BIT - 1U );
        for ( const auto offset : newBlockOffsets ) {
            if ( headMatches( offset, nBitsBefore ) && tailMatches( offset + stringToFind.size(), nBitsAfter ) ) {
                blockOffsets.push_back( offset * CHAR_BIT - nBitsBefore );
            }
        }
    }

    return blockOffsets;
}


template<uint8_t bitStringSize>
std::optional<uint64_t>
BitStringFinder<bitStringSize>::find()
{
    if ( bitStringSize == 0 ) {
        return std::nullopt;
    }

    if ( !m_offsetsInBuffer.empty() ) {
        const auto offset = m_nTotalBytesRead * CHAR_BIT + m_offsetsInBuffer.back();
        m_offsetsInBuffer.pop_back();
        return offset;
    }

    while ( !eof() )
    {
        if ( bufferEof() ) {
            const auto nBytesRead = refillBuffer();
            if ( nBytesRead == 0 ) {
                return std::nullopt;
            }
        }

        m_offsetsInBuffer = findBitStrings( { m_buffer.data(), m_buffer.size() }, m_bitStringToFind );

        const auto firstBitsToIgnore = static_cast<uint8_t>( m_bufferBitsRead % CHAR_BIT );
        m_offsetsInBuffer.erase(
            std::remove_if( m_offsetsInBuffer.begin(), m_offsetsInBuffer.end(),
                            [firstBitsToIgnore] ( auto offset ) { return offset < firstBitsToIgnore; } ),
            m_offsetsInBuffer.end()
        );

        /* Sort descending so that we can pop from back successively. */
        std::sort( m_offsetsInBuffer.begin(), m_offsetsInBuffer.end(), [] ( auto a, auto b ) { return a > b; } );

        m_bufferBitsRead = m_buffer.size() * CHAR_BIT;
        if ( !m_offsetsInBuffer.empty() ) {
            const auto offset = m_nTotalBytesRead * CHAR_BIT + m_offsetsInBuffer.back();
            m_offsetsInBuffer.pop_back();
            return offset;
        }
    }

    return std::nullopt;
}


template<uint8_t bitStringSize>
size_t
BitStringFinder<bitStringSize>::refillBuffer()
{
    if ( !m_fileReader || m_fileReader->eof() ) {
        m_nTotalBytesRead += m_buffer.size();
        m_buffer.clear();
        return 0;
    }

    /* Read chunk of data from file into buffer. */
    size_t nBytesRead = 0;
    if ( m_buffer.empty() ) {
        assert( m_nTotalBytesRead == 0 );
        assert( m_bufferBitsRead == 0 );

        m_buffer.resize( m_fileChunksInBytes );
        nBytesRead = m_fileReader->read( m_buffer.data(), m_buffer.size() );
        m_buffer.resize( nBytesRead );
    } else {
        m_nTotalBytesRead += m_buffer.size() - m_movingBytesToKeep;
        m_bufferBitsRead = m_movingBytesToKeep * CHAR_BIT - m_movingBitsToKeep;

        /* On subsequent refills, keep the last bits in order to find bit strings on buffer boundaries. */
        std::memmove( m_buffer.data(), m_buffer.data() + m_buffer.size() - m_movingBytesToKeep, m_movingBytesToKeep );

        const auto nBytesToRead = m_buffer.size() - m_movingBytesToKeep;
        nBytesRead = m_fileReader->read( m_buffer.data() + m_movingBytesToKeep, nBytesToRead );
        m_buffer.resize( m_movingBytesToKeep + nBytesRead );
    }

    return nBytesRead;
}
}  // namespace rapidgzip
