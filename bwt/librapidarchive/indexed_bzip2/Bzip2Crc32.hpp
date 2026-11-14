#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

/* This guard selects the compilers and architectures that provide the intrinsic headers and the
 * function-level target attribute. The translation unit stays on the baseline ISA; the PCLMUL/SSSE3
 * functions below are compiled with target attributes and selected by __builtin_cpu_supports at
 * runtime. A compile-time __PCLMUL__ check would require -mpclmul for the whole translation unit and
 * defeat the runtime dispatch. */
#if ( defined( __GNUC__ ) || defined( __clang__ ) ) && ( defined( __x86_64__ ) || defined( __i386__ ) )
    #include <immintrin.h>
    #include <tmmintrin.h>
    #include <wmmintrin.h>
    #define LIBRAPIDARCHIVE_BZIP2_CRC_PCLMUL 1
#else
    #define LIBRAPIDARCHIVE_BZIP2_CRC_PCLMUL 0
#endif

namespace bzip2
{
using CRC32LookupTable = std::array<uint32_t, 256>;


[[nodiscard]] constexpr CRC32LookupTable
createCRC32LookupTable() noexcept
{
    constexpr auto littleEndian{ false };
    CRC32LookupTable table{};
    for ( uint32_t i = 0; i < table.size(); ++i ) {
        uint32_t c = littleEndian ? i : i << 24U;
        for ( int j = 0; j < 8; ++j ) {
            if ( littleEndian ) {
                c = ( ( c & 1U ) != 0U ) ? ( c >> 1U ) ^ 0xEDB8'8320U : c >> 1U;
            } else {
                c = ( ( ( c & 0x8000'0000U ) != 0U ) ) ? ( c << 1U ) ^ 0x04C1'1DB7U : ( c << 1U );
            }
        }
        table[i] = c;
    }
    return table;
}


static constexpr int CRC32_LOOKUP_TABLE_SIZE = 256;

/* a small lookup table: raw data -> CRC32 value to speed up CRC calculation */
alignas( 64 ) constexpr static CRC32LookupTable CRC32_TABLE = createCRC32LookupTable();


[[nodiscard]] constexpr uint32_t
updateCRC32( uint32_t crc,
             uint8_t  data ) noexcept
{
    //return ( crc >> 8U ) ^ CRC32_TABLE[( crc ^ data ) & 0xFFU];
    return ( crc << 8U ) ^ CRC32_TABLE[( ( crc >> 24U ) ^ data ) & 0xFFU];
}


/**
 * Number of bytes processed per iteration by the bzip2 CRC. bzip2 uses the non-reflected (MSB-first)
 * CRC32 with polynomial 0x04C11DB7, so it cannot use the CRC32 CPU instruction and needs slicing tables.
 * Larger slices are faster in isolation but need more lookup-table cache lines.
 */
#ifndef LIBRAPIDARCHIVE_BZIP2_CRC_SLICES
    #define LIBRAPIDARCHIVE_BZIP2_CRC_SLICES 8
#endif

static constexpr unsigned int BZIP2_CRC_SLICES = LIBRAPIDARCHIVE_BZIP2_CRC_SLICES;
static_assert( ( BZIP2_CRC_SLICES == 1 ) || ( BZIP2_CRC_SLICES == 4 ) || ( BZIP2_CRC_SLICES == 8 ),
               "LIBRAPIDARCHIVE_BZIP2_CRC_SLICES must be 1, 4, or 8." );


/**
 * @param SLICE_SIZE number of bytes the table is used for.
 * @return tables where table[k][byte] is the CRC of @p byte followed by k zero bytes, so that table[0]
 *         is the ordinary byte-wise table.
 */
template<unsigned int SLICE_SIZE>
[[nodiscard]] constexpr std::array<CRC32LookupTable, SLICE_SIZE>
createCRC32SliceTables() noexcept
{
    std::array<CRC32LookupTable, SLICE_SIZE> tables{};
    tables[0] = createCRC32LookupTable();
    for ( unsigned int i = 1; i < SLICE_SIZE; ++i ) {
        for ( unsigned int byte = 0; byte < 256; ++byte ) {
            tables[i][byte] = ( tables[i - 1][byte] << 8U ) ^ tables[0][tables[i - 1][byte] >> 24U];
        }
    }
    return tables;
}


/* Keep the tables at namespace scope, like the gzip CRC tables. */
template<unsigned int SLICE_SIZE>
alignas( 64 ) static constexpr std::array<CRC32LookupTable, SLICE_SIZE> CRC32_SLICE_TABLES =
    createCRC32SliceTables<SLICE_SIZE>();


/**
 * Slicing-by-@p SLICE_SIZE update for the non-reflected bzip2 CRC32.
 *
 * For a slice of N data bytes d[0..N-1], the first four bytes are XORed with the four current CRC
 * bytes and looked up in the tables for the remaining N-1..N-4 zero bytes, while the later data
 * bytes only depend on the tables for the smaller remaining shifts. The final @c size % N bytes use
 * the byte-wise table as the tail.
 */
template<unsigned int SLICE_SIZE>
[[nodiscard]] uint32_t
updateCRC32SliceByN( uint32_t       crc,
                     const uint8_t* data,
                     size_t         size ) noexcept
{
    static_assert( ( SLICE_SIZE == 1 ) || ( SLICE_SIZE == 4 ) || ( SLICE_SIZE == 8 ),
                   "SLICE_SIZE must be 1, 4, or 8." );

    if constexpr ( SLICE_SIZE == 1 ) {
        for ( size_t i = 0; i < size; ++i ) {
            crc = updateCRC32( crc, data[i] );
        }
        return crc;
    } else {
        constexpr const auto& TABLES = CRC32_SLICE_TABLES<SLICE_SIZE>;

        size_t i = 0;
        for ( ; i + SLICE_SIZE <= size; i += SLICE_SIZE ) {
            uint32_t result = 0;
        #if defined( __GNUC__ ) || defined( __clang__ )
            #pragma GCC unroll 8
        #endif
            for ( unsigned int j = 0; j < SLICE_SIZE; ++j ) {
                const auto byte = data[i + j];
                const auto index = j < 4 ? static_cast<uint8_t>( ( crc >> ( 24U - 8U * j ) ) ^ byte ) : byte;
                result ^= TABLES[SLICE_SIZE - 1 - j][index];
            }
            crc = result;
        }

        for ( ; i < size; ++i ) {
            crc = updateCRC32( crc, data[i] );
        }
        return crc;
    }
}


#if LIBRAPIDARCHIVE_BZIP2_CRC_PCLMUL
/**
 * The PCLMUL kernel interprets the CRC register and the message as polynomials over GF(2) with bit i
 * holding the coefficient of x^i. bzip2 uses the non-reflected polynomial P(x) = x^32 + 0x04C11DB7,
 * MSB-first bit order, so a 16-byte message block must be byte-reversed before it can be folded.
 *
 * Folding keeps a 128-bit polynomial that is congruent to the processed prefix modulo P. The final
 * 128-bit value is then fed as a 16-byte message into the scalar byte-wise CRC with initial state 0,
 * which performs the last reduction to 32 bits and appends any remaining tail bytes.
 */

/* x^exponent mod P for the non-reflected bzip2 polynomial. */
[[nodiscard]] constexpr uint32_t
crc32Bzip2XPower( unsigned int exponent ) noexcept
{
    uint32_t value = 1;  /* x^0 */
    for ( unsigned int i = 0; i < exponent; ++i ) {
        const auto shifted = static_cast<uint64_t>( value ) << 1U;
        value = static_cast<uint32_t>( shifted ^ ( ( shifted >> 32U ) != 0U ? 0x1'04C1'1DB7ULL : 0ULL ) );
    }
    return value;
}

static constexpr uint32_t CRC32_BZIP2_X128 = crc32Bzip2XPower( 128 );
static constexpr uint32_t CRC32_BZIP2_X192 = crc32Bzip2XPower( 192 );
static constexpr uint32_t CRC32_BZIP2_X512 = crc32Bzip2XPower( 512 );
static constexpr uint32_t CRC32_BZIP2_X576 = crc32Bzip2XPower( 576 );

/* Minimum buffer size above which the setup and the final 16-byte scalar reduction are amortized. */
static constexpr size_t CRC32_PCLMUL_MIN_SIZE = 256;

#define LIBRAPIDARCHIVE_BZIP2_CRC_TARGET __attribute__( ( target( "sse2,pclmul,ssse3" ) ) )

/* value * x^shift mod P. @p factorLow is x^shift and @p factorHigh is x^(shift + 64). */
[[nodiscard]]
LIBRAPIDARCHIVE_BZIP2_CRC_TARGET inline __m128i
crc32Bzip2Fold( __m128i value,
                __m128i factorLow,
                __m128i factorHigh ) noexcept
{
    return _mm_xor_si128( _mm_clmulepi64_si128( value, factorHigh, 0x01 ),
                          _mm_clmulepi64_si128( value, factorLow, 0x00 ) );
}


[[nodiscard]]
LIBRAPIDARCHIVE_BZIP2_CRC_TARGET inline __m128i
crc32Bzip2LoadBlock( const uint8_t* source,
                     __m128i        reverseBytes ) noexcept
{
    return _mm_shuffle_epi8( _mm_loadu_si128( reinterpret_cast<const __m128i*>( source ) ), reverseBytes );
}


[[nodiscard]]
LIBRAPIDARCHIVE_BZIP2_CRC_TARGET inline uint32_t
updateCRC32PCLMUL( uint32_t       crc,
                   const uint8_t* data,
                   size_t         size ) noexcept
{
    if ( size < 16 ) {
        return updateCRC32SliceByN<BZIP2_CRC_SLICES>( crc, data, size );
    }

    const auto reverseBytes = _mm_setr_epi8( 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 );
    const auto fold128Low = _mm_set_epi64x( 0, CRC32_BZIP2_X128 );
    const auto fold128High = _mm_set_epi64x( 0, CRC32_BZIP2_X192 );
    const auto fold512Low = _mm_set_epi64x( 0, CRC32_BZIP2_X512 );
    const auto fold512High = _mm_set_epi64x( 0, CRC32_BZIP2_X576 );
    /* The initial CRC is shifted by x^96 into the first block, as derived for the MSB-first reduction. */
    const auto initialCRC = _mm_set_epi32( static_cast<int>( crc ), 0, 0, 0 );

    const auto blockCount = size / 16;
    size_t block = 0;
    __m128i accumulator;

    if ( blockCount >= 4 ) {
        auto lane0 = _mm_xor_si128( crc32Bzip2LoadBlock( data, reverseBytes ), initialCRC );
        auto lane1 = crc32Bzip2LoadBlock( data + 16, reverseBytes );
        auto lane2 = crc32Bzip2LoadBlock( data + 32, reverseBytes );
        auto lane3 = crc32Bzip2LoadBlock( data + 48, reverseBytes );
        block = 4;
        /* Four independent folding chains hide the CLMUL latency; each lane advances by four blocks. */
        for ( ; block + 4 <= blockCount; block += 4 ) {
            const auto* const source = data + block * 16;
            lane0 = _mm_xor_si128( crc32Bzip2Fold( lane0, fold512Low, fold512High ),
                                   crc32Bzip2LoadBlock( source, reverseBytes ) );
            lane1 = _mm_xor_si128( crc32Bzip2Fold( lane1, fold512Low, fold512High ),
                                   crc32Bzip2LoadBlock( source + 16, reverseBytes ) );
            lane2 = _mm_xor_si128( crc32Bzip2Fold( lane2, fold512Low, fold512High ),
                                   crc32Bzip2LoadBlock( source + 32, reverseBytes ) );
            lane3 = _mm_xor_si128( crc32Bzip2Fold( lane3, fold512Low, fold512High ),
                                   crc32Bzip2LoadBlock( source + 48, reverseBytes ) );
        }
        /* The lanes are offset by one block each, so folding them together with a 128-bit stride restores
         * the sequential order. */
        accumulator = _mm_xor_si128( crc32Bzip2Fold( lane0, fold128Low, fold128High ), lane1 );
        accumulator = _mm_xor_si128( crc32Bzip2Fold( accumulator, fold128Low, fold128High ), lane2 );
        accumulator = _mm_xor_si128( crc32Bzip2Fold( accumulator, fold128Low, fold128High ), lane3 );
    } else {
        accumulator = _mm_xor_si128( crc32Bzip2LoadBlock( data, reverseBytes ), initialCRC );
        block = 1;
    }

    for ( ; block < blockCount; ++block ) {
        accumulator = _mm_xor_si128(
            crc32Bzip2Fold( accumulator, fold128Low, fold128High ),
            crc32Bzip2LoadBlock( data + block * 16, reverseBytes ) );
    }

    /* Serialize the folded polynomial in message order and let the scalar CRC perform the final reduction
     * and process the remaining bytes. */
    alignas( 32 ) uint8_t trailing[32];
    _mm_store_si128( reinterpret_cast<__m128i*>( trailing ),
                     _mm_shuffle_epi8( accumulator, reverseBytes ) );
    const auto tailSize = size - blockCount * 16;
    if ( tailSize > 0 ) {
        std::memcpy( trailing + 16, data + blockCount * 16, tailSize );
    }
    return updateCRC32SliceByN<BZIP2_CRC_SLICES>( 0, trailing, 16 + tailSize );
}
#undef LIBRAPIDARCHIVE_BZIP2_CRC_TARGET
#endif  /* LIBRAPIDARCHIVE_BZIP2_CRC_PCLMUL */


#if LIBRAPIDARCHIVE_BZIP2_CRC_PCLMUL \
    && ( ( defined( __GNUC__ ) && !defined( __clang__ ) && ( __GNUC__ >= 11 ) ) \
         || ( defined( __clang__ ) && ( __clang_major__ >= 12 ) ) )
    #define LIBRAPIDARCHIVE_BZIP2_CRC_VPCLMUL 1
#else
    #define LIBRAPIDARCHIVE_BZIP2_CRC_VPCLMUL 0
#endif

#if LIBRAPIDARCHIVE_BZIP2_CRC_VPCLMUL
/* The 256-bit variant folds two 128-bit lanes per register and twice the CLMUL work per instruction. */
static constexpr uint32_t CRC32_BZIP2_X1024 = crc32Bzip2XPower( 1024 );
static constexpr uint32_t CRC32_BZIP2_X1088 = crc32Bzip2XPower( 1088 );

#define LIBRAPIDARCHIVE_BZIP2_CRC_AVX2_TARGET __attribute__( ( target( "sse2,avx2,vpclmulqdq,pclmul,ssse3" ) ) )

[[nodiscard]]
LIBRAPIDARCHIVE_BZIP2_CRC_AVX2_TARGET inline __m256i
crc32Bzip2Fold256( __m256i value,
                   __m256i factorLow,
                   __m256i factorHigh ) noexcept
{
    return _mm256_xor_si256( _mm256_clmulepi64_epi128( value, factorHigh, 0x01 ),
                             _mm256_clmulepi64_epi128( value, factorLow, 0x00 ) );
}


[[nodiscard]]
LIBRAPIDARCHIVE_BZIP2_CRC_AVX2_TARGET inline __m256i
crc32Bzip2LoadBlocks256( const uint8_t* source,
                         __m256i        reverseBytes ) noexcept
{
    return _mm256_shuffle_epi8( _mm256_loadu_si256( reinterpret_cast<const __m256i*>( source ) ),
                                reverseBytes );
}


[[nodiscard]]
LIBRAPIDARCHIVE_BZIP2_CRC_AVX2_TARGET inline uint32_t
updateCRC32VPCLMUL( uint32_t       crc,
                    const uint8_t* data,
                    size_t         size ) noexcept
{
    const auto blockCount = size / 16;
    if ( blockCount < 8 ) {
        return updateCRC32PCLMUL( crc, data, size );
    }

    const auto reverseBytes128 = _mm_setr_epi8( 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 );
    const auto reverseBytes = _mm256_broadcastsi128_si256( reverseBytes128 );
    const auto fold1024Low = _mm256_set_epi64x( 0, CRC32_BZIP2_X1024, 0, CRC32_BZIP2_X1024 );
    const auto fold1024High = _mm256_set_epi64x( 0, CRC32_BZIP2_X1088, 0, CRC32_BZIP2_X1088 );
    const auto initialCRC = _mm256_setr_epi32( 0, 0, 0, static_cast<int>( crc ), 0, 0, 0, 0 );

    auto lane0 = _mm256_xor_si256( crc32Bzip2LoadBlocks256( data, reverseBytes ), initialCRC );
    auto lane1 = crc32Bzip2LoadBlocks256( data + 32, reverseBytes );
    auto lane2 = crc32Bzip2LoadBlocks256( data + 64, reverseBytes );
    auto lane3 = crc32Bzip2LoadBlocks256( data + 96, reverseBytes );
    size_t block = 8;
    for ( ; block + 8 <= blockCount; block += 8 ) {
        const auto* const source = data + block * 16;
        lane0 = _mm256_xor_si256( crc32Bzip2Fold256( lane0, fold1024Low, fold1024High ),
                                  crc32Bzip2LoadBlocks256( source, reverseBytes ) );
        lane1 = _mm256_xor_si256( crc32Bzip2Fold256( lane1, fold1024Low, fold1024High ),
                                  crc32Bzip2LoadBlocks256( source + 32, reverseBytes ) );
        lane2 = _mm256_xor_si256( crc32Bzip2Fold256( lane2, fold1024Low, fold1024High ),
                                  crc32Bzip2LoadBlocks256( source + 64, reverseBytes ) );
        lane3 = _mm256_xor_si256( crc32Bzip2Fold256( lane3, fold1024Low, fold1024High ),
                                  crc32Bzip2LoadBlocks256( source + 96, reverseBytes ) );
    }

    const auto fold128Low = _mm_set_epi64x( 0, CRC32_BZIP2_X128 );
    const auto fold128High = _mm_set_epi64x( 0, CRC32_BZIP2_X192 );
    __m128i accumulator = _mm256_castsi256_si128( lane0 );
    accumulator = _mm_xor_si128( crc32Bzip2Fold( accumulator, fold128Low, fold128High ),
                                 _mm256_extracti128_si256( lane0, 1 ) );
    accumulator = _mm_xor_si128( crc32Bzip2Fold( accumulator, fold128Low, fold128High ),
                                 _mm256_castsi256_si128( lane1 ) );
    accumulator = _mm_xor_si128( crc32Bzip2Fold( accumulator, fold128Low, fold128High ),
                                 _mm256_extracti128_si256( lane1, 1 ) );
    accumulator = _mm_xor_si128( crc32Bzip2Fold( accumulator, fold128Low, fold128High ),
                                 _mm256_castsi256_si128( lane2 ) );
    accumulator = _mm_xor_si128( crc32Bzip2Fold( accumulator, fold128Low, fold128High ),
                                 _mm256_extracti128_si256( lane2, 1 ) );
    accumulator = _mm_xor_si128( crc32Bzip2Fold( accumulator, fold128Low, fold128High ),
                                 _mm256_castsi256_si128( lane3 ) );
    accumulator = _mm_xor_si128( crc32Bzip2Fold( accumulator, fold128Low, fold128High ),
                                 _mm256_extracti128_si256( lane3, 1 ) );

    for ( ; block < blockCount; ++block ) {
        accumulator = _mm_xor_si128(
            crc32Bzip2Fold( accumulator, fold128Low, fold128High ),
            _mm_shuffle_epi8(
                _mm_loadu_si128( reinterpret_cast<const __m128i*>( data + block * 16 ) ),
                reverseBytes128 ) );
    }

    alignas( 32 ) uint8_t trailing[32];
    _mm_store_si128( reinterpret_cast<__m128i*>( trailing ),
                     _mm_shuffle_epi8( accumulator, reverseBytes128 ) );
    const auto tailSize = size - blockCount * 16;
    if ( tailSize > 0 ) {
        std::memcpy( trailing + 16, data + blockCount * 16, tailSize );
    }
    return updateCRC32SliceByN<BZIP2_CRC_SLICES>( 0, trailing, 16 + tailSize );
}
#undef LIBRAPIDARCHIVE_BZIP2_CRC_AVX2_TARGET
#endif  /* LIBRAPIDARCHIVE_BZIP2_CRC_VPCLMUL */


[[nodiscard]] inline bool
hasPCLMULCRC32() noexcept
{
#if LIBRAPIDARCHIVE_BZIP2_CRC_PCLMUL
    return ( __builtin_cpu_supports( "pclmul" ) != 0 ) && ( __builtin_cpu_supports( "ssse3" ) != 0 );
#else
    return false;
#endif
}


[[nodiscard]] inline bool
hasVPCLMULCRC32() noexcept
{
#if LIBRAPIDARCHIVE_BZIP2_CRC_VPCLMUL
    return ( __builtin_cpu_supports( "avx2" ) != 0 )
           && ( __builtin_cpu_supports( "vpclmulqdq" ) != 0 )
           && ( __builtin_cpu_supports( "pclmul" ) != 0 )
           && ( __builtin_cpu_supports( "ssse3" ) != 0 );
#else
    return false;
#endif
}


[[nodiscard]] inline uint32_t
updateCRC32( uint32_t    crc,
             const char* data,
             size_t      size ) noexcept
{
#if LIBRAPIDARCHIVE_BZIP2_CRC_VPCLMUL
    if ( ( size >= CRC32_PCLMUL_MIN_SIZE ) && hasVPCLMULCRC32() ) {
        return updateCRC32VPCLMUL( crc, reinterpret_cast<const uint8_t*>( data ), size );
    }
#endif
#if LIBRAPIDARCHIVE_BZIP2_CRC_PCLMUL
    if ( ( size >= CRC32_PCLMUL_MIN_SIZE ) && hasPCLMULCRC32() ) {
        return updateCRC32PCLMUL( crc, reinterpret_cast<const uint8_t*>( data ), size );
    }
#endif
    return updateCRC32SliceByN<BZIP2_CRC_SLICES>(
        crc, reinterpret_cast<const uint8_t*>( data ), size );
}
}  // namespace bzip2
