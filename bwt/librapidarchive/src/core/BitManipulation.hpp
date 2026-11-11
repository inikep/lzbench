#pragma once

#include <array>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef _MSC_VER
    #include <intrin.h>
#endif

namespace rapidgzip
{
[[nodiscard]] inline bool
isLittleEndian()
{
    constexpr uint16_t endianTestNumber = 1;
    return *reinterpret_cast<const uint8_t*>( &endianTestNumber ) == 1;
}


/**
 * @return The number of trailing zero bits in @p value. The result is undefined for zero, matching the
 *         hardware instruction. Callers must ensure that at least one bit is set.
 *
 * C++20 provides std::countr_zero, which is portable and well-defined for zero. Use it once the project
 * requires C++20. std::bitset and std::simd do not map this operation to a single scalar instruction.
 */
[[nodiscard]] inline unsigned int
trailingZeroCount( const uint32_t value ) noexcept
{
#ifdef _MSC_VER
#if defined( _M_IX86 ) || ( defined( _M_X64 ) && !defined( _M_ARM64EC ) )
    unsigned long int result = 0;
    _BitScanForward( &result, value );
    return static_cast<unsigned int>( result );
#else
        /* ARM64 and ARM64EC have no x86 bit-scan intrinsic, so use the portable fallback. */
    uint32_t remaining = value;
    unsigned int result = 0;
    for ( ; ( remaining & 1U ) == 0; remaining >>= 1U ) {
        ++result;
    }
    return result;
#endif
#else
    return static_cast<unsigned int>( __builtin_ctz( value ) );
#endif
}


[[nodiscard]] constexpr uint64_t
byteSwap( uint64_t value )
{
#if defined( __GNUC__ ) || defined( __clang__ )
    return __builtin_bswap64( value );
#else
    /* _byteswap_uint64 is not constexpr on MSVC, so keep the shift/mask form there. */
    value = ( ( value & uint64_t( 0x0000'0000'FFFF'FFFFULL ) ) << 32U ) |
            ( ( value & uint64_t( 0xFFFF'FFFF'0000'0000ULL ) ) >> 32U );
    value = ( ( value & uint64_t( 0x0000'FFFF'0000'FFFFULL ) ) << 16U ) |
            ( ( value & uint64_t( 0xFFFF'0000'FFFF'0000ULL ) ) >> 16U );
    value = ( ( value & uint64_t( 0x00FF'00FF'00FF'00FFULL ) ) << 8U  ) |
            ( ( value & uint64_t( 0xFF00'FF00'FF00'FF00ULL ) ) >> 8U  );
    return value;
#endif
}


[[nodiscard]] constexpr uint32_t
byteSwap( uint32_t value )
{
    value = ( ( value & uint32_t( 0x0000'FFFFUL ) ) << 16U ) | ( ( value & uint32_t( 0xFFFF'0000UL ) ) >> 16U );
    value = ( ( value & uint32_t( 0x00FF'00FFUL ) ) << 8U  ) | ( ( value & uint32_t( 0xFF00'FF00UL ) ) >> 8U  );
    return value;
}


[[nodiscard]] constexpr uint16_t
byteSwap( uint16_t value )
{
    return static_cast<uint16_t>( ( ( static_cast<uint32_t>( value ) & 0x00FFU ) << 8U ) |
                                  ( ( static_cast<uint32_t>( value ) & 0xFF00U ) >> 8U ) );
}


/**
 * @verbatim
 * 63                48                  32                  16        8         0
 * |                 |                   |                   |         |         |
 * 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 1111 1111 1111
 *                                                                  <------------>
 *                                                                   nBitsSet = 12
 * @endverbatim
 *
 * @param nBitsSet the number of lowest bits which should be 1 (rest are 0)
 */
template<typename T>
[[nodiscard]] constexpr T
nLowestBitsSet( uint8_t nBitsSet )
{
    static_assert( std::is_unsigned_v<T>, "Type must be unsigned!" );
    if ( nBitsSet == 0 ) {
        return T( 0 );
    }
    if ( nBitsSet >= std::numeric_limits<T>::digits ) {
        return static_cast<T>( ~T( 0 ) );
    }
    const auto nZeroBits = static_cast<uint8_t>( std::max( 0, std::numeric_limits<T>::digits - nBitsSet ) );
    return static_cast<T>( static_cast<T>( ~T( 0 ) ) >> nZeroBits );
}


template<typename T, uint8_t nBitsSet>
[[nodiscard]] constexpr T
nLowestBitsSet()
{
    static_assert( std::is_unsigned_v<T>, "Type must be unsigned!" );
    if constexpr ( nBitsSet == 0 ) {
        return T( 0 );
    } else if constexpr ( nBitsSet >= std::numeric_limits<T>::digits ) {
        return static_cast<T>( ~T( 0 ) );
    } else {
        const auto nZeroBits = static_cast<uint8_t>( std::max( 0, std::numeric_limits<T>::digits - nBitsSet ) );
        return static_cast<T>( static_cast<T>( ~T( 0 ) ) >> nZeroBits );
    }
}


/* Size: uint64_t: 16 KiB, uint32_t: 8 KiB, ... */
template<typename T>
static constexpr std::array<T, 256U> N_LOWEST_BITS_SET_LUT =
    [] ()
    {
        std::array<T, 256U> result{};
        for ( size_t i = 0; i < result.size(); ++i ) {
            result[i] = nLowestBitsSet<T>( static_cast<uint8_t>( i ) );
        }
        return result;
    }();


template<typename T>
[[nodiscard]] constexpr T
nHighestBitsSet( uint8_t nBitsSet )
{
    static_assert( std::is_unsigned_v<T>, "Type must be unsigned!" );
    if ( nBitsSet == 0 ) {
        return T( 0 );
    }
    if ( nBitsSet >= std::numeric_limits<T>::digits ) {
        return static_cast<T>( ~T( 0 ) );
    }
    const auto nZeroBits = static_cast<uint8_t>( std::max( 0, std::numeric_limits<T>::digits - nBitsSet ) );
    return static_cast<T>( static_cast<T>( ~T( 0 ) ) << nZeroBits );
}


/* Size: uint64_t: 16 KiB, uint32_t: 8 KiB, ... */
template<typename T>
static constexpr std::array<T, 256U> N_HIGHEST_BITS_SET_LUT =
    [] ()
    {
        std::array<T, 256U> result{};
        for ( size_t i = 0; i < result.size(); ++i ) {
            result[i] = nHighestBitsSet<T>( i );
        }
        return result;
    }();


template<typename T, uint8_t nBitsSet>
[[nodiscard]] constexpr T
nHighestBitsSet()
{
    static_assert( std::is_unsigned_v<T>, "Type must be unsigned!" );
    if constexpr ( nBitsSet == 0 ) {
        return T( 0 );
    } else if constexpr ( nBitsSet >= std::numeric_limits<T>::digits ) {
        return static_cast<T>( ~T( 0 ) );
    } else {
        const auto nZeroBits = std::max( 0, std::numeric_limits<T>::digits - nBitsSet );
        return static_cast<T>( static_cast<T>( ~T( 0 ) ) << nZeroBits );
    }
}


[[nodiscard]] constexpr uint8_t
reverseBitsWithoutLUT( uint8_t data )
{
    /* Reverse bits using bit-parallelism in a recursive fashion, i.e., first swap every bit with its neighbor,
     * then swap each half nibble with its neighbor, then each nibble. */
    constexpr std::array<uint8_t, 3> masks = { 0b0101'0101U, 0b0011'0011U, 0b0000'1111U };
    for ( size_t i = 0; i < masks.size(); ++i ) {
        const uint32_t mask{ masks[i] };
        data = static_cast<uint8_t>( ( ( uint32_t( data ) &  mask ) << ( 1U << i ) ) |
                                     ( ( uint32_t( data ) & ~mask ) >> ( 1U << i ) ) );
    }
    return data;
}


[[nodiscard]] constexpr uint16_t
reverseBitsWithoutLUT( uint16_t data )
{
    /* Reverse bits using bit-parallelism in a recursive fashion, i.e., first swap every bit with its neighbor,
     * then swap each half nibble with its neighbor, then each nibble, then each byte. */
    constexpr std::array<uint16_t, 4> masks = {
        0b0101'0101'0101'0101U,
        0b0011'0011'0011'0011U,
        0b0000'1111'0000'1111U,
        0b0000'0000'1111'1111U,
    };
    for ( size_t i = 0; i < masks.size(); ++i ) {
        const uint32_t mask{ masks[i] };
        data = static_cast<uint16_t>( ( ( uint32_t( data ) &  mask ) << ( 1U << i ) ) |
                                      ( ( uint32_t( data ) & ~mask ) >> ( 1U << i ) ) );
    }
    return data;
}


[[nodiscard]] constexpr uint32_t
reverseBitsWithoutLUT( uint32_t data )
{
    /* Reverse bits using bit-parallelism in a recursive fashion, i.e., first swap every bit with its neighbor,
     * then swap each half nibble with its neighbor, then each nibble, then each byte, then each word. */
    constexpr std::array<uint32_t, 5> masks = {
        0b0101'0101'0101'0101'0101'0101'0101'0101U,
        0b0011'0011'0011'0011'0011'0011'0011'0011U,
        0b0000'1111'0000'1111'0000'1111'0000'1111U,
        0b0000'0000'1111'1111'0000'0000'1111'1111U,
        0b0000'0000'0000'0000'1111'1111'1111'1111U,
    };
    for ( size_t i = 0; i < masks.size(); ++i ) {
        data = ( ( data &  masks[i] ) << ( 1U << i ) ) |
               ( ( data & ~masks[i] ) >> ( 1U << i ) );
    }
    return data;
}


[[nodiscard]] constexpr uint64_t
reverseBitsWithoutLUT( uint64_t data )
{
    /* Reverse bits using bit-parallelism in a recursive fashion, i.e., first swap every bit with its neighbor,
     * then swap each half nibble with its neighbor, then each nibble, then each byte, then each word, then each
     * double word. */
    constexpr std::array<uint64_t, 6> masks = {
        0b0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101'0101U,
        0b0011'0011'0011'0011'0011'0011'0011'0011'0011'0011'0011'0011'0011'0011'0011'0011U,
        0b0000'1111'0000'1111'0000'1111'0000'1111'0000'1111'0000'1111'0000'1111'0000'1111U,
        0b0000'0000'1111'1111'0000'0000'1111'1111'0000'0000'1111'1111'0000'0000'1111'1111U,
        0b0000'0000'0000'0000'1111'1111'1111'1111'0000'0000'0000'0000'1111'1111'1111'1111U,
        0b0000'0000'0000'0000'0000'0000'0000'0000'1111'1111'1111'1111'1111'1111'1111'1111U,
    };
    /* Using godbolt, shows that both gcc 8.6 and clang will unroll this loop! Clang 13 seems to produce even shorter
     * code by somehow skipping word and dword swaps, maybe implementing those via moves instead. Both, evaluate
     * the 1U << i and ~masks[i] expressions at compile-time. Note that ~mask is -mask on two's-complement platforms! */
    for ( size_t i = 0; i < masks.size(); ++i ) {
        data = ( ( data &  masks[i] ) << ( 1U << i ) ) |
               ( ( data & ~masks[i] ) >> ( 1U << i ) );
    }
    return data;
}


template<typename T>
[[nodiscard]] constexpr std::array<T, 1ULL << static_cast<uint8_t>( std::numeric_limits<T>::digits )>
createReversedBitsLUT();

template<>
[[nodiscard]] constexpr std::array<uint8_t, 1ULL << 8U>
createReversedBitsLUT<uint8_t>()
{
    using T = uint8_t;
    static_assert( std::is_unsigned_v<T> && std::is_integral_v<T> );

    std::array<T, 1ULL << static_cast<uint8_t>( std::numeric_limits<T>::digits )> result{};
    for ( size_t i = 0; i < result.size(); ++i ) {
        result[i] = reverseBitsWithoutLUT( static_cast<T>( i ) );
    }
    return result;
}

template<>
[[nodiscard]] constexpr std::array<uint16_t, 1ULL << 16U>
createReversedBitsLUT<uint16_t>()
{
    using T = uint16_t;
    static_assert( std::is_unsigned_v<T> && std::is_integral_v<T> );

    constexpr auto REVERSED_BITS_LUT = createReversedBitsLUT<uint8_t>();

    std::array<T, 1ULL << static_cast<uint8_t>( std::numeric_limits<T>::digits )> result{};
    for ( size_t i = 0; i < result.size(); ++i ) {
        result[i] = ( REVERSED_BITS_LUT[i & 0xFFU] << 8U ) | REVERSED_BITS_LUT[( i >> 8U ) & 0xFFU];
    }
    return result;
}


/* Size: uint16_t: 64 KiB, uint8_t: 256 B */
template<typename T>
alignas( 64 ) static constexpr auto REVERSED_BITS_LUT = createReversedBitsLUT<T>();


template<typename T>
[[nodiscard]] constexpr T
reverseBits( T value )
{
    static_assert( std::is_unsigned_v<T> && std::is_integral_v<T> );

    if constexpr ( sizeof( T ) <= 2 ) {
        return REVERSED_BITS_LUT<T>[value];
    } else {
        return reverseBitsWithoutLUT( value );
    }
}


/**
 * Reverses the lowest @p bitCount bits. The highest bits are set to 0 and are assumed to be zero in the input.
 * @param bitCount Must be greater than zero or else the applied bitshift is undefined behavior as per C++ standard.
 */
template<typename T>
[[nodiscard]] constexpr T
reverseBits( T       value,
             uint8_t bitCount )
{
    return reverseBits<T>( value ) >> static_cast<uint8_t>( std::numeric_limits<T>::digits - bitCount );
}


/**
 * Basically ceil(log2(stateCount)) with an exception for 0.
 */
[[nodiscard]] constexpr uint8_t
requiredBits( const uint64_t stateCount )
{
    if ( stateCount == 0 ) {
        return 0;
    }
    if ( stateCount == 1 ) {
        return 1;
    }

    uint8_t result{ 0 };
    for ( auto maxValue = stateCount - 1; maxValue != 0; maxValue >>= 1U ) {
        ++result;
    }
    return result;
}
}  // namespace rapidgzip
