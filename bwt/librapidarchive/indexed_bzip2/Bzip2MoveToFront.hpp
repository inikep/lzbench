/**
 * @file The Move-to-Front (MTF) list used by the bzip2 block decoder.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include <core/common.hpp>

/* ARM64EC also defines _M_X64 and _M_AMD64, but it is an ARM64 target and must not take the x64
 * SSE2 path. libsais uses the same condition. */
#if ( defined( __SSE2__ ) || ( defined( _M_IX86_FP ) && ( _M_IX86_FP >= 2 ) ) \
      || defined( _M_AMD64 ) || defined( _M_X64 ) ) && !defined( _M_ARM64EC )
    #define LIBRAPIDARCHIVE_BZIP2_HAS_SSE2 1
    #include <emmintrin.h>
#else
    #define LIBRAPIDARCHIVE_BZIP2_HAS_SSE2 0
#endif
#if defined( __ARM_NEON )
    #include <arm_neon.h>
#endif


namespace bzip2
{
/**
 * bzip2's Move-to-Front (MTF) list for the block data. Eight symbols are packed into one 64-bit word, with the
 * symbol at list position p stored in byte p % 8 of word p / 8. This avoids the variable-length memmove that
 * otherwise dominates the literal path for alphabets with a uniform MTF index distribution such as base64.
 */
using PackedMTF = std::array<uint64_t, 256 / 8>;


#if defined( __ARM_NEON ) && defined( __BYTE_ORDER__ ) && ( __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ )
/* Compares each of the first 64 list positions against the insert position. */
constexpr std::array<uint8_t, 64> MTF_INDEX_BYTES =
    [] ()
    {
        std::array<uint8_t, 64> result{};
        for ( size_t i = 0; i < result.size(); ++i ) {
            result[i] = static_cast<uint8_t>( i );
        }
        return result;
    }();
#endif


[[nodiscard]] constexpr PackedMTF
createPackedMTF() noexcept
{
    PackedMTF result{};
    for ( size_t word = 0; word < result.size(); ++word ) {
        uint64_t value = 0;
        for ( size_t byte = 0; byte < 8; ++byte ) {
            value |= static_cast<uint64_t>( word * 8 + byte ) << ( byte * 8 );
        }
        result[word] = value;
    }
    return result;
}


/**
 * @param bytes The byte values that are used in the block in ascending order.
 * @return the initial Move-to-Front list of a block, which contains byte values instead of symbol indexes.
 */
[[nodiscard]] inline PackedMTF
createPackedMTF( const uint8_t* const bytes,
                 const size_t         count ) noexcept
{
    PackedMTF result{};
    for ( size_t i = 0; ( i < count ) && ( i < 256 ); ++i ) {
        result[i / 8] |= static_cast<uint64_t>( bytes[i] ) << ( ( i % 8 ) * 8U );
    }
    return result;
}


/**
 * bzip2's Move-to-Front (MTF) list is a recency-ordered cache of the 256 byte values.
 * Encoding a symbol writes its current list position and then moves it to the front.
 * For the input "bzipbzip" and the reduced alphabet "abc...z" the positions are:
 * 1, 25, 9, 16, 3, 3, 3, 3
 *
 * @verbatim
 *        |  SSE 128 Bit  |...
 *        |uint64 |uint64 |...
 *   b -> "a>b<c d|e f g h|..."     -> position  1, list becomes "bac..."
 *   z -> "b a c d|e f g h|...>z<"  -> position 25, list becomes "zbac..."
 *   i -> "z b a d|e f g h|...>i<"  -> position  9, list becomes "izba..."
 *   p -> "i z b a|d f g h|... p"   -> position 16, list becomes "pizba..."
 *   b -> "p i z>b<a f g h|... z"   -> position  3, list becomes "bpiza..."
 *   ...
 * @endverbatim
 *
 * After the first "bzip" the list starts with "p i z b a ...", so the second "bzip" finds every
 * symbol three positions behind the front. The list is therefore a diff against the recent symbol
 * order: the most recent symbol is always at position 0, the one before it at position 1, and so
 * on. Repeating a symbol yields a small index. The decoder only has to reverse the same update.
 *
 * The list is packed as 32 uint64 words with 8 symbols per word (see PackedMTF), so list position p
 * lives in byte p % 8 of word p / 8. A plain array MTF update would shift a variable-length prefix.
 * This function instead updates the first 64 positions unconditionally: it saves the
 * selected symbol, shifts the whole 64-byte prefix one position towards the back, puts the saved
 * symbol at the front, and keeps the suffix after the selected position unchanged. That is exactly
 * the list update, but without a data-dependent loop exit.
 *
 * The SSE2 path treats the 64 bytes as four 16-byte vectors. _mm_slli_si128 shifts every vector by
 * one byte towards higher list positions; _mm_srli_si128(previous, 15) moves the last byte of the
 * previous vector into the low byte of the next one, so the four operations form one continuous
 * 64-byte shift. limit = position + 1 and the index vectors 0..63 produce a 0xFF mask for the
 * positions <= position and 0x00 above it. The blend therefore selects the shifted prefix [0,
 * position] and the original suffix (position, 64), which is the same result as moving the selected
 * symbol to the front and shifting only the symbols in between. The saved symbol is inserted at
 * position 0 by _mm_cvtsi32_si128.
 *
 * The branchless path matters for alphabets with a uniform MTF index distribution such as base64
 * after BWT: the scalar loop exits after a data-dependent number of words on almost every literal,
 * which the branch predictor cannot learn. Shifting all 64 positions costs some extra work for small
 * indices but removes that misprediction. Positions >= 64 and targets without SSE2 use the scalar
 * word loop below, which propagates the carry byte through the packed words.
 *
 * An update at position p only changes the prefix [0, p]; positions above p are unchanged.
 * A following index q > p therefore reads an unchanged list element and could be gathered before the update.
 * This implementation still stores the full 64-byte prefix unconditionally, so the next load aliases that store.
 * Batching increasing MTF indices and gathering their symbols from the unchanged suffix is a more promising
 * optimization than widening the vectors: AVX2 and AVX-512 variants kept the same store/load dependency and
 * measured slower than this SSE2 version.
 *
 * @return the symbol byte that is now at the front of the list.
 */
[[nodiscard]] forceinline uint8_t
moveToFront( PackedMTF&   mtf,
             const size_t position ) noexcept
{
#if LIBRAPIDARCHIVE_BZIP2_HAS_SSE2
    if ( position < 64 ) {
        auto* const data = reinterpret_cast<uint8_t*>( mtf.data() );
        /* Keep the selected symbol because shifting overwrites its old list position. */
        const auto head = data[position];

        /* Load 64 B (four 16-byte vectors) out of the maximum 256 B and shift it unconditionally by 1 B
         * to make place at the front. */
        const auto in0 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( data ) );
        const auto in1 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( data + sizeof( __m128i ) ) );
        const auto in2 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( data + 2 * sizeof( __m128i ) ) );
        const auto in3 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( data + 3 * sizeof( __m128i ) ) );

        /* The high byte of one vector becomes the low byte of the next, so these four shifts form
         * one continuous 64-byte shift. The selected symbol fills the freed position 0. */
        auto out0 = _mm_slli_si128( in0, 1 );
        out0 = _mm_or_si128( out0, _mm_cvtsi32_si128( head ) );
        const auto out1 = _mm_or_si128( _mm_slli_si128( in1, 1 ), _mm_srli_si128( in0, sizeof( __m128i ) - 1 ) );
        const auto out2 = _mm_or_si128( _mm_slli_si128( in2, 1 ), _mm_srli_si128( in1, sizeof( __m128i ) - 1 ) );
        const auto out3 = _mm_or_si128( _mm_slli_si128( in3, 1 ), _mm_srli_si128( in2, sizeof( __m128i ) - 1 ) );

        /* The idea here is to vector-compare index <= position, where we build all indexes like std::iota to get
         * masks. 0xFF for positions <= position selects the shifted prefix; 0x00 keeps the suffix above. */
        const auto limit = _mm_set1_epi8( static_cast<char>( position + 1 ) );
        const auto index0 = _mm_setr_epi8( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
        const auto index1 = _mm_setr_epi8( 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 );
        const auto index2 = _mm_setr_epi8( 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 );
        const auto index3 = _mm_setr_epi8( 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63 );

        const auto mask0 = _mm_cmpgt_epi8( limit, index0 );
        const auto mask1 = _mm_cmpgt_epi8( limit, index1 );
        const auto mask2 = _mm_cmpgt_epi8( limit, index2 );
        const auto mask3 = _mm_cmpgt_epi8( limit, index3 );

        /**
         * Now we have some mask = [mask0, mask1, mask2, mask3] = 1111..1 (@p position) 0..00000 that can be
         * used to merge the shifted version of the alphabet with the unshifted version after @p position.
         * Representation with 8 instead of 64 symbol/byte shifts:
         * @verbatim
         *         Symbol f at position 5 to be moved to front
         *                              v
         * Index            : 0 1 2 3 4 5 6 7
         * Alphabet         : a b c d e f g h
         * Alphabet >> 1    : _ a b c d e f g
         * Mask (idx <= p)  : 1 1 1 1 1 1 0 0
         * Mask & (Al >> 1) : _ a b c d e _ _
         * ~Mask & Alphabet : _ _ _ _ _ _ g h
         * OR               : _ a b c d e g h
         * Insert head      : f a b c d e g h
         * @endverbatim
         * Beware! The shift direction and exact position is confusing because of little endian storage order! SSE is
         * used to shift left by 1, so that the byte representation is shifted to the "right" in usual index order.
         */
        _mm_storeu_si128( reinterpret_cast<__m128i*>( data ),
                          _mm_or_si128( _mm_and_si128( mask0, out0 ),
                                        _mm_andnot_si128( mask0, in0 ) ) );
        _mm_storeu_si128( reinterpret_cast<__m128i*>( data + 16 ),
                          _mm_or_si128( _mm_and_si128( mask1, out1 ),
                                        _mm_andnot_si128( mask1, in1 ) ) );
        _mm_storeu_si128( reinterpret_cast<__m128i*>( data + 32 ),
                          _mm_or_si128( _mm_and_si128( mask2, out2 ),
                                        _mm_andnot_si128( mask2, in2 ) ) );
        _mm_storeu_si128( reinterpret_cast<__m128i*>( data + 48 ),
                          _mm_or_si128( _mm_and_si128( mask3, out3 ),
                                        _mm_andnot_si128( mask3, in3 ) ) );
        return head;
    }

    /* Positions 64 to 255 previously used the scalar word loop below, which carries one byte through
     * up to 31 words. The packed list is 256 bytes, so the same 64-byte shift can be applied to whole
     * 64-byte chunks and only the chunk containing the selected position needs the prefix mask. */
    if ( position < mtf.size() * sizeof( uint64_t ) ) {
        auto* const data = reinterpret_cast<uint8_t*>( mtf.data() );
        const auto head = data[position];
        const auto fullChunks = position / 64;
        auto carry = head;

        const auto shiftFullChunk =
            [] ( uint8_t* const chunk, const uint8_t incomingByte ) noexcept
            {
                const auto in0 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( chunk ) );
                const auto in1 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( chunk + sizeof( __m128i ) ) );
                const auto in2 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( chunk + 2 * sizeof( __m128i ) ) );
                const auto in3 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( chunk + 3 * sizeof( __m128i ) ) );
                _mm_storeu_si128( reinterpret_cast<__m128i*>( chunk ),
                                  _mm_or_si128( _mm_slli_si128( in0, 1 ), _mm_cvtsi32_si128( incomingByte ) ) );
                _mm_storeu_si128(
                    reinterpret_cast<__m128i*>( chunk + sizeof( __m128i ) ),
                    _mm_or_si128( _mm_slli_si128( in1, 1 ), _mm_srli_si128( in0, sizeof( __m128i ) - 1 ) ) );
                _mm_storeu_si128(
                    reinterpret_cast<__m128i*>( chunk + 2 * sizeof( __m128i ) ),
                    _mm_or_si128( _mm_slli_si128( in2, 1 ), _mm_srli_si128( in1, sizeof( __m128i ) - 1 ) ) );
                _mm_storeu_si128(
                    reinterpret_cast<__m128i*>( chunk + 3 * sizeof( __m128i ) ),
                    _mm_or_si128( _mm_slli_si128( in3, 1 ), _mm_srli_si128( in2, sizeof( __m128i ) - 1 ) ) );
                return static_cast<uint8_t>(
                    _mm_cvtsi128_si32( _mm_srli_si128( in3, sizeof( __m128i ) - 1 ) ) );
            };

        for ( size_t chunk = 0; chunk < fullChunks; ++chunk ) {
            carry = shiftFullChunk( data + chunk * 64, carry );
        }

        /* The chunk containing the selected position is shifted up to and including that position. */
        auto* const chunk = data + fullChunks * 64;
        const auto in0 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( chunk ) );
        const auto in1 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( chunk + sizeof( __m128i ) ) );
        const auto in2 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( chunk + 2 * sizeof( __m128i ) ) );
        const auto in3 = _mm_loadu_si128( reinterpret_cast<const __m128i*>( chunk + 3 * sizeof( __m128i ) ) );
        auto out0 = _mm_slli_si128( in0, 1 );
        out0 = _mm_or_si128( out0, _mm_cvtsi32_si128( carry ) );
        const auto out1 = _mm_or_si128( _mm_slli_si128( in1, 1 ), _mm_srli_si128( in0, sizeof( __m128i ) - 1 ) );
        const auto out2 = _mm_or_si128( _mm_slli_si128( in2, 1 ), _mm_srli_si128( in1, sizeof( __m128i ) - 1 ) );
        const auto out3 = _mm_or_si128( _mm_slli_si128( in3, 1 ), _mm_srli_si128( in2, sizeof( __m128i ) - 1 ) );

        const auto limit = _mm_set1_epi8( static_cast<char>( ( position % 64 ) + 1 ) );
        const auto index0 = _mm_setr_epi8( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 );
        const auto index1 = _mm_setr_epi8( 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 );
        const auto index2 = _mm_setr_epi8( 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 );
        const auto index3 = _mm_setr_epi8( 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63 );
        const auto mask0 = _mm_cmpgt_epi8( limit, index0 );
        const auto mask1 = _mm_cmpgt_epi8( limit, index1 );
        const auto mask2 = _mm_cmpgt_epi8( limit, index2 );
        const auto mask3 = _mm_cmpgt_epi8( limit, index3 );
        _mm_storeu_si128( reinterpret_cast<__m128i*>( chunk ),
                          _mm_or_si128( _mm_and_si128( mask0, out0 ), _mm_andnot_si128( mask0, in0 ) ) );
        _mm_storeu_si128( reinterpret_cast<__m128i*>( chunk + sizeof( __m128i ) ),
                          _mm_or_si128( _mm_and_si128( mask1, out1 ), _mm_andnot_si128( mask1, in1 ) ) );
        _mm_storeu_si128( reinterpret_cast<__m128i*>( chunk + 2 * sizeof( __m128i ) ),
                          _mm_or_si128( _mm_and_si128( mask2, out2 ), _mm_andnot_si128( mask2, in2 ) ) );
        _mm_storeu_si128( reinterpret_cast<__m128i*>( chunk + 3 * sizeof( __m128i ) ),
                          _mm_or_si128( _mm_and_si128( mask3, out3 ), _mm_andnot_si128( mask3, in3 ) ) );
        return head;
    }
#endif
#if defined( __ARM_NEON ) && defined( __BYTE_ORDER__ ) && ( __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ )
    if ( position < 64 ) {
        auto* const data = reinterpret_cast<uint8_t*>( mtf.data() );
        const auto head = data[position];

        const auto in0 = vld1q_u8( data );
        const auto in1 = vld1q_u8( data + 16 );
        const auto in2 = vld1q_u8( data + 32 );
        const auto in3 = vld1q_u8( data + 48 );

        /* vext concatenates its operands, so [a, b] at offset 15 yields [a[15], b[0..14]]. */
        auto headVector = vdupq_n_u8( 0 );
        headVector = vsetq_lane_u8( head, headVector, 15 );
        const auto out0 = vextq_u8( headVector, in0, 15 );
        const auto out1 = vextq_u8( in0, in1, 15 );
        const auto out2 = vextq_u8( in1, in2, 15 );
        const auto out3 = vextq_u8( in2, in3, 15 );

        const auto limit = vdupq_n_u8( static_cast<uint8_t>( position + 1 ) );
        const auto index0 = vld1q_u8( MTF_INDEX_BYTES.data() );
        const auto index1 = vld1q_u8( MTF_INDEX_BYTES.data() + 16 );
        const auto index2 = vld1q_u8( MTF_INDEX_BYTES.data() + 32 );
        const auto index3 = vld1q_u8( MTF_INDEX_BYTES.data() + 48 );

        const auto mask0 = vcgtq_u8( limit, index0 );
        const auto mask1 = vcgtq_u8( limit, index1 );
        const auto mask2 = vcgtq_u8( limit, index2 );
        const auto mask3 = vcgtq_u8( limit, index3 );

        vst1q_u8( data, vbslq_u8( mask0, out0, in0 ) );
        vst1q_u8( data + 16, vbslq_u8( mask1, out1, in1 ) );
        vst1q_u8( data + 32, vbslq_u8( mask2, out2, in2 ) );
        vst1q_u8( data + 48, vbslq_u8( mask3, out3, in3 ) );
        return head;
    }
#endif

    const auto wordIndex = position >> 3;  // equivalent to division by sizeof( PackedMTF::value_type )
    const auto shift = static_cast<uint32_t>( ( position & 7U ) * 8U );
    uint64_t previous = ( mtf[wordIndex] >> shift ) & 0xFFU;

    for ( size_t i = 0; i < wordIndex; ++i ) {
        const auto current = mtf[i];
        mtf[i] = ( current << 8U ) | previous;
        previous = current >> 56U;
    }

    const auto current = mtf[wordIndex];
    /* 0x100 << 56 overflows to zero for unsigned, which would give the correct all-ones mask,
     * but spell it out so that the shift is never out of range. */
    const auto mask = shift == 56U ? ~uint64_t( 0 ) : ( ( uint64_t( 0x100 ) << shift ) - 1U );
    mtf[wordIndex] = ( current & ~mask ) | ( ( ( current << 8U ) | previous ) & mask );
    return static_cast<uint8_t>( mtf.front() );
}
}  // namespace bzip2
