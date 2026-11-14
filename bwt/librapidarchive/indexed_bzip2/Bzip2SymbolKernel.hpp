/**
 * @file The hot loop that decodes the Huffman-coded symbols of a bzip2 block into run records.
 *
 * This file intentionally has no include guard and no namespace. Bzip2SymbolDecoder.hpp includes it once per
 * kernel variant from inside namespace bzip2 after defining:
 *
 *  - LIBRAPIDARCHIVE_BZIP2_KERNEL_NAME: the name of the function to define.
 *  - LIBRAPIDARCHIVE_BZIP2_KERNEL_TARGET: a function target attribute or nothing.
 *  - LIBRAPIDARCHIVE_BZIP2_KERNEL_VARIANT: 0 for portable C++, 1 for SSE4.1, 2 for AVX2.
 *
 * Compiling the same loop several times with different target attributes is what makes runtime dispatch
 * possible without building the whole translation unit for a newer instruction set. A template cannot do this:
 * the target attribute applies to a function, not to an instantiation, and GCC and Clang refuse to inline
 * intrinsics such as _mm256_shuffle_epi8 into a function that was not compiled for their instruction set.
 * The per-variant parts are therefore macros, which also keeps all state in the registers of one function.
 *
 * All variants keep the decoder state in local variables, i.e., registers, and only branch for rare events:
 *
 *  - The bit buffer is refilled unconditionally with one unaligned 64-bit load for every two symbols.
 *  - A run symbol only changes the length of the current record. It is handled with masks instead of a branch
 *    because the sequence of literals and run symbols is close to random for the branch predictor.
 *  - The SIMD variants hold the first 32 entries of the Move-to-Front list in registers. Moving one of those
 *    to the front is a table-driven byte shuffle with a latency of a few cycles that does not go through memory.
 *    Only deeper list positions take a branch and shift the rest of the list in memory. That loop runs over
 *    all chunks used by the block's alphabet and masks the bytes behind the accessed position so that its trip
 *    count does not depend on the position and does not mispredict.
 */

#if LIBRAPIDARCHIVE_BZIP2_KERNEL_VARIANT == 2

    #define BZIP2_KERNEL_CHUNK_SIZE 32U

    #define BZIP2_KERNEL_LOAD_FRONT() \
            __m256i front = _mm256_load_si256( reinterpret_cast<const __m256i*>( mtf ) )

    #define BZIP2_KERNEL_STORE_FRONT() \
            _mm256_store_si256( reinterpret_cast<__m256i*>( mtf ), front )

    /* The second shuffle works on the swapped halves so that bytes can cross the 128-bit lanes. */
    #define BZIP2_KERNEL_FRONT_UPDATE( SYMBOL ) \
            do { \
                const auto* const shuffleRow = SYMBOL_KERNEL_FRONT_SHUFFLES.data() + ( SYMBOL ) * 64U; \
                front = _mm256_or_si256( \
                    _mm256_shuffle_epi8( front, _mm256_load_si256( reinterpret_cast<const __m256i*>( shuffleRow ) ) ), \
                    _mm256_shuffle_epi8( _mm256_permute4x64_epi64( front, 0x4E ), \
                                         _mm256_load_si256( reinterpret_cast<const __m256i*>( shuffleRow + 32 ) ) ) ); \
            } while ( false )

    #define BZIP2_KERNEL_FRONT_BYTE() \
            static_cast<uint32_t>( _mm_cvtsi128_si32( _mm256_castsi256_si128( front ) ) )

    /* Shift list positions [0, POSITION) one towards the back and put BYTE at the front. The memory part of the
     * list starts behind the register part. Chunks entirely behind POSITION are rewritten unchanged. */
    #define BZIP2_KERNEL_MOVE_TAIL_TO_FRONT( POSITION, BYTE ) \
            do { \
                const auto biasedPosition = _mm256_set1_epi8( static_cast<char>( static_cast<uint32_t>( POSITION ) - \
                                                                                 128U ) ); \
                auto previousChunk = front; \
                for ( uint32_t chunkIndex = 1; chunkIndex < chunkCount; ++chunkIndex ) { \
                    auto* const chunkData = reinterpret_cast<__m256i*>( mtf + 32U * chunkIndex ); \
                    const auto currentChunk = _mm256_load_si256( chunkData ); \
                    const auto* const biasedPositions = reinterpret_cast<const __m256i*>( \
                        SYMBOL_KERNEL_BIASED_POSITIONS.data() + 32U * chunkIndex ); \
                    const auto keepMask = _mm256_cmpgt_epi8( _mm256_load_si256( biasedPositions ), biasedPosition ); \
                    const auto shiftedChunk = _mm256_alignr_epi8( \
                        currentChunk, _mm256_permute2x128_si256( currentChunk, previousChunk, 0x03 ), 15 ); \
                    _mm256_store_si256( chunkData, _mm256_blendv_epi8( shiftedChunk, currentChunk, keepMask ) ); \
                    previousChunk = currentChunk; \
                } \
                const auto byteValue = _mm256_castsi128_si256( \
                    _mm_slli_si128( _mm_cvtsi32_si128( static_cast<int>( BYTE ) ), 15 ) ); \
                front = _mm256_alignr_epi8( front, _mm256_permute2x128_si256( front, byteValue, 0x02 ), 15 ); \
            } while ( false )

#elif LIBRAPIDARCHIVE_BZIP2_KERNEL_VARIANT == 1

    #define BZIP2_KERNEL_CHUNK_SIZE 16U

    #define BZIP2_KERNEL_LOAD_FRONT() \
            __m128i front0 = _mm_load_si128( reinterpret_cast<const __m128i*>( mtf ) ); \
            __m128i front1 = _mm_load_si128( reinterpret_cast<const __m128i*>( mtf + 16 ) )

    #define BZIP2_KERNEL_STORE_FRONT() \
            _mm_store_si128( reinterpret_cast<__m128i*>( mtf ), front0 ); \
            _mm_store_si128( reinterpret_cast<__m128i*>( mtf + 16 ), front1 )

    #define BZIP2_KERNEL_FRONT_UPDATE( SYMBOL ) \
            do { \
                const auto* const shuffleRow = SYMBOL_KERNEL_FRONT_SHUFFLES.data() + ( SYMBOL ) * 64U; \
                const auto control0 = _mm_load_si128( reinterpret_cast<const __m128i*>( shuffleRow ) ); \
                const auto control1 = _mm_load_si128( reinterpret_cast<const __m128i*>( shuffleRow + 16 ) ); \
                const auto control2 = _mm_load_si128( reinterpret_cast<const __m128i*>( shuffleRow + 32 ) ); \
                const auto control3 = _mm_load_si128( reinterpret_cast<const __m128i*>( shuffleRow + 48 ) ); \
                const auto newFront0 = _mm_or_si128( _mm_shuffle_epi8( front0, control0 ), \
                                                     _mm_shuffle_epi8( front1, control2 ) ); \
                front1 = _mm_or_si128( _mm_shuffle_epi8( front1, control1 ), \
                                       _mm_shuffle_epi8( front0, control3 ) ); \
                front0 = newFront0; \
            } while ( false )

    #define BZIP2_KERNEL_FRONT_BYTE() static_cast<uint32_t>( _mm_cvtsi128_si32( front0 ) )

    #define BZIP2_KERNEL_MOVE_TAIL_TO_FRONT( POSITION, BYTE ) \
            do { \
                const auto biasedPosition = _mm_set1_epi8( \
                    static_cast<char>( static_cast<uint32_t>( POSITION ) - 128U ) ); \
                auto previousChunk = front1; \
                for ( uint32_t chunkIndex = 2; chunkIndex < chunkCount; ++chunkIndex ) { \
                    auto* const chunkData = reinterpret_cast<__m128i*>( mtf + 16U * chunkIndex ); \
                    const auto currentChunk = _mm_load_si128( chunkData ); \
                    const auto* const biasedPositions = reinterpret_cast<const __m128i*>( \
                        SYMBOL_KERNEL_BIASED_POSITIONS.data() + 16U * chunkIndex ); \
                    const auto keepMask = _mm_cmpgt_epi8( _mm_load_si128( biasedPositions ), biasedPosition ); \
                    _mm_store_si128( chunkData, _mm_blendv_epi8( _mm_alignr_epi8( currentChunk, previousChunk, 15 ), \
                                                                 currentChunk, keepMask ) ); \
                    previousChunk = currentChunk; \
                } \
                front1 = _mm_alignr_epi8( front1, front0, 15 ); \
                const auto insertedByte = _mm_cvtsi32_si128( static_cast<int>( BYTE ) ); \
                front0 = _mm_or_si128( _mm_slli_si128( front0, 1 ), insertedByte ); \
            } while ( false )

#elif LIBRAPIDARCHIVE_BZIP2_KERNEL_VARIANT == 3

    #define BZIP2_KERNEL_CHUNK_SIZE 16U

    #define BZIP2_KERNEL_LOAD_FRONT() \
            uint8x16_t front0 = vld1q_u8( mtf ); \
            uint8x16_t front1 = vld1q_u8( mtf + 16 )

    #define BZIP2_KERNEL_STORE_FRONT() \
            vst1q_u8( mtf, front0 ); \
            vst1q_u8( mtf + 16, front1 )

    #define BZIP2_KERNEL_FRONT_UPDATE( SYMBOL ) \
            do { \
                const auto* const control = SYMBOL_KERNEL_FRONT_SHUFFLES_NEON.data() + ( SYMBOL ) * 32U; \
                uint8x16x2_t table; \
                table.val[0] = front0; \
                table.val[1] = front1; \
                front0 = vqtbl2q_u8( table, vld1q_u8( control ) ); \
                front1 = vqtbl2q_u8( table, vld1q_u8( control + 16 ) ); \
            } while ( false )

    #define BZIP2_KERNEL_FRONT_BYTE() static_cast<uint32_t>( vgetq_lane_u8( front0, 0 ) )

    /* See the AVX2 variant above. vbslq_u8 selects the first operand where the mask is set. */
    #define BZIP2_KERNEL_MOVE_TAIL_TO_FRONT( POSITION, BYTE ) \
            do { \
                auto previousChunk = front1; \
                for ( uint32_t chunkIndex = 2; chunkIndex < chunkCount; ++chunkIndex ) { \
                    auto* const chunkData = mtf + 16U * chunkIndex; \
                    const auto currentChunk = vld1q_u8( chunkData ); \
                    const auto* const positions = SYMBOL_KERNEL_POSITIONS_NEON.data() + 16U * chunkIndex; \
                    const auto keepMask = vcgtq_u8( \
                        vld1q_u8( positions ), vdupq_n_u8( static_cast<uint8_t>( POSITION ) ) ); \
                    const auto shiftedChunk = vextq_u8( previousChunk, currentChunk, 15 ); \
                    vst1q_u8( chunkData, vbslq_u8( keepMask, currentChunk, shiftedChunk ) ); \
                    previousChunk = currentChunk; \
                } \
                front1 = vextq_u8( front0, front1, 15 ); \
                front0 = vextq_u8( vdupq_n_u8( static_cast<uint8_t>( BYTE ) ), front0, 15 ); \
            } while ( false )

#endif

#if LIBRAPIDARCHIVE_BZIP2_KERNEL_VARIANT != 0
    #define BZIP2_KERNEL_FRONT_SYMBOLS SIMD_KERNEL_FRONT_SIZE
#endif


[[nodiscard]] LIBRAPIDARCHIVE_BZIP2_KERNEL_TARGET inline SymbolDecoderStatus
LIBRAPIDARCHIVE_BZIP2_KERNEL_NAME( SymbolDecoderState&  state,
                                   const uint8_t* const buffer,
                                   const size_t         bufferSize,
                                   size_t&              bitOffset ) noexcept
{
    /* A group of GROUP_SIZE symbols is decoded without any bounds checks on the input. */
    const auto byteOffset = bitOffset >> 3U;
    if ( ( bufferSize < SYMBOL_KERNEL_MINIMUM_INPUT ) || ( byteOffset > bufferSize - SYMBOL_KERNEL_MINIMUM_INPUT ) ) {
        return SymbolDecoderStatus::NEED_INPUT;
    }

    /* The next unread bit is the highest bit of bitBuffer, bitCount many bits are valid, and the bits behind
     * those are a preview of the byte that @p in points to. See Fabian Giesen's "Reading bits in far too many
     * ways", variant 4. */
    const uint8_t* in = buffer + byteOffset;
    const uint8_t* const inLimit = buffer + ( bufferSize - SYMBOL_KERNEL_MINIMUM_INPUT );
    uint64_t bitBuffer = 0;
    uint32_t bitCount = 0;

#define BZIP2_KERNEL_REFILL() \
        do { \
            bitBuffer |= loadBigEndian64( in ) >> bitCount; \
            in += ( 63U - bitCount ) >> 3U; \
            bitCount |= 56U; \
        } while ( false )

    BZIP2_KERNEL_REFILL();
    bitBuffer <<= ( bitOffset & 7U );
    bitCount -= static_cast<uint32_t>( bitOffset & 7U );

    uint8_t* LIBRAPIDARCHIVE_BZIP2_RESTRICT record = state.record;
    const uint8_t* const recordLimit = state.recordLimit;
    uint32_t length = state.length;
    uint32_t runWeight = state.runWeight;
    const uint8_t* selector = state.selector;
    const uint8_t* const selectorEnd = state.selectorEnd;
    const SymbolTable* const tables = state.tables;
    auto status = SymbolDecoderStatus::NEED_INPUT;

#if LIBRAPIDARCHIVE_BZIP2_KERNEL_VARIANT != 0
    auto* const mtf = reinterpret_cast<uint8_t*>( state.mtf.data() );
    /* Number of vector-sized chunks of the list that the alphabet of this block uses. */
    const uint32_t chunkCount = ( state.symbolCount + BZIP2_KERNEL_CHUNK_SIZE - 1U ) / BZIP2_KERNEL_CHUNK_SIZE;
    BZIP2_KERNEL_LOAD_FRONT();
#endif

/* A record is limited to 255 bytes. Longer runs are split into several records for the same byte.
 * splitLongRecord gets copies because taking the address of the hot loop variables would keep them in memory. */
#define BZIP2_KERNEL_CHECK_RECORD_LENGTH() \
        do { \
            if ( UNLIKELY( length > 255U ) ) [[unlikely]] { \
                auto splitRecord = record; \
                auto splitLength = length; \
                if ( !splitLongRecord( splitRecord, recordLimit, splitLength ) ) { \
                    goto blockTooLarge; \
                } \
                record = splitRecord; \
                length = splitLength; \
            } \
        } while ( false )

#define BZIP2_KERNEL_START_RECORD( BYTE ) \
        do { \
            record += RUN_RECORD_SIZE; \
            record[0] = static_cast<uint8_t>( BYTE ); \
            record[1] = 1; \
            length = 1; \
            runWeight = 1; \
        } while ( false )

#if LIBRAPIDARCHIVE_BZIP2_KERNEL_VARIANT == 0

/* The portable variant branches between run symbols and literals and uses the packed list in memory. */
#define BZIP2_KERNEL_DECODE_SYMBOL() \
        do { \
            uint32_t entryValue = table->lookup( bitBuffer ); \
            if ( UNLIKELY( ( entryValue & SymbolTable::LENGTH_MASK ) == 0 ) ) [[unlikely]] { \
                entryValue = table->decodeLong( bitBuffer ); \
                if ( entryValue == SymbolTable::INVALID_ENTRY ) { \
                    goto invalidCode; \
                } \
            } \
            bitBuffer <<= ( entryValue & SymbolTable::LENGTH_MASK ); \
            bitCount -= entryValue & SymbolTable::LENGTH_MASK; \
            const uint32_t symbolValue = entryValue >> SymbolTable::LENGTH_BITS; \
            if ( symbolValue <= SYMBOL_RUNB ) { \
                length += runWeight << symbolValue; \
                runWeight <<= 1U; \
                record[1] = static_cast<uint8_t>( length ); \
                BZIP2_KERNEL_CHECK_RECORD_LENGTH(); \
            } else if ( LIKELY( symbolValue != SymbolTable::END_OF_BLOCK ) ) [[likely]] { \
                const auto byteValue = moveToFront( state.mtf, symbolValue - 1U ); \
                BZIP2_KERNEL_START_RECORD( byteValue ); \
            } else { \
                goto endOfBlock; \
            } \
        } while ( false )

#else

/* Handles RUNA, RUNB, and literals at list positions below BZIP2_KERNEL_FRONT_SYMBOLS without a branch:
 *  - The list update is the identity for RUNA and RUNB, so it is done for all symbols.
 *  - runMask is all ones for RUNA (0) and RUNB (1) and zero for literals. It relies on the arithmetic right
 *    shift of negative numbers, which GCC, Clang, and MSVC implement and which C++20 guarantees.
 *  - A run symbol adds one or two times the run weight to the current record and doubles the weight. Multiplying
 *    with SYMBOL + 1 instead of shifting keeps the shift count register free for the bit buffer.
 *  - A literal starts a new record with length one. The record pointer only advances for literals, so a run
 *    symbol rewrites the current record. */
#define BZIP2_KERNEL_FRONT_SYMBOL( SYMBOL ) \
        do { \
            BZIP2_KERNEL_FRONT_UPDATE( SYMBOL ); \
            const uint32_t byteValue = BZIP2_KERNEL_FRONT_BYTE(); \
            const uint32_t runMask = static_cast<uint32_t>( static_cast<int32_t>( static_cast<uint32_t>( SYMBOL ) - \
                                                                                  2U ) >> 31U ); \
            const uint32_t isLiteral = runMask + 1U; \
            length = ( ( length + runWeight * ( static_cast<uint32_t>( SYMBOL ) + 1U ) )&runMask ) + isLiteral; \
            runWeight = ( ( runWeight + runWeight )&runMask ) + isLiteral; \
            record += RUN_RECORD_SIZE * isLiteral; \
            record[0] = static_cast<uint8_t>( byteValue ); \
            record[1] = static_cast<uint8_t>( length ); \
            BZIP2_KERNEL_CHECK_RECORD_LENGTH(); \
        } while ( false )

/* Codes that are too long for the lookup table, deep list positions, and the end of the block all have an
 * entry above the limit so that the common case only needs this one branch. */
#define BZIP2_KERNEL_DECODE_SYMBOL() \
        do { \
            uint32_t entryValue = table->lookup( bitBuffer ); \
            if ( UNLIKELY( entryValue > ( ( BZIP2_KERNEL_FRONT_SYMBOLS << SymbolTable::LENGTH_BITS ) \
                                  | SymbolTable::LENGTH_MASK ) ) ) [[unlikely]] { \
                if ( ( entryValue & SymbolTable::LENGTH_MASK ) == 0 ) { \
                    entryValue = table->decodeLong( bitBuffer ); \
                    if ( entryValue == SymbolTable::INVALID_ENTRY ) { \
                        goto invalidCode; \
                    } \
                } \
                bitBuffer <<= ( entryValue & SymbolTable::LENGTH_MASK ); \
                bitCount -= entryValue & SymbolTable::LENGTH_MASK; \
                const uint32_t symbolValue = entryValue >> SymbolTable::LENGTH_BITS; \
                if ( symbolValue == SymbolTable::END_OF_BLOCK ) { \
                    goto endOfBlock; \
                } \
                if ( symbolValue > BZIP2_KERNEL_FRONT_SYMBOLS ) { \
                    const uint32_t tailPosition = symbolValue - 1U; \
                    const uint32_t tailByte = mtf[tailPosition]; \
                    BZIP2_KERNEL_MOVE_TAIL_TO_FRONT( tailPosition, tailByte ); \
                    BZIP2_KERNEL_START_RECORD( tailByte ); \
                } else { \
                    BZIP2_KERNEL_FRONT_SYMBOL( symbolValue ); \
                } \
            } else { \
                bitBuffer <<= ( entryValue & SymbolTable::LENGTH_MASK ); \
                bitCount -= entryValue & SymbolTable::LENGTH_MASK; \
                const uint32_t symbolValue = entryValue >> SymbolTable::LENGTH_BITS; \
                BZIP2_KERNEL_FRONT_SYMBOL( symbolValue ); \
            } \
        } while ( false )

#endif

    while ( true ) {
        if ( in > inLimit ) {
            break;
        }
        if ( UNLIKELY( selector == selectorEnd ) ) [[unlikely]] {
            status = SymbolDecoderStatus::OUT_OF_SELECTORS;
            break;
        }
        if ( UNLIKELY( record > recordLimit ) ) [[unlikely]] {
            goto blockTooLarge;
        }
        const SymbolTable* const table = tables + *selector++;

        /* One refill yields at least 56 valid bits, which suffices for two codes of MAX_HUFCODE_BITS = 20 bits.
         * The input checks happen once per group, which is why the buffer needs SYMBOL_KERNEL_MINIMUM_INPUT. */
        for ( int pair = 0; pair < GROUP_SIZE / 2; ++pair ) {
            BZIP2_KERNEL_REFILL();
            BZIP2_KERNEL_DECODE_SYMBOL();
            BZIP2_KERNEL_DECODE_SYMBOL();
        }
    }
    goto finished;

    /* Error and end states. The kernel always returns at a group boundary or at the end of the block, so the
     * state can be handed to decodeSymbolGroup or to the next call. */
endOfBlock:
    status = SymbolDecoderStatus::END_OF_BLOCK;
    goto finished;

invalidCode:
    status = SymbolDecoderStatus::INVALID_CODE;
    goto finished;

blockTooLarge:
    status = SymbolDecoderStatus::BLOCK_TOO_LARGE;

finished:
#if LIBRAPIDARCHIVE_BZIP2_KERNEL_VARIANT != 0
    BZIP2_KERNEL_STORE_FRONT();
#endif
    state.record = record;
    state.length = length;
    state.runWeight = runWeight;
    state.selector = selector;
    bitOffset = static_cast<size_t>( in - buffer ) * 8U - bitCount;
    return status;
}


#undef BZIP2_KERNEL_REFILL
#undef BZIP2_KERNEL_CHECK_RECORD_LENGTH
#undef BZIP2_KERNEL_START_RECORD
#undef BZIP2_KERNEL_DECODE_SYMBOL
#if LIBRAPIDARCHIVE_BZIP2_KERNEL_VARIANT != 0
    #undef BZIP2_KERNEL_FRONT_SYMBOLS
    #undef BZIP2_KERNEL_CHUNK_SIZE
    #undef BZIP2_KERNEL_LOAD_FRONT
    #undef BZIP2_KERNEL_STORE_FRONT
    #undef BZIP2_KERNEL_FRONT_UPDATE
    #undef BZIP2_KERNEL_FRONT_BYTE
    #undef BZIP2_KERNEL_MOVE_TAIL_TO_FRONT
    #undef BZIP2_KERNEL_FRONT_SYMBOL
#endif
