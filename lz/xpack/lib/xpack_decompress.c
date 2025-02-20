/*
 * xpack_decompress.c - decompressor for the XPACK compression format
 *
 * Copyright 2016 Eric Biggers
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifdef __SSE2__
#  include <emmintrin.h>
#endif

#include "xpack_common.h"
#include "x86_cpu_features.h"

/*
 * If the expression passed to SAFETY_CHECK() evaluates to false, then the
 * decompression routine immediately returns DECOMPRESS_BAD_DATA, indicating the
 * compressed data is invalid.
 *
 * Theoretically, these checks could be disabled for specialized applications
 * where all input to the decompressor will be trusted.
 */
#if 0
#  pragma message("UNSAFE DECOMPRESSION IS ENABLED. THIS MUST ONLY BE USED IF THE DECOMPRESSOR INPUT WILL ALWAYS BE TRUSTED!")
#  define SAFETY_CHECK(expr)	(void)(expr)
#else
#  define SAFETY_CHECK(expr)	if (unlikely(!(expr))) return DECOMPRESS_BAD_DATA
#endif

/*
 * An entry in a FSE decode table.  The index of the entry in the table is the
 * state with which the entry is associated.
 *
 * For efficiency we sometimes access this struct as the u32 value and sometimes
 * as the individual fields.
 */
typedef struct {
	union {
		u32 entry;
		struct { /* for big endian systems */
			u16 destination_range_start;
			u8 num_bits;
			u8 symbol;
		} be;
		struct { /* for little endian systems */
			u8 symbol;
			u8 num_bits;
			u16 destination_range_start;
		} le;
	};
} fse_decode_entry_t;

/*
 * DECODE_SYMBOL() - Macro to decode a FSE-encoded symbol.  The decoded symbol
 * is obtained from the decode table entry for the current state.  The state is
 * then updated to the next state, which is obtained by indexing the current
 * state's "destination range" with the next 'num_bits' bits of input data.
 */
#if 1 /* Optimized version which accesses the entry as a u32 */
#define DECODE_SYMBOL(state, decode_table)				\
(									\
	sym = decode_table[state].entry,				\
	state = (sym >> 16) + POP_BITS((sym >> 8) & 0xFF),		\
	sym & 0xFF							\
)

#else /* Unoptimized version which accesses individual struct members */
#define DECODE_SYMBOL(state, decode_table)				\
(									\
	sym = CPU_IS_LITTLE_ENDIAN() ?					\
	        decode_table[state].le.symbol :				\
	        decode_table[state].be.symbol,				\
	state = CPU_IS_LITTLE_ENDIAN() ?				\
		  decode_table[state].le.destination_range_start +	\
			POP_BITS(decode_table[state].le.num_bits) :	\
		  decode_table[state].be.destination_range_start +	\
			POP_BITS(decode_table[state].be.num_bits),	\
	sym								\
)
#endif

/*
 * Build the FSE decode table for an alphabet.
 *
 * @decode_table [out]
 *	The decode table to build.
 * @state_counts [in but invalidated]
 *	An array which provides, for each symbol in the alphabet, the number of
 *	states which should be assigned to that symbol.
 * @alphabet_size [in]
 *	The number of symbols in the alphabet.
 * @log2_num_states [in]
 *	The log base 2 of the number of states, which is also the number of
 *	entries in the decode table being built.
 *
 * Returns true if the state counts were valid or false if they were not.
 */
static bool
build_fse_decode_table(fse_decode_entry_t decode_table[], u16 state_counts[],
		       unsigned alphabet_size, unsigned log2_num_states)
{
	/*
	 * Assign a symbol to each state such that each symbol 'sym' gets
	 * assigned to exactly 'state_counts[sym]' states.  To do this, assign
	 * states to symbols in order of increasing symbol value while visiting
	 * all states in a special order.
	 */
	const unsigned num_states = 1 << log2_num_states;
	const unsigned state_generator = get_state_generator(num_states);
	const unsigned state_mask = num_states - 1;
	unsigned state = 0;
	u32 total_count = 0;
	unsigned sym;

	for (sym = 0; sym < alphabet_size; sym++) {
		unsigned count = state_counts[sym];
		if (count == 0) /* Unused symbol? */
			continue;
		total_count += count;
		do {
			decode_table[state].entry = sym;
			state = (state + state_generator) & state_mask;
		} while (--count);
	}

	/*
	 * Verify that the sum of the state counts really was
	 * 2**log2_num_states.  With a bad input, the sum might be lower than
	 * expected (in which case not all states were visited) or higher than
	 * expected (in which case some states were visited multiple times).
	 * Both cases are strictly forbidden.
	 */
	if (unlikely(total_count != num_states))
		return false;

	/*
	 * Now, set 'num_bits' and 'destination_range_start' for each decode
	 * table entry.  This works as follows.  First, a little background:
	 * given a symbol that is assigned 'count' states out of a total of
	 * 'num_states' states, the entropy, in bits, of an occurrence of that
	 * symbol is:
	 *
	 *		  log2(1/probability)
	 *		= log2(1/(count/num_states))
	 *		= log2(num_states/count)
	 *		= log2(num_states) - log2(count)
	 *
	 * This may be a non-integer value.  The rounded-down value is:
	 *
	 *     min_bits = floor(log2(num_states) - log2(count))
	 *		= log2(num_states) - ceil(log2(count))
	 *
	 * With finite state entropy coding, we will sometimes code the symbol
	 * using 'min_bits' bits and sometimes using 'min_bits + 1' bits.  Each
	 * of the symbol's 'count' states will be associated with one of these
	 * two choices of 'num_bits'.  In addition, each state will point to a
	 * "destination range" of length '2**num_bits'.  The destination range
	 * is the range of states which the encoder may have been in prior to
	 * encoding the symbol and entering a given state.
	 *
	 * The precise mapping of a symbol's states to bit counts and
	 * destination ranges is defined as follows.  For some 'X < count', the
	 * numerically first 'X' states are each assigned 'min_bits + 1' bits
	 * and are mapped consecutively to a series of destination ranges that
	 * ends with state 'num_states - 1'.  The remaining 'count - X' states
	 * are each assigned 'min_bits' bits and are mapped consecutively to a
	 * series of destination ranges that starts with state 0.  Since the
	 * destination ranges must exactly cover all 'num_states' states (this
	 * is required, in general, for encoding to have been possible), we can
	 * solve for 'X':
	 *
	 *	(2**(min_bits+1))X + (2**min_bits)(count - X) = num_states
	 *	(2**min_bits)(2X + count - X) = num_states
	 *	(2**min_bits)(X + count) = num_states
	 *	X + count = num_states / (2**min_bits)
	 *	X = num_states / (2**min_bits) - count
	 *
	 * As an example, with num_states = 256 and count = 23, then min_bits =
	 * log2(256) - ceil(log2(23)) = 8 - 5 = 3.  So each of the symbol's 23
	 * states will be assigned 3 ('min_bits') or 4 ('min_bits + 1') bits.
	 * Processing the 23 states in ascending numerical order, the first X
	 * states will each be assigned 4 bits and the next 23 - X states will
	 * each be assigned 3 bits.  X is:
	 *
	 *	X = num_states / (2**min_bits) - count
	 *	  = 256 / (2**3) - 23
	 *	  = 9
	 *
	 * Hence, the first 9 states will each be assigned 4 bits and have
	 * destination ranges covering the last 9 * 2**4 = 144 of the 256
	 * states, and the remaining 23 - 9 = 14 states will each be assigned 3
	 * bits and have destination ranges covering the first 14 * 2**3 = 112
	 * of the 256 states.
	 *
	 * There are a few possible implementations for actually computing
	 * 'num_bits' and 'destination_range_start' for each of a symbol's
	 * states.  What we do is iterate through *all* states in ascending
	 * order.  This interleaves states for different symbols but guarantees
	 * that all states for each symbol are visited in ascending order.
	 * 'state_counts[sym]' is re-used as a counter which is incremented each
	 * time after a state for symbol 'sym' is visited.  'X' is just the
	 * distance between the initial value of 'state_counts[sym]' and the
	 * closest power of 2 greater than or equal to 'state_counts[sym]'.
	 * When the counter reaches this power of 2, then the number of bits
	 * required, as computed by 'log2(num_states) - floor(log2(counter))',
	 * decreases from 'min_bits + 1' to 'min_bits'.  In addition, the
	 * destination range start for each state is easily computed from the
	 * value of the counter and num_bits at that state.
	 */
	for (state = 0; state < num_states; state++) {

		u32 sym = decode_table[state].entry;
		u32 counter = state_counts[sym]++;
		unsigned num_bits = log2_num_states - bsr32(counter);
		u32 destination_range_start = (counter << num_bits) - num_states;

		if (CPU_IS_LITTLE_ENDIAN()) {
			decode_table[state].le.num_bits = num_bits;
			decode_table[state].le.destination_range_start = destination_range_start;
		} else {
			decode_table[state].be.num_bits = num_bits;
			decode_table[state].be.destination_range_start = destination_range_start;
		}
	}

	return true;
}

/* Copy a word from @src to @dst, making no assumptions about alignment. */
static forceinline void
copy_word_unaligned(const u8 *src, u8 *dst)
{
	store_word_unaligned(load_word_unaligned(src), dst);
}

/* Copy 16 bytes from @src to @dst, making no assumptions about alignment. */
static forceinline void
copy_16_bytes_unaligned(const u8 *src, u8 *dst)
{
#ifdef __SSE2__
	__m128i v = _mm_loadu_si128((const __m128i *)src);
	_mm_storeu_si128((__m128i *)dst, v);
#else
	STATIC_ASSERT(WORDBYTES == 4 || WORDBYTES == 8);
	if (WORDBYTES == 4) {
		copy_word_unaligned(src + 0, dst + 0);
		copy_word_unaligned(src + 4, dst + 4);
		copy_word_unaligned(src + 8, dst + 8);
		copy_word_unaligned(src + 12, dst + 12);
	} else {
		copy_word_unaligned(src + 0, dst + 0);
		copy_word_unaligned(src + 8, dst + 8);
	}
#endif
}

/* Build a word which consists of the byte @b repeated. */
static forceinline machine_word_t
repeat_byte(u8 b)
{
	machine_word_t v;

	STATIC_ASSERT(WORDBITS == 32 || WORDBITS == 64);

	v = b;
	v |= v << 8;
	v |= v << 16;
	v |= v << ((WORDBITS == 64) ? 32 : 0);
	return v;
}


/******************************************************************************
 *				Input bitstream                               *
 ******************************************************************************/

/*
 * The state of the "input bitstream" consists of the following variables:
 *
 *	- in_next: pointer to the next unread byte in the input buffer
 *
 *	- in_end: pointer just past the end of the input buffer
 *
 *	- bitbuf: a word-sized variable containing bits that have been read from
 *		  the input buffer.  The buffered bits are right-aligned
 *		  (they're the low-order bits).
 *
 *	- bitsleft: number of bits in 'bitbuf' that are valid.
 *
 * To make it easier for the compiler to optimize the code by keeping variables
 * in registers, these are declared as normal variables and manipulated using
 * macros.
 */

/*
 * The maximum number of bits that can be requested to be in the bitbuffer
 * variable.  This is the maximum value of 'n' that can be passed to
 * ENSURE_BITS(n).
 *
 * This not equal to WORDBITS because we never read less than one byte at a
 * time.  If the bitbuffer variable contains more than (WORDBITS - 8) bits, then
 * we can't read another byte without first consuming some bits.  So the maximum
 * count we can ensure is (WORDBITS - 7).
 */
#define MAX_ENSURE	(WORDBITS - 7)

/*
 * Evaluates to true if 'n' is a valid argument to ENSURE_BITS(n), or false if
 * 'n' is too large to be passed to ENSURE_BITS(n).  Note: if 'n' is a compile
 * time constant, then this expression will be a compile-type constant.
 * Therefore, CAN_ENSURE() can be used choose between alternative
 * implementations at compile time.
 */
#define CAN_ENSURE(n)	((n) <= MAX_ENSURE)

/*
 * Fill the bitbuffer variable, reading one byte at a time.
 *
 * Note: if we would overrun the input buffer, we just don't read anything,
 * leaving the bits as 0 but marking them as filled.  This makes the
 * implementation simpler because this removes the need to distinguish between
 * "real" overruns and overruns that occur because of our own lookahead during
 * decompression.  The disadvantage is that a "real" overrun can go undetected,
 * and the decompressor may return a success status rather than the expected
 * failure status if one occurs.  However, this is not too important because
 * even if this specific case were to be handled "correctly", one could easily
 * come up with a different case where the compressed data would be corrupted in
 * such a way that fully retains its validity from the point of view of the
 * decompressor.  Users should run a checksum against the decompressed data if
 * they wish to detect corruptions.
 */
#define FILL_BITS_BYTEWISE()						  \
do {									  \
	do {								  \
		if (likely(in_next != in_end))				  \
			bitbuf |= (machine_word_t)*in_next++ << bitsleft; \
		else							  \
			overrun_count++;				  \
		bitsleft += 8;						  \
	} while (bitsleft <= WORDBITS - 8);				  \
} while (0)

/*
 * Fill the bitbuffer variable by reading the next word from the input buffer.
 * This can be significantly faster than FILL_BITS_BYTEWISE().  However, for
 * this to work correctly, the word must be interpreted in little-endian format.
 * In addition, the memory access may be unaligned.  Therefore, this method is
 * most efficient on little-endian architectures that support fast unaligned
 * access, such as x86 and x86_64.
 */
#define FILL_BITS_WORDWISE()						\
do {									\
	bitbuf |= get_unaligned_leword(in_next) << bitsleft;		\
	in_next += (WORDBITS - bitsleft) >> 3;				\
	bitsleft += (WORDBITS - bitsleft) & ~7;				\
} while (0)

/*
 * Load more bits from the input buffer until the specified number of bits is
 * present in the bitbuffer variable.  'n' must be <= MAX_ENSURE.
 */
#define ENSURE_BITS(n)							\
do {									\
	if (bitsleft < (n)) {						\
		if (UNALIGNED_ACCESS_IS_FAST &&				\
		    likely(in_end - in_next >= WORDBYTES))		\
			FILL_BITS_WORDWISE();				\
		else							\
			FILL_BITS_BYTEWISE();				\
	}								\
} while (0)

/* Remove and return the next 'n' bits from the bitbuffer variable. */
#define POP_BITS(n)							\
(									\
	bits = (u32)bitbuf & (((u32)1 << (n)) - 1),			\
	bitbuf >>= (n),							\
	bitsleft -= (n),						\
	bits								\
)

/*
 * Align the input to the next byte boundary, discarding any remaining bits in
 * the current byte.
 *
 * Note that if the bitbuffer variable currently contains more than 8 bits, then
 * we must rewind 'in_next', effectively putting those bits back.  Only the bits
 * in what would be the "current" byte if we were reading one byte at a time can
 * be actually discarded.
 */
#define ALIGN_INPUT()							\
do {									\
	in_next -= (bitsleft >> 3) - MIN(overrun_count, bitsleft >> 3);	\
	bitbuf = 0;							\
	bitsleft = 0;							\
} while (0)


/* The main decompressor structure */
struct xpack_decompressor {

	/*
	 * The FSE decoding table for each alphabet.  The literal table can be
	 * in union with the other tables because all literal symbols are
	 * decoded first.
	 */
	union {
		fse_decode_entry_t literal_decode_table
				[1 << MAX_LOG2_NUM_LITERAL_STATES];
		struct {
			fse_decode_entry_t litrunlen_decode_table
					[1 << MAX_LOG2_NUM_LITRUNLEN_STATES];
			fse_decode_entry_t length_decode_table
					[1 << MAX_LOG2_NUM_LENGTH_STATES];
			fse_decode_entry_t offset_decode_table
					[1 << MAX_LOG2_NUM_OFFSET_STATES];
			fse_decode_entry_t aligned_decode_table
					[1 << MAX_LOG2_NUM_ALIGNED_STATES];
		};
	};

	/* The FSE state counts for each alphabet */
	union {
		u16 state_counts[LITERAL_ALPHABET_SIZE +
				 LITRUNLEN_ALPHABET_SIZE +
				 LENGTH_ALPHABET_SIZE +
				 MAX_OFFSET_ALPHABET_SIZE +
				 ALIGNED_ALPHABET_SIZE];
		struct {
			u16 literal_state_counts[LITERAL_ALPHABET_SIZE];
			u16 litrunlen_state_counts[LITRUNLEN_ALPHABET_SIZE];
			u16 length_state_counts[LENGTH_ALPHABET_SIZE];
			u16 offset_state_counts[MAX_OFFSET_ALPHABET_SIZE];
			u16 aligned_state_counts[ALIGNED_ALPHABET_SIZE];
		};
	};
};

#define FUNCNAME xpack_decompress_default
#define ATTRIBUTES
#include "decompress_impl.h"
#undef FUNCNAME
#undef ATTRIBUTES

#if X86_CPU_FEATURES_ENABLED && \
	COMPILER_SUPPORTS_BMI2_TARGET && !defined(__BMI2__)
#  define FUNCNAME xpack_decompress_bmi2
#  define ATTRIBUTES __attribute__((target("bmi2")))
#  include "decompress_impl.h"
#  undef FUNCNAME
#  undef ATTRIBUTES
#  define DISPATCH_ENABLED 1
#else
#  define DISPATCH_ENABLED 0
#endif

#if DISPATCH_ENABLED

static enum decompress_result
dispatch(struct xpack_decompressor *d, const void *in, size_t in_nbytes,
	 void *out, size_t out_nbytes_avail, size_t *actual_out_nbytes_ret);

typedef enum decompress_result (*decompress_func_t)
	(struct xpack_decompressor *d, const void *in, size_t in_nbytes,
	 void *out, size_t out_nbytes_avail, size_t *actual_out_nbytes_ret);

static decompress_func_t decompress_impl = dispatch;

static enum decompress_result
dispatch(struct xpack_decompressor *d, const void *in, size_t in_nbytes,
	 void *out, size_t out_nbytes_avail, size_t *actual_out_nbytes_ret)
{
	decompress_func_t f = xpack_decompress_default;
#if X86_CPU_FEATURES_ENABLED
	if (x86_have_cpu_feature(X86_CPU_FEATURE_BMI2))
		f = xpack_decompress_bmi2;
#endif
	decompress_impl = f;
	return (*f)(d, in, in_nbytes, out, out_nbytes_avail,
		    actual_out_nbytes_ret);
}
#endif /* DISPATCH_ENABLED */

/*
 * This is the main decompression routine.  See libxpack.h for the
 * documentation.
 *
 * Note that the real code is in decompress_impl.h.  The part here just handles
 * calling the appropriate implementation depending on the CPU features at
 * runtime.
 */
LIBEXPORT enum decompress_result
xpack_decompress(struct xpack_decompressor *d, const void *in, size_t in_nbytes,
		 void *out, size_t out_nbytes_avail,
		 size_t *actual_out_nbytes_ret)
{
#if DISPATCH_ENABLED
	return (*decompress_impl)(d, in, in_nbytes, out, out_nbytes_avail,
				  actual_out_nbytes_ret);
#else
	return xpack_decompress_default(d, in, in_nbytes, out, out_nbytes_avail,
					actual_out_nbytes_ret);
#endif
}

LIBEXPORT struct xpack_decompressor *
xpack_alloc_decompressor(void)
{
	return malloc(sizeof(struct xpack_decompressor));
}

LIBEXPORT void
xpack_free_decompressor(struct xpack_decompressor *d)
{
	free(d);
}
