/*
 * xpack_compress.c - compressor for the XPACK compression format
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

#ifndef DECOMPRESSION_ONLY

#ifdef __SSE2__
#  include <emmintrin.h>
#endif
#ifdef __SSE4_1__
#  include <smmintrin.h>
#endif

#include "hc_matchfinder.h"
#include "lz_extend.h"
#include "xpack_common.h"

/*
 * The compressor always chooses a block of at least MIN_BLOCK_LENGTH bytes,
 * except if the last block has to be shorter.
 */
#define MIN_BLOCK_LENGTH	10000

/*
 * The compressor attempts to end blocks after SOFT_MAX_BLOCK_LENGTH bytes, but
 * the final size might be larger due to matches extending beyond the end of the
 * block.  Specifically:
 *
 *  - The greedy parser may choose an arbitrarily long match starting at the
 *    SOFT_MAX_BLOCK_LENGTH'th byte.
 *
 *  - The lazy parser may choose a sequence of literals starting at the
 *    SOFT_MAX_BLOCK_LENGTH'th byte when it sees a sequence of increasing good
 *    matches.  The final match may be of arbitrary length.  The length of the
 *    literal sequence is approximately limited by the "nice match length"
 *    parameter.  The actual limit is related to match scores and may be
 *    slightly different.  We overestimate the limit as EXTRA_LITERAL_SPACE.
 */
#define SOFT_MAX_BLOCK_LENGTH	300000
#define EXTRA_LITERAL_SPACE	512

/* Holds the symbols and extra offset bits needed to represent a match */
struct match {
	u8 litrunlen_sym;
	u8 length_sym;
	u8 offset_sym;
	u32 extra_offset_bits;
};

/* Frequency counters for each alphabet */
struct freqs {
	u32 literal[LITERAL_ALPHABET_SIZE];
	u32 litrunlen[LITRUNLEN_ALPHABET_SIZE];
	u32 length[LENGTH_ALPHABET_SIZE];
	u32 offset[MAX_OFFSET_ALPHABET_SIZE];
	u32 aligned[ALIGNED_ALPHABET_SIZE];
};

/* Finite State Entropy encoding information for a symbol */
struct fse_symbol_encoding_info {
	u32 adjusted_num_states_in_big_ranges;
	s32 next_states_begin;
};

/* Finite State Entropy encoding information for each alphabet */
struct codes {
	struct fse_symbol_encoding_info literal_sym_encinfo[LITERAL_ALPHABET_SIZE];
	struct fse_symbol_encoding_info litrunlen_sym_encinfo[LITRUNLEN_ALPHABET_SIZE];
	struct fse_symbol_encoding_info length_sym_encinfo[LENGTH_ALPHABET_SIZE];
	struct fse_symbol_encoding_info offset_sym_encinfo[MAX_OFFSET_ALPHABET_SIZE];
	struct fse_symbol_encoding_info aligned_sym_encinfo[ALIGNED_ALPHABET_SIZE];

	u16 literal_next_statesx[1 << MAX_LOG2_NUM_LITERAL_STATES];
	u16 litrunlen_next_statesx[1 << MAX_LOG2_NUM_LITRUNLEN_STATES];
	u16 length_next_statesx[1 << MAX_LOG2_NUM_LENGTH_STATES];
	u16 offset_next_statesx[1 << MAX_LOG2_NUM_OFFSET_STATES];
	u16 aligned_next_statesx[1 << MAX_LOG2_NUM_ALIGNED_STATES];

	unsigned log2_num_literal_states;
	unsigned log2_num_litrunlen_states;
	unsigned log2_num_length_states;
	unsigned log2_num_offset_states;
	unsigned log2_num_aligned_states;

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

/* Block split statistics.  See "Block splitting algorithm" below. */
#define NUM_LITERAL_OBSERVATION_TYPES 8
#define NUM_MATCH_OBSERVATION_TYPES 2
#define NUM_OBSERVATION_TYPES (NUM_LITERAL_OBSERVATION_TYPES + NUM_MATCH_OBSERVATION_TYPES)
struct block_split_stats {
	u32 new_observations[NUM_OBSERVATION_TYPES];
	u32 observations[NUM_OBSERVATION_TYPES];
	u32 num_new_observations;
	u32 num_observations;
};

/* The main compressor structure */
struct xpack_compressor {

	unsigned nice_match_length;
	unsigned max_search_depth;
	u8 *in_buffer;
	size_t in_nbytes;
	size_t max_buffer_size;
	size_t (*impl)(struct xpack_compressor *, void *, size_t);

	struct freqs freqs;
	struct block_split_stats split_stats;
	struct codes codes;

	unsigned cumul_state_counts[MAX_ALPHABET_SIZE];
	u8 state_to_symbol[MAX_NUM_STATES];

	u32 num_literals;
	u32 num_matches;
	u32 num_extra_bytes;

	u8 literals[SOFT_MAX_BLOCK_LENGTH + EXTRA_LITERAL_SPACE];
	struct match matches[DIV_ROUND_UP(SOFT_MAX_BLOCK_LENGTH, MIN_MATCH_LEN) + 1];
	u8 extra_bytes[6 + /* extra for actual block length > soft max */
		MAX4(1 * DIV_ROUND_UP(SOFT_MAX_BLOCK_LENGTH,
				      MIN_MATCH_LEN + LENGTH_ALPHABET_SIZE - 1),
		     3 * DIV_ROUND_UP(SOFT_MAX_BLOCK_LENGTH,
				      MIN_MATCH_LEN + LENGTH_ALPHABET_SIZE - 1 + 0xFF),
		     1 * DIV_ROUND_UP(SOFT_MAX_BLOCK_LENGTH,
				      LITRUNLEN_ALPHABET_SIZE - 1),
		     3 * DIV_ROUND_UP(SOFT_MAX_BLOCK_LENGTH,
				      LITRUNLEN_ALPHABET_SIZE - 1 + 0xFF))];

	/* Hash chains matchfinder (MUST BE LAST!!!) */
	struct hc_matchfinder hc_mf;
};

/* Return the log base 2 of 'n', rounded up to the nearest integer. */
static forceinline unsigned
ilog2_ceil(u32 n)
{
	if (n <= 1)
		return 0;
	return 1 + bsr32(n - 1);
}

/* Select the log2(num_states) to use for an alphabet. */
static unsigned
select_log2_num_states(u32 total_freq, unsigned num_used_syms,
		       unsigned max_log2_num_states)
{
	unsigned num_states = 1 << max_log2_num_states;  /* Default value */

	/*
	 * If there are not many symbols to be encoded, then it's not helpful to
	 * use many states.
	 */
	num_states = MIN(num_states, total_freq / 4);

	/*
	 * There must be at least as many states as distinct used symbols.
	 * Note: we're guaranteed num_used_syms > 0 here because of the earlier
	 * check, which implies that this calculation produces num_states > 0.
	 */
	num_states = MAX(num_states, num_used_syms);

	return ilog2_ceil(num_states);
}

/* Remove states from symbols until the correct number of states is used. */
static void
adjust_state_counts(u16 state_counts[], unsigned num_states_overrun,
		    unsigned alphabet_size)
{
	unsigned shift;
	unsigned sym;
	unsigned n;

	for (shift = 3; num_states_overrun != 0; shift--) {
		for (sym = 0; sym < alphabet_size; sym++) {
			if (state_counts[sym] > 1) {
				n = MIN((state_counts[sym] - 1) >> shift,
					num_states_overrun);
				state_counts[sym] -= n;
				num_states_overrun -= n;
				if (num_states_overrun == 0)
					break;
			}
		}
	}
}

/*
 * Determine how many states to assign to each symbol.
 *
 * Basically, for each symbol 'sym' we need to take the real number
 *
 *	freqs[sym] * (num_states / total_freq)
 *
 * and round it up or down to the nearest integer as appropriate to make all the
 * state_counts[] sum to num_states, while still approximating the real entropy
 * well.  However, this implementation does *not* compute the entropy-optimal
 * state counts.
 */
static void
compute_state_counts(const u32 freqs[], const u32 total_freq,
		     u16 state_counts[], const unsigned alphabet_size,
		     const unsigned log2_num_states)
{
	signed int remaining_states = 1 << log2_num_states;
	unsigned max_state_count = 0;
	unsigned sym_with_max_state_count = 0;
	unsigned sym = 0;

#if 0
	const float scale_factor = (float)(1 << log2_num_states) / (float)total_freq;
	const __m128 v_scale_factor = _mm_set1_ps(scale_factor);
	const __m128i v_lowcount_cutoff = _mm_set1_epi16(0x7FFF - (u16)(0.5 / scale_factor));
	__m128i v_num_states_used = _mm_set1_epi16(0);
	__m128i v_max_state_count = _mm_set1_epi16(0);
	__m128i v_sym_with_max_state_count = _mm_set1_epi16(0);
	__m128i v_syms = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);

	/* Process 8 freqs at a time  */
	for (; sym < (alphabet_size & ~7); sym += 8) {

		/* Load the next freqs. */
		__m128i v_freq1 = _mm_loadu_si128((const __m128i *)&freqs[sym + 0]);
		__m128i v_freq2 = _mm_loadu_si128((const __m128i *)&freqs[sym + 4]);

		/* Prepare adjustment for the 'state_count == 0 && freq != 0' case  */
		__m128i v_freqpack_saturated = _mm_packs_epi32(v_freq1, v_freq2);
		__m128i v_freqpack_saturated_adjusted = _mm_add_epi16(v_freqpack_saturated,
								      v_lowcount_cutoff);
		__m128i v_negative_adjustment = _mm_cmpgt_epi16(v_freqpack_saturated_adjusted,
								v_lowcount_cutoff);

		/* Compute: state_count = round(count * (num_states / total_freq))  */
		__m128 v_freqf1 = _mm_cvtepi32_ps(v_freq1);
		__m128 v_freqf2 = _mm_cvtepi32_ps(v_freq2);
		__m128 v_mul1 = _mm_mul_ps(v_freqf1, v_scale_factor);
		__m128 v_mul2 = _mm_mul_ps(v_freqf2, v_scale_factor);
		__m128i v_muli1 = _mm_cvtps_epi32(v_mul1);
		__m128i v_muli2 = _mm_cvtps_epi32(v_mul2);
		__m128i v_state_count = _mm_packs_epi32(v_muli1, v_muli2);

		/* If state_count == 0 but freq != 0, set state_count=1. */
		v_state_count = _mm_sub_epi16(v_state_count, v_negative_adjustment);

		/* Save the state counts  */
		_mm_storeu_si128((__m128i *)&state_counts[sym], v_state_count);

		/* Update num_states_used  */
		v_num_states_used = _mm_add_epi16(v_num_states_used, v_state_count);

		/* Update max_state_count  */
		v_max_state_count = _mm_max_epi16(v_max_state_count, v_state_count);

		/* Update sym_with_max_state_count  */
		__m128i v_is_new_max = _mm_cmpeq_epi16(v_state_count, v_max_state_count);
	#ifdef __SSE4_1__
		v_sym_with_max_state_count = _mm_blendv_epi8(v_sym_with_max_state_count,
							     v_syms, v_is_new_max);
	#else
		__m128i v_old_syms_to_keep = _mm_andnot_si128(v_is_new_max, v_sym_with_max_state_count);
		__m128i v_new_syms_to_set = _mm_and_si128(v_is_new_max, v_syms);
		v_sym_with_max_state_count = _mm_or_si128(v_old_syms_to_keep, v_new_syms_to_set);
	#endif

		v_syms = _mm_add_epi32(v_syms, _mm_set1_epi16(8));
	}

	for (int i = 0; i < 8; i++) {
		remaining_states -= ((__v8hi)v_num_states_used)[i];
		if (((__v8hi)v_max_state_count)[i] > max_state_count) {
			max_state_count = ((__v8hi)v_max_state_count)[i];
			sym_with_max_state_count = ((__v8hi)v_sym_with_max_state_count)[i];
		}
	}
#endif /* __SSE2__ */

	const u32 highprec_step = ((u32)1 << 31) / total_freq;
	const unsigned shift = 31 - log2_num_states - 1;

	for (; sym < alphabet_size; sym++) {
		 /*
		  * Rescale the frequency.  Round up if the fractional part is
		  * greater than or equal to 0.5.  Otherwise, round down.
		  */
		unsigned state_count =
			(((freqs[sym] * highprec_step) >> shift) + 1) >> 1;

		if (state_count == 0 && freqs[sym] != 0)
			state_count = 1;

		state_counts[sym] = state_count;
		remaining_states -= state_count;

		if (state_count > max_state_count) {
			max_state_count = state_count;
			sym_with_max_state_count = sym;
		}
	}

	/*
	 * If there are still states to assign, assign them to the most common
	 * symbol.  Or if we assigned more states than were actually available,
	 * then either subtract from the most common symbol (for minor overruns)
	 * or use the slower adjustment algorithm (for major overruns).
	 */
	if (-remaining_states < (signed int)(max_state_count >> 2)) {
		state_counts[sym_with_max_state_count] += remaining_states;
	} else {
		adjust_state_counts(state_counts, -remaining_states,
				    alphabet_size);
	}
}

/* Build the FSE encoding tables for an alphabet, given the state counts. */
static void
build_fse_encoding_tables(struct xpack_compressor *c,
			  struct fse_symbol_encoding_info sym_encinfo[],
			  u16 next_statesx[],
			  const u16 state_counts[],
			  const unsigned alphabet_size,
			  const unsigned log2_num_states)
{
	const unsigned num_states = 1 << log2_num_states;
	const unsigned state_generator = get_state_generator(num_states);
	const unsigned state_mask = num_states - 1;
	unsigned cumul_total;
	unsigned sym;
	unsigned state;
	unsigned count;
	unsigned max_bits;

	/*
	 * Build sym_encinfo[], which provides encoding information for each
	 * used symbol.  At the same time, build cumul_state_counts[], which for
	 * each symbol provides the total state count of the symbols that
	 * numerically precede it.
	 */
	cumul_total = 0;
	for (sym = 0; sym < alphabet_size; sym++) {

		count = state_counts[sym];

		if (count == 0) /* Unused symbol? */
			continue;

		c->cumul_state_counts[sym] = cumul_total;

		/*
		 * Each encoding of this symbol requires either 'min_bits' or
		 * 'max_bits = min_bits + 1' bits, where 'min_bits' is the
		 * entropy of this symbol rounded down to the nearest integer:
		 *
		 *	min_bits = floor(log2(1/probability))
		 *	min_bits = floor(log2(1/(count/num_states)))
		 *	min_bits = floor(log2(num_states/count))
		 *	min_bits = floor(log2(num_states) - log2(count))
		 *	min_bits = log2(num_states) - ceil(log2(count))
		 */
		max_bits = log2_num_states - ilog2_ceil(count) + 1;

		/*
		 * Save a value that makes it possible to branchlessly find the
		 * num_bits for a given state.  See encode_symbol() for details.
		 */
		sym_encinfo[sym].adjusted_num_states_in_big_ranges =
			((u32)max_bits << MAX_LOG2_NUM_STATES) -
			((u32)count << max_bits);

		/*
		 * When we need to encode an instance of this symbol, we'll have
		 * a "current state".  We'll need to find which destination
		 * range the current state is in, and which state --- the "next
		 * state" from the encoder's point of view but the "previous
		 * state" from the decoder's point of view --- maps to that
		 * destination range.  How can we do this efficiently?
		 *
		 * The solution relies on these facts:
		 *
		 *   - We'll know the number of bits to use.  Consequently,
		 *     we'll know the length of the destination range.
		 *   - The 'min_bits' destination ranges all precede the
		 *     'max_bits' destination ranges.
		 *
		 * What we'll do is maintain the state adjusted upwards by
		 * 'num_states'.  Then, we'll right-shift it by the number of
		 * bits that need to be used.  If 'min_bits' were required, then
		 * the result will be 'num_states >> min_bits' plus the index of
		 * the destination range in the list of 'min_bits' destination
		 * ranges.  But if 'max_bits' were required, then the result
		 * will be 'num_states >> min_bits' minus the number of
		 * 'max_bits' destination ranges, plus the index of the
		 * destination range in the list of 'max_bits' destination
		 * ranges.  Result: we map states to consecutive integers, each
		 * of which identifies a destination range.  We can use these
		 * integers as indices into a lookup table for the next state.
		 *
		 * Below, 'cumul_total' is the index at which the entries will
		 * actually begin in 'next_statesx[]'.  'count' is the beginning
		 * of the sequence of destination range identifiers.  This is
		 * 'num_states >> min_bits' minus the number of 'max_bits'
		 * destination ranges, which is also the same as the number of
		 * states (or destination ranges).  Note that the result of the
		 * subtraction may be a negative number.
		 */
		sym_encinfo[sym].next_states_begin = (s32)cumul_total - (s32)count;

		cumul_total += count;
	}

	/* Assign states to symbols. */
	state = 0;
	for (sym = 0; sym < alphabet_size; sym++) {
		count = state_counts[sym];
		while (count--) {
			c->state_to_symbol[state] = sym;
			state = (state + state_generator) & state_mask;
		}
	}

	/*
	 * Build next_statesx[].  This array maps symbol occurrences in the
	 * state table, ordered primarily by increasing symbol value and
	 * secondarily by increasing state, to their states, adjusted upwards by
	 * num_states.
	 */
	for (state = 0; state < num_states; state++) {
		unsigned symbol = c->state_to_symbol[state];
		unsigned position = c->cumul_state_counts[symbol]++;
		next_statesx[position] = num_states + state;
	}
}

/*
 * Choose the FSE state counts for the specified alphabet, where each symbol has
 * the frequency given in @freqs.
 */
static unsigned
choose_state_counts(const u32 freqs[], unsigned alphabet_size,
		    unsigned max_log2_num_states, u16 state_counts[])
{
	u32 total_freq = 0;
	unsigned num_used_syms = 0;
	unsigned log2_num_states;
	unsigned sym;

	/* Compute the total frequency and the number of used symbols. */
	for (sym = 0; sym < alphabet_size; sym++) {
		if (freqs[sym] != 0) {
			num_used_syms++;
			total_freq += freqs[sym];
		}
	}

	/*
	 * If no symbols from this alphabet were used, then output a code that
	 * contains an arbitrary unused symbol.
	 */
	if (total_freq == 0) {
		state_counts[0] = 1;
		for (sym = 1; sym < alphabet_size; sym++)
			state_counts[sym] = 0;
		return 0;
	}

	/* Select the number of states to use. */
	log2_num_states = select_log2_num_states(total_freq, num_used_syms,
						 max_log2_num_states);

	/* Decide how many states to assign to each symbol. */
	compute_state_counts(freqs, total_freq, state_counts,
			     alphabet_size, log2_num_states);

	return log2_num_states;
}

/* Output stream for header (writes in forwards direction) */
struct header_ostream {
	machine_word_t bitbuf;
	unsigned bitcount;
	u8 *begin;
	u8 *next;
	u8 *end;
};

static void
header_ostream_init(struct header_ostream *os,
		    void *out, size_t out_nbytes_avail)
{
	os->bitbuf = 0;
	os->bitcount = 0;
	os->begin = out;
	os->next = os->begin;
	os->end = os->next + out_nbytes_avail;
}

static void
header_ostream_write_bits(struct header_ostream *os,
			  machine_word_t bits, unsigned num_bits)
{
	/*
	 * We only flush 'bitbuf' when it completely fills up.  This improves
	 * performance.
	 */
	os->bitbuf |= bits << os->bitcount;
	os->bitcount += num_bits;
	if (os->bitcount >= WORDBITS) {
		if (os->end - os->next >= WORDBYTES) {
			put_unaligned_leword(os->bitbuf, os->next);
			os->next += WORDBYTES;
		} else {
			os->next = os->end;
		}
		os->bitcount -= WORDBITS;
		os->bitbuf = bits >> (num_bits - os->bitcount);
	}
}

static size_t
header_ostream_flush(struct header_ostream *os)
{
	while ((int)os->bitcount > 0) {
		if (os->next != os->end)
			*os->next++ = os->bitbuf;
		os->bitcount -= 8;
		os->bitbuf >>= 8;
	}

	if (os->next == os->end)  /* overflow? */
		return 0;

	return os->next - os->begin;
}

/*
 * Output the state counts.  Return the number of bytes written, or 0 if the
 * output buffer is too small.
 */
static void
write_state_counts(struct header_ostream *os,
		   const u16 state_counts[], unsigned num_state_counts)
{
	unsigned sym = 0;

	while (sym < num_state_counts) {
		unsigned count = state_counts[sym++];
		unsigned bits;
		unsigned num_bits;

		if (count == 0) {
			unsigned start = sym - 1;
			unsigned num_zeroes;

			while (sym < num_state_counts && state_counts[sym] == 0)
				sym++;
			num_zeroes = sym - start;

			while (num_zeroes >= ZEROCODE2_MIN) {
				unsigned count = MIN(num_zeroes, ZEROCODE2_MAX);
				bits = ((count - ZEROCODE2_MIN) << CODEBITS) | ZEROCODE2;
				num_bits = ZEROCODE2_NBITS + CODEBITS;
				header_ostream_write_bits(os, bits, num_bits);
				num_zeroes -= count;
			}

			if (num_zeroes < ZEROCODE1_MIN)
				continue;

			bits = ((num_zeroes - ZEROCODE1_MIN) << CODEBITS) | ZEROCODE1;
			num_bits = ZEROCODE1_NBITS + CODEBITS;
		} else {
			unsigned order = bsr32(count);
			bits = ((count ^ (1 << order)) << CODEBITS) | order;
			num_bits = order + CODEBITS;
		}
		header_ostream_write_bits(os, bits, num_bits);
	}
}

/* Output stream for encoded symbols (writes in backwards direction) */
struct symbol_ostream {
	machine_word_t bitbuf;
	unsigned bitcount;
	u8 *begin;
	u8 *next;
	u8 *end;
};

static void
symbol_ostream_init(struct symbol_ostream *os, void *buffer, size_t size)
{
	os->bitbuf = 0;
	os->bitcount = 0;
	os->begin = buffer;
	os->end = os->begin + size;
	os->next = os->end - MIN(WORDBYTES, size);
}

/*
 * Add bits to the bitbuffer variable, without flushing.  The caller must ensure
 * there is enough space.
 */
static forceinline void
symbol_ostream_add_bits(struct symbol_ostream *os, machine_word_t bits, unsigned num_bits)
{
	os->bitbuf = (os->bitbuf << num_bits) | bits;
	os->bitcount += num_bits;
}

/*
 * Flush bits from the bitbuffer variable to the output buffer.  After calling
 * this, the bitbuffer variable is guaranteed to contain fewer than 8 bits.
 */
static forceinline void
symbol_ostream_flush_bits(struct symbol_ostream *os)
{
	machine_word_t bits = os->bitbuf <<
		((WORDBITS - os->bitcount) & (WORDBITS - 1));

	put_unaligned_leword(bits, os->next);
	os->next -= MIN(os->next - os->begin, os->bitcount >> 3);
	os->bitcount &= 7;
}

/*
 * Flush any remaining bits to the output buffer and terminate the bitstream.
 * Return the total number of bytes written to the output buffer, or 0 if there
 * was not enough space available in the output buffer to write everything.
 */
static size_t
symbol_ostream_flush(struct symbol_ostream *os)
{
	symbol_ostream_flush_bits(os);

	if (os->next == os->begin) /* Not enough space? */
		return 0;

	/*
	 * Terminate the last byte with a '1' bit so that the decoder knows
	 * where to start from.
	 */
	os->bitbuf <<= 8 - os->bitcount;
	os->bitbuf |= (1 << (7 - os->bitcount));
	os->next += WORDBYTES - 1;
	*os->next = (u8)os->bitbuf;

	return os->end - os->next;
}

static forceinline void
encode_initial_state(struct symbol_ostream *os, unsigned initial_statex,
		     unsigned log2_num_states)
{
	symbol_ostream_add_bits(os, initial_statex - (1 << log2_num_states),
				log2_num_states);
	symbol_ostream_flush_bits(os);
}

/* Encode a symbol using Finite State Entropy encoding */
static forceinline unsigned
encode_symbol(unsigned symbol, unsigned cur_statex, struct symbol_ostream *os,
	      const struct fse_symbol_encoding_info sym_encinfo[],
	      const u16 next_statesx[])
{
	unsigned num_bits;

	/*
	 * Calculate the number of bits required to encode this symbol when in
	 * the current state.  'adjusted_num_states_in_big_ranges' was set to
	 * (max_bits << MAX_LOG2_NUM_STATES) - 2*num_states + (number of states
	 * in max_bits destination ranges).  If we add cur_statex (which is
	 * num_states plus the current state) to this value, then we get a
	 * number less than max_bits << MAX_LOG2_NUM_STATES iff the current
	 * state is in a min_bits destination range (as opposed to a 'max_bits =
	 * min_bits + 1' destination range).  Then the correct num_bits, which
	 * is always either min_bits or max_bits, is simply that value right
	 * shifted by MAX_LOG2_NUM_STATES.
	 */
	num_bits = (sym_encinfo[symbol].adjusted_num_states_in_big_ranges +
		    cur_statex) >> MAX_LOG2_NUM_STATES;

	/* Output the appropriate number of bits of the state. */
	symbol_ostream_add_bits(os, cur_statex & ((1 << num_bits) - 1), num_bits);

	/* Look up the next state using the high bits of the current state. */
	return next_statesx[sym_encinfo[symbol].next_states_begin +
			    (cur_statex >> num_bits)];
}

/*
 * Encode the matches and literals.  Note that the encoding order is backwards
 * from the decoding order!
 */
static size_t
encode_items(const struct xpack_compressor *c, void *out, size_t out_nbytes_avail,
	     bool is_aligned_block)
{
	struct symbol_ostream os;
	size_t nbytes;
	unsigned order;
	unsigned litrunlen_statex;
	unsigned length_statex;
	unsigned offset_statex;
	unsigned aligned_statex;
#if NUM_LITERAL_STREAMS == 2
	unsigned literal_statex_1;
	unsigned literal_statex_2;
#else
	unsigned literal_statex;
#endif
	s32 i;

	symbol_ostream_init(&os, out, out_nbytes_avail);

	/* Encode the matches and literal run lengths */

	litrunlen_statex = 1 << c->codes.log2_num_litrunlen_states;
	length_statex = 1 << c->codes.log2_num_length_states;
	offset_statex = 1 << c->codes.log2_num_offset_states;
	aligned_statex = 1 << c->codes.log2_num_aligned_states;

	i = c->num_matches - 1;
	if (i >= 0 && c->matches[i].offset_sym == MAX_OFFSET_ALPHABET_SIZE) {
		/* Terminating literal run length, with no following match */
		litrunlen_statex = encode_symbol(c->matches[i].litrunlen_sym,
						 litrunlen_statex,
						 &os,
						 c->codes.litrunlen_sym_encinfo,
						 c->codes.litrunlen_next_statesx);
		symbol_ostream_flush_bits(&os);
		i--;
	}

	for (; i >= 0; i--) {

		const struct match *match = &c->matches[i];

		if (match->offset_sym >= NUM_REPS) {

			unsigned offset_log2 = match->offset_sym - NUM_REPS;

			if (is_aligned_block && offset_log2 >= NUM_ALIGNED_BITS) {
				symbol_ostream_add_bits(&os,
							match->extra_offset_bits >> NUM_ALIGNED_BITS,
							offset_log2 - NUM_ALIGNED_BITS);
				aligned_statex = encode_symbol(match->extra_offset_bits & (ALIGNED_ALPHABET_SIZE - 1),
							       aligned_statex,
							       &os,
							       c->codes.aligned_sym_encinfo,
							       c->codes.aligned_next_statesx);
			} else {
				symbol_ostream_add_bits(&os, match->extra_offset_bits, offset_log2);
			}
			symbol_ostream_flush_bits(&os);
		}

		offset_statex = encode_symbol(match->offset_sym,
					      offset_statex,
					      &os,
					      c->codes.offset_sym_encinfo,
					      c->codes.offset_next_statesx);
		symbol_ostream_flush_bits(&os);

		length_statex = encode_symbol(match->length_sym,
					      length_statex,
					      &os,
					      c->codes.length_sym_encinfo,
					      c->codes.length_next_statesx);

		litrunlen_statex = encode_symbol(match->litrunlen_sym,
						 litrunlen_statex,
						 &os,
						 c->codes.litrunlen_sym_encinfo,
						 c->codes.litrunlen_next_statesx);
		symbol_ostream_flush_bits(&os);
	}

	/* Encode the inital states for matches and literal run lengths */

	if (is_aligned_block)
		encode_initial_state(&os, aligned_statex, c->codes.log2_num_aligned_states);
	encode_initial_state(&os, offset_statex, c->codes.log2_num_offset_states);
	encode_initial_state(&os, length_statex, c->codes.log2_num_length_states);
	encode_initial_state(&os, litrunlen_statex, c->codes.log2_num_litrunlen_states);

	/* Encode the literals */

#if NUM_LITERAL_STREAMS == 2
	literal_statex_1 = 1 << c->codes.log2_num_literal_states;
	literal_statex_2 = 1 << c->codes.log2_num_literal_states;

	for (i = c->num_literals - 1; i >= 1; i -= 2) {

		literal_statex_1 = encode_symbol(c->literals[i],
						 literal_statex_1,
						 &os,
						 c->codes.literal_sym_encinfo,
						 c->codes.literal_next_statesx);

		literal_statex_2 = encode_symbol(c->literals[i - 1],
						 literal_statex_2,
						 &os,
						 c->codes.literal_sym_encinfo,
						 c->codes.literal_next_statesx);
		symbol_ostream_flush_bits(&os);
	}

	if (c->num_literals & 1) {
		literal_statex_1 = encode_symbol(c->literals[0],
						 literal_statex_1,
						 &os,
						 c->codes.literal_sym_encinfo,
						 c->codes.literal_next_statesx);
		symbol_ostream_flush_bits(&os);

		/* last state the encoder used is state_1
		 * => first state the encoder will see is state_1
		 * => numbering will be the same
		 * => encoder must output state_2, then state_1 */
		encode_initial_state(&os, literal_statex_2, c->codes.log2_num_literal_states);
		encode_initial_state(&os, literal_statex_1, c->codes.log2_num_literal_states);
	} else {
		/* Reversed numbering */
		encode_initial_state(&os, literal_statex_1, c->codes.log2_num_literal_states);
		encode_initial_state(&os, literal_statex_2, c->codes.log2_num_literal_states);
	}

#else /* NUM_LITERAL_STREAMS == 2 */

	literal_statex = 1 << c->codes.log2_num_literal_states;

	for (i = c->num_literals - 1; i >= 0; i--) {

		literal_statex = encode_symbol(c->literals[i],
					       literal_statex,
					       &os,
					       c->codes.literal_sym_encinfo,
					       c->codes.literal_next_statesx);

		symbol_ostream_flush_bits(&os);
	}

	encode_initial_state(&os, literal_statex, c->codes.log2_num_literal_states);
#endif /* NUM_LITERAL_STREAMS != 2 */

	/* Literal count */
	order = bsr32(c->num_literals + 1);
	symbol_ostream_add_bits(&os, (c->num_literals + 1) -
				((u32)1 << order), order);
	symbol_ostream_add_bits(&os, order, 5);

	nbytes = symbol_ostream_flush(&os);
	if (nbytes == 0)
		return 0;

	/*
	 * We wrote the data at the end of the output space going backwards.
	 * Now move the data to the beginning.
	 */
	memmove(out, os.next, nbytes);

	return nbytes;
}

static void
write_block_size(struct header_ostream *os, u32 block_size)
{
	u32 bits;
	int num_bits;

	if (block_size == DEFAULT_BLOCK_SIZE) {
		bits = 1;
		num_bits = 1;
	} else {
		bits = block_size << 1;
		num_bits = 1 + NUM_BLOCKSIZE_BITS;
	}

	header_ostream_write_bits(os, bits, num_bits);
}

/* Heuristic for using ALIGNED blocks */
static int
choose_block_type(struct xpack_compressor *c)
{
	u32 min_count = -1;
	u32 max_count = 0;
	unsigned sym;

	for (sym = 0; sym < ALIGNED_ALPHABET_SIZE; sym++) {
		min_count = MIN(min_count, c->freqs.aligned[sym]);
		max_count = MAX(max_count, c->freqs.aligned[sym]);
	}

	if (min_count * 3 < max_count) /* unbalanced? */
		return BLOCKTYPE_ALIGNED;
	else
		return BLOCKTYPE_VERBATIM;
}

/******************************************************************************/

/*
 * Block splitting algorithm.  The problem is to decide when it is worthwhile to
 * start a new block with new entropy codes.  There is a theoretically optimal
 * solution: recursively consider every possible block split, considering the
 * exact cost of each block, and choose the minimum cost approach.  But this is
 * far too slow.  Instead, as an approximation, we can count symbols and after
 * every N symbols, compare the expected distribution of symbols based on the
 * previous data with the actual distribution.  If they differ "by enough", then
 * start a new block.
 *
 * As an optimization and heuristic, we don't distinguish between every symbol
 * but rather we combine many symbols into a single "observation type".  For
 * literals we only look at the high bits and low bits, and for matches we only
 * look at whether the match is long or not.  The assumption is that for typical
 * "real" data, places that are good block boundaries will tend to be noticable
 * based only on changes in these aggregate frequencies, without looking for
 * subtle differences in individual symbols.  For example, a change from ASCII
 * bytes to non-ASCII bytes, or from few matches (generally less compressible)
 * to many matches (generally more compressible), would be easily noticed based
 * on the aggregates.
 *
 * For determining whether the frequency distributions are "different enough" to
 * start a new block, the simply heuristic of splitting when the sum of absolute
 * differences exceeds a constant seems to be good enough.  We also add a number
 * proportional to the block size so that the algorithm is more likely to end
 * large blocks than small blocks.  This reflects the general expectation that
 * it will become increasingly beneficial to start a new block as the current
 * blocks grows larger.
 *
 * Finally, for an approximation, it is not strictly necessary that the exact
 * symbols being used are considered.  With "near-optimal parsing", for example,
 * the actual symbols that will be used are unknown until after the block
 * boundary is chosen and the block has been optimized.  Since the final choices
 * cannot be used, we can use preliminary "greedy" choices instead.
 */

/* Initialize the block split statistics when starting a new block. */
static void
init_block_split_stats(struct block_split_stats *stats)
{
	int i;

	for (i = 0; i < NUM_OBSERVATION_TYPES; i++) {
		stats->new_observations[i] = 0;
		stats->observations[i] = 0;
	}
	stats->num_new_observations = 0;
	stats->num_observations = 0;
}

/* Literal observation.  Heuristic: use the top 2 bits and low 1 bits of the
 * literal, for 8 possible literal observation types.  */
static forceinline void
observe_literal(struct block_split_stats *stats, u8 lit)
{
	stats->new_observations[((lit >> 5) & 0x6) | (lit & 1)]++;
	stats->num_new_observations++;
}

/* Match observation.  Heuristic: use one observation type for "short match" and
 * one observation type for "long match".  */
static forceinline void
observe_match(struct block_split_stats *stats, unsigned length)
{
	stats->new_observations[NUM_LITERAL_OBSERVATION_TYPES + (length >= 9)]++;
	stats->num_new_observations++;
}

static bool
do_end_block_check(struct block_split_stats *stats, u32 block_size)
{
	int i;

	if (stats->num_observations > 0) {

		/* Note: to avoid slow divisions, we do not divide by
		 * 'num_observations', but rather do all math with the numbers
		 * multiplied by 'num_observations'.  */
		u32 total_delta = 0;
		for (i = 0; i < NUM_OBSERVATION_TYPES; i++) {
			u32 expected = stats->observations[i] * stats->num_new_observations;
			u32 actual = stats->new_observations[i] * stats->num_observations;
			u32 delta = (actual > expected) ? actual - expected :
							  expected - actual;
			total_delta += delta;
		}

		/* Ready to end the block? */
		if (total_delta + (block_size >> 12) * stats->num_observations >=
		    200 * stats->num_observations)
			return true;
	}

	for (i = 0; i < NUM_OBSERVATION_TYPES; i++) {
		stats->num_observations += stats->new_observations[i];
		stats->observations[i] += stats->new_observations[i];
		stats->new_observations[i] = 0;
	}
	stats->num_new_observations = 0;
	return false;
}

static forceinline bool
should_end_block(struct block_split_stats *stats,
		 const u8 *in_block_begin, const u8 *in_next, const u8 *in_end)
{
	/* Ready to check block split statistics? */
	if (stats->num_new_observations < 512 ||
	    in_next - in_block_begin < MIN_BLOCK_LENGTH ||
	    in_end - in_next < 16384)
		return false;

	return do_end_block_check(stats, in_next - in_block_begin);
}

/******************************************************************************/

static void
begin_block(struct xpack_compressor *c)
{
	memset(&c->freqs, 0, sizeof(c->freqs));
	c->num_literals = 0;
	c->num_matches = 0;
	c->num_extra_bytes = 0;
	init_block_split_stats(&c->split_stats);
}

static void
record_literal(struct xpack_compressor *c, u8 literal)
{
	c->literals[c->num_literals++] = literal;
	c->freqs.literal[literal]++;
}

static void
record_litrunlen(struct xpack_compressor *c, struct match *match, u32 litrunlen)
{
	unsigned litrunlen_sym;

	if (litrunlen >= LITRUNLEN_ALPHABET_SIZE - 1) {
		u32 v = litrunlen - (LITRUNLEN_ALPHABET_SIZE - 1);
		if (v < 0xFF) {
			c->extra_bytes[c->num_extra_bytes++] = v;
		} else {
			v -= 0xFF;
			c->extra_bytes[c->num_extra_bytes++] = 0xFF;
			c->extra_bytes[c->num_extra_bytes++] = (u8)(v >> 0);
			c->extra_bytes[c->num_extra_bytes++] = (u8)(v >> 8);
			c->extra_bytes[c->num_extra_bytes++] = (u8)(v >> 16);
		}
		litrunlen_sym = LITRUNLEN_ALPHABET_SIZE - 1;
	} else {
		litrunlen_sym = litrunlen;
	}

	match->litrunlen_sym = litrunlen_sym;
	c->freqs.litrunlen[litrunlen_sym]++;
}

static void
record_length(struct xpack_compressor *c, struct match *match, u32 length)
{
	unsigned length_sym;

	length -= MIN_MATCH_LEN;

	if (length >= LENGTH_ALPHABET_SIZE - 1) {
		u32 v = length - (LENGTH_ALPHABET_SIZE - 1);
		if (v < 0xFF) {
			c->extra_bytes[c->num_extra_bytes++] = v;
		} else {
			v -= 0xFF;
			c->extra_bytes[c->num_extra_bytes++] = 0xFF;
			c->extra_bytes[c->num_extra_bytes++] = (u8)(v >> 0);
			c->extra_bytes[c->num_extra_bytes++] = (u8)(v >> 8);
			c->extra_bytes[c->num_extra_bytes++] = (u8)(v >> 16);
		}
		length_sym = LENGTH_ALPHABET_SIZE - 1;
	} else {
		length_sym = length;
	}

	match->length_sym = length_sym;
	c->freqs.length[length_sym]++;
}

static void
record_explicit_offset(struct xpack_compressor *c, struct match *match,
		       u32 offset)
{
	unsigned offset_log2 = bsr32(offset);
	unsigned offset_sym = NUM_REPS + offset_log2;

	match->offset_sym = offset_sym;
	c->freqs.offset[offset_sym]++;
	match->extra_offset_bits = offset - ((u32)1 << offset_log2);
	if (offset_log2 >= NUM_ALIGNED_BITS)
		c->freqs.aligned[offset & (ALIGNED_ALPHABET_SIZE - 1)]++;
}

static void
record_repeat_offset(struct xpack_compressor *c, struct match *match,
		     unsigned rep_idx)
{
	match->offset_sym = rep_idx;
	c->freqs.offset[rep_idx]++;
}

static size_t
write_block(struct xpack_compressor *c, void *out, size_t out_nbytes_avail,
	    u32 block_size, u32 last_litrunlen, bool is_final_block)
{
	struct header_ostream os;
	size_t header_size;
	size_t items_size;
	int block_type;
	unsigned num_state_counts;
	unsigned order;

	/* Final litrunlen */
	record_litrunlen(c, &c->matches[c->num_matches], last_litrunlen);
	c->matches[c->num_matches].offset_sym = MAX_OFFSET_ALPHABET_SIZE;
	c->num_matches++;

	/* Choose the block type */
	block_type = choose_block_type(c);

	header_ostream_init(&os, out, out_nbytes_avail);

	/* Output the "final block" flag */
	header_ostream_write_bits(&os, is_final_block, 1);

	/* Output the block type */
	header_ostream_write_bits(&os, block_type, NUM_BLOCKTYPE_BITS);

	/* Output the block size */
	write_block_size(&os, block_size);

	/* Compute FSE state counts for each alphabet */

	c->codes.log2_num_literal_states =
		choose_state_counts(c->freqs.literal,
				    LITERAL_ALPHABET_SIZE,
				    MAX_LOG2_NUM_LITERAL_STATES,
				    c->codes.literal_state_counts);

	c->codes.log2_num_litrunlen_states =
		choose_state_counts(c->freqs.litrunlen,
				    LITRUNLEN_ALPHABET_SIZE,
				    MAX_LOG2_NUM_LITRUNLEN_STATES,
				    c->codes.litrunlen_state_counts);

	c->codes.log2_num_length_states =
		choose_state_counts(c->freqs.length,
				    LENGTH_ALPHABET_SIZE,
				    MAX_LOG2_NUM_LENGTH_STATES,
				    c->codes.length_state_counts);

	c->codes.log2_num_offset_states =
		choose_state_counts(c->freqs.offset,
				    MAX_OFFSET_ALPHABET_SIZE,
				    MAX_LOG2_NUM_OFFSET_STATES,
				    c->codes.offset_state_counts);

	if (block_type == BLOCKTYPE_ALIGNED) {
		c->codes.log2_num_aligned_states =
			choose_state_counts(c->freqs.aligned,
					    ALIGNED_ALPHABET_SIZE,
					    MAX_LOG2_NUM_ALIGNED_STATES,
					    c->codes.aligned_state_counts);
	}

	/* Output the FSE state counts for each alphabet */
	header_ostream_write_bits(&os, c->codes.log2_num_literal_states, 4);
	header_ostream_write_bits(&os, c->codes.log2_num_litrunlen_states, 4);
	header_ostream_write_bits(&os, c->codes.log2_num_length_states, 4);
	header_ostream_write_bits(&os, c->codes.log2_num_offset_states, 4);
	if (block_type == BLOCKTYPE_ALIGNED)
		header_ostream_write_bits(&os, c->codes.log2_num_aligned_states, 4);

#ifndef _MSC_VER
	STATIC_ASSERT(offsetof(struct codes,
			       aligned_state_counts[ALIGNED_ALPHABET_SIZE]) ==
		      offsetof(struct codes, state_counts) + sizeof(c->codes.state_counts));
#endif
	num_state_counts = ARRAY_LEN(c->codes.state_counts);
	if (block_type != BLOCKTYPE_ALIGNED)
		num_state_counts -= ALIGNED_ALPHABET_SIZE;

	write_state_counts(&os, c->codes.state_counts, num_state_counts);

	/* Output the number of extra bytes */
	order = bsr32(c->num_extra_bytes + 1);
	header_ostream_write_bits(&os, order, 5);
	header_ostream_write_bits(&os,
				  (c->num_extra_bytes + 1) - ((u32)1 << order),
				  order);

	/* Align to the next byte boundary */
	header_size = header_ostream_flush(&os);
	if (header_size == 0)
		return 0;

	/* Add the extra bytes */
	if (c->num_extra_bytes >= out_nbytes_avail - header_size)
		return 0;
	memcpy((u8 *)out + header_size, c->extra_bytes, c->num_extra_bytes);
	header_size += c->num_extra_bytes;

	/* Build the FSE encoding tables for each alphabet */

	build_fse_encoding_tables(c, c->codes.literal_sym_encinfo,
				  c->codes.literal_next_statesx,
				  c->codes.literal_state_counts,
				  LITERAL_ALPHABET_SIZE,
				  c->codes.log2_num_literal_states);

	build_fse_encoding_tables(c, c->codes.litrunlen_sym_encinfo,
				  c->codes.litrunlen_next_statesx,
				  c->codes.litrunlen_state_counts,
				  LITRUNLEN_ALPHABET_SIZE,
				  c->codes.log2_num_litrunlen_states);

	build_fse_encoding_tables(c, c->codes.length_sym_encinfo,
				  c->codes.length_next_statesx,
				  c->codes.length_state_counts,
				  LENGTH_ALPHABET_SIZE,
				  c->codes.log2_num_length_states);

	build_fse_encoding_tables(c, c->codes.offset_sym_encinfo,
				  c->codes.offset_next_statesx,
				  c->codes.offset_state_counts,
				  MAX_OFFSET_ALPHABET_SIZE,
				  c->codes.log2_num_offset_states);

	if (block_type == BLOCKTYPE_ALIGNED) {
		build_fse_encoding_tables(c, c->codes.aligned_sym_encinfo,
					  c->codes.aligned_next_statesx,
					  c->codes.aligned_state_counts,
					  ALIGNED_ALPHABET_SIZE,
					  c->codes.log2_num_aligned_states);
	}

	/* Encode the items */

	items_size = encode_items(c, (u8 *)out + header_size,
				  out_nbytes_avail - header_size,
				  block_type == BLOCKTYPE_ALIGNED);
	if (items_size == 0)
		return 0;

	return header_size + items_size;
}

static size_t
compress_greedy(struct xpack_compressor *c, void *out, size_t out_nbytes_avail)
{
	u8 * const out_begin = out;
	u8 * out_next = out_begin;
	u8 * const out_end = out_begin + out_nbytes_avail;
	const u8 * const in_begin = c->in_buffer;
	const u8 *	 in_next = in_begin;
	const u8 * const in_end  = in_begin + c->in_nbytes;
	u32 max_len = MIN(c->in_nbytes, UINT32_MAX);
	u32 nice_len = MIN(c->nice_match_length, max_len);
	u32 next_hashes[2] = {0, 0};
	u32 recent_offsets[NUM_REPS];

	init_recent_offsets(recent_offsets);
	hc_matchfinder_init(&c->hc_mf);

	do {
		/* Starting a new block */

		const u8 * const in_block_begin = in_next;
		const u8 * const in_max_block_end =
			in_next + MIN(SOFT_MAX_BLOCK_LENGTH, in_end - in_next);
		u32 length;
		u32 offset;
		size_t nbytes;
		u32 litrunlen = 0;

		begin_block(c);

		do {
			if (unlikely(max_len > in_end - in_next)) {
				max_len = in_end - in_next;
				nice_len = MIN(max_len, nice_len);
			}

			/* Find the longest match at the current position. */

			length = hc_matchfinder_longest_match(&c->hc_mf,
							      in_begin,
							      in_next - in_begin,
							#if MIN_MATCH_LEN == 4
							      3,
							#else
							      2,
							#endif
							      max_len,
							      nice_len,
							      c->max_search_depth,
							      next_hashes,
							      &offset);
		#if MIN_MATCH_LEN == 4
			if (length < 4) {
		#else
			if (length < 3 || (length == 3 && offset >= 4096)) {
		#endif
				/* Literal */
				observe_literal(&c->split_stats, *in_next);
				record_literal(c, *in_next);
				in_next++;
				litrunlen++;
			} else {
				/* Match */
				struct match *match = &c->matches[c->num_matches++];

				STATIC_ASSERT(NUM_REPS >= 1 && NUM_REPS <= 4);

				observe_match(&c->split_stats, length);

				if (offset == recent_offsets[0]) {
					record_repeat_offset(c, match, 0);
				}
			#if NUM_REPS >= 2
				else if (offset == recent_offsets[1]) {
					recent_offsets[1] = recent_offsets[0];
					record_repeat_offset(c, match, 1);
				}
			#endif
			#if NUM_REPS >= 3
				else if (offset == recent_offsets[2]) {
					recent_offsets[2] = recent_offsets[0];
					record_repeat_offset(c, match, 2);
				}
			#endif
			#if NUM_REPS >= 4
				else if (offset == recent_offsets[3]) {
					recent_offsets[3] = recent_offsets[0];
					record_repeat_offset(c, match, 3);
				}
			#endif
				else {
					record_explicit_offset(c, match, offset);
				#if NUM_REPS >= 4
					recent_offsets[3] = recent_offsets[2];
				#endif
				#if NUM_REPS >= 3
					recent_offsets[2] = recent_offsets[1];
				#endif
				#if NUM_REPS >= 2
					recent_offsets[1] = recent_offsets[0];
				#endif
				}
				recent_offsets[0] = offset;
				record_litrunlen(c, match, litrunlen);
				record_length(c, match, length);

				in_next = hc_matchfinder_skip_positions(&c->hc_mf,
									in_begin,
									in_next + 1 - in_begin,
									in_end - in_begin,
									length - 1,
									next_hashes);
				litrunlen = 0;
			}
		} while (in_next < in_max_block_end &&
			 !should_end_block(&c->split_stats, in_block_begin, in_next, in_end));

		nbytes = write_block(c, out_next, out_end - out_next,
				     in_next - in_block_begin, litrunlen,
				     in_next == in_end);
		if (nbytes == 0)
			return 0;

		out_next += nbytes;

	} while (in_next != in_end);

	return out_next - out_begin;
}

/*
 * Given a pointer to the current byte sequence and the current list of recent
 * match offsets, find the longest repeat offset match.
 *
 * If no match of at least MIN_MATCH_LEN bytes is found, then return 0.
 *
 * If a match of at least MIN_MATCH_LEN bytes is found, then return its length
 * and set *rep_max_idx_ret to the index of its offset in @queue.
 */
static u32
find_longest_repeat_offset_match(const u8 * const in_next,
				 const u32 max_len,
				 const u32 recent_offsets[],
				 unsigned *rep_max_idx_ret)
{
#if MIN_MATCH_LEN == 2
#  define load_initial  load_u16_unaligned
#elif MIN_MATCH_LEN == 3
#  define load_initial	load_u24_unaligned
#elif MIN_MATCH_LEN == 4
#  define load_initial	load_u32_unaligned
#else
#  error "unsupported MIN_MATCH_LEN"
#endif
	const u32 next_bytes = load_initial(in_next);
	const u8 *matchptr;
	u32 rep_len;
	u32 rep_max_len;
	unsigned rep_max_idx;

	STATIC_ASSERT(NUM_REPS >= 1 && NUM_REPS <= 4);

	matchptr = in_next - recent_offsets[0];
	if (load_initial(matchptr) == next_bytes)
		rep_max_len = lz_extend(in_next, matchptr, MIN_MATCH_LEN, max_len);
	else
		rep_max_len = 0;
	rep_max_idx = 0;

#if NUM_REPS >= 2
	matchptr = in_next - recent_offsets[1];
	if (load_initial(matchptr) == next_bytes) {
		rep_len = lz_extend(in_next, matchptr, MIN_MATCH_LEN, max_len);
		if (rep_len > rep_max_len) {
			rep_max_len = rep_len;
			rep_max_idx = 1;
		}
	}
#endif

#if NUM_REPS >= 3
	matchptr = in_next - recent_offsets[2];
	if (load_initial(matchptr) == next_bytes) {
		rep_len = lz_extend(in_next, matchptr, MIN_MATCH_LEN, max_len);
		if (rep_len > rep_max_len) {
			rep_max_len = rep_len;
			rep_max_idx = 2;
		}
	}
#endif

#if NUM_REPS >= 4
	matchptr = in_next - recent_offsets[3];
	if (load_initial(matchptr) == next_bytes) {
		rep_len = lz_extend(in_next, matchptr, MIN_MATCH_LEN, max_len);
		if (rep_len > rep_max_len) {
			rep_max_len = rep_len;
			rep_max_idx = 3;
		}
	}
#endif

	*rep_max_idx_ret = rep_max_idx;
	return rep_max_len;
}

/* Fast heuristic scoring for lazy parsing: how "good" is this match? */
static forceinline u32
explicit_offset_match_score(u32 len, u32 adjusted_offset)
{
	u32 score = len;

	if (adjusted_offset < 4096)
		score++;

	if (adjusted_offset < 256)
		score++;

	return score;
}

static forceinline u32
repeat_offset_match_score(u32 rep_len, unsigned rep_idx)
{
	return rep_len + 3;
}

static size_t
compress_lazy(struct xpack_compressor *c, void *out, size_t out_nbytes_avail)
{
	u8 * const out_begin = out;
	u8 * out_next = out_begin;
	u8 * const out_end = out_begin + out_nbytes_avail;
	const u8 * const in_begin = c->in_buffer;
	const u8 *	 in_next = in_begin;
	const u8 * const in_end  = in_begin + c->in_nbytes;
	u32 max_len = MIN(c->in_nbytes, UINT32_MAX);
	u32 nice_len = MIN(c->nice_match_length, max_len);
	u32 next_hashes[2] = {0, 0};
	u32 recent_offsets[NUM_REPS];

	init_recent_offsets(recent_offsets);
	hc_matchfinder_init(&c->hc_mf);

	do {
		/* Starting a new block */

		const u8 * const in_block_begin = in_next;
		const u8 * const in_max_block_end =
			in_next + MIN(SOFT_MAX_BLOCK_LENGTH, in_end - in_next);
		u32 cur_len;
		u32 cur_offset;
		u32 cur_offset_data;
		u32 cur_score;
		u32 next_len;
		u32 next_offset;
		u32 next_offset_data;
		u32 next_score;
		u32 rep_max_len;
		unsigned rep_max_idx;
		u32 rep_score;
		u32 skip_len;
		u32 litrunlen = 0;
		size_t nbytes;
		struct match *match;

		begin_block(c);

		do {
			if (unlikely(max_len > in_end - in_next)) {
				max_len = in_end - in_next;
				nice_len = MIN(max_len, nice_len);
			}

			/* Find the longest match at the current position. */

			cur_len = hc_matchfinder_longest_match(&c->hc_mf,
							       in_begin,
							       in_next - in_begin,
							#if MIN_MATCH_LEN == 4
							       3,
							#else
							       2,
							#endif
							       max_len,
							       nice_len,
							       c->max_search_depth,
							       next_hashes,
							       &cur_offset);
		#if MIN_MATCH_LEN == 4
			if (cur_len < 4) {
		#else
			if (cur_len < 3 || (cur_len == 3 && cur_offset >= 4096)) {
		#endif
				/*
				 * There was no match found, or the only match
				 * found was a distant length 3 match.  Output a
				 * literal.
				 */
				observe_literal(&c->split_stats, *in_next);
				record_literal(c, *in_next);
				in_next++;
				litrunlen++;
				continue;
			}

			observe_match(&c->split_stats, cur_len);

			if (cur_offset == recent_offsets[0]) {
				in_next++;
				cur_offset_data = 0;
				skip_len = cur_len - 1;
				goto choose_cur_match;
			}

			cur_offset_data = cur_offset + (NUM_REPS - 1);
			cur_score = explicit_offset_match_score(cur_len, cur_offset_data);

			/* Consider a repeat offset match. */
			rep_max_len = find_longest_repeat_offset_match(in_next,
								       in_end - in_next,
								       recent_offsets,
								       &rep_max_idx);
			in_next++;

			if (rep_max_len >= 3 &&
			    (rep_score = repeat_offset_match_score(rep_max_len,
								   rep_max_idx)) >= cur_score)
			{
				cur_len = rep_max_len;
				cur_offset_data = rep_max_idx;
				skip_len = rep_max_len - 1;
				goto choose_cur_match;
			}

		have_cur_match:

			/* We have a match at the current position. */

			/* If we have a very long match, choose it immediately. */
			if (cur_len >= nice_len) {
				skip_len = cur_len - 1;
				goto choose_cur_match;
			}

			/* See if there's a better match at the next position. */

			if (unlikely(max_len > in_end - in_next)) {
				max_len = in_end - in_next;
				nice_len = MIN(max_len, nice_len);
			}

			next_len = hc_matchfinder_longest_match(&c->hc_mf,
								in_begin,
								in_next - in_begin,
							#if MIN_MATCH_LEN == 2
								cur_len - 2,
							#else
								cur_len - 1,
							#endif
								max_len,
								nice_len,
								c->max_search_depth / 2,
								next_hashes,
								&next_offset);

		#if MIN_MATCH_LEN == 2
			if (next_len <= cur_len - 2) {
		#else
			if (next_len <= cur_len - 1) {
		#endif
				in_next++;
				skip_len = cur_len - 2;
				goto choose_cur_match;
			}

			next_offset_data = next_offset + (NUM_REPS - 1);
			next_score = explicit_offset_match_score(next_len, next_offset_data);

			rep_max_len = find_longest_repeat_offset_match(in_next,
								       in_end - in_next,
								       recent_offsets,
								       &rep_max_idx);
			in_next++;

			if (rep_max_len >= 3 &&
			    (rep_score = repeat_offset_match_score(rep_max_len,
								   rep_max_idx)) >= next_score)
			{

				if (rep_score > cur_score) {
					/*
					 * The next match is better, and it's a
					 * repeat offset match.
					 */
					record_literal(c, *(in_next - 2));
					litrunlen++;
					cur_len = rep_max_len;
					cur_offset_data = rep_max_idx;
					skip_len = cur_len - 1;
					goto choose_cur_match;
				}
			} else {
				if (next_score > cur_score) {
					/*
					 * The next match is better, and it's an
					 * explicit offset match.
					 */
					record_literal(c, *(in_next - 2));
					litrunlen++;
					cur_len = next_len;
					cur_offset_data = next_offset_data;
					cur_score = next_score;
					goto have_cur_match;
				}
			}

			/* The original match was better. */
			skip_len = cur_len - 2;

		choose_cur_match:
			match = &c->matches[c->num_matches++];
			if (cur_offset_data < NUM_REPS) {
				u32 offset;

				record_repeat_offset(c, match, cur_offset_data);

				offset = recent_offsets[cur_offset_data];
				recent_offsets[cur_offset_data] = recent_offsets[0];
				recent_offsets[0] = offset;
			} else {
				record_explicit_offset(c, match,
						       cur_offset_data - (NUM_REPS - 1));
				STATIC_ASSERT(NUM_REPS >= 1 && NUM_REPS <= 4);
			#if NUM_REPS >= 4
				recent_offsets[3] = recent_offsets[2];
			#endif
			#if NUM_REPS >= 3
				recent_offsets[2] = recent_offsets[1];
			#endif
			#if NUM_REPS >= 2
				recent_offsets[1] = recent_offsets[0];
			#endif
				recent_offsets[0] = cur_offset_data - (NUM_REPS - 1);
			}
			record_litrunlen(c, match, litrunlen);
			record_length(c, match, cur_len);
			litrunlen = 0;

			in_next = hc_matchfinder_skip_positions(&c->hc_mf,
								in_begin,
								in_next - in_begin,
								in_end - in_begin,
								skip_len,
								next_hashes);
		} while (in_next < in_max_block_end &&
			 !should_end_block(&c->split_stats, in_block_begin, in_next, in_end));

		nbytes = write_block(c, out_next, out_end - out_next,
				     in_next - in_block_begin, litrunlen,
				     in_next == in_end);
		if (nbytes == 0)
			return 0;

		out_next += nbytes;

	} while (in_next != in_end);

	return out_next - out_begin;
}

LIBEXPORT struct xpack_compressor *
xpack_alloc_compressor(size_t max_buffer_size, int compression_level)
{
	struct xpack_compressor *c;

	c = malloc(offsetof(struct xpack_compressor, hc_mf) +
		   hc_matchfinder_size(max_buffer_size));
	if (!c)
		goto err0;

#ifdef ENABLE_PREPROCESSING
	c->in_buffer = malloc(max_buffer_size);
	if (!c->in_buffer)
		goto err1;
#endif

	c->max_buffer_size = max_buffer_size;

	switch (compression_level) {
	case 1:
		c->impl = compress_greedy;
		c->max_search_depth = 1;
		c->nice_match_length = MIN_MATCH_LEN;
		break;
	case 2:
		c->impl = compress_greedy;
		c->max_search_depth = 8;
		c->nice_match_length = 8;
		break;
	case 3:
		c->impl = compress_greedy;
		c->max_search_depth = 16;
		c->nice_match_length = 16;
		break;
	case 4:
		c->impl = compress_lazy;
		c->max_search_depth = 8;
		c->nice_match_length = 12;
		break;
	case 5:
		c->impl = compress_lazy;
		c->max_search_depth = 16;
		c->nice_match_length = 24;
		break;
	case 6:
		c->impl = compress_lazy;
		c->max_search_depth = 32;
		c->nice_match_length = 48;
		break;
	case 7:
		c->impl = compress_lazy;
		c->max_search_depth = 64;
		c->nice_match_length = 96;
		break;
	case 8:
		c->impl = compress_lazy;
		c->max_search_depth = 128;
		c->nice_match_length = 192;
		break;
	case 9:
		c->impl = compress_lazy;
		c->max_search_depth = 256;
		c->nice_match_length = 384;
		STATIC_ASSERT(EXTRA_LITERAL_SPACE >= 384 * 4 / 3);
		break;
	default:
		goto err2;
	}

	/* max_search_depth == 0 is invalid */
	if (c->max_search_depth < 1)
		c->max_search_depth = 1;

	return c;

err2:
#ifdef ENABLE_PREPROCESSING
	free(c->in_buffer);
err1:
#endif
	free(c);
err0:
	return NULL;
}

LIBEXPORT size_t
xpack_compress(struct xpack_compressor *c, const void *in, size_t in_nbytes,
	       void *out, size_t out_nbytes_avail)
{
	/* Don't bother trying to compress very small inputs. */
	if (in_nbytes < 100)
		return 0;

	/* Safety check */
	if (unlikely(in_nbytes > c->max_buffer_size))
		return 0;

#ifdef ENABLE_PREPROCESSING
	/* Copy the input data into the internal buffer and preprocess it. */
	memcpy(c->in_buffer, in, in_nbytes);
	c->in_nbytes = in_nbytes;
	preprocess(c->in_buffer, in_nbytes);
#else
	/* Preprocessing is disabled.  No internal buffer is needed. */
	c->in_buffer = (void *)in;
	c->in_nbytes = in_nbytes;
#endif

	return (*c->impl)(c, out, out_nbytes_avail);
}

LIBEXPORT void
xpack_free_compressor(struct xpack_compressor *c)
{
	if (c) {
	#ifdef ENABLE_PREPROCESSING
		free(c->in_buffer);
	#endif
		free(c);
	}
}

#endif /* !DECOMPRESSION_ONLY */
