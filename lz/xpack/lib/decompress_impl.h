/*
 * decompress_impl.h - XPACK decompression implementation
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

/*
 * This is the actual decompression routine, lifted out of xpack_decompress.c so
 * that it can be compiled with different target instruction sets.
 */

static enum decompress_result ATTRIBUTES
FUNCNAME(struct xpack_decompressor * restrict d,
	 const void * restrict in, size_t in_nbytes,
	 void * restrict out, size_t out_nbytes_avail,
	 size_t *actual_out_nbytes_ret)
{
	const u8 *in_next = in;
	const u8 * const in_end = in_next + in_nbytes;
	u8 *out_next = out;
	u8 * const out_end = out_next + out_nbytes_avail;
	u8 *out_block_end;
	u32 recent_offsets[NUM_REPS];
#ifdef ENABLE_PREPROCESSING
	unsigned preprocessed = 0;
#endif
	machine_word_t bitbuf = 0;
	unsigned bitsleft = 0;
	size_t overrun_count = 0;
	unsigned num_padding_bits;
	unsigned num_state_counts;
	unsigned is_final_block;
	unsigned block_type;
	size_t block_usize;
	s32 num_literals;
	u8 *literals;
	u8 *lits;
	u8 *lits_end;
	unsigned order;
	const u8 *extra_bytes;
	u32 num_extra_bytes;
	unsigned log2_num_literal_states;
	unsigned log2_num_litrunlen_states;
	unsigned log2_num_length_states;
	unsigned log2_num_offset_states;
	unsigned log2_num_aligned_states;
#if NUM_LITERAL_STREAMS == 2
	unsigned literal_state_1;
	unsigned literal_state_2;
#else
	unsigned literal_state;
#endif
	unsigned litrunlen_state;
	unsigned length_state;
	unsigned offset_state;
	unsigned aligned_state;
	unsigned i;
	u32 sym;
	u32 bits;

	init_recent_offsets(recent_offsets);

next_block:
	/* Starting to decompress the next block */

	ENSURE_BITS(1 + NUM_BLOCKTYPE_BITS + 1 + NUM_BLOCKSIZE_BITS);

	/* "final block" flag */
	is_final_block = POP_BITS(1);

	/* block type */
	block_type = POP_BITS(NUM_BLOCKTYPE_BITS);

	/* block uncompressed size */
	if (POP_BITS(1))
		block_usize = DEFAULT_BLOCK_SIZE;
	else
		block_usize = POP_BITS(NUM_BLOCKSIZE_BITS);

	SAFETY_CHECK(block_type == BLOCKTYPE_ALIGNED ||
		     block_type == BLOCKTYPE_VERBATIM);

	if (unlikely(block_usize > out_end - out_next))
		return DECOMPRESS_INSUFFICIENT_SPACE;

	SAFETY_CHECK(block_usize > 0);

	out_block_end = out_next + block_usize;

	/* Read the FSE state counts for each alphabet. */
	ENSURE_BITS(20);
	log2_num_literal_states = POP_BITS(4);
	log2_num_litrunlen_states = POP_BITS(4);
	log2_num_length_states = POP_BITS(4);
	log2_num_offset_states = POP_BITS(4);
	if (block_type == BLOCKTYPE_ALIGNED)
		log2_num_aligned_states = POP_BITS(4);
	else
		log2_num_aligned_states = 0;

	SAFETY_CHECK(log2_num_literal_states <= MAX_LOG2_NUM_LITERAL_STATES &&
		     log2_num_litrunlen_states <= MAX_LOG2_NUM_LITRUNLEN_STATES &&
		     log2_num_length_states <= MAX_LOG2_NUM_LENGTH_STATES &&
		     log2_num_offset_states <= MAX_LOG2_NUM_OFFSET_STATES &&
		     log2_num_aligned_states <= MAX_LOG2_NUM_ALIGNED_STATES);

#ifndef _MSC_VER
	STATIC_ASSERT(offsetof(struct xpack_decompressor,
			aligned_state_counts[ALIGNED_ALPHABET_SIZE]) ==
		      offsetof(struct xpack_decompressor, state_counts) +
			sizeof(d->state_counts));
#endif

	num_state_counts = ARRAY_LEN(d->state_counts);
	if (block_type != BLOCKTYPE_ALIGNED)
		num_state_counts -= ALIGNED_ALPHABET_SIZE;

	for (i = 0; i < num_state_counts; ) {
		unsigned code;

		ENSURE_BITS(CODEBITS + MAX_EXTRA_CODEBITS);

		code = POP_BITS(CODEBITS);

		if (code < ZEROCODE1) {
			/* single nonzero count */
			d->state_counts[i++] = (1 << code) + POP_BITS(code);
		} else {
			unsigned num_zeroes;

			if (code == ZEROCODE1) {
				/* a few zeroes */
				num_zeroes = ZEROCODE1_MIN +
					     POP_BITS(ZEROCODE1_NBITS);
			} else {
				/* many zeroes */
				num_zeroes = ZEROCODE2_MIN +
					     POP_BITS(ZEROCODE2_NBITS);
			}
			SAFETY_CHECK(num_zeroes <= num_state_counts - i);
			do {
				d->state_counts[i++] = 0;
			} while (--num_zeroes);
		}
	}

#ifdef ENABLE_PREPROCESSING
	preprocessed |= d->literal_state_counts[0xE8];
#endif

	/* Prepare the extra_bytes pointer. */

	ENSURE_BITS(5);
	order = POP_BITS(5);
	STATIC_ASSERT(CAN_ENSURE(25));
	SAFETY_CHECK(order <= 25);
	ENSURE_BITS(order);
	num_extra_bytes = ((u32)1 << order) + POP_BITS(order) - 1;
	ALIGN_INPUT();
	SAFETY_CHECK(num_extra_bytes < in_end - in_next);
	extra_bytes = in_next;
	in_next += num_extra_bytes;

	/* Set up the FSE symbol input stream. */
	SAFETY_CHECK(*in_next != 0);
	num_padding_bits = 1 + bsf32(*in_next);
	bitbuf = *in_next++ >> num_padding_bits;
	bitsleft = 8 - num_padding_bits;

	/* Decode the literals. */

	ENSURE_BITS(5);
	order = POP_BITS(5);
	STATIC_ASSERT(CAN_ENSURE(25));
	SAFETY_CHECK(order <= 25);
	ENSURE_BITS(order);
	num_literals = ((u32)1 << order) + POP_BITS(order) - 1;
	SAFETY_CHECK(num_literals <= out_block_end - out_next);
	literals = out_block_end - num_literals;

	SAFETY_CHECK(build_fse_decode_table(d->literal_decode_table,
					    d->literal_state_counts,
					    LITERAL_ALPHABET_SIZE,
					    log2_num_literal_states));

#if NUM_LITERAL_STREAMS == 2
	ENSURE_BITS(2 * MAX_LOG2_NUM_LITERAL_STATES);
	literal_state_1 = POP_BITS(log2_num_literal_states);
	literal_state_2 = POP_BITS(log2_num_literal_states);
	lits = literals;
	lits_end = literals + (num_literals & ~1);
	while (lits != lits_end) {
		ENSURE_BITS(2 * MAX_LOG2_NUM_LITERAL_STATES);
		*lits++ = DECODE_SYMBOL(literal_state_1, d->literal_decode_table);
		*lits++ = DECODE_SYMBOL(literal_state_2, d->literal_decode_table);
	}
	if (lits_end != out_block_end) {
		ENSURE_BITS(MAX_LOG2_NUM_LITERAL_STATES);
		*lits++ = DECODE_SYMBOL(literal_state_1, d->literal_decode_table);
	}
	SAFETY_CHECK(literal_state_1 == 0 && literal_state_2 == 0);
#else
	ENSURE_BITS(MAX_LOG2_NUM_LITERAL_STATES);
	literal_state = POP_BITS(log2_num_literal_states);
	lits = literals;
	lits_end = literals + num_literals;
	while (lits != lits_end) {
		ENSURE_BITS(MAX_LOG2_NUM_LITERAL_STATES);
		*lits++ = DECODE_SYMBOL(literal_state, d->literal_decode_table);
	}
	SAFETY_CHECK(literal_state == 0);
#endif

	/* Prepare to decode literal runs and matches */

	ENSURE_BITS(MAX_LOG2_NUM_LITRUNLEN_STATES + MAX_LOG2_NUM_LENGTH_STATES);
	litrunlen_state = POP_BITS(log2_num_litrunlen_states);
	length_state = POP_BITS(log2_num_length_states);

	ENSURE_BITS(MAX_LOG2_NUM_OFFSET_STATES + MAX_LOG2_NUM_ALIGNED_STATES);
	offset_state = POP_BITS(log2_num_offset_states);
	aligned_state = 0;
	if (block_type == BLOCKTYPE_ALIGNED)
		aligned_state = POP_BITS(log2_num_aligned_states);

	SAFETY_CHECK(build_fse_decode_table(d->litrunlen_decode_table,
					    d->litrunlen_state_counts,
					    LITRUNLEN_ALPHABET_SIZE,
					    log2_num_litrunlen_states));

	SAFETY_CHECK(build_fse_decode_table(d->length_decode_table,
					    d->length_state_counts,
					    LENGTH_ALPHABET_SIZE,
					    log2_num_length_states));

	SAFETY_CHECK(build_fse_decode_table(d->offset_decode_table,
					    d->offset_state_counts,
					    MAX_OFFSET_ALPHABET_SIZE,
					    log2_num_offset_states));

	if (block_type == BLOCKTYPE_ALIGNED) {
		SAFETY_CHECK(build_fse_decode_table(d->aligned_decode_table,
						    d->aligned_state_counts,
						    ALIGNED_ALPHABET_SIZE,
						    log2_num_aligned_states));
	}

	/* Decode literal runs and matches */

	for (;;) {
		u32 litrunlen;
		u32 length;
		u32 offset;
		unsigned offset_sym;

		STATIC_ASSERT(MAX_LOG2_NUM_LITRUNLEN_STATES +
			      MAX_LOG2_NUM_LENGTH_STATES +
			      MAX_LOG2_NUM_OFFSET_STATES <= 32);
		if (CAN_ENSURE(32))
			ENSURE_BITS(32);
		else
			ENSURE_BITS(MAX_LOG2_NUM_LITRUNLEN_STATES +
				    MAX_LOG2_NUM_LENGTH_STATES);

		/* BEGIN decode literal run */

		/* Decode the literal run length and copy the literals. */
		litrunlen = DECODE_SYMBOL(litrunlen_state,
					  d->litrunlen_decode_table);

	#if 0	/* Unoptimized version */
		if (litrunlen == LITRUNLEN_ALPHABET_SIZE - 1) {
			SAFETY_CHECK(extra_bytes < in_end);
			litrunlen += *extra_bytes++;
			if (litrunlen == 0xFF + LITRUNLEN_ALPHABET_SIZE - 1) {
				SAFETY_CHECK(in_end - extra_bytes >= 3);
				litrunlen += (u32)*extra_bytes++ << 0;
				litrunlen += (u32)*extra_bytes++ << 8;
				litrunlen += (u32)*extra_bytes++ << 16;
			}
		}
		num_literals -= litrunlen;
		SAFETY_CHECK(num_literals >= 0);
		SAFETY_CHECK(out_next <= literals);
		while (litrunlen--)
			*out_next++ = *literals++;
		if (out_next == out_block_end) /* End of block? */
			break;
	#else
		STATIC_ASSERT(LITRUNLEN_ALPHABET_SIZE - 2 <= 15);
		if (UNALIGNED_ACCESS_IS_FAST &&
		    likely(num_literals >= 16 && literals - out_next >= 16 &&
			   litrunlen != LITRUNLEN_ALPHABET_SIZE - 1))
		{
			/* Fast case */
			copy_16_bytes_unaligned(literals, out_next);
			out_next += litrunlen;
			literals += litrunlen;
			num_literals -= litrunlen;
		} else {
			/* Slow case */
			const u32 cutoff = LITRUNLEN_ALPHABET_SIZE - 1;
			if (litrunlen == cutoff) {
				SAFETY_CHECK(extra_bytes < in_end);
				litrunlen += *extra_bytes++;
				if (litrunlen == 0xFF + cutoff) {
					SAFETY_CHECK(in_end - extra_bytes >= 3);
					litrunlen += (u32)*extra_bytes++ << 0;
					litrunlen += (u32)*extra_bytes++ << 8;
					litrunlen += (u32)*extra_bytes++ << 16;
				}
			}

			num_literals -= litrunlen;
			SAFETY_CHECK(num_literals >= 0);

			if (UNALIGNED_ACCESS_IS_FAST &&
			    likely(litrunlen + WORDBYTES <= literals - out_next &&
				   num_literals >= WORDBYTES))
			{
				const u8 *src = literals;
				u8 *dst = out_next;

				out_next += litrunlen;
				literals += litrunlen;
				do {
					copy_word_unaligned(src, dst);
					src += WORDBYTES;
					dst += WORDBYTES;
					litrunlen -= WORDBYTES;
				} while ((s32)litrunlen > 0);
			} else {
				while (litrunlen--)
					*out_next++ = *literals++;
			}

			if (out_next == out_block_end) /* End of block? */
				break;
		}
	#endif
		/* END decode literal run */

		/* BEGIN decode match */

		/* Decode the length symbol */

		length = DECODE_SYMBOL(length_state, d->length_decode_table);

		/* Decode the offset symbol */

		if (!CAN_ENSURE(32))
			ENSURE_BITS(MAX_LOG2_NUM_OFFSET_STATES);
		offset_sym = DECODE_SYMBOL(offset_state, d->offset_decode_table);

		/* Decode the rest of the offset */

		if (offset_sym >= NUM_REPS) {

			/* Explicit offset */

			unsigned offset_log2 = offset_sym - NUM_REPS;

			offset = (u32)1 << offset_log2;

			if (block_type == BLOCKTYPE_ALIGNED &&
			    offset_log2 >= NUM_ALIGNED_BITS)
			{
				ENSURE_BITS(MAX_LOG2_NUM_ALIGNED_STATES +
					    offset_log2 - NUM_ALIGNED_BITS);

				offset += DECODE_SYMBOL(aligned_state,
							d->aligned_decode_table);
				offset += POP_BITS(offset_log2 -
						   NUM_ALIGNED_BITS) <<
							NUM_ALIGNED_BITS;
			} else {
				ENSURE_BITS(offset_log2);
				offset += POP_BITS(offset_log2);
			}

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
		} else {
			/* Repeat offset */
			offset = recent_offsets[offset_sym];
			recent_offsets[offset_sym] = recent_offsets[0];
		}

		recent_offsets[0] = offset;

		SAFETY_CHECK(offset <= out_next - (u8 *)out);

		/* Decode the remainder of the length and copy the match. */

		length += MIN_MATCH_LEN;

		if (UNALIGNED_ACCESS_IS_FAST && length <= 16 &&
		    offset >= length && literals - out_next >= 16)
		{
			/*
			 * Fast case: short length, no overlap, and we aren't
			 * getting too close to the literals portion of the
			 * output buffer.
			 */
			copy_16_bytes_unaligned(out_next - offset, out_next);
		} else {
			/*
			 * "Slow case" (but still very important): long length,
			 * or small offset, or we're getting close to the
			 * literals portion of the output buffer.
			 */
			const u32 cutoff = LENGTH_ALPHABET_SIZE - 1 + MIN_MATCH_LEN;
			const u8 *src;
			u8 *dst, *end;
			if (length == cutoff) {
				SAFETY_CHECK(extra_bytes < in_end);
				length += *extra_bytes++;
				if (length == 0xFF + cutoff) {
					SAFETY_CHECK(in_end - extra_bytes >= 3);
					length += (u32)*extra_bytes++ << 0;
					length += (u32)*extra_bytes++ << 8;
					length += (u32)*extra_bytes++ << 16;
				}
			}

			SAFETY_CHECK(length <= literals - out_next);

			src = out_next - offset;
			dst = out_next;
			end = out_next + length;

			if (UNALIGNED_ACCESS_IS_FAST &&
			    likely(literals - end >= WORDBYTES)) {
				if (offset >= WORDBYTES) {
					copy_word_unaligned(src, dst);
					src += WORDBYTES;
					dst += WORDBYTES;
					if (dst < end) {
						do {
							copy_word_unaligned(src, dst);
							src += WORDBYTES;
							dst += WORDBYTES;
						} while (dst < end);
					}
				} else if (offset == 1) {
					machine_word_t v = repeat_byte(*(dst - 1));
					do {
						store_word_unaligned(v, dst);
						src += WORDBYTES;
						dst += WORDBYTES;
					} while (dst < end);
				} else {
					do {
						*dst++ = *src++;
					} while (dst < end);
				}
			} else {
				do {
					*dst++ = *src++;
				} while (dst < end);
			}
		}

		out_next += length;

		/* END decode match */
	}

	SAFETY_CHECK(litrunlen_state == 0 && length_state == 0 &&
		     offset_state == 0 && aligned_state == 0);

	ALIGN_INPUT();

	/* Finished decompressing a block. */
	if (!is_final_block)
		goto next_block;

	/* That was the final block. */

#ifdef ENABLE_PREPROCESSING
	/* Postprocess the data if needed. */
	if (preprocessed)
		postprocess(out, out_nbytes_avail);
#endif

	if (actual_out_nbytes_ret) {
		*actual_out_nbytes_ret = out_next - (u8 *)out;
	} else {
		if (out_next != out_end)
			return DECOMPRESS_SHORT_OUTPUT;
	}
	return DECOMPRESS_SUCCESS;
}
