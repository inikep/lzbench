/*
 * xpack_common.c - common code for XPACK compression and decompression
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

#ifdef ENABLE_PREPROCESSING

#include <string.h>

#ifdef __SSE2__
#  include <emmintrin.h>
#endif

#ifdef __AVX2__
#  include <immintrin.h>
#endif

#include "xpack_common.h"
#include "unaligned.h"

static void
do_translate_target(void *target, s32 input_pos)
{
	s32 abs_offset, rel_offset;

	rel_offset = get_unaligned_le32(target);
	if (rel_offset >= -input_pos && rel_offset < MAGIC_FILESIZE) {
		if (rel_offset < MAGIC_FILESIZE - input_pos) {
			/* "good translation" */
			abs_offset = rel_offset + input_pos;
		} else {
			/* "compensating translation" */
			abs_offset = rel_offset - MAGIC_FILESIZE;
		}
		put_unaligned_le32(abs_offset, target);
	}
}

static void
undo_translate_target(void *target, s32 input_pos)
{
	s32 abs_offset, rel_offset;

	abs_offset = get_unaligned_le32(target);
	if (abs_offset >= 0) {
		if (abs_offset < MAGIC_FILESIZE) {
			/* "good translation" */
			rel_offset = abs_offset - input_pos;
			put_unaligned_le32(rel_offset, target);
		}
	} else {
		if (abs_offset >= -input_pos) {
			/* "compensating translation" */
			rel_offset = abs_offset + MAGIC_FILESIZE;
			put_unaligned_le32(rel_offset, target);
		}
	}
}

static void
e8_filter(u8 *data, u32 size, void (*process_target)(void *, s32))
{

#if !defined(__SSE2__) && !defined(__AVX2__)
	/*
	 * A worthwhile optimization is to push the end-of-buffer check into the
	 * relatively rare E8 case.  This is possible if we replace the last six
	 * bytes of data with E8 bytes; then we are guaranteed to hit an E8 byte
	 * before reaching end-of-buffer.  In addition, this scheme guarantees
	 * that no translation can begin following an E8 byte in the last 10
	 * bytes because a 4-byte offset containing E8 as its high byte is a
	 * large negative number that is not valid for translation.  That is
	 * exactly what we need.
	 */
	u8 *tail;
	u8 saved_bytes[6];
	u8 *p;

	if (size <= 10)
		return;

	tail = &data[size - 6];
	memcpy(saved_bytes, tail, 6);
	memset(tail, 0xE8, 6);
	p = data;
	for (;;) {
		while (*p != 0xE8)
			p++;
		if (p >= tail)
			break;
		(*process_target)(p + 1, p - data);
		p += 5;
	}
	memcpy(tail, saved_bytes, 6);
#else
	/* SSE2 or AVX-2 optimized version for x86_64 */

	u8 *p = data;
	u64 valid_mask = ~0;

	if (size <= 10)
		return;
#ifdef __AVX2__
#  define ALIGNMENT_REQUIRED 32
#else
#  define ALIGNMENT_REQUIRED 16
#endif

	/* Process one byte at a time until the pointer is properly aligned. */
	while ((uintptr_t)p % ALIGNMENT_REQUIRED != 0) {
		if (p >= data + size - 10)
			return;
		if (*p == 0xE8 && (valid_mask & 1)) {
			(*process_target)(p + 1, p - data);
			valid_mask &= ~0x1F;
		}
		p++;
		valid_mask >>= 1;
		valid_mask |= (u64)1 << 63;
	}

	if (data + size - p >= 64) {

		/* Vectorized processing */

		/* Note: we use a "trap" E8 byte to eliminate the need to check
		 * for end-of-buffer in the inner loop.  This byte is carefully
		 * positioned so that it will never be changed by a previous
		 * translation before it is detected. */

		u8 *trap = p + ((data + size - p) & ~31) - 32 + 4;
		u8 saved_byte = *trap;
		*trap = 0xE8;

		for (;;) {
			u32 e8_mask;
			u8 *orig_p = p;
		#ifdef __AVX2__
			const __m256i e8_bytes = _mm256_set1_epi8(0xE8);
			for (;;) {
				__m256i bytes = *(const __m256i *)p;
				__m256i cmpresult = _mm256_cmpeq_epi8(bytes, e8_bytes);
				e8_mask = _mm256_movemask_epi8(cmpresult);
				if (e8_mask)
					break;
				p += 32;
			}
		#else
			const __m128i e8_bytes = _mm_set1_epi8(0xE8);
			for (;;) {
				/* Read the next 32 bytes of data and test them
				 * for E8 bytes. */
				__m128i bytes1 = *(const __m128i *)p;
				__m128i bytes2 = *(const __m128i *)(p + 16);
				__m128i cmpresult1 = _mm_cmpeq_epi8(bytes1, e8_bytes);
				__m128i cmpresult2 = _mm_cmpeq_epi8(bytes2, e8_bytes);
				u32 mask1 = _mm_movemask_epi8(cmpresult1);
				u32 mask2 = _mm_movemask_epi8(cmpresult2);
				/* The masks have a bit set for each E8 byte.
				 * We stay in this fast inner loop as long as
				 * there are no E8 bytes. */
				if (mask1 | mask2) {
					e8_mask = mask1 | (mask2 << 16);
					break;
				}
				p += 32;
			}
		#endif

			/* Did we pass over data with no E8 bytes? */
			if (p != orig_p)
				valid_mask = ~0;

			/* Are we nearing end-of-buffer? */
			if (p == trap - 4)
				break;

			/* Process the E8 bytes.  However, the AND with
			 * 'valid_mask' ensures we never process an E8 byte that
			 * was itself part of a translation target. */
			while ((e8_mask &= valid_mask)) {
				unsigned bit = bsf32(e8_mask);
				(*process_target)(p + bit + 1, p + bit - data);
				valid_mask &= ~((u64)0x1F << bit);
			}

			valid_mask >>= 32;
			valid_mask |= 0xFFFFFFFF00000000;
			p += 32;
		}

		*trap = saved_byte;
	}

	/* Approaching the end of the buffer; process one byte a time. */
	while (p < data + size - 10) {
		if (*p == 0xE8 && (valid_mask & 1)) {
			(*process_target)(p + 1, p - data);
			valid_mask &= ~0x1F;
		}
		p++;
		valid_mask >>= 1;
		valid_mask |= (u64)1 << 63;
	}
#endif /* __SSE2__ || __AVX2__ */
}

void
preprocess(void *data, u32 size)
{
	e8_filter(data, size, do_translate_target);
}

void
postprocess(void *data, u32 size)
{
	e8_filter(data, size, undo_translate_target);
}

#endif /* ENABLE_PREPROCESSING */
