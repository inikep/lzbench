/*
 * lz_extend.h - fast match extension for Lempel-Ziv matchfinding
 */

#ifndef LIB_LZ_EXTEND_H
#define LIB_LZ_EXTEND_H

#include "unaligned.h"

/*
 * Return the number of bytes at @matchptr that match the bytes at @strptr, up
 * to a maximum of @max_len.  Initially, @start_len bytes are matched.
 */
static forceinline u32
lz_extend(const u8 * const strptr, const u8 * const matchptr,
	  const u32 start_len, const u32 max_len)
{
	u32 len = start_len;
	machine_word_t v_word;

	if (UNALIGNED_ACCESS_IS_FAST) {

		if (likely(max_len - len >= 4 * WORDBYTES)) {

		#define COMPARE_WORD_STEP					\
			v_word = load_word_unaligned(&matchptr[len]) ^		\
				 load_word_unaligned(&strptr[len]);		\
			if (v_word != 0)					\
				goto word_differs;				\
			len += WORDBYTES;					\

			COMPARE_WORD_STEP
			COMPARE_WORD_STEP
			COMPARE_WORD_STEP
			COMPARE_WORD_STEP
		#undef COMPARE_WORD_STEP
		}

		while (len + WORDBYTES <= max_len) {
			v_word = load_word_unaligned(&matchptr[len]) ^
				 load_word_unaligned(&strptr[len]);
			if (v_word != 0)
				goto word_differs;
			len += WORDBYTES;
		}
	}

	while (len < max_len && matchptr[len] == strptr[len])
		len++;
	return len;

word_differs:
	if (CPU_IS_LITTLE_ENDIAN())
		len += (bsfw(v_word) >> 3);
	else
		len += (8 * WORDBYTES - 1 - bsrw(v_word)) >> 3;
	return len;
}

#endif /* LIB_LZ_EXTEND_H */
