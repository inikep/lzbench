/* inftrees.h -- header to use inftrees.c
 * Copyright (C) 1995-2026 Mark Adler
 * Modifications Copyright (C) 2023-2026, Advanced Micro Devices. All rights reserved.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* WARNING: this file should *not* be used by applications. It is
   part of the implementation of the compression library and is
   subject to change. Applications should only use zlib.h.
 */

/* Structure for decoding tables.  Each entry provides either the
   information needed to do the operation requested by the code that
   indexed that table entry, or it provides a pointer to another
   table that indexes more bits of the code.  op indicates whether
   the entry is a pointer to another table, a literal, a length or
   distance, an end-of-block, or an invalid code.  For a table
   pointer, the low four bits of op is the number of index bits of
   that table.  For a length or distance, the low four bits of op
   is the number of extra bits to get after the code.  bits is
   the number of bits in this code or part of the code to drop off
   of the bit buffer.  val is the actual byte to output in the case
   of a literal, the base length or distance, or the offset from
   the current table to the next table.  Each entry is four bytes. */
typedef struct {
    unsigned char op;           /* operation, extra bits, table bits */
    unsigned char bits;         /* bits in this part of the code */
    unsigned short val;         /* offset in table or code value */
} code;

/* WARNING: on the combined format (non-GCC, see AOCL_COMBINED_CODE_FORMAT) op/bits
   are REDEFINED for length/distance leaves (op = 16|code_bits, bits =
   code_bits+extra). Layout is unchanged, but reading here.op & 15 or here.bits
   raw MISREADS such entries; always use CODE_BITS()/CODE_EXTRA()/EXTRA_BITS(). */

/* op values as set by inflate_table():
    00000000 - literal
    0000tttt - table link, tttt != 0 is the number of table index bits
    0001eeee - length or distance, eeee is the number of extra bits
    01100000 - end of block
    01000000 - invalid code
 */

#ifndef AOCL_ZLIB_OPT
/* Maximum size of the dynamic table.  The maximum number of code structures is
   1444, which is the sum of 852 for literal/length codes and 592 for distance
   codes.  These values were found by exhaustive searches using the program
   examples/enough.c found in the zlib distribution.  The arguments to that
   program are the number of symbols, the initial root table size, and the
   maximum bit length of a code.  "enough 286 9 15" for literal/length codes
   returns 852, and "enough 30 6 15" for distance codes returns 592. The
   initial root table size (9 or 6) is found in the fifth argument of the
   inflate_table() calls in inflate.c and infback.c.  If the root table size is
   changed, then these maximum sizes would be need to be recalculated and
   updated. */
#define ENOUGH_LENS 852
#else
/* Maximum size of the dynamic table.  The maximum number of code structures is
   1924, which is the sum of 1332 for literal/length codes and 592 for distance
   codes.  These values were found by exhaustive searches using the program
   examples/enough.c found in the zlib distribution.  The arguments to that
   program are the number of symbols, the initial root table size, and the
   maximum bit length of a code.  "enough 286 10 15" for literal/length codes
   returns returns 1332, and "enough 30 9 15" for distance codes returns 592.
   The initial root table size (10 or 9) is found in the fifth argument of the
   inflate_table() calls in inflate.c and infback.c.  If the root table size is
   changed, then these maximum sizes would be need to be recalculated and
   updated. */
#define ENOUGH_LENS 1332
#endif /* AOCL_ZLIB_OPT */
#define ENOUGH_DISTS 592
#define ENOUGH (ENOUGH_LENS+ENOUGH_DISTS)

/* Type of code to build for inflate_table() */
typedef enum {
    CODES,
    LENS,
    DISTS
} codetype;

/* Combined table-entry format (ported from zlib-ng), enabled on non-GCC only.
   Under GCC the classic format is kept and these accessors reduce to the
   plain fields.

   Classic format (GCC): for a length/distance code, op = 16 | extra_bits,
     bits = code_bits, and the decoder does DROPBITS(code_bits) then BITS(extra).
   Combined format (non-GCC): op = 16 | code_bits, bits = code_bits + extra_bits.
     The decoder saves `old = hold`, does ONE DROPBITS(bits), and extracts the
     extra bits from `old` with EXTRA_BITS (a single shift, no 2nd DROPBITS). */
#ifndef AOCL_COMBINED_CODE_FORMAT
#  if defined(__GNUC__) && !defined(__clang__)
#    define AOCL_COMBINED_CODE_FORMAT 0
#  else
#    define AOCL_COMBINED_CODE_FORMAT 1
#  endif
#endif

#if AOCL_COMBINED_CODE_FORMAT
#include <stdint.h>   /* uint64_t is used by EXTRA_BITS below; inftrees.h can be
                         included (e.g. by infcover.c) without a prior stdint. */
/* Build op/bits for a length/distance entry from classic (extra, code_bits). */
#  define COMBINE_OP(extra, code_bits) \
     ((unsigned char)((extra) & 16 ? (unsigned char)((code_bits) | 16) : (unsigned char)(extra)))
/* Fold the extra-bit count into here.bits, but ONLY for real length/distance
   leaves (classic op has the 0x10 flag). Invalid length codes carry lext[]
   markers such as 199 and 75 (see inftrees.c) that lack the 0x10 flag; adding
   (extra & 15) for those would inflate here.bits and make the decoder drop too
   many bits before it reports the invalid code. */
#  define COMBINE_BITS(code_bits, extra) \
     ((unsigned char)((extra) & 16 ? ((code_bits) + ((extra) & 15)) : (code_bits)))
/* Number of CODE bits (not extra) to consume for this entry. For a length or
   distance leaf (op&16) that is op&15; otherwise here.bits (literal/link).
   The `optoff` argument is inflate_state.opt_off (the format the table was
   built in), not the live global zlibOptOff, so a mid-stream flip is harmless. */
#  define CODE_BITS(here, optoff)  ((optoff) \
     ? ((unsigned)((here).bits)) \
     : ((unsigned)(((here).op & 16) ? ((here).op & 15) : (here).bits)))
/* Extra-bit count for a length/distance entry (0 otherwise). */
#  define CODE_EXTRA(here, optoff) ((optoff) \
     ? ((unsigned)((here).op) & 15) \
     : ((unsigned)(((here).op & 16) ? ((here).bits - ((here).op & 15)) : 0)))
/* Extract the extra-bit value from the pre-DROPBITS accumulator `old`.
   Invariant: (here).bits <= 63. It is the shift count in (1 << (here).bits),
   so it must stay below 64 or the shift is undefined. In the combined format
   here.bits = code_bits + extra_bits, and both are bounded (code_bits <=
   MAXBITS == 15, extra_bits <= 13 for distances / 5 for lengths), so here.bits
   never exceeds ~28. Keep this true if the code/entry layout ever changes. */
#  define EXTRA_BITS(old, here, op) \
     (((old) & ((((uint64_t)1) << (here).bits) - 1)) >> ((op) & 15))
#else
/* Classic layout (GCC): format never varies, so the optoff argument is ignored
   but kept for a single call-site signature across both builds. */
#  define CODE_BITS(here, optoff)  ((void)(optoff), (unsigned)((here).bits))
#  define CODE_EXTRA(here, optoff) ((void)(optoff), (unsigned)((here).op) & 15)
#endif

int ZLIB_INTERNAL inflate_table(codetype type, unsigned short FAR *lens,
                                unsigned codes, code FAR * FAR *table,
                                unsigned FAR *bits, unsigned short FAR *work);
struct inflate_state;
void ZLIB_INTERNAL inflate_fixed(struct inflate_state FAR *state);
