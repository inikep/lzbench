/**
 * Copyright (C) 2023-2024, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef AOCL_SEND_BITS_H
#define AOCL_SEND_BITS_H

#include "deflate.h"

 /* Dynamically opting AOCL variant of send_bits */
#ifdef AOCL_ZLIB_OPT
#define OPT_send_bits(s, c, v) \
    { if (zlibOptOff == 1) { send_bits((s), (c), (v)); } else { AOCL_send_bits((s), (c), (v)); } }
#else
#define OPT_send_bits(s, c, v) send_bits((s), (c), (v))
#endif /* AOCL_ZLIB_OPT */

#ifndef ZLIB_DEBUG
#  define send_code(s, c, tree) OPT_send_bits(s, tree[c].Code, tree[c].Len);
 /* Send a code of the given tree. c and tree must not have side effects */

#else /* ZLIB_DEBUG */
#  define send_code(s, c, tree) \
     { if (z_verbose>2) fprintf(stderr,"\ncd %3d ",(c)); \
       OPT_send_bits(s, tree[c].Code, tree[c].Len); }
#endif


/* ===========================================================================
 * Output a uint64_t LSB first on the stream.
 * IN assertion: there is enough room in pendingBuf.
 */
#define AOCL_put_uInt64_t(s, w) { \
    zmemcpy(&s->pending_buf[s->pending], &w, sizeof(w)); \
    s->pending += 8; \
}

/* AOCL variant of send_bits marco for 64 bit size of member variable of deflate state (s->bi_buf) */
#if defined(ZLIB_DEBUG)
static void AOCL_send_bits(deflate_state* s, int value, int length) {
    Tracevv((stderr, " l %2d v %4x ", length, value));
    Assert(length > 0 && length <= 63, "invalid length");
    s->bits_sent += (ulg)length;

    /* If not enough room in bi_buf, use (valid) bits from bi_buf and
     * (64 - bi_valid) bits from value, leaving (width - (64 - bi_valid))
     * unused bits in value.
     */
    if (s->bi_valid == AOCL_Buf_size) {
        AOCL_put_uInt64_t(s, s->bi_buf);
        s->bi_buf = value;
        s->bi_valid = length;
    }
    else if (s->bi_valid > (int)AOCL_Buf_size - length) {
        s->bi_buf |= (uint64_t)value << s->bi_valid;
        AOCL_put_uInt64_t(s, s->bi_buf);
        s->bi_buf = (uint64_t)value >> (AOCL_Buf_size - s->bi_valid);
        s->bi_valid += length - AOCL_Buf_size;
    }
    else {
        s->bi_buf |= (uint64_t)value << s->bi_valid;
        s->bi_valid += length;
    }
}

#else

/* AOCL variant of send_bits marco for 64 bit size of member variable of deflate state (s->bi_buf) */
#define AOCL_send_bits(s, value, length) \
{ int len = length; \
  if (s->bi_valid == AOCL_Buf_size) { \
      AOCL_put_uInt64_t(s, s->bi_buf); \
      s->bi_buf = value; \
      s->bi_valid = len; \
  }\
  else if (s->bi_valid > (int)AOCL_Buf_size - len) { \
    int val = (int)value; \
    s->bi_buf |= (uint64_t)val << s->bi_valid; \
    AOCL_put_uInt64_t(s, s->bi_buf); \
    s->bi_buf = (uint64_t)val >> (AOCL_Buf_size - s->bi_valid); \
    s->bi_valid += len - AOCL_Buf_size; \
  } else { \
    s->bi_buf |= (uint64_t)(value) << s->bi_valid; \
    s->bi_valid += len; \
  } \
}
#endif /* ZLIB_DEBUG */

#endif /* AOCL_SEND_BITS_H */
