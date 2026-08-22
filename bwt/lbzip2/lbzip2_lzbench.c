/*
  lbzip2_lzbench.c -- buffer-to-buffer wrapper around lbzip2's codec

  lbzip2 is a bzip2-compatible compressor whose command-line tool splits the
  work over threads.  lzbench benchmarks buffer-to-buffer calls and does its
  own threading, so this drives the same low-level encoder and decoder
  sequentially, one 100k-900k block at a time, exactly as compress.c and
  expand.c do around their scheduler.  The bytes produced are the same.

  This file is part of lzbench and, like lbzip2, is distributed under the GNU
  General Public License, version 3 or later.
*/

#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "decode.h"
#include "encode.h"
#include "lbzip2_lzbench.h"

/* decode.c allocates through main.c's xmalloc(), which is the only thing it
   needs from the command-line tool. */
void *
xmalloc(size_t size)
{
  void *ptr = malloc(size);

  if (ptr == NULL)
    abort();

  return ptr;
}


size_t
lbzip2_buf_compress(const void *inbuf, size_t insize, void *outbuf,
                    size_t outsize, int level)
{
  const uint8_t *in = inbuf;
  uint8_t *out = outbuf;
  struct encoder_state *enc;
  size_t left = insize;
  size_t pos = HEADER_SIZE;
  uint32_t combined_crc = 0;

  if (level < 1 || level > 9 || outsize < HEADER_SIZE + TRAILER_SIZE)
    return 0;

  out[0] = 0x42;
  out[1] = 0x5A;
  out[2] = 0x68;
  out[3] = 0x30 + level;

  enc = malloc(encoder_alloc_size(level * 100000u));
  if (enc == NULL)
    return 0;

  while (left > 0) {
    uint32_t crc;
    size_t size;
    const void *block;

    encoder_init(enc, level * 100000u, CLUSTER_FACTOR);

    /* One call fills the block, or exhausts the input trying. */
    (void)collect(enc, in + (insize - left), &left);

    size = encode(enc, &crc);
    block = transmit(enc, NULL);
    combined_crc = combine_crc(combined_crc, crc);

    /* Every block is padded to a whole number of bytes by the encoder, so
       blocks concatenate without any bit shifting. */
    if (pos + size + TRAILER_SIZE > outsize) {
      free(enc);
      return 0;
    }
    memcpy(out + pos, block, size);
    pos += size;
  }

  free(enc);

  out[pos++] = 0x17;
  out[pos++] = 0x72;
  out[pos++] = 0x45;
  out[pos++] = 0x38;
  out[pos++] = 0x50;
  out[pos++] = 0x90;
  out[pos++] = combined_crc >> 24;
  out[pos++] = (combined_crc >> 16) & 0xFF;
  out[pos++] = (combined_crc >> 8) & 0xFF;
  out[pos++] = combined_crc & 0xFF;

  return pos;
}


size_t
lbzip2_buf_decompress(const void *inbuf, size_t insize, void *outbuf,
                      size_t outsize)
{
  const uint8_t *in = inbuf;
  struct parser_state ps;
  struct bitstream bs;
  struct header hd;
  uint32_t *words;
  size_t nwords;
  size_t produced = 0;
  unsigned garbage;
  int rv;

  /* The stream header is what tells the parser the block size; the tool
     reads it before starting the expansion, and so do we. */
  if (insize < HEADER_SIZE + TRAILER_SIZE || in[0] != 0x42 || in[1] != 0x5A ||
      in[2] != 0x68 || in[3] < 0x31 || in[3] > 0x39)
    return 0;

  /* The decoder reads the stream as 32-bit words, so copy the rest of the
     input into an aligned buffer zero-padded to a word boundary.  Compressed
     input is small next to the work of decoding it. */
  nwords = (insize - HEADER_SIZE + 3) / 4;
  words = calloc(nwords, sizeof(uint32_t));
  if (words == NULL)
    return 0;
  memcpy(words, in + HEADER_SIZE, insize - HEADER_SIZE);

  bs.live = 0;
  bs.buff = 0;
  bs.block = NULL;
  bs.data = words;
  bs.limit = words + nwords;
  bs.eof = true;

  parser_init(&ps, in[3] - 0x30, 0);

  for (;;) {
    struct decoder_state ds;

    rv = parse(&ps, &hd, &bs, &garbage);
    if (rv == FINISH)
      break;
    if (rv != OK)
      goto err;

    decoder_init(&ds);

    rv = retrieve(&ds, &bs);
    if (rv == OK) {
      decode(&ds);

      do {
        size_t avail = outsize - produced;

        if (avail == 0) {
          rv = ERR_OVERFLOW;
          break;
        }
        rv = emit(&ds, (char *)outbuf + produced, &avail);
        produced = outsize - avail;
      }
      while (rv == MORE);
    }

    if (rv == OK && ds.crc != hd.crc)
      rv = ERR_BLKCRC;

    decoder_free(&ds);

    if (rv != OK)
      goto err;
  }

  free(words);
  return produced;

err:
  free(words);
  return 0;
}
