/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 *
 * lz_codecs.cpp: LZ-based codecs that offer fast decompression speeds
 */

#include "codecs.h"

#include <stdint.h>
#include <stdio.h> // printf
#include <string.h> // memcpy
#include <algorithm> // std::max



int64_t lzbench_memcpy(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    memcpy(outbuf, inbuf, insize);
    return insize;
}



#ifndef BENCH_REMOVE_MEMLZ
#include "lz/memlz/memlz.h"

char* lzbench_memlz_init(size_t insize, size_t level, size_t)
{
    return (char*)malloc(sizeof(memlz_state));
}

void lzbench_memlz_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_memlz_compress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t* codec_options)
{
    if (!codec_options->work_mem)
        return 0;

    memlz_reset((memlz_state*)codec_options->work_mem);
    return memlz_stream_compress(outbuf, inbuf, insize, (memlz_state*)codec_options->work_mem);
}

int64_t lzbench_memlz_decompress(char* inbuf, size_t insize, char* outbuf, size_t outsize, codec_options_t* codec_options)
{
    memlz_reset((memlz_state*)codec_options->work_mem);
    return (int64_t)memlz_stream_decompress(outbuf, inbuf, (memlz_state*)codec_options->work_mem);
}

#endif // BENCH_REMOVE_MEMLZ



#ifndef BENCH_REMOVE_BRIEFLZ
#include "lz/brieflz/brieflz.h"

char* lzbench_brieflz_init(size_t insize, size_t level, size_t)
{
    return (char*) malloc(blz_workmem_size_level(insize, level));
}

void lzbench_brieflz_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_brieflz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (!codec_options->work_mem)
        return 0;

    int64_t res = blz_pack_level(inbuf, outbuf, insize, (void*)codec_options->work_mem, codec_options->level);

    return res;
}

int64_t lzbench_brieflz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return blz_depack(inbuf, outbuf, outsize);
}

#endif // BENCH_REMOVE_BRIEFLZ



#ifndef BENCH_REMOVE_BROTLI
#include "brotli/encode.h"
#include "brotli/decode.h"

int64_t lzbench_brotli_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int windowLog = codec_options->additional_param;
    if (!windowLog) windowLog = BROTLI_DEFAULT_WINDOW; // sliding window size. Range is 10 to 24.

    size_t actual_osize = outsize;
    return BrotliEncoderCompress(codec_options->level, windowLog, BROTLI_DEFAULT_MODE, insize, (const uint8_t*)inbuf, &actual_osize, (uint8_t*)outbuf) == 0 ? 0 : actual_osize;
}
int64_t lzbench_brotli_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    size_t actual_osize = outsize;
    return BrotliDecoderDecompress(insize, (const uint8_t*)inbuf, &actual_osize, (uint8_t*)outbuf) == BROTLI_DECODER_RESULT_ERROR ? 0 : actual_osize;
}

#endif // BENCH_REMOVE_BROTLI



#ifndef BENCH_REMOVE_CRUSH
#include "lz/crush/crush.hpp"

int64_t lzbench_crush_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return crush::compress(codec_options->level, (uint8_t*)inbuf, insize, (uint8_t*)outbuf);
}

int64_t lzbench_crush_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return crush::decompress((uint8_t*)inbuf, (uint8_t*)outbuf, outsize);
}

#endif // BENCH_REMOVE_CRUSH



#ifndef BENCH_REMOVE_FASTLZ
extern "C"
{
    #include "lz/fastlz/fastlz.h"
}

int64_t lzbench_fastlz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return fastlz_compress_level(codec_options->level, inbuf, insize, outbuf);
}

int64_t lzbench_fastlz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return fastlz_decompress(inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_FASTLZMA2
#include "lz/fast-lzma2/fast-lzma2.h"

int64_t lzbench_fastlzma2_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    size_t ret = FL2_compressMt(outbuf, outsize, inbuf, insize, codec_options->level, codec_options->threads);
    if (FL2_isError(ret)) return 0;
    return ret;
}

int64_t lzbench_fastlzma2_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    size_t ret = FL2_decompressMt(outbuf, outsize, inbuf, insize, codec_options->threads);
    if (FL2_isError(ret)) return 0;
    return ret;
}
#endif // BENCH_REMOVE_FASTLZMA2



#ifndef BENCH_REMOVE_KANZI
#include "misc/kanzi-cpp/src/types.hpp"
#include "misc/kanzi-cpp/src/InputStream.hpp"
#include "misc/kanzi-cpp/src/OutputStream.hpp"
#include "misc/kanzi-cpp/src/io/CompressedInputStream.hpp"
#include "misc/kanzi-cpp/src/io/CompressedOutputStream.hpp"
#include "misc/kanzi-cpp/src/util/fixedbuf.hpp"

int64_t lzbench_kanzi_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    std::string entropy;
    std::string transform;
    kanzi::uint szBlock;

    switch (codec_options->level) {
    case 0:
        transform = "NONE";
        entropy = "NONE";
        szBlock = 4 * 1024 * 1024;
        break;
    case 1:
        transform = "LZX";
        entropy = "NONE";
        szBlock = 4 * 1024 * 1024;
        break;
    case 2:
        transform = "DNA+LZ";
        entropy = "HUFFMAN";
        szBlock = 4 * 1024 * 1024;
        break;
    case 3:
        transform = "TEXT+UTF+PACK+MM+LZX";
        entropy = "HUFFMAN";
        szBlock = 4 * 1024 * 1024;
        break;
    case 4:
        transform = "TEXT+UTF+EXE+PACK+MM+ROLZ";
        entropy = "NONE";
        szBlock = 4 * 1024 * 1024;
        break;
    case 5:
        transform = "TEXT+UTF+BWT+RANK+ZRLT";
        entropy = "ANS0";
        szBlock = 4 * 1024 * 1024;
        break;
    case 6:
        transform = "TEXT+UTF+BWT+SRT+ZRLT";
        entropy = "FPAQ";
        szBlock = 8 * 1024 * 1024;
        break;
    case 7:
        transform = "LZP+TEXT+UTF+BWT+LZP";
        entropy = "CM";
        szBlock = 16 * 1024 * 1024;
        break;
    case 8:
        transform = "EXE+RLT+TEXT+UTF+DNA";
        entropy = "TPAQ";
        szBlock = 16 * 1024 * 1024;
        break;
    case 9:
        transform = "EXE+RLT+TEXT+UTF+DNA";
        entropy = "TPAQX";
        szBlock = 32 * 1024 * 1024;
        break;
    default:
        return -1;
    }

    ofixedbuf buf(outbuf, outsize);
    std::iostream os(&buf);
    kanzi::CompressedOutputStream cos(os, codec_options->threads, entropy, transform, szBlock);
    const size_t max_io_size = size_t(1) << 30;
    size_t remaining = insize;
    char* next = inbuf;

    while (remaining > 0) {
        const size_t chunk = std::min(remaining, max_io_size);
        cos.write(next, static_cast<std::streamsize>(chunk));
        next += chunk;
        remaining -= chunk;
    }

    cos.close();
    return cos.getWritten();
}

int64_t lzbench_kanzi_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ifixedbuf buf(inbuf, insize);
    std::iostream is(&buf);
    kanzi::CompressedInputStream cis(is, codec_options->threads);
    const size_t max_io_size = size_t(1) << 30;
    size_t total = 0;

    while (total < outsize) {
        const size_t chunk = std::min(outsize - total, max_io_size);
        cis.read(outbuf + total, static_cast<std::streamsize>(chunk));
        const size_t decoded = static_cast<size_t>(cis.gcount());
        total += decoded;

        if (decoded != chunk)
            break;
    }

    cis.close();
    return total;
}
#endif // BENCH_REMOVE_KANZI



#ifndef BENCH_REMOVE_LIBDEFLATE
#include "lz/libdeflate/libdeflate.h"
int64_t lzbench_libdeflate_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    struct libdeflate_compressor *compressor = libdeflate_alloc_compressor(codec_options->level);
    if (!compressor)
        return 0;
    int64_t res = libdeflate_deflate_compress(compressor, inbuf, insize, outbuf, outsize);
    libdeflate_free_compressor(compressor);
    return res;
}
int64_t lzbench_libdeflate_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    struct libdeflate_decompressor *decompressor = libdeflate_alloc_decompressor();
    if (!decompressor)
        return 0;
    size_t res = 0;
    if (libdeflate_deflate_decompress(decompressor, inbuf, insize, outbuf, outsize, &res) != LIBDEFLATE_SUCCESS) {
        return 0;
    }
    return res;
}
#endif



#ifndef BENCH_REMOVE_LIZARD
#include "lz/lizard/lizard_compress.h"
#include "lz/lizard/lizard_decompress.h"

int64_t lzbench_lizard_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return Lizard_compress(inbuf, outbuf, insize, outsize, codec_options->level);
}

int64_t lzbench_lizard_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return Lizard_decompress_safe(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZ4
#include "lz/lz4/lib/lz4.h"
#include "lz/lz4/lib/lz4hc.h"

int64_t lzbench_lz4_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZ4_compress_default(inbuf, outbuf, insize, outsize);
}

int64_t lzbench_lz4fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZ4_compress_fast(inbuf, outbuf, insize, outsize, codec_options->level);
}

int64_t lzbench_lz4hc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZ4_compress_HC(inbuf, outbuf, insize, outsize, codec_options->level);
}

int64_t lzbench_lz4_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZ4_decompress_safe(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZAV
#include "lz/lzav/lzav.h"

int64_t lzbench_lzav_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (codec_options->level == 1)
        return lzav_compress_default(inbuf, outbuf, insize, outsize);
    return lzav_compress_hi(inbuf, outbuf, insize, outsize);
}

int64_t lzbench_lzav_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return lzav_decompress(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZF
extern "C"
{
    #include "lz/lzf/lzf.h"
}

int64_t lzbench_lzf_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (codec_options->level == 0)
        return lzf_compress(inbuf, insize, outbuf, outsize);
    return lzf_compress_very(inbuf, insize, outbuf, outsize);
}

int64_t lzbench_lzf_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return lzf_decompress(inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZFSE
extern "C"
{
    #include "lz/lzfse/lzfse.h"
}

char* lzbench_lzfse_init(size_t insize, size_t level, size_t)
{
    return (char*) malloc(std::max(lzfse_encode_scratch_size(), lzfse_decode_scratch_size()));
}

void lzbench_lzfse_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_lzfse_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return lzfse_encode_buffer((uint8_t*)outbuf, outsize, (uint8_t*)inbuf, insize, codec_options->work_mem);
}

int64_t lzbench_lzfse_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return lzfse_decode_buffer((uint8_t*)outbuf, outsize, (uint8_t*)inbuf, insize, codec_options->work_mem);
}

#endif



#ifndef BENCH_REMOVE_LZFSE
extern "C"
{
    #include "lz/lzfse/lzvn.h"
}

char* lzbench_lzvn_init(size_t insize, size_t level, size_t)
{
    return (char*) malloc(std::max(lzvn_encode_scratch_size(), lzvn_decode_scratch_size()));
}

void lzbench_lzvn_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_lzvn_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return lzvn_encode_buffer((uint8_t*)outbuf, outsize, (uint8_t*)inbuf, insize, codec_options->work_mem);
}

int64_t lzbench_lzvn_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return lzvn_decode_buffer_scratch((uint8_t*)outbuf, outsize, (uint8_t*)inbuf, insize, codec_options->work_mem);
}

#endif



#ifndef BENCH_REMOVE_LZG
#include "lz/liblzg/lzg.h"

int64_t lzbench_lzg_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzg_encoder_config_t cfg;
    cfg.level = codec_options->level;
    cfg.fast = LZG_TRUE;
    cfg.progressfun = NULL;
    cfg.userdata = NULL;
    return LZG_Encode((const unsigned char*)inbuf, insize, (unsigned char*)outbuf, outsize, &cfg);
}

int64_t lzbench_lzg_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZG_Decode((const unsigned char*)inbuf, insize, (unsigned char*)outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZHAM
#include "lz/lzham/include/lzham.h"
#include <memory.h>

int64_t lzbench_lzham_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int dict_size_log = codec_options->additional_param;
    lzham_compress_params comp_params;

    memset(&comp_params, 0, sizeof(comp_params));
    comp_params.m_struct_size = sizeof(lzham_compress_params);
    comp_params.m_dict_size_log2 = dict_size_log?dict_size_log:26;
    comp_params.m_max_helper_threads = codec_options->threads > 1 ? codec_options->threads : 0;
    comp_params.m_level = (lzham_compress_level)codec_options->level;

    lzham_compress_status_t comp_status;
    lzham_uint32 comp_adler32 = 0;

    if ((comp_status = lzham_compress_memory(&comp_params, (uint8_t*)outbuf, &outsize, (const lzham_uint8 *)inbuf, insize, &comp_adler32)) != LZHAM_COMP_STATUS_SUCCESS)
    {
        printf("Compression test failed with status %i!\n", comp_status);
        return 0;
    }

    return outsize;
}

int64_t lzbench_lzham_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int dict_size_log = codec_options->additional_param;
    lzham_uint32 comp_adler32 = 0;
    lzham_decompress_params decomp_params;

    memset(&decomp_params, 0, sizeof(decomp_params));
    decomp_params.m_struct_size = sizeof(decomp_params);
    decomp_params.m_dict_size_log2 = dict_size_log?dict_size_log:26;

    lzham_decompress_memory(&decomp_params, (uint8_t*)outbuf, &outsize, (const lzham_uint8 *)inbuf, insize, &comp_adler32);
    return outsize;
}

#endif



#ifndef BENCH_REMOVE_LZLIB
#include "lz/lzlib/lzlib.h"

int64_t lzbench_lzlib_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
  struct Lzma_options
  {
      int dictionary_size;		/* 4 KiB .. 512 MiB */
      int match_len_limit;		/* 5 .. 273 */
  };

  const struct Lzma_options option_mapping[10] = {
    {   65535,  16 },		/* -0 */
    { 1 << 20,   5 },		/* -1 */
    { 3 << 19,   6 },		/* -2 */
    { 1 << 21,   8 },		/* -3 */
    { 3 << 20,  12 },		/* -4 */
    { 1 << 22,  20 },		/* -5 */
    { 1 << 23,  36 },		/* -6 */
    { 1 << 24,  68 },		/* -7 */
    { 3 << 23, 132 },		/* -8 */
    { 1 << 25, 273 } };		/* -9 */

  struct LZ_Encoder * encoder;
  const int match_len_limit = option_mapping[codec_options->level].match_len_limit;
  const unsigned long long member_size = 0x7FFFFFFFFFFFFFFFULL;	/* INT64_MAX */
  int new_pos = 0;
  int written = 0;
  bool error = false;
  int dict_size = option_mapping[codec_options->level].dictionary_size;
  uint8_t *buf = (uint8_t*)inbuf;
  uint8_t *obuf = (uint8_t*)outbuf;


  if( dict_size > insize ) dict_size = insize;		/* saves memory */
  if( dict_size < LZ_min_dictionary_size() )
    dict_size = LZ_min_dictionary_size();
  encoder = LZ_compress_open( dict_size, match_len_limit, member_size );
  if( !encoder || LZ_compress_errno( encoder ) != LZ_ok )
    { LZ_compress_close( encoder ); return 0; }

  while( true )
    {
    int rd;
    if( LZ_compress_write_size( encoder ) > 0 )
      {
      if( written < insize )
        {
        const int wr = LZ_compress_write( encoder, buf + written, insize - written );
        if( wr < 0 ) { error = true; break; }
        written += wr;
        }
      if( written >= insize ) LZ_compress_finish( encoder );
      }
    rd = LZ_compress_read( encoder, obuf + new_pos, outsize - new_pos );
    if( rd < 0 ) { error = true; break; }
    new_pos += rd;
    if( LZ_compress_finished( encoder ) == 1 ) break;
    }

  if( LZ_compress_close( encoder ) < 0 ) error = true;
  if (error) return 0;

  return new_pos;
}


int64_t lzbench_lzlib_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
  struct LZ_Decoder * const decoder = LZ_decompress_open();
  uint8_t * new_data = (uint8_t*)outbuf;
  int new_data_size = outsize;		/* initial size */
  int new_pos = 0;
  int written = 0;
  bool error = false;
  uint8_t *data = (uint8_t*)inbuf;


  if( !decoder || LZ_decompress_errno( decoder ) != LZ_ok )
    { LZ_decompress_close( decoder ); return 0; }

  while( true )
    {
    int rd;
    if( LZ_decompress_write_size( decoder ) > 0 )
      {
      if( written < insize )
        {
        const int wr = LZ_decompress_write( decoder, data + written, insize - written );
     //   printf("write=%d written=%d left=%d\n", wr, written, insize - written);
        if( wr < 0 ) { error = true; break; }
        written += wr;
        }
      if( written >= insize ) LZ_decompress_finish( decoder );
      }
    rd = LZ_decompress_read( decoder, new_data + new_pos, new_data_size - new_pos );
  //  printf("read=%d new_pos=%d\n", rd, new_pos);
    if( rd < 0 ) { error = true; break; }
    new_pos += rd;
    if( LZ_decompress_finished( decoder ) == 1 ) break;
    }

  if( LZ_decompress_close( decoder ) < 0 ) error = true;

  return new_pos;
}

#endif



#ifndef BENCH_REMOVE_LZMA

#include <string.h>
#include "misc/7-zip/Alloc.h"
#include "misc/7-zip/Lzma2Dec.h"
#include "misc/7-zip/Lzma2DecMt.h"
#include "misc/7-zip/Lzma2Enc.h"

int64_t lzbench_lzma_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    CLzma2EncProps props;
    CLzma2EncHandle enc;
    SRes res;
    SizeT out_len = outsize;

    Lzma2EncProps_Init(&props);
    props.lzmaProps.level = codec_options->level;
    props.numTotalThreads = codec_options->threads;

    enc = Lzma2Enc_Create(&g_Alloc, &g_Alloc);
    if (enc == NULL) return -1;

    res = Lzma2Enc_SetProps(enc, &props);
    if (res != SZ_OK) {
        Lzma2Enc_Destroy(enc);
        return -2;
    }

    outbuf[0] = Lzma2Enc_WriteProperties(enc);;

    res = Lzma2Enc_Encode2(enc, NULL, (Byte*)outbuf + 1, &out_len, NULL, (const Byte*)inbuf, insize, NULL);
    Lzma2Enc_Destroy(enc);
    if (res != SZ_OK) return -3;

    return out_len + 1;
}

// ISeqInStream implementation for an in-memory buffer
typedef struct {
    ISeqInStream vt;
    const Byte *data;
    size_t size;
} CBufInStream;

static SRes MyRead(void *p, void *buf, size_t *size) {
    CBufInStream *s = (CBufInStream *)p;
    size_t toRead = *size;
    if (toRead > s->size) {
        toRead = s->size;
    }
    memcpy(buf, s->data, toRead);
    s->data += toRead;
    s->size -= toRead;
    *size = toRead;
    return SZ_OK;
}

// ISeqOutStream implementation for an in-memory buffer
typedef struct {
    ISeqOutStream vt;
    Byte *data;
    size_t size;
} CBufOutStream;

static size_t MyWrite(void *p, const void *buf, size_t size) {
    CBufOutStream *s = (CBufOutStream *)p;
    size_t toWrite = size;
    if (toWrite > s->size) {
        toWrite = s->size;
    }
    memcpy(s->data, buf, toWrite);
    s->data += toWrite;
    s->size -= toWrite;
    return toWrite;
}

int64_t lzbench_lzma_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options) {
    CLzma2DecMtHandle dec_handle;
    CLzma2DecMtProps props;
    UInt64 out_size_defined = (UInt64)outsize;
    UInt64 in_processed = 0;
    int is_mt = 0;
    SRes res;
    Byte prop_byte;

    if (insize == 0) return -1;
    prop_byte = (Byte)inbuf[0];

    CBufInStream inStream;
    inStream.vt.Read = (SRes (*)(ISeqInStreamPtr, void*, size_t*))MyRead;
    inStream.data = (const Byte *)inbuf + 1;
    inStream.size = insize - 1;

    CBufOutStream outStream;
    outStream.vt.Write = (size_t (*)(ISeqOutStreamPtr, const void*, size_t))MyWrite;
    outStream.data = (Byte *)outbuf;
    outStream.size = outsize;

    dec_handle = Lzma2DecMt_Create(&g_Alloc, &g_Alloc);
    if (!dec_handle) return -2;

    Lzma2DecMtProps_Init(&props);
    props.numThreads = codec_options->threads;

    res = Lzma2DecMt_Decode(
        dec_handle,
        prop_byte,
        &props,
        &outStream.vt,
        &out_size_defined,
        1,
        &inStream.vt,
        &in_processed,
        &is_mt,
        NULL
    );

    Lzma2DecMt_Destroy(dec_handle);
    if (res != SZ_OK) return -3;

    return (int64_t)(outsize - outStream.size);
}

#endif



#ifndef BENCH_REMOVE_LZO
#include "lz/lzo/lzo1.h"
#include "lz/lzo/lzo1a.h"
#include "lz/lzo/lzo1b.h"
#include "lz/lzo/lzo1c.h"
#include "lz/lzo/lzo1f.h"
#include "lz/lzo/lzo1x.h"
#include "lz/lzo/lzo1y.h"
#include "lz/lzo/lzo1z.h"
#include "lz/lzo/lzo2a.h"

char* lzbench_lzo_init(size_t, size_t, size_t)
{
    lzo_init();

    return (char*) malloc(LZO1B_999_MEM_COMPRESS);
}

void lzbench_lzo_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_lzo1_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    if (!codec_options->work_mem)
        return 0;

    if (codec_options->level == 99)
        res = lzo1_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)codec_options->work_mem);
    else
        res = lzo1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)codec_options->work_mem);

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1a_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;

    if (!codec_options->work_mem)
        return 0;

    if (codec_options->level == 99)
        res = lzo1a_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)codec_options->work_mem);
    else
        res = lzo1a_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)codec_options->work_mem);

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1a_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1a_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1b_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    switch (codec_options->level)
    {
        default:
        case 1: res = lzo1b_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 2: res = lzo1b_2_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 3: res = lzo1b_3_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 4: res = lzo1b_4_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 5: res = lzo1b_5_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 6: res = lzo1b_6_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 7: res = lzo1b_7_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 8: res = lzo1b_8_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 9: res = lzo1b_9_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 99: res = lzo1b_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 999: res = lzo1b_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
    }

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1b_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1b_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1c_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    switch (codec_options->level)
    {
        default:
        case 1: res = lzo1c_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 2: res = lzo1c_2_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 3: res = lzo1c_3_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 4: res = lzo1c_4_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 5: res = lzo1c_5_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 6: res = lzo1c_6_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 7: res = lzo1c_7_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 8: res = lzo1c_8_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 9: res = lzo1c_9_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 99: res = lzo1c_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 999: res = lzo1c_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
    }

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1c_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1c_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1f_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    if (codec_options->level == 999)
        res = lzo1f_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    else
        res = lzo1f_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1f_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1f_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1x_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    switch (codec_options->level)
    {
        default:
        case 1: res = lzo1x_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 11: res = lzo1x_1_11_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 12: res = lzo1x_1_12_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 15: res = lzo1x_1_15_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
        case 999: res = lzo1x_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
    }

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1x_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1x_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1y_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    if (codec_options->level == 999)
        res = lzo1y_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    else
        res = lzo1y_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1y_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1y_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

int64_t lzbench_lzo1z_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    res = lzo1z_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo1z_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo1z_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}


int64_t lzbench_lzo2a_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint lzo_complen = 0;
    int res;
    char* workmem = codec_options->work_mem;
    if (!workmem)
        return 0;

    res = lzo2a_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);

    if (res != LZO_E_OK) return 0;

    return lzo_complen;
}

int64_t lzbench_lzo2a_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzo_uint decomplen = 0;

    if (lzo2a_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

    return decomplen;
}

#endif



#ifndef BENCH_REMOVE_LZSSE
#include "lzsse/lzsse2/lzsse2.h"

char* lzbench_lzsse2_init(size_t insize, size_t, size_t)
{
    return (char*) LZSSE2_MakeOptimalParseState(insize);
}

void lzbench_lzsse2_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE2_FreeOptimalParseState((LZSSE2_OptimalParseState*) workmem);
}

int64_t lzbench_lzsse2_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (!codec_options->work_mem) return 0;

    return LZSSE2_CompressOptimalParse((LZSSE2_OptimalParseState*) codec_options->work_mem, inbuf, insize, outbuf, outsize, codec_options->level);
}

int64_t lzbench_lzsse2_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZSSE2_Decompress(inbuf, insize, outbuf, outsize);
}


#include "lzsse/lzsse4/lzsse4.h"

char* lzbench_lzsse4_init(size_t insize, size_t, size_t)
{
    return (char*) LZSSE4_MakeOptimalParseState(insize);
}

void lzbench_lzsse4_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE4_FreeOptimalParseState((LZSSE4_OptimalParseState*) workmem);
}

int64_t lzbench_lzsse4_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (!codec_options->work_mem) return 0;

    return LZSSE4_CompressOptimalParse((LZSSE4_OptimalParseState*) codec_options->work_mem, inbuf, insize, outbuf, outsize, codec_options->level);
}

int64_t lzbench_lzsse4_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZSSE4_Decompress(inbuf, insize, outbuf, outsize);
}

char* lzbench_lzsse4fast_init(size_t, size_t, size_t)
{
    return (char*) LZSSE4_MakeFastParseState();
}

void lzbench_lzsse4fast_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE4_FreeFastParseState((LZSSE4_FastParseState*) workmem);
}

int64_t lzbench_lzsse4fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (!codec_options->work_mem) return 0;

    return LZSSE4_CompressFast((LZSSE4_FastParseState*) codec_options->work_mem, inbuf, insize, outbuf, outsize);
}


#include "lzsse/lzsse8/lzsse8.h"

char* lzbench_lzsse8_init(size_t insize, size_t, size_t)
{
    return (char*) LZSSE8_MakeOptimalParseState(insize);
}

void lzbench_lzsse8_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE8_FreeOptimalParseState((LZSSE8_OptimalParseState*) workmem);
}

int64_t lzbench_lzsse8_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (!codec_options->work_mem) return 0;

    return LZSSE8_CompressOptimalParse((LZSSE8_OptimalParseState*) codec_options->work_mem, inbuf, insize, outbuf, outsize, codec_options->level);
}

int64_t lzbench_lzsse8_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return LZSSE8_Decompress(inbuf, insize, outbuf, outsize);
}

char* lzbench_lzsse8fast_init(size_t, size_t, size_t)
{
    return (char*) LZSSE8_MakeFastParseState();
}

void lzbench_lzsse8fast_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE8_FreeFastParseState((LZSSE8_FastParseState*) workmem);
}

int64_t lzbench_lzsse8fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    if (!codec_options->work_mem) return 0;

    return LZSSE8_CompressFast((LZSSE8_FastParseState*) codec_options->work_mem, inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_QUICKLZ
#include "quicklz/quicklz151b7.h"

int64_t lzbench_quicklz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int64_t res;
    qlz_state_compress* state = (qlz_state_compress*) calloc(1, std::max(qlz151_get_setting_3(1),std::max(qlz151_get_setting_1(1), qlz151_get_setting_2(1))));
    if (!state)
        return 0;


    switch (codec_options->level)
    {
        default:
        case 1:	res = qlz151_compress_1(inbuf, outbuf, insize, (qlz_state_compress*)state); break;
        case 2:	res = qlz151_compress_2(inbuf, outbuf, insize, (qlz_state_compress*)state); break;
        case 3:	res = qlz151_compress_3(inbuf, outbuf, insize, (qlz_state_compress*)state); break;
    }

    free(state);
    return res;
}

int64_t lzbench_quicklz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int64_t res;
    qlz_state_compress* dstate = (qlz_state_compress*) calloc(1, std::max(qlz151_get_setting_3(2),std::max(qlz151_get_setting_1(2), qlz151_get_setting_2(2))));
    if (!dstate)
        return 0;

    switch (codec_options->level)
    {
        default:
        case 1: res = qlz151_decompress_1(inbuf, outbuf, (qlz_state_decompress*)dstate); break;
        case 2: res = qlz151_decompress_2(inbuf, outbuf, (qlz_state_decompress*)dstate); break;
        case 3: res = qlz151_decompress_3(inbuf, outbuf, (qlz_state_decompress*)dstate); break;
    }

    free(dstate);
    return res;
}

#endif



#ifndef BENCH_REMOVE_SNAPPY
#include "snappy/snappy.h"

int64_t lzbench_snappy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    snappy::RawCompress(inbuf, insize, outbuf, &outsize);
    return outsize;
}

int64_t lzbench_snappy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    snappy::RawUncompress(inbuf, insize, outbuf);
    return outsize;
}

#endif



#ifndef BENCH_REMOVE_TORNADO
#include "tornado/tor_test.h"

int64_t lzbench_tornado_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return tor_compress(codec_options->level, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, outsize);
}

int64_t lzbench_tornado_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return tor_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_UCL
#include "ucl/ucl.h"

int64_t lzbench_ucl_nrv2b_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ucl_uint complen;
    int res = ucl_nrv2b_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, codec_options->level, NULL, NULL);

    if (res != UCL_E_OK) return 0;
    return complen;
}

int64_t lzbench_ucl_nrv2b_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ucl_uint decomplen;
    int res = ucl_nrv2b_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL);

    if (res != UCL_E_OK) return 0;
    return decomplen;
}

int64_t lzbench_ucl_nrv2d_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ucl_uint complen;
    int res = ucl_nrv2d_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, codec_options->level, NULL, NULL);

    if (res != UCL_E_OK) return 0;
    return complen;
}

int64_t lzbench_ucl_nrv2d_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ucl_uint decomplen;
    int res = ucl_nrv2d_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL);

    if (res != UCL_E_OK) return 0;
    return decomplen;
}

int64_t lzbench_ucl_nrv2e_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ucl_uint complen;
    int res = ucl_nrv2e_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, codec_options->level, NULL, NULL);

    if (res != UCL_E_OK) return 0;
    return complen;
}

int64_t lzbench_ucl_nrv2e_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    ucl_uint decomplen;
    int res = ucl_nrv2e_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL);

    if (res != UCL_E_OK) return 0;
    return decomplen;
}

#endif



#ifndef BENCH_REMOVE_ZLIB
#include "zlib/zlib.h"

int64_t lzbench_zlib_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    uLongf zcomplen = outsize;
    int err = compress2((uint8_t*)outbuf, &zcomplen, (uint8_t*)inbuf, insize, codec_options->level);
    if (err != Z_OK)
        return 0;
    return zcomplen;
}

int64_t lzbench_zlib_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    uLongf zdecomplen = outsize;
    int err = uncompress((uint8_t*)outbuf, &zdecomplen, (uint8_t*)inbuf, insize);
    if (err != Z_OK)
        return 0;
    return zdecomplen;
}

#endif



#ifndef BENCH_REMOVE_ZLIB_NG

#undef z_const
#undef Z_NULL

#define in_func zlibng_in_func
#include "lz/zlib-ng/zlib-ng.h"
#undef in_func

int64_t lzbench_zlib_ng_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    size_t zcomplen = outsize;
    int err = zng_compress2((uint8_t*)outbuf, &zcomplen, (uint8_t*)inbuf, insize, codec_options->level);
    if (err != Z_OK)
        return 0;
    return zcomplen;
}

int64_t lzbench_zlib_ng_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    size_t zdecomplen = outsize;
    int err = zng_uncompress((uint8_t*)outbuf, &zdecomplen, (uint8_t*)inbuf, insize);
    if (err != Z_OK)
        return 0;
    return zdecomplen;
}

#endif



#if !defined(BENCH_REMOVE_SLZ) && !defined(BENCH_REMOVE_ZLIB)
extern "C"
{
    #include "slz/src/slz.h"
}

int64_t lzbench_slz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    struct slz_stream strm;
    size_t outlen = 0;
    size_t window = 8192 << ((codec_options->level & 3) * 2);
    size_t len;
    size_t blk;

    if (codec_options->additional_param == 0)
        slz_init(&strm, !!codec_options->level, SLZ_FMT_GZIP);
    else if (codec_options->additional_param == 1)
        slz_init(&strm, !!codec_options->level, SLZ_FMT_ZLIB);
    else
        slz_init(&strm, !!codec_options->level, SLZ_FMT_DEFLATE);

    do {
        blk = std::min(insize, window);

        len = slz_encode(&strm, outbuf, inbuf, blk, insize > blk);
        outlen += len;
        outbuf += len;
        inbuf += blk;
        insize -= blk;
    } while (insize > 0);

    outlen += slz_finish(&strm, outbuf);
    return outlen;
}

/* uses zlib to perform the decompression */
int64_t lzbench_slz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    z_stream stream;
    int err;

    stream.zalloc    = NULL;
    stream.zfree     = NULL;

    stream.next_in   = (unsigned char *)inbuf;
    stream.avail_in  = insize;
    stream.next_out  = (unsigned char *)outbuf;
    stream.avail_out = outsize;

    outsize = 0;

    if (codec_options->additional_param == 0)      // gzip
        err = inflateInit2(&stream, 15 + 16);
    else if (codec_options->additional_param == 1) // zlip
        err = inflateInit2(&stream, 15);
    else                  // deflate
        err = inflateInit2(&stream, -15);

    if (err == Z_OK) {
        if (inflate(&stream, Z_FINISH) == Z_STREAM_END)
            outsize = stream.total_out;
        inflateEnd(&stream);
    }
    return outsize;
}
#endif



#ifndef BENCH_REMOVE_XZ
#include "lz/xz/src/liblzma/api/lzma.h"

int64_t lzbench_xz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret;

    // Prepare multithreaded compression settings
    lzma_mt mt_options = {0};

    // Compression level: default to 6 if codec_options->level is unset
    mt_options.preset = (codec_options && codec_options->level >= 0 && codec_options->level <= 9)
                          ? (uint32_t)codec_options->level
                          : LZMA_PRESET_DEFAULT;

    // Check type (CRC64 is default and common)
    mt_options.check = LZMA_CHECK_NONE;
    //mt_options.check = LZMA_CHECK_CRC32;

    // Number of threads
    mt_options.threads = codec_options->threads;
    mt_options.block_size = 0;

    // lzma_stream_encoder_mt supports .xz format with multithreading
    ret = lzma_stream_encoder_mt(&strm, &mt_options);
    if (ret != LZMA_OK) {
        return -1;
    }

    strm.next_in = (const uint8_t *)inbuf;
    strm.avail_in = insize;
    strm.next_out = (uint8_t *)outbuf;
    strm.avail_out = outsize;

    // Compress in one shot
    ret = lzma_code(&strm, LZMA_FINISH);
    if (ret != LZMA_STREAM_END) {
        lzma_end(&strm);
        return -2;
    }

    size_t compressed_size = strm.total_out;

    lzma_end(&strm);
    return (int64_t)compressed_size;
}

int64_t lzbench_xz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret;

    // Configure multithreaded decoder options
    lzma_mt mt_options = {0};
    mt_options.threads = codec_options->threads;

    // Use unlimited memory for decoder
    mt_options.memlimit_stop = UINT64_MAX;
    mt_options.flags = LZMA_CONCATENATED | LZMA_IGNORE_CHECK;

    // Use multithreaded decoder (available in XZ Utils 5.4.0+)
    ret = lzma_stream_decoder_mt(&strm, &mt_options);
    if (ret != LZMA_OK) {
        lzma_end(&strm);
        return -1;
    }

    strm.next_in = (const uint8_t *)inbuf;
    strm.avail_in = insize;
    strm.next_out = (uint8_t *)outbuf;
    strm.avail_out = outsize;

    // Perform decompression
    ret = lzma_code(&strm, LZMA_FINISH);
    if (ret != LZMA_STREAM_END) {
        lzma_end(&strm);
        return -2;
    }

    size_t decompressed_size = strm.total_out;
    lzma_end(&strm);
    return (int64_t)decompressed_size;
}

#endif // BENCH_REMOVE_XZ



#ifndef BENCH_REMOVE_ZLING
#include "lz/libzling/libzling.h"

namespace baidu {
namespace zling {

struct MemInputter: public baidu::zling::Inputter {
    MemInputter(uint8_t* buffer, size_t buflen) :
        m_buffer(buffer),
        m_buflen(buflen),
        m_total_read(0) {}

    size_t GetData(unsigned char* buf, size_t len) {
        if (len > m_buflen - m_total_read)
            len = m_buflen - m_total_read;

        memcpy(buf, m_buffer + m_total_read, len);
        m_total_read += len;
        return len;
    }
    bool   IsEnd() { return m_total_read >= m_buflen; }
    bool   IsErr() { return false; }
    size_t GetInputSize() { return m_total_read; }

private:
    uint8_t* m_buffer;
    size_t m_buflen, m_total_read;
};

struct MemOutputter : public baidu::zling::Outputter {
    MemOutputter(uint8_t* buffer, size_t buflen) :
        m_buffer(buffer),
        m_buflen(buflen),
        m_total_write(0) {}

    size_t PutData(unsigned char* buf, size_t len) {
        if (len > m_buflen - m_total_write)
            len = m_buflen - m_total_write;

        memcpy(m_buffer + m_total_write, buf, len);
        m_total_write += len;
        return len;
    }
    bool   IsErr() { return m_total_write > m_buflen; }
    size_t GetOutputSize() { return m_total_write; }

private:
    FILE*  m_fp;
    uint8_t* m_buffer;
    size_t m_buflen, m_total_write;
};

}  // namespace zling
}  // namespace baidu

int64_t lzbench_zling_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    baidu::zling::MemInputter  inputter((uint8_t*)inbuf, insize);
    baidu::zling::MemOutputter outputter((uint8_t*)outbuf, outsize);
    baidu::zling::Encode(&inputter, &outputter, NULL, codec_options->level);

    return outputter.GetOutputSize();
}

int64_t lzbench_zling_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    baidu::zling::MemInputter  inputter((uint8_t*)inbuf, insize);
    baidu::zling::MemOutputter outputter((uint8_t*)outbuf, outsize);
    baidu::zling::Decode(&inputter, &outputter);

    return outputter.GetOutputSize();
}

#endif



#ifndef BENCH_REMOVE_ZSTD
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd/lib/zstd.h"

typedef struct {
    ZSTD_CCtx* cctx;
    ZSTD_DCtx* dctx;
    ZSTD_CDict* cdict;
    ZSTD_parameters zparams;
    ZSTD_customMem cmem;
} zstd_params_s;

char* lzbench_zstd_init(size_t insize, size_t level, size_t windowLog)
{
    zstd_params_s* zstd_params = (zstd_params_s*) malloc(sizeof(zstd_params_s));
    if (!zstd_params) return NULL;
    zstd_params->cctx = ZSTD_createCCtx();
    zstd_params->dctx = ZSTD_createDCtx();
#if 1
    zstd_params->cdict = NULL;
#else
    zstd_params->zparams = ZSTD_getParams(level, insize, 0);
    zstd_params->cmem = { NULL, NULL, NULL };
    if (windowLog && zstd_params->zparams.cParams.windowLog > windowLog) {
        zstd_params->zparams.cParams.windowLog = windowLog;
        zstd_params->zparams.cParams.chainLog = windowLog + ((zstd_params->zparams.cParams.strategy == ZSTD_btlazy2) | (zstd_params->zparams.cParams.strategy == ZSTD_btopt) | (zstd_params->zparams.cParams.strategy == ZSTD_btopt2));
    }
    zstd_params->cdict = ZSTD_createCDict_advanced(NULL, 0, zstd_params->zparams, zstd_params->cmem);
#endif

    return (char*) zstd_params;
}

void lzbench_zstd_deinit(char* workmem)
{
    zstd_params_s* zstd_params = (zstd_params_s*) workmem;
    if (!zstd_params) return;
    if (zstd_params->cctx) ZSTD_freeCCtx(zstd_params->cctx);
    if (zstd_params->dctx) ZSTD_freeDCtx(zstd_params->dctx);
    if (zstd_params->cdict) ZSTD_freeCDict(zstd_params->cdict);
    free(workmem);
}

int64_t lzbench_zstd_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    size_t res;
    int windowLog = codec_options->additional_param;
    zstd_params_s* zstd_params = (zstd_params_s*) codec_options->work_mem;

    if (!zstd_params || !zstd_params->cctx) return 0;

#if 1
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, codec_options->level);
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_contentSizeFlag, 1);

    if (codec_options->threads > 1)
        ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_nbWorkers, codec_options->threads);

    if (windowLog) {
        size_t currentWindowLog = ZSTD_getParams(codec_options->level, insize, 0).cParams.windowLog;
        if (currentWindowLog > windowLog) {
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, windowLog);
            int strategy = ZSTD_getParams(codec_options->level, insize, 0).cParams.strategy;
            int chainLog = windowLog + ((strategy == ZSTD_btlazy2) || (strategy == ZSTD_btopt) || (strategy == ZSTD_btultra));
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, chainLog);
        }
    }

    res = ZSTD_compress2(zstd_params->cctx, outbuf, outsize, inbuf, insize);
#else
    if (!zstd_params->cdict) return 0;
    res = ZSTD_compress_usingCDict(zstd_params->cctx, outbuf, outsize, inbuf, insize, zstd_params->cdict);
#endif
    if (ZSTD_isError(res)) return res;

    return res;
}

int64_t lzbench_zstd_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    zstd_params_s* zstd_params = (zstd_params_s*) codec_options->work_mem;
    if (!zstd_params || !zstd_params->dctx) return 0;

    return ZSTD_decompressDCtx(zstd_params->dctx, outbuf, outsize, inbuf, insize);
}

char* lzbench_zstd_LDM_init(size_t insize, size_t level, size_t windowLog)
{
    zstd_params_s* zstd_params = (zstd_params_s*) lzbench_zstd_init(insize, level, windowLog);
    if (!zstd_params) return NULL;
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_enableLongDistanceMatching, 1);
    return (char*) zstd_params;
}

int64_t lzbench_zstd_LDM_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    zstd_params_s* zstd_params = (zstd_params_s*) codec_options->work_mem;
    if (!zstd_params || !zstd_params->cctx) return 0;
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_enableLongDistanceMatching, 1);
    return lzbench_zstd_compress(inbuf, insize, outbuf, outsize, codec_options);
}

/* zstd tuned — 自定义参数变体，填补 zstd -2 和 -4 之间的 Pareto 空位 */
char* lzbench_zstd_tuned_init(size_t insize, size_t level, size_t windowLog)
{
    zstd_params_s* zstd_params = (zstd_params_s*) malloc(sizeof(zstd_params_s));
    if (!zstd_params) return NULL;
    zstd_params->cctx = ZSTD_createCCtx();
    zstd_params->dctx = ZSTD_createDCtx();
    zstd_params->cdict = NULL;
    (void)insize;
    (void)level;
    (void)windowLog;
    return (char*) zstd_params;
}

int64_t lzbench_zstd_tuned_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    zstd_params_s* zstd_params = (zstd_params_s*) codec_options->work_mem;
    if (!zstd_params || !zstd_params->cctx) return 0;

    int level = (int)codec_options->level;

    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 1);
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_contentSizeFlag, 1);
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_checksumFlag, 0);

    if (codec_options->threads > 1)
        ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_nbWorkers, (int)codec_options->threads);

    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);

    /* 每个 level 对应一组调优参数
     *
     * === 关键 Pareto 最优点 (Silesia corpus, i5-13420H) ===
     * L90:  b5(lazy2),  w=20, h=20, c=18       → 126 MB/s, 1027 MB/s, 29.58%  严格支配 zstd -5
     * L102: b6(btlazy2), w=22, h=20, c=18       →  86 MB/s, 1060 MB/s, 28.55%  比率优先 (用户选择)
     * L140: b5(lazy2),  w=22, h=21, c=20, s=4   →  88 MB/s,  930 MB/s, 28.88%  速度优先
     * L148: b5(lazy2),  w=22, h=21, c=20, s=6   →  57 MB/s, 1024 MB/s, 28.40%  比率突破
     * L144: b5(lazy2),  w=22, h=22, c=16, s=6   →  41 MB/s,  922 MB/s, 28.30%  极限比率
     *
     * zstd -5 参考: 133 MB/s, 1021 MB/s, 29.60%
     */
    switch (level) {
        case 1: /* 接近 zstd -2: dfast, hash=17, chain=16, search=1 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 2: /* 中间配置: dfast, hash=17, chain=17, search=1 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 3: /* 偏向比率: dfast, hash=18, chain=17, search=1 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 4: /* 偏向比率: dfast, hash=17, chain=18, search=1 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 5: /* 深度搜索: dfast, hash=17, chain=17, search=2 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 2);
            break;
        case 6: /* lazy 策略: lazy, hash=17, chain=17, search=1 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 7: /* lazy2 策略: lazy2, hash=17, chain=17, search=2 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy2);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 2);
            break;
        case 8: /* dfast, hash=18, chain=18, search=1 — 组合 L3+L4 最优参数 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 9: /* dfast, hash=17, chain=18, search=2 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 2);
            break;
        case 10: /* dfast, hash=18, chain=17, search=2 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 2);
            break;
        case 11: /* greedy, hash=18, chain=17, search=1 — 更快策略 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_greedy);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 12: /* lazy, hash=18, chain=18, search=1 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 13: /* dfast, hash=18, chain=19, search=1 — 极限 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 14: /* dfast, hash=19, chain=18, search=1 — 极限 hash */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 15: /* dfast, hash=19, chain=19, search=1 — 最大 hash/chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 16: /* dfast, hash=18, chain=18, search=2 — L8 加深搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 2);
            break;
        /* === 第二阶段：进攻 <30% 比率 === */
        case 17: /* lazy, hash=18, chain=19, search=1 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 18: /* lazy2, hash=18, chain=18, search=2 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy2);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 2);
            break;
        case 19: /* lazy2, hash=19, chain=19, search=1 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy2);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 20: /* btlazy2, hash=18, chain=18, search=1 — 二叉树搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_btlazy2);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 21: /* dfast, hash=19, chain=19, search=1, window=21 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            break;
        case 22: /* greedy, hash=18, chain=18, search=1 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_greedy);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 23: /* lazy, hash=18, chain=18, search=1, window=21 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            break;
        case 24: /* lazy2, hash=18, chain=18, search=1, window=21 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy2);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            break;
        /* === 第三阶段：优化 lazy2 速度 / 探索 greedy 极限 === */
        case 25: /* lazy2, hash=17, chain=18, s=1 — 减少 hash 提速 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy2);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 26: /* lazy2, hash=18, chain=17, s=1 — 减少 chain 提速 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy2);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 27: /* greedy, hash=18, chain=19, s=1 — 极限比率 greedy */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_greedy);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 28: /* greedy, hash=19, chain=19, s=1 — 最大 greedy */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_greedy);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 29: /* lazy, hash=18, chain=20, s=1 — 深化 lazy */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 30: /* lazy2, hash=18, chain=19, s=1 — 深化 lazy2 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy2);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 31: /* lazy2, hash=17, chain=19, s=1 — 少 hash 多 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy2);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        /* === 第四阶段：精调 lazy2 突破 30% + 100MB/s === */
        case 32: /* lazy2, h=17, c=17, s=1 — 最小 lazy2 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy2);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 33: /* lazy2, h=16, c=18, s=1 — 极小 hash */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy2);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 16);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 34: /* lazy2, h=18, c=17, s=0 — 无搜索(lazy2基线) */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_lazy2);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 0);
            break;
        case 35: /* dfast, h=19, c=19, s=2 — dfast 极限 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 2);
            break;
        /* === 第五阶段：等级继承 — 高 level 基线 + 参数覆盖 === */
        case 36: /* base=3(lazy), w=20, c=19 — level 3 深化 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 3);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            break;
        case 37: /* base=4(lazy2), w=20 — level 4 小窗口提速 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 4);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            break;
        case 38: /* base=4(lazy2), w=20, c=19 — level 4 深化 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 4);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            break;
        case 39: /* base=3(lazy), w=20, h=19, c=19 — level 3 最大 hash/chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 3);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            break;
        case 40: /* base=4(lazy2), w=20, h=19, c=19 — level 4 最大 hash/chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 4);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            break;
        case 41: /* base=5(lazy2), w=20 — level 5 小窗口 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            break;
        case 42: /* base=5(lazy2), w=20, h=18, c=18 — level 5 减 hash/chain 提速 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        /* === 第六阶段：L41 深化 — 追平 zstd -5 比率 === */
        case 43: /* base=5(lazy2), w=20, c=20 — 深化 chain 提比率 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            break;
        case 44: /* base=5(lazy2), w=20, h=19, c=19 — 显式最大 hash/chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            break;
        case 45: /* base=5(lazy2), w=20, h=18, c=20 — 多 chain 少 hash */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            break;
        case 46: /* base=6(btlazy2), w=20 — 二叉树搜索 + 小窗口 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            break;
        case 47: /* base=6(btlazy2), w=20, h=18, c=18 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 48: /* base=5(lazy2), w=21 — 略大窗口 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            break;
        /* === 第七阶段：base=6 降 hash/chain 提速，base=5 加 search === */
        case 49: /* base=6(btlazy2), w=20, h=16, c=16 — 极小 hash/chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 16);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            break;
        case 50: /* base=6(btlazy2), w=20, h=17, c=17 — 适中 hash/chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 51: /* base=5(lazy2), w=20, s=1 — L41 + 搜索深度 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 52: /* base=6(btlazy2), w=20, h=16, c=17 — 少 hash 多 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 16);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 53: /* base=6(btlazy2), w=20, h=17, c=16 — 多 hash 少 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            break;
        case 54: /* base=5(lazy2), w=21, h=17 — L48 减 hash 提速 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            break;
        /* === 第八阶段：L50 精调 — 追 29.60% + 守 100 MB/s === */
        case 55: /* base=6(btlazy2), w=20, h=17, c=18 — L50 + 深化 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 56: /* base=6(btlazy2), w=20, h=16, c=18 — 极小 hash + 深 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 16);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 57: /* base=6(btlazy2), w=21 — 大窗口 + btlazy2 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            break;
        case 58: /* base=6(btlazy2), w=21, h=17, c=17 — L50 大窗口版 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 59: /* base=6(btlazy2), w=20, h=18, c=17 — 多 hash L50 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 60: /* base=6(btlazy2), w=20, h=17, c=19 — L50 深挖 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            break;
        /* === 第九阶段：进攻更低比率 — base=7/8 + 大窗口 === */
        case 61: /* base=7, w=20 — level 7 + 小窗口 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 7);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            break;
        case 62: /* base=7, w=20, h=17, c=17 — 控制开销 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 7);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 63: /* base=7, w=21 — 扩大窗口 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 7);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            break;
        case 64: /* base=6, w=22 — 更大窗口压比率 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            break;
        case 65: /* base=6, w=22, h=17, c=17 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 66: /* base=6, w=20, h=19, c=19 — 最大 hash/chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            break;
        case 67: /* base=8, w=20 — level 8 + 小窗口 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 8);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            break;
        case 68: /* base=8, w=20, h=17, c=17 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 8);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        /* === 第十阶段：searchLog + 精调 hash/chain 比例 === */
        case 69: /* b6, w=20, h=19, c=18 — L66 减 chain 提压速 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 70: /* b6, w=20, h=18, c=19 — 少 hash 深 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            break;
        case 71: /* b6, w=20, h=17, c=19, s=1 — L60 + 搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 72: /* b6, w=20, h=18, c=18, s=2 — L47 + 深度搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 2);
            break;
        case 73: /* b6, w=21, h=17, c=18 — L57/L58 混合 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 74: /* b6, w=20, h=18, c=17, s=1 — L59 + 搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 75: /* b6, w=20, h=17, s=2 — 少 chain 多 search */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 2);
            break;
        case 76: /* b6, w=20, h=18, c=19, s=1 — 深 chain + 搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        /* === 第十一阶段：极限非对称 + minMatch + targetLength === */
        case 77: /* b6, w=20, h=19, c=17 — 极端少 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 78: /* b6, w=20, h=20, c=18 — hashLog 20 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 79: /* b6, w=20, h=19, c=16 — 极小 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            break;
        case 80: /* b6, w=20, h=19, c=18, minMatch=4 — L69 + 更大 minMatch */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_minMatch, 4);
            break;
        case 81: /* b6, w=20, h=19, c=18, targetLen=8 — L69 + targetLength */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_targetLength, 8);
            break;
        case 82: /* b6, w=20, h=19, c=18, targetLen=16 — 更长目标匹配 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_targetLength, 16);
            break;
        case 83: /* b6, w=20, h=18, c=20 — 极端深 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            break;
        case 84: /* b6, w=20, h=19, c=18, minMatch=3 — 最小 minMatch */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_minMatch, 3);
            break;
        /* === 第十二阶段：h=20 精调 + w=19 + base=5 极限 === */
        case 85: /* b6, w=20, h=20, c=17 — L78 减 chain 提压速 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 86: /* b6, w=20, h=20, c=16 — 极小 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            break;
        case 87: /* b6, w=19, h=18, c=18 — 更小窗口 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 88: /* b6, w=19, h=19, c=18 — 小窗口大 hash */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 89: /* b5(lazy2), w=20, h=19, c=19 — lazy2 极限 hash/chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            break;
        case 90: /* b5(lazy2), w=20, h=20, c=18 — lazy2 + h20 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 91: /* b6, w=20, h=20, c=19 — h=20 深 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            break;
        case 92: /* b6, w=20, h=20, s=1 — 纯 hash + 搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        case 93: /* b6, w=20, h=17, c=20 — 最大不对称 h17/c20 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            break;
        case 94: /* b6, w=21, h=19, c=18 — L57 + 大 hash */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 95: /* b6, w=21, h=20, c=18 — 大窗口 + 最大 hash */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 96: /* b6, w=20, h=20, c=18, s=1 — L78 + 搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
        /* === 第十三阶段：L90/L95 精调 === */
        case 97: /* b5(lazy2), w=20, h=20, c=17 — L90 减 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 98: /* b5(lazy2), w=20, h=20, c=19 — L90 深 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            break;
        case 99: /* b5(lazy2), w=21, h=20, c=18 — L90 大窗口 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 100: /* b6, w=21, h=20, c=17 — L95 减 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 101: /* b6, w=21, h=20, c=19 — L95 深 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 19);
            break;
        case 102: /* b6, w=22, h=20, c=18 — 超大窗口 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 103: /* b6, w=21, h=21, c=18 — hashLog=21 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 104: /* b6, w=21, h=20, c=20 — 全极限 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            break;
        /* === 第十四阶段：L102 加速 — 降 hash/chain/window === */
        case 105: /* b6, w=22, h=19, c=18 — L102 减 hash */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 106: /* b6, w=22, h=20, c=17 — L102 减 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 107: /* b6, w=22, h=20, c=16 — L102 极小 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            break;
        case 108: /* b6, w=22, h=19, c=17 — L102 减 hash + chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 109: /* b6, w=21, h=21, c=18 — w=21 + 极限 hash */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 110: /* b6, w=21, h=20, c=16 — w=21 极小 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            break;
        case 111: /* b6, w=22, h=19, c=16 — 最少 hash + chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 19);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            break;
        case 112: /* b6, w=21, h=21, c=17 — h=21 + 少 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        /* === 第十五阶段：L102 加速 — lazy2 策略 + 大 hash === */
        case 113: /* b5(lazy2), w=22, h=21, c=18 — lazy2 + 超大 hash */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 114: /* b5(lazy2), w=22, h=20, c=18 — lazy2 直接替换 L102 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 115: /* b5(lazy2), w=22, h=21, c=17 — lazy2 + 大 hash + 少 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 116: /* b6(btlazy2), w=22, h=21, c=17 — L102 + h=21 + 减 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 117: /* b5(lazy2), w=22, h=20, c=17 — lazy2 + 少 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            break;
        case 118: /* b5(lazy2), w=21, h=20, c=18 — lazy2 + w=21 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 119: /* b6(btlazy2), w=22, h=21, c=18 — L102 + h=21 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 120: /* b6(btlazy2), w=22, h=21, c=16 — L116 再减 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            break;
        /* === 第十六阶段：LDM 长距离匹配 — 用 w=20 达到 w=22 的比率 === */
        case 121: /* b6, w=20, h=20, c=18, LDM(h=18,m=64) — 基础 LDM */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_enableLongDistanceMatching, 1);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_ldmHashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_ldmMinMatch, 64);
            break;
        case 122: /* b6, w=20, h=20, c=18, LDM(h=20,m=64) — 大 LDM hash */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_enableLongDistanceMatching, 1);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_ldmHashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_ldmMinMatch, 64);
            break;
        case 123: /* b6, w=20, h=20, c=18, LDM(h=18,m=32) — 激进 LDM */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_enableLongDistanceMatching, 1);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_ldmHashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_ldmMinMatch, 32);
            break;
        case 124: /* b6, w=21, h=20, c=18, LDM(h=18,m=64) — w=21 + LDM */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_enableLongDistanceMatching, 1);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_ldmHashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_ldmMinMatch, 64);
            break;
        case 125: /* b6, w=20, h=20, c=17, LDM(h=18,m=64) — 少 chain + LDM */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_enableLongDistanceMatching, 1);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_ldmHashLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_ldmMinMatch, 64);
            break;
        case 126: /* b6, w=20, h=20, c=18, LDM(h=17,m=128) — 轻量 LDM */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 6);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_enableLongDistanceMatching, 1);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_ldmHashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_ldmMinMatch, 128);
            break;
        /* === 第十七阶段：更快策略 + 超大窗口/哈希 — 逼近 L102 比率 === */
        case 127: /* b3(dfast), w=22, h=21, c=14 — 最快策略 + 超大 hash */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 3);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 14);
            break;
        case 128: /* b3(dfast), w=22, h=21, c=16 — dfast + 更深 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 3);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            break;
        case 129: /* b4(lazy), w=22, h=21, c=16 — lazy + 超大 hash */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 4);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            break;
        case 130: /* b4(lazy), w=22, h=21, c=18 — lazy + 大 hash + 深 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 4);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 131: /* b5(lazy2), w=22, h=21, c=20 — lazy2 极限 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            break;
        case 132: /* b5(lazy2), w=22, h=21, c=22 — lazy2 超级极限 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 22);
            break;
        /* === 第十八阶段：lazy2 + searchLog / 极限 hashLog — 突破 29.16% 天花板 === */
        case 133: /* b5(lazy2), w=22, h=21, c=18, s=2 — lazy2 + 搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 2);
            break;
        case 134: /* b5(lazy2), w=22, h=21, c=18, s=4 — lazy2 + 深搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 4);
            break;
        case 135: /* b5(lazy2), w=22, h=22, c=18 — lazy2 + 超大 hash (4M) */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            break;
        case 136: /* b5(lazy2), w=22, h=22, c=20 — lazy2 + h=22 + 深 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            break;
        /* === 第十九阶段：lazy2 + searchLog 深挖 — 从 28.88% 推向 28.55% === */
        case 137: /* b5(lazy2), w=22, h=21, c=18, s=5 — 更深搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 5);
            break;
        case 138: /* b5(lazy2), w=22, h=21, c=18, s=6 — 极限搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 6);
            break;
        case 139: /* b5(lazy2), w=22, h=22, c=18, s=4 — L134 + h=22 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 4);
            break;
        case 140: /* b5(lazy2), w=22, h=21, c=20, s=4 — L134 + 深 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 4);
            break;
        case 141: /* b5(lazy2), w=22, h=22, c=20, s=4 — 全极限组合 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 4);
            break;
        /* === 第二十阶段：优化 L138 (28.40%) 速度 — h=22 + s=5/6 === */
        case 142: /* b5(lazy2), w=22, h=22, c=18, s=5 — h22 减搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 5);
            break;
        case 143: /* b5(lazy2), w=22, h=22, c=18, s=6 — L138 + h=22 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 18);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 6);
            break;
        case 144: /* b5(lazy2), w=22, h=22, c=16, s=6 — L143 少 chain */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 6);
            break;
        case 145: /* b5(lazy2), w=22, h=21, c=20, s=5 — 中搜索 + 深链 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 5);
            break;
        case 146: /* b5(lazy2), w=22, h=21, c=22, s=5 — 极限链 + 中搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 5);
            break;
        /* === 第二十一阶段：L145 精炼 — h22 + s6 组合冲锋 === */
        case 147: /* b5(lazy2), w=22, h=22, c=20, s=5 — L145 + h=22 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 5);
            break;
        case 148: /* b5(lazy2), w=22, h=21, c=20, s=6 — L145 + s=6 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 20);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 6);
            break;
        case 149: /* b5(lazy2), w=22, h=22, c=16, s=5 — h22 + 少链 + 中搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 16);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 5);
            break;
        case 150: /* b5(lazy2), w=22, h=21, c=22, s=6 — 极限链 + 最大搜索 */
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, 5);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 21);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 22);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 6);
            break;
        default:
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_strategy, ZSTD_dfast);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_hashLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_chainLog, 17);
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_searchLog, 1);
            break;
    }

    size_t res = ZSTD_compress2(zstd_params->cctx, outbuf, outsize, inbuf, insize);
    if (ZSTD_isError(res)) return (int64_t)res;
    return (int64_t)res;
}
#endif

#ifndef BENCH_REMOVE_ZXC
#include "zxc/include/zxc.h"

typedef struct {
    zxc_cctx *cctx;
    zxc_dctx *dctx;
    int level;
} zxc_bench_t;

char *lzbench_zxc_init(size_t insize, size_t level, size_t)
{
    zxc_bench_t *bench = (zxc_bench_t *)malloc(sizeof(zxc_bench_t));
    if (!bench)
        return NULL;

    bench->level = (int)level;

    zxc_compress_opts_t copts = {0};
    copts.level = (int)level;

    /* ZXC block_size must be a power of 2 in [4KB, 2MB].
     * Valid values:  4096  (4KB)    1 << 12
     *                8192  (8KB)    1 << 13
     *               16384  (16KB)   1 << 14
     *               32768  (32KB)   1 << 15
     *               65536  (64KB)   1 << 16
     *              131072  (128KB)  1 << 17
     *              262144  (256KB)  1 << 18
     *              524288  (512KB)  1 << 19  (default)
     *             1048576  (1MB)    1 << 20
     *             2097152  (2MB)    1 << 21
     * Set to 0 to use the default (512KB). */
    copts.block_size = 0;

    bench->cctx = zxc_create_cctx(&copts);
    bench->dctx = zxc_create_dctx();

    if (!bench->cctx || !bench->dctx)
    {
        if (bench->cctx) zxc_free_cctx(bench->cctx);
        if (bench->dctx) zxc_free_dctx(bench->dctx);
        free(bench);
        return NULL;
    }
    return (char *)bench;
}

void lzbench_zxc_deinit(char *workmem)
{
    zxc_bench_t *bench = (zxc_bench_t *)workmem;
    if (!bench)
        return;
    if (bench->cctx) zxc_free_cctx(bench->cctx);
    if (bench->dctx) zxc_free_dctx(bench->dctx);
    free(bench);
}

int64_t lzbench_zxc_compress(char *inbuf, size_t insize, char *outbuf,
                             size_t outsize, codec_options_t *codec_options)
{
    zxc_bench_t *bench = (zxc_bench_t *)codec_options->work_mem;
    if (!bench || !bench->cctx) return 0;

    int64_t res = zxc_compress_cctx(bench->cctx, inbuf, insize,
                                     outbuf, outsize, NULL);
    return (res > 0) ? res : 0;
}

int64_t lzbench_zxc_decompress(char *inbuf, size_t insize, char *outbuf,
                               size_t outsize, codec_options_t *codec_options)
{
    zxc_bench_t *bench = (zxc_bench_t *)codec_options->work_mem;
    if (!bench || !bench->dctx) return 0;

    int64_t res = zxc_decompress_dctx(bench->dctx, inbuf, insize,
                                       outbuf, outsize, NULL);
    return (res > 0) ? res : 0;
}
#endif
