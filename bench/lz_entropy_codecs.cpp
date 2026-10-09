/*
 * Copyright (c) Przemyslaw Skibinski <inikep@gmail.com>
 * All rights reserved.
 *
 * This source code is dual-licensed under the GPLv2 and GPLv3 licenses.
 * For additional details, refer to the LICENSE file located in the root
 * directory of this source tree.
 *
 * lz_entropy_codecs.cpp: LZ codecs with an entropy-coding stage: Huffman, FSE/ANS or range
 *     coding (lz+entropy/, plus kanzi and lzma from misc/)
 */

#include "codecs.h"

#include <stdint.h>
#include <stdio.h> // printf
#include <string.h> // memcpy
#include <algorithm> // std::max



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



#ifndef BENCH_REMOVE_MBROTLI
#include "mbrotli/mbrotli-ffi/include/mbrotli.h"

int64_t lzbench_mbrotli_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int windowLog = codec_options->additional_param;
    if (!windowLog) windowLog = 22; // sliding window size. Range is 10 to 24.

    size_t actual_osize = outsize;
    return mbrotli_compress((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, &actual_osize, codec_options->level, windowLog) == MBROTLI_OK ? actual_osize : 0;
}
int64_t lzbench_mbrotli_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    size_t actual_osize = outsize;
    return mbrotli_decompress((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, &actual_osize) == MBROTLI_OK ? actual_osize : 0;
}

#endif // BENCH_REMOVE_MBROTLI



#ifndef BENCH_REMOVE_FASTLZMA2
#include "lz+entropy/fast-lzma2/fast-lzma2.h"

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
#include "lz+entropy/libdeflate/libdeflate.h"
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
        libdeflate_free_decompressor(decompressor);
        return 0;
    }
    libdeflate_free_decompressor(decompressor);
    return res;
}
#endif



#ifndef BENCH_REMOVE_WZIP
#include "lz+entropy/wzip/WZIP.h"

int64_t lzbench_wzip_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int cap = outsize > 0x7FFFFFFF ? 0x7FFFFFFF : (int)outsize;
    return wzip_compress_mt(inbuf, (int)insize, outbuf, &cap, codec_options->level, codec_options->threads);
}

int64_t lzbench_wzip_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    int cap = (int)outsize;
    return wzip_decompress(inbuf, (int)insize, outbuf, &cap);
}

#endif



#ifndef BENCH_REMOVE_LIZARD
#include "lz+entropy/lizard/lizard_compress.h"
#include "lz+entropy/lizard/lizard_decompress.h"

int64_t lzbench_lizard_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return Lizard_compress(inbuf, outbuf, insize, outsize, codec_options->level);
}

int64_t lzbench_lizard_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    return Lizard_decompress_safe(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZFSE
extern "C"
{
    #include "lz+entropy/lzfse/lzfse.h"
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
    #include "lz+entropy/lzfse/lzvn.h"
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



#ifndef BENCH_REMOVE_LZHAM
#include "lz+entropy/lzham/include/lzham.h"
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
#include "lz+entropy/lzlib/lzlib.h"

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
#include "lz+entropy/zlib-ng/zlib-ng.h"
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
#include "lz+entropy/xz/src/liblzma/api/lzma.h"

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
#include "lz+entropy/libzling/libzling.h"

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



#ifndef BENCH_REMOVE_OPENZL
#include <openzl/openzl.h>
#include <openzl/codecs/zl_segmenters.h>

// The OpenZL format version used for the compression in lzbench
#define LZBENCH_OPENZL_FORMAT_VERSION 24

typedef struct {
    ZL_Compressor* cgraph;
    ZL_CCtx* cctx;
    ZL_DCtx* dctx;
    size_t eltWidth;  // element width of the integer profiles, 1 for the others
} openzl_params_s;

static char* lzbench_openzl_init_base(size_t insize, size_t level, size_t windowLog)
{
    openzl_params_s* params = (openzl_params_s*) malloc(sizeof(openzl_params_s));

    params->eltWidth = 1;
    params->cgraph = ZL_Compressor_create();
    assert(params->cgraph);
    params->cctx = ZL_CCtx_create();
    assert(params->cctx);
    params->dctx = ZL_DCtx_create();
    assert(params->dctx);

    ZL_Report report = ZL_Compressor_setParameter(params->cgraph, ZL_CParam_formatVersion, LZBENCH_OPENZL_FORMAT_VERSION);
    if (ZL_isError(report)) {
      printf("OpenZL initialisation error: %s\n", ZL_Compressor_getErrorContextString(params->cgraph, report));
      abort();
    }

    return (char*) params;
}

char* lzbench_openzl_init_serial(size_t insize, size_t level, size_t windowLog)
{
    openzl_params_s* params = (openzl_params_s*) lzbench_openzl_init_base(insize, level, windowLog);

    // ZL_GRAPH_LZ: standard graph for LZ compression, offer performance similar to Zstd.
    // Used for serial data (aka raw bytes).
    ZL_Report report = ZL_Compressor_selectStartingGraphID(params->cgraph, ZL_GRAPH_LZ);
    if (ZL_isError(report)) {
      printf("OpenZL initialisation error: %s\n", ZL_Compressor_getErrorContextString(params->cgraph, report));
      abort();
    }

    return (char*) params;
}

template <typename TInteger>
char* lzbench_openzl_init_integer_t(size_t insize, size_t level, size_t windowLog)
{
    openzl_params_s* params = (openzl_params_s*) lzbench_openzl_init_base(insize, level, windowLog);

    // Build a graph to compress signed or unsigned integers (Little Endian).
    // Adapted from OpenZL buildIntProfile() source code in cli/utils/compress_profiles.cpp .
    ZL_GraphID graph = ZL_GRAPH_FIELD_LZ;
    if (std::is_signed<TInteger>::value) {
        graph = ZL_Compressor_registerStaticGraph_fromNode1o(params->cgraph, ZL_NODE_ZIGZAG, graph);
    }
    graph = ZL_Compressor_registerStaticGraph_fromNode1o(params->cgraph, ZL_Node_interpretAsLE(8*sizeof(TInteger)), graph);
    graph = ZL_Compressor_buildNumFromSerialSegmenter(params->cgraph, sizeof(TInteger), 0, graph);

    ZL_Report report = ZL_Compressor_selectStartingGraphID(params->cgraph, graph);
    if (ZL_isError(report)) {
      printf("OpenZL initialisation error: %s\n", ZL_Compressor_getErrorContextString(params->cgraph, report));
      abort();
    }

    params->eltWidth = sizeof(TInteger);
    return (char*) params;
}

// Explicit definition of the template specialisations referenced in lzbench.h comp_desc[].
template char* lzbench_openzl_init_integer_t<uint8_t>(size_t insize, size_t level, size_t windowLog);
template char* lzbench_openzl_init_integer_t<int8_t>(size_t insize, size_t level, size_t windowLog);
template char* lzbench_openzl_init_integer_t<uint16_t>(size_t insize, size_t level, size_t windowLog);
template char* lzbench_openzl_init_integer_t<int16_t>(size_t insize, size_t level, size_t windowLog);
template char* lzbench_openzl_init_integer_t<uint32_t>(size_t insize, size_t level, size_t windowLog);
template char* lzbench_openzl_init_integer_t<int32_t>(size_t insize, size_t level, size_t windowLog);
template char* lzbench_openzl_init_integer_t<uint64_t>(size_t insize, size_t level, size_t windowLog);
template char* lzbench_openzl_init_integer_t<int64_t>(size_t insize, size_t level, size_t windowLog);

char* lzbench_openzl_init_generic(size_t insize, size_t level, size_t windowLog)
{
    openzl_params_s* params = (openzl_params_s*) lzbench_openzl_init_base(insize, level, windowLog);

    // ZL_GRAPH_COMPRESS_GENERIC: "default" generic compression suitable for any stream type.
    // Used as a fallback if a compressor does not match the characteristics of the data.
    // Currently corresponds to Zstd level 6.
    ZL_Report report = ZL_Compressor_selectStartingGraphID(params->cgraph, ZL_GRAPH_COMPRESS_GENERIC);
    if (ZL_isError(report)) {
      printf("OpenZL initialisation error: %s\n", ZL_Compressor_getErrorContextString(params->cgraph, report));
      abort();
    }

    return (char*) params;
}

char* lzbench_openzl_init_zstd(size_t insize, size_t level, size_t windowLog)
{
    openzl_params_s* params = (openzl_params_s*) lzbench_openzl_init_base(insize, level, windowLog);

    // ZL_GRAPH_ZSTD: Zstd compression.
    ZL_Report report = ZL_Compressor_selectStartingGraphID(params->cgraph, ZL_GRAPH_ZSTD);
    if (ZL_isError(report)) {
      printf("OpenZL initialisation error: %s\n", ZL_Compressor_getErrorContextString(params->cgraph, report));
      abort();
    }

    // OpenZL does not validate the compression level; it is forwarded to zstd, which
    // clamps it to [ZSTD_minCLevel(), ZSTD_maxCLevel()], i.e. [-131072, 22].
    // Level 0 requests the default behaviour, which corresponds to level 6.
    // lzbench limits the range to [-99, 22]; -99 is an arbitrary practical floor.
    report = ZL_Compressor_setParameter(params->cgraph, ZL_CParam_compressionLevel, level);
    if (ZL_isError(report)) {
      printf("OpenZL initialisation error: %s\n", ZL_Compressor_getErrorContextString(params->cgraph, report));
      abort();
    }

    return (char*) params;
}

char* lzbench_openzl_init_lz4(size_t insize, size_t level, size_t windowLog)
{
    openzl_params_s* params = (openzl_params_s*) lzbench_openzl_init_base(insize, level, windowLog);

    // ZL_GRAPH_LZ4: LZ4 compression.
    ZL_Report report = ZL_Compressor_selectStartingGraphID(params->cgraph, ZL_GRAPH_LZ4);
    if (ZL_isError(report)) {
      printf("OpenZL initialisation error: %s\n", ZL_Compressor_getErrorContextString(params->cgraph, report));
      abort();
    }

    // OpenZL does not validate the compression level: the lz4 graph maps levels <= 1 to
    // LZ4_compress_fast() with acceleration = 1 - level (lz4 clamps the acceleration to
    // LZ4_ACCELERATION_MAX, i.e. 65537), and levels >= 2 to LZ4_compress_HC(), whose
    // maximum is LZ4HC_CLEVEL_MAX, i.e. 12.
    // Level 0 requests the default behaviour, which corresponds to level 6.
    // lzbench limits the range to [-99, 12]; -99 is an arbitrary practical floor.
    report = ZL_Compressor_setParameter(params->cgraph, ZL_CParam_compressionLevel, level);
    if (ZL_isError(report)) {
      printf("OpenZL initialisation error: %s\n", ZL_Compressor_getErrorContextString(params->cgraph, report));
      abort();
    }

    return (char*) params;
}

void lzbench_openzl_deinit(char* workmem)
{
    openzl_params_s* params = (openzl_params_s*) workmem;
    if (!params) return;
    if (params->dctx) ZL_DCtx_free(params->dctx);
    if (params->cctx) ZL_CCtx_free(params->cctx);
    if (params->cgraph) ZL_Compressor_free(params->cgraph);
    free(workmem);
}

int64_t lzbench_openzl_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    openzl_params_s* params = (openzl_params_s*) codec_options->work_mem;
    if (not params or not params->cctx or not params->cgraph) return 0;

    ZL_Report report = ZL_CCtx_refCompressor(params->cctx, params->cgraph);
    if (ZL_isError(report)) {
      printf("OpenZL compression error: %s\n", ZL_CCtx_getErrorContextString(params->cctx, report));
      return 0;
    }

    // The integer profiles fail if the input is not a whole number of elements
    // (OpenZL in strict mode, the library default; facebook/openzl#1085), e.g. on
    // most files or the last -b block. Compress the whole elements and store the
    // 1-7 trailing bytes uncompressed after the frame.
    size_t tail = insize % params->eltWidth;
    if (outsize < tail) return 0;

    report = ZL_CCtx_compress(params->cctx, outbuf, outsize - tail, inbuf, insize - tail);
    if (ZL_isError(report)) {
      printf("OpenZL compression error: %s\n", ZL_CCtx_getErrorContextString(params->cctx, report));
      return 0;
    }

    size_t clen = ZL_validResult(report);
    memcpy(outbuf + clen, inbuf + insize - tail, tail);
    return (int64_t) (clen + tail);
}

int64_t lzbench_openzl_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options)
{
    openzl_params_s* params = (openzl_params_s*) codec_options->work_mem;
    if (not params or not params->dctx) return 0;

    // outsize is the original size, so it gives the number of trailing bytes
    // stored after the frame (see lzbench_openzl_compress)
    size_t tail = outsize % params->eltWidth;
    if (insize < tail) return 0;

    ZL_Report report = ZL_DCtx_decompress(params->dctx, outbuf, outsize - tail, inbuf, insize - tail);
    if (ZL_isError(report)) {
      printf("OpenZL decompression error: %s\n", ZL_DCtx_getErrorContextString(params->dctx, report));
      return 0;
    }

    size_t dlen = ZL_validResult(report);
    if (dlen != outsize - tail) return 0;
    memcpy(outbuf + dlen, inbuf + insize - tail, tail);
    return (int64_t) (dlen + tail);
}

#endif // BENCH_REMOVE_OPENZL



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
