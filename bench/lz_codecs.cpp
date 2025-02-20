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



int64_t lzbench_memcpy(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char* )
{
    memcpy(outbuf, inbuf, insize);
    return insize;
}



#ifndef BENCH_REMOVE_BLOSCLZ
#include "lz/blosclz/blosc/blosclz.h"

int64_t lzbench_blosclz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
    return blosclz_compress(level, inbuf, insize, outbuf, outsize, 1);
}

int64_t lzbench_blosclz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char*)
{
    return blosclz_decompress(inbuf, insize, outbuf, outsize);
}

#endif // BENCH_REMOVE_BLOSCLZ



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

int64_t lzbench_brieflz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
    if (!workmem)
        return 0;

    int64_t res = blz_pack_level(inbuf, outbuf, insize, (void*)workmem, level);

    return res;
}

int64_t lzbench_brieflz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
    return blz_depack(inbuf, outbuf, outsize);
}

#endif // BENCH_REMOVE_BRIEFLZ



#ifndef BENCH_REMOVE_BROTLI
#include "brotli/encode.h"
#include "brotli/decode.h"

int64_t lzbench_brotli_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t windowLog, char*)
{
    if (!windowLog) windowLog = BROTLI_DEFAULT_WINDOW; // sliding window size. Range is 10 to 24.

    size_t actual_osize = outsize;
    return BrotliEncoderCompress(level, windowLog, BROTLI_DEFAULT_MODE, insize, (const uint8_t*)inbuf, &actual_osize, (uint8_t*)outbuf) == 0 ? 0 : actual_osize;
}
int64_t lzbench_brotli_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
    size_t actual_osize = outsize;
    return BrotliDecoderDecompress(insize, (const uint8_t*)inbuf, &actual_osize, (uint8_t*)outbuf) == BROTLI_DECODER_RESULT_ERROR ? 0 : actual_osize;
}

#endif // BENCH_REMOVE_BROTLI



#ifndef BENCH_REMOVE_CRUSH
#include "lz/crush/crush.hpp"

int64_t lzbench_crush_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return crush::compress(level, (uint8_t*)inbuf, insize, (uint8_t*)outbuf);
}

int64_t lzbench_crush_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return crush::decompress((uint8_t*)inbuf, (uint8_t*)outbuf, outsize);
}

#endif // BENCH_REMOVE_CRUSH



#ifndef BENCH_REMOVE_FASTLZ
extern "C"
{
	#include "lz/fastlz/fastlz.h"
}

int64_t lzbench_fastlz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return fastlz_compress_level(level, inbuf, insize, outbuf);
}

int64_t lzbench_fastlz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return fastlz_decompress(inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_FASTLZMA2
#include "lz/fast-lzma2/fast-lzma2.h"

int64_t lzbench_fastlzma2_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t windowLog, char* workmem)
{
    size_t ret = FL2_compress(outbuf, outsize, inbuf, insize, level);
    if (FL2_isError(ret)) return 0;
    return ret;
}

int64_t lzbench_fastlzma2_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char* workmem)
{
    size_t ret = FL2_decompress(outbuf, outsize, inbuf, insize);
    if (FL2_isError(ret)) return 0;
    return ret;
}
#endif // BENCH_REMOVE_FASTLZMA2



#ifndef BENCH_REMOVE_KANZI
#include "misc/kanzi-cpp/src/types.hpp"
#include "misc/kanzi-cpp/src/util.hpp"
#include "misc/kanzi-cpp/src/InputStream.hpp"
#include "misc/kanzi-cpp/src/OutputStream.hpp"
#include "misc/kanzi-cpp/src/io/CompressedInputStream.hpp"
#include "misc/kanzi-cpp/src/io/CompressedOutputStream.hpp"

int64_t lzbench_kanzi_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t windowLog, char*)
{
  std::string entropy;
  std::string transform;
  kanzi::uint szBlock;

  switch (level) {
     case 0:
         transform = "NONE";
         entropy = "NONE";
         szBlock = 4 * 1024 * 1024;
         break;
     case 1:
         transform = "PACK+LZ";
         entropy = "NONE";
         szBlock = 4 * 1024 * 1024;
         break;
     case 2:
         transform = "PACK+LZ";
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
         transform = "EXE+RLT+TEXT+UTF";
         entropy = "TPAQ";
         szBlock = 16 * 1024 * 1024;
         break;
     case 9:
         transform = "EXE+RLT+TEXT+UTF";
         entropy = "TPAQX";
         szBlock = 32 * 1024 * 1024;
         break;
     default:
         return -1;
  }

  ostreambuf<char> buf(outbuf, outsize);
  std::iostream os(&buf);
  int cores = 1; ///std::max(int(std::thread::hardware_concurrency()) / 2, 1);
  kanzi::CompressedOutputStream cos(os, entropy, transform, szBlock, false, std::min(cores, 64));
  cos.write(inbuf, insize);
  cos.close();
  return cos.getWritten();
}

int64_t lzbench_kanzi_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
  istreambuf<char> buf(inbuf, insize);
  std::iostream is(&buf);
  int cores = 1; ///std::max(int(std::thread::hardware_concurrency()) / 2, 1);
  kanzi::CompressedInputStream cis(is, std::min(cores, 64));
  cis.read(outbuf, outsize);
  cis.close();
  return outsize;//cis.getRead();
}
#endif // BENCH_REMOVE_KANZI



#ifndef BENCH_REMOVE_LIBDEFLATE
#include "lz/libdeflate/libdeflate.h"
int64_t lzbench_libdeflate_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
    struct libdeflate_compressor *compressor = libdeflate_alloc_compressor(level);
    if (!compressor)
        return 0;
    int64_t res = libdeflate_deflate_compress(compressor, inbuf, insize, outbuf, outsize);
    libdeflate_free_compressor(compressor);
    return res;
}
int64_t lzbench_libdeflate_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
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

int64_t lzbench_lizard_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return Lizard_compress(inbuf, outbuf, insize, outsize, level);
}

int64_t lzbench_lizard_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return Lizard_decompress_safe(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZ4
#include "lz/lz4/lib/lz4.h"
#include "lz/lz4/lib/lz4hc.h"

int64_t lzbench_lz4_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return LZ4_compress_default(inbuf, outbuf, insize, outsize);
}

int64_t lzbench_lz4fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return LZ4_compress_fast(inbuf, outbuf, insize, outsize, level);
}

int64_t lzbench_lz4hc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return LZ4_compress_HC(inbuf, outbuf, insize, outsize, level);
}

int64_t lzbench_lz4_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return LZ4_decompress_safe(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZAV
#include "lz/lzav/lzav.h"

int64_t lzbench_lzav_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	if (level == 1)
		return lzav_compress_default(inbuf, outbuf, insize, outsize);
	return lzav_compress_hi(inbuf, outbuf, insize, outsize);
}

int64_t lzbench_lzav_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return lzav_decompress(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZF
extern "C"
{
	#include "lz/lzf/lzf.h"
}

int64_t lzbench_lzf_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	if (level == 0)
		return lzf_compress(inbuf, insize, outbuf, outsize);
	return lzf_compress_very(inbuf, insize, outbuf, outsize);
}

int64_t lzbench_lzf_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
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

int64_t lzbench_lzfse_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
	return lzfse_encode_buffer((uint8_t*)outbuf, outsize, (uint8_t*)inbuf, insize, workmem);
}

int64_t lzbench_lzfse_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char* workmem)
{
	return lzfse_decode_buffer((uint8_t*)outbuf, outsize, (uint8_t*)inbuf, insize, workmem);
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

int64_t lzbench_lzvn_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
	return lzvn_encode_buffer((uint8_t*)outbuf, outsize, (uint8_t*)inbuf, insize, workmem);
}

int64_t lzbench_lzvn_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char* workmem)
{
	return lzvn_decode_buffer_scratch((uint8_t*)outbuf, outsize, (uint8_t*)inbuf, insize, workmem);
}

#endif



#ifndef BENCH_REMOVE_LZG
#include "lz/liblzg/lzg.h"

int64_t lzbench_lzg_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
    lzg_encoder_config_t cfg;
    cfg.level = level;
    cfg.fast = LZG_TRUE;
    cfg.progressfun = NULL;
    cfg.userdata = NULL;
    return LZG_Encode((const unsigned char*)inbuf, insize, (unsigned char*)outbuf, outsize, &cfg);
}

int64_t lzbench_lzg_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
    return LZG_Decode((const unsigned char*)inbuf, insize, (unsigned char*)outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZHAM
#include "lz/lzham/lzham.h"
#include <memory.h>

int64_t lzbench_lzham_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t dict_size_log, char*)
{
	lzham_compress_params comp_params;
	memset(&comp_params, 0, sizeof(comp_params));
	comp_params.m_struct_size = sizeof(lzham_compress_params);
	comp_params.m_dict_size_log2 = dict_size_log?dict_size_log:26;
	comp_params.m_max_helper_threads = 0;
	comp_params.m_level = (lzham_compress_level)level;

	lzham_compress_status_t comp_status;
	lzham_uint32 comp_adler32 = 0;

	if ((comp_status = lzham_compress_memory(&comp_params, (uint8_t*)outbuf, &outsize, (const lzham_uint8 *)inbuf, insize, &comp_adler32)) != LZHAM_COMP_STATUS_SUCCESS)
	{
		printf("Compression test failed with status %i!\n", comp_status);
		return 0;
	}

	return outsize;
}

int64_t lzbench_lzham_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t dict_size_log, char*)
{
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

int64_t lzbench_lzlib_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
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
  const int match_len_limit = option_mapping[level].match_len_limit;
  const unsigned long long member_size = 0x7FFFFFFFFFFFFFFFULL;	/* INT64_MAX */
  int new_pos = 0;
  int written = 0;
  bool error = false;
  int dict_size = option_mapping[level].dictionary_size;
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
 

int64_t lzbench_lzlib_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
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
#include "misc/7-zip/LzmaDec.h"
#include "misc/7-zip/LzmaEnc.h"

#ifndef BENCH_REMOVE_TORNADO
static void *SzAlloc(ISzAllocPtr p, size_t size) { (void)p; return MyAlloc(size); }
static void SzFree(ISzAllocPtr p, void *address) { (void)p; MyFree(address); }
const ISzAlloc g_Alloc = { SzAlloc, SzFree };
#endif

int64_t lzbench_lzma_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	CLzmaEncProps props;
	int res;
    size_t headerSize = LZMA_PROPS_SIZE;
	SizeT out_len = outsize - LZMA_PROPS_SIZE;
	
	LzmaEncProps_Init(&props);
	props.level = level;
	props.numThreads = 1;
	LzmaEncProps_Normalize(&props);
  /*
  p->level = 5;
  p->dictSize = p->mc = 0;
  p->reduceSize = (UInt64)(Int64)-1;
  p->lc = p->lp = p->pb = p->algo = p->fb = p->btMode = p->numHashBytes = p->numThreads = -1;
  p->writeEndMark = 0;
  */
  
  	res = LzmaEncode((uint8_t*)outbuf+LZMA_PROPS_SIZE, &out_len, (uint8_t*)inbuf, insize, &props, (uint8_t*)outbuf, &headerSize, 0/*int writeEndMark*/, NULL, &g_Alloc, &g_Alloc);
	if (res != SZ_OK) return 0;
	
//	printf("out_len=%u LZMA_PROPS_SIZE=%d headerSize=%d\n", (int)(out_len + LZMA_PROPS_SIZE), LZMA_PROPS_SIZE, (int)headerSize);
	return LZMA_PROPS_SIZE + out_len;
}

int64_t lzbench_lzma_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	int res;
	SizeT out_len = outsize;
	SizeT src_len = insize - LZMA_PROPS_SIZE;
	ELzmaStatus status;

//	SRes LzmaDecode(Byte *dest, SizeT *destLen, const Byte *src, SizeT *srcLen, const Byte *propData, unsigned propSize, ELzmaFinishMode finishMode, ELzmaStatus *status, ISzAlloc *alloc)
	res = LzmaDecode((uint8_t*)outbuf, &out_len, (uint8_t*)inbuf+LZMA_PROPS_SIZE, &src_len, (uint8_t*)inbuf, LZMA_PROPS_SIZE, LZMA_FINISH_END, &status, &g_Alloc);
	if (res != SZ_OK) return 0;
	
//	printf("out_len=%u\n", (int)(out_len + LZMA_PROPS_SIZE));	
    return out_len;
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

int64_t lzbench_lzo1_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
	lzo_uint lzo_complen = 0;
	int res;

    if (!workmem)
        return 0;

	if (level == 99)
		res = lzo1_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    else
		res = lzo1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    
	if (res != LZO_E_OK) return 0;
		
	return lzo_complen; 
}

int64_t lzbench_lzo1_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	lzo_uint decomplen = 0;

    if (lzo1_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1a_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
	lzo_uint lzo_complen = 0;
	int res;

    if (!workmem)
        return 0;

	if (level == 99)
		res = lzo1a_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    else
		res = lzo1a_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    
	if (res != LZO_E_OK) return 0;
		
	return lzo_complen; 
}

int64_t lzbench_lzo1a_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	lzo_uint decomplen = 0;

    if (lzo1a_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1b_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
	lzo_uint lzo_complen = 0;
	int res;

    if (!workmem)
        return 0;

	switch (level)
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

int64_t lzbench_lzo1b_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	lzo_uint decomplen = 0;

    if (lzo1b_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1c_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
	lzo_uint lzo_complen = 0;
	int res;

    if (!workmem)
        return 0;

	switch (level)
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

int64_t lzbench_lzo1c_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	lzo_uint decomplen = 0;

    if (lzo1c_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1f_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
	lzo_uint lzo_complen = 0;
	int res;

    if (!workmem)
        return 0;

	if (level == 999)
		res = lzo1f_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    else
		res = lzo1f_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    
	if (res != LZO_E_OK) return 0;
		
	return lzo_complen; 
}

int64_t lzbench_lzo1f_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	lzo_uint decomplen = 0;

    if (lzo1f_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1x_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
	lzo_uint lzo_complen = 0;
	int res;

    if (!workmem)
        return 0;

	switch (level)
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

int64_t lzbench_lzo1x_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	lzo_uint decomplen = 0;

    if (lzo1x_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1y_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
	lzo_uint lzo_complen = 0;
	int res;

    if (!workmem)
        return 0;

	if (level == 999)
		res = lzo1y_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    else
		res = lzo1y_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    
	if (res != LZO_E_OK) return 0;
		
	return lzo_complen; 
}

int64_t lzbench_lzo1y_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	lzo_uint decomplen = 0;

    if (lzo1y_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1z_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
	lzo_uint lzo_complen = 0;
	int res;

    if (!workmem)
        return 0;

    res = lzo1z_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    
	if (res != LZO_E_OK) return 0;
		
	return lzo_complen; 
}

int64_t lzbench_lzo1z_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	lzo_uint decomplen = 0;

    if (lzo1z_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}


int64_t lzbench_lzo2a_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
	lzo_uint lzo_complen = 0;
	int res;

    if (!workmem)
        return 0;

    res = lzo2a_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    
	if (res != LZO_E_OK) return 0;
		
	return lzo_complen; 
}

int64_t lzbench_lzo2a_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
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

int64_t lzbench_lzsse2_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
    if (!workmem) return 0;

    return LZSSE2_CompressOptimalParse((LZSSE2_OptimalParseState*) workmem, inbuf, insize, outbuf, outsize, level);
}

int64_t lzbench_lzsse2_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
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

int64_t lzbench_lzsse4_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
    if (!workmem) return 0;

    return LZSSE4_CompressOptimalParse((LZSSE4_OptimalParseState*) workmem, inbuf, insize, outbuf, outsize, level);
}

int64_t lzbench_lzsse4_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
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

int64_t lzbench_lzsse4fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char* workmem)
{
    if (!workmem) return 0;

    return LZSSE4_CompressFast((LZSSE4_FastParseState*) workmem, inbuf, insize, outbuf, outsize);
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

int64_t lzbench_lzsse8_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
    if (!workmem) return 0;

    return LZSSE8_CompressOptimalParse((LZSSE8_OptimalParseState*) workmem, inbuf, insize, outbuf, outsize, level);
}

int64_t lzbench_lzsse8_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
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

int64_t lzbench_lzsse8fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char* workmem)
{
    if (!workmem) return 0;

    return LZSSE8_CompressFast((LZSSE8_FastParseState*) workmem, inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_QUICKLZ
#include "quicklz/quicklz151b7.h"
#include "quicklz/quicklz.h"

int64_t lzbench_quicklz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t , char*)
{
    int64_t res;
    qlz150_state_compress* state = (qlz150_state_compress*) calloc(1, std::max(qlz_get_setting_3(1),std::max(qlz_get_setting_1(1), qlz_get_setting_2(1))));
    if (!state)
        return 0;

    
	switch (level)
	{
		default:
		case 1:	res = qlz_compress_1(inbuf, outbuf, insize, (qlz150_state_compress*)state); break;
		case 2:	res = qlz_compress_2(inbuf, outbuf, insize, (qlz150_state_compress*)state); break;
		case 3:	res = qlz_compress_3(inbuf, outbuf, insize, (qlz150_state_compress*)state); break;
		case 4:	res = qlz_compress(inbuf, outbuf, insize, (qlz_state_compress*)state); break;
	}
    
    free(state);
    return res;
}

int64_t lzbench_quicklz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t , char*)
{
    int64_t res;
    qlz150_state_compress* dstate = (qlz150_state_compress*) calloc(1, std::max(qlz_get_setting_3(2),std::max(qlz_get_setting_1(2), qlz_get_setting_2(2))));
    if (!dstate)
        return 0;
        
	switch (level)
	{
		default:
		case 1: res = qlz_decompress_1(inbuf, outbuf, (qlz150_state_decompress*)dstate); break;
		case 2: res = qlz_decompress_2(inbuf, outbuf, (qlz150_state_decompress*)dstate); break;
		case 3: res = qlz_decompress_3(inbuf, outbuf, (qlz150_state_decompress*)dstate); break;
		case 4: res = qlz_decompress(inbuf, outbuf, (qlz_state_decompress*)dstate); break;
	}

    free(dstate);
    return res;
}

#endif



#ifndef BENCH_REMOVE_SNAPPY
#include "snappy/snappy.h"

int64_t lzbench_snappy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	snappy::RawCompress(inbuf, insize, outbuf, &outsize);
	return outsize;
}

int64_t lzbench_snappy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	snappy::RawUncompress(inbuf, insize, outbuf);
	return outsize;
}

#endif



#ifndef BENCH_REMOVE_TORNADO
#include "tornado/tor_test.h"

int64_t lzbench_tornado_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return tor_compress(level, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, outsize); 
}

int64_t lzbench_tornado_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return tor_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, outsize); 
}

#endif



#ifndef BENCH_REMOVE_UCL
#include "ucl/ucl.h"

int64_t lzbench_ucl_nrv2b_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	ucl_uint complen;
	int res = ucl_nrv2b_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, level, NULL, NULL);

	if (res != UCL_E_OK) return 0;
	return complen;
}

int64_t lzbench_ucl_nrv2b_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	ucl_uint decomplen;
	int res = ucl_nrv2b_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL);

	if (res != UCL_E_OK) return 0;
	return decomplen;
}

int64_t lzbench_ucl_nrv2d_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	ucl_uint complen;
	int res = ucl_nrv2d_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, level, NULL, NULL);

	if (res != UCL_E_OK) return 0;
	return complen;
}

int64_t lzbench_ucl_nrv2d_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	ucl_uint decomplen;
	int res = ucl_nrv2d_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL);

	if (res != UCL_E_OK) return 0;
	return decomplen;
}

int64_t lzbench_ucl_nrv2e_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	ucl_uint complen;
	int res = ucl_nrv2e_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, level, NULL, NULL);

	if (res != UCL_E_OK) return 0;
	return complen;
}

int64_t lzbench_ucl_nrv2e_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	ucl_uint decomplen;
	int res = ucl_nrv2e_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL);

	if (res != UCL_E_OK) return 0;
	return decomplen;
}

#endif



#ifndef BENCH_REMOVE_XPACK
#include "lz/xpack/lib/libxpack.h"

typedef struct {
    struct xpack_compressor *xpackc;
    struct xpack_decompressor *xpackd;
} xpack_params_s;

char* lzbench_xpack_init(size_t insize, size_t level, size_t)
{
    xpack_params_s* xpack_params = (xpack_params_s*) malloc(sizeof(xpack_params_s));
    if (!xpack_params) return NULL;
    xpack_params->xpackc = xpack_alloc_compressor(insize, level);
    xpack_params->xpackd = xpack_alloc_decompressor(); 

    return (char*) xpack_params;
}

void lzbench_xpack_deinit(char* workmem)
{
    xpack_params_s* xpack_params = (xpack_params_s*) workmem;
    if (!xpack_params) return;
    if (xpack_params->xpackc) xpack_free_compressor(xpack_params->xpackc);
    if (xpack_params->xpackd) xpack_free_decompressor(xpack_params->xpackd);
    free(workmem);
}

int64_t lzbench_xpack_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
    xpack_params_s* xpack_params = (xpack_params_s*) workmem;
    if (!xpack_params || !xpack_params->xpackc) return 0;

    return xpack_compress(xpack_params->xpackc, inbuf, insize, outbuf, outsize);
}

int64_t lzbench_xpack_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char* workmem)
{
    xpack_params_s* xpack_params = (xpack_params_s*) workmem;
    if (!xpack_params || !xpack_params->xpackd) return 0;

    size_t res = xpack_decompress(xpack_params->xpackd, inbuf, insize, outbuf, outsize, NULL);
    if (res != 0) return 0;

    return outsize;
}

#endif



#ifndef BENCH_REMOVE_ZLIB
#include "zlib/zlib.h"

int64_t lzbench_zlib_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	uLongf zcomplen = insize;
	int err = compress2((uint8_t*)outbuf, &zcomplen, (uint8_t*)inbuf, insize, level);
	if (err != Z_OK)
		return 0;
	return zcomplen;
}

int64_t lzbench_zlib_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	uLongf zdecomplen = outsize;
	int err = uncompress((uint8_t*)outbuf, &zdecomplen, (uint8_t*)inbuf, insize); 
	if (err != Z_OK)
		return 0;
	return outsize;
}

#endif



#ifndef BENCH_REMOVE_ZLIB_NG

#undef z_const
#undef Z_NULL

#include "zlib-ng/zlib-ng.h"

int64_t lzbench_zlib_ng_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	size_t zcomplen = insize;
	int err = zng_compress2((uint8_t*)outbuf, &zcomplen, (uint8_t*)inbuf, insize, level);
	if (err != Z_OK)
		return 0;
	return zcomplen;
}

int64_t lzbench_zlib_ng_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	size_t zdecomplen = outsize;
	int err = zng_uncompress((uint8_t*)outbuf, &zdecomplen, (uint8_t*)inbuf, insize);
	if (err != Z_OK)
		return 0;
	return outsize;
}

#endif



#if !defined(BENCH_REMOVE_SLZ) && !defined(BENCH_REMOVE_ZLIB)
extern "C"
{
	#include "slz/src/slz.h"
}

int64_t lzbench_slz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t param2, char*)
{
	struct slz_stream strm;
	size_t outlen = 0;
	size_t window = 8192 << ((level & 3) * 2);
	size_t len;
	size_t blk;

	if (param2 == 0)
		slz_init(&strm, !!level, SLZ_FMT_GZIP);
	else if (param2 == 1)
		slz_init(&strm, !!level, SLZ_FMT_ZLIB);
	else
		slz_init(&strm, !!level, SLZ_FMT_DEFLATE);

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
int64_t lzbench_slz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t param2, char*)
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

	if (param2 == 0)      // gzip
		err = inflateInit2(&stream, 15 + 16);
	else if (param2 == 1) // zlip
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

int64_t lzbench_xz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t ignore1, char* ignore2)
{
    lzma_options_lzma opt_lzma;
    lzma_stream strm = LZMA_STREAM_INIT;
    uint32_t preset = level; // preset |= LZMA_PRESET_EXTREME;

    if (lzma_lzma_preset(&opt_lzma, preset))
        return 0;

    lzma_ret ret = lzma_alone_encoder(&strm, &opt_lzma);
    if (ret != LZMA_OK)
        return 0;

    strm.next_in = (uint8_t*)inbuf;
    strm.avail_in = insize;
    strm.next_out = (uint8_t*)outbuf;
    strm.avail_out = outsize;

    ret = lzma_code(&strm, LZMA_FINISH);
    if (ret != LZMA_STREAM_END)
        return 0;

    lzma_end(&strm);

    return (char*)strm.next_out - outbuf;
}

int64_t lzbench_xz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t ignore1, size_t ignore2, char* ignore3)
{
    lzma_stream strm = LZMA_STREAM_INIT;

    lzma_ret ret = lzma_alone_decoder(&strm, UINT64_MAX);
    if (ret != LZMA_OK)
        return 0;

    strm.next_in = (uint8_t*)inbuf;
    strm.avail_in = insize;
    strm.next_out = (uint8_t*)outbuf;
    strm.avail_out = outsize;

    ret = lzma_code(&strm, LZMA_FINISH);
    if (ret != LZMA_STREAM_END)
        return 0;

    lzma_end(&strm);

    return (char*)strm.next_out - outbuf;
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

int64_t lzbench_zling_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	baidu::zling::MemInputter  inputter((uint8_t*)inbuf, insize);
	baidu::zling::MemOutputter outputter((uint8_t*)outbuf, outsize);
	baidu::zling::Encode(&inputter, &outputter, NULL, level);

	return outputter.GetOutputSize();
}

int64_t lzbench_zling_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
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

int64_t lzbench_zstd_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t windowLog, char* workmem)
{
    size_t res;

    zstd_params_s* zstd_params = (zstd_params_s*) workmem;
    if (!zstd_params || !zstd_params->cctx) return 0;

#if 1
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, level);
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_contentSizeFlag, 1);

    if (windowLog) {
        size_t currentWindowLog = ZSTD_getParams(level, insize, 0).cParams.windowLog;
        if (currentWindowLog > windowLog) {
            ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, windowLog);
            int strategy = ZSTD_getParams(level, insize, 0).cParams.strategy;
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

int64_t lzbench_zstd_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char* workmem)
{
    zstd_params_s* zstd_params = (zstd_params_s*) workmem;
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

int64_t lzbench_zstd_LDM_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t windowLog, char* workmem)
{
    zstd_params_s* zstd_params = (zstd_params_s*) workmem;
    if (!zstd_params || !zstd_params->cctx) return 0;
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_enableLongDistanceMatching, 1);
    return lzbench_zstd_compress(inbuf, insize, outbuf, outsize, level, windowLog, (char*) zstd_params);
}
#endif
