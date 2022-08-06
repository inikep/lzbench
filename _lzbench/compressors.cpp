#include "compressors.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h> // memcpy

#ifndef MAX
    #define MAX(a,b) ((a)>(b))?(a):(b)
#endif
#ifndef MIN
	#define MIN(a,b) ((a)<(b)?(a):(b))
#endif


int64_t lzbench_memcpy(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char*, bool)
{
    memcpy(outbuf, inbuf, insize);
    return insize;
}

int64_t lzbench_return_0(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char*, bool)
{
    return 0;
}


#ifndef BENCH_REMOVE_BLOSCLZ
#include "blosclz/blosclz.h"

int64_t lzbench_blosclz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
    return blosclz_compress(level, inbuf, insize, outbuf, outsize, 1);
}

int64_t lzbench_blosclz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char*, bool)
{
    return blosclz_decompress(inbuf, insize, outbuf, outsize);
}

#endif // BENCH_REMOVE_BLOSCLZ


#ifndef BENCH_REMOVE_BRIEFLZ
#include "brieflz/brieflz.h"

char* lzbench_brieflz_init(size_t insize, size_t level, size_t, const std::string&)
{
    return (char*) malloc(blz_workmem_size_level(insize, level));
}

void lzbench_brieflz_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_brieflz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
{
    if (!workmem)
        return 0;

    int64_t res = blz_pack_level(inbuf, outbuf, insize, (void*)workmem, level);

    return res;
}

int64_t lzbench_brieflz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
    return blz_depack(inbuf, outbuf, outsize);
}

#endif // BENCH_REMOVE_BRIEFLZ



#ifndef BENCH_REMOVE_BROTLI
#include "brotli/encode.h"
#include "brotli/decode.h"

int64_t lzbench_brotli_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t windowLog, char*, bool)
{
    size_t actual_osize = outsize;
    return BrotliEncoderCompress(level, windowLog, BROTLI_DEFAULT_MODE, insize, (const uint8_t*)inbuf, &actual_osize, (uint8_t*)outbuf) == 0 ? 0 : actual_osize;
}

int64_t lzbench_brotli_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
    size_t actual_osize = outsize;
    return BrotliDecoderDecompress(insize, (const uint8_t*)inbuf, &actual_osize, (uint8_t*)outbuf) == BROTLI_DECODER_RESULT_ERROR ? 0 : actual_osize;
}

#endif // BENCH_REMOVE_BROTLI



#ifndef BENCH_REMOVE_BZIP2
#include "bzip2/bzlib.h"

int64_t lzbench_bzip2_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t windowLog, char*, bool)
{
   unsigned int a_outsize = outsize;
   return BZ2_bzBuffToBuffCompress((char *)outbuf, &a_outsize, (char *)inbuf, (unsigned int)insize, level, 0, 0)==BZ_OK?a_outsize:-1;
}

int64_t lzbench_bzip2_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
   unsigned int a_outsize = outsize;
   return BZ2_bzBuffToBuffDecompress((char *)outbuf, &a_outsize, (char *)inbuf, (unsigned int)insize, 0, 0)==BZ_OK?a_outsize:-1;
}

#endif // BENCH_REMOVE_BZIP2



#ifndef BENCH_REMOVE_CRUSH
#include "crush/crush.hpp"

int64_t lzbench_crush_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	return crush::compress(level, (uint8_t*)inbuf, insize, (uint8_t*)outbuf);
}

int64_t lzbench_crush_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	return crush::decompress((uint8_t*)inbuf, (uint8_t*)outbuf, outsize);
}

#endif // BENCH_REMOVE_CRUSH




#ifndef BENCH_REMOVE_DENSITY
extern "C"
{
	#include "density/density_api.h"
}

char* lzbench_density_init(size_t insize, size_t level, size_t, const std::string&)
{
    return (char*) malloc(MAX(density_compress_safe_size(insize), density_decompress_safe_size(insize)));
}

void lzbench_density_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_density_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	density_processing_result result = density_compress((uint8_t *)inbuf, insize, (uint8_t *)outbuf, density_compress_safe_size(outsize), (DENSITY_ALGORITHM)level);
	if (result.state) 
		return 0;
		
	return result.bytesWritten;
}

int64_t lzbench_density_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	density_processing_result result = density_decompress((uint8_t *)inbuf, insize, (uint8_t *)outbuf, density_decompress_safe_size(outsize));
	if (result.state) 
		return 0;
		
	return result.bytesWritten;
}

#endif // BENCH_REMOVE_DENSITY



#ifndef BENCH_REMOVE_FASTLZ
extern "C"
{
	#include "fastlz/fastlz.h"
}

int64_t lzbench_fastlz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	return fastlz_compress_level(level, inbuf, insize, outbuf);
}

int64_t lzbench_fastlz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	return fastlz_decompress(inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_FASTLZMA2
#include "fast-lzma2/fast-lzma2.h"

int64_t lzbench_fastlzma2_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t windowLog, char* workmem, bool)
{
    size_t ret = FL2_compress(outbuf, outsize, inbuf, insize, level);
    if (FL2_isError(ret)) return 0;
    return ret;
}

int64_t lzbench_fastlzma2_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char* workmem, bool)
{
    size_t ret = FL2_decompress(outbuf, outsize, inbuf, insize);
    if (FL2_isError(ret)) return 0;
    return ret;
}
#endif // BENCH_REMOVE_FASTLZMA2


#ifndef BENCH_REMOVE_GIPFELI
#include "gipfeli/gipfeli.h"

int64_t lzbench_gipfeli_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
    int64_t res;
    util::compression::Compressor *gipfeli = util::compression::NewGipfeliCompressor();
    if (gipfeli)
    {
        util::compression::UncheckedByteArraySink sink((char*)outbuf);
        util::compression::ByteArraySource src((const char*)inbuf, insize);
        res = gipfeli->CompressStream(&src, &sink); 
        delete gipfeli; 
    }
    else res=0;
    return res;
}

int64_t lzbench_gipfeli_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
    int64_t res = 0;
    util::compression::Compressor *gipfeli = util::compression::NewGipfeliCompressor();
    if (gipfeli)
    {
        util::compression::UncheckedByteArraySink sink((char*)outbuf);
        util::compression::ByteArraySource src((const char*)inbuf, insize);
        if (gipfeli->UncompressStream(&src, &sink))
            res = outsize;
        delete gipfeli;
    }
    return res;
}

#endif



#ifndef BENCH_REMOVE_GLZA
#include "glza/GLZAcomp.h"
#include "glza/GLZAdecode.h"

int64_t lzbench_glza_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	if (GLZAcomp(insize, (uint8_t *)inbuf, &outsize, (uint8_t *)outbuf, (FILE *)0, NULL) == 0) return(0);
	return outsize;
}

int64_t lzbench_glza_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	if (GLZAdecode(insize, (uint8_t *)inbuf, &outsize, (uint8_t *)outbuf, (FILE *)0) == 0) return(0);
	return outsize;
}

#endif



#ifndef BENCH_REMOVE_LIBDEFLATE
#include "libdeflate/libdeflate.h"
int64_t lzbench_libdeflate_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
    struct libdeflate_compressor *compressor = libdeflate_alloc_compressor(level);
    if (!compressor)
        return 0;
    int64_t res = libdeflate_deflate_compress(compressor, inbuf, insize, outbuf, outsize);
    libdeflate_free_compressor(compressor);
    return res;
}
int64_t lzbench_libdeflate_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
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
#include "lizard/lizard_compress.h"
#include "lizard/lizard_decompress.h"

int64_t lzbench_lizard_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	return Lizard_compress(inbuf, outbuf, insize, outsize, level);
}

int64_t lzbench_lizard_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	return Lizard_decompress_safe(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZ4
#include "lz4/lz4.h"
#include "lz4/lz4hc.h"

int64_t lzbench_lz4_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	return LZ4_compress_default(inbuf, outbuf, insize, outsize);
}

int64_t lzbench_lz4fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	return LZ4_compress_fast(inbuf, outbuf, insize, outsize, level);
}

int64_t lzbench_lz4hc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	return LZ4_compress_HC(inbuf, outbuf, insize, outsize, level);
}

int64_t lzbench_lz4_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	return LZ4_decompress_safe(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZF
extern "C"
{
	#include "lzf/lzf.h"
}

int64_t lzbench_lzf_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	if (level == 0)
		return lzf_compress(inbuf, insize, outbuf, outsize);
	return lzf_compress_very(inbuf, insize, outbuf, outsize);
}

int64_t lzbench_lzf_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	return lzf_decompress(inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZFSE
extern "C"
{
	#include "lzfse/lzfse.h"
}

char* lzbench_lzfse_init(size_t insize, size_t level, size_t, const std::string&)
{
    return (char*) malloc(MAX(lzfse_encode_scratch_size(), lzfse_decode_scratch_size()));
}

void lzbench_lzfse_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_lzfse_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
{
	return lzfse_encode_buffer((uint8_t*)outbuf, outsize, (uint8_t*)inbuf, insize, workmem);
}

int64_t lzbench_lzfse_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char* workmem, bool)
{
	return lzfse_decode_buffer((uint8_t*)outbuf, outsize, (uint8_t*)inbuf, insize, workmem);
}

#endif



#ifndef BENCH_REMOVE_LZVN
extern "C"
{
	#include "lzfse/lzvn.h"
}

char* lzbench_lzvn_init(size_t insize, size_t level, size_t, const std::string&)
{
    return (char*) malloc(MAX(lzvn_encode_scratch_size(), lzvn_decode_scratch_size()));
}

void lzbench_lzvn_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_lzvn_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
{
	return lzvn_encode_buffer((uint8_t*)outbuf, outsize, (uint8_t*)inbuf, insize, workmem);
}

int64_t lzbench_lzvn_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char* workmem, bool)
{
	return lzvn_decode_buffer_scratch((uint8_t*)outbuf, outsize, (uint8_t*)inbuf, insize, workmem);
}

#endif



#ifndef BENCH_REMOVE_LZG
#include "liblzg/lzg.h"

int64_t lzbench_lzg_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
    lzg_encoder_config_t cfg;
    cfg.level = level;
    cfg.fast = LZG_TRUE;
    cfg.progressfun = NULL;
    cfg.userdata = NULL;
    return LZG_Encode((const unsigned char*)inbuf, insize, (unsigned char*)outbuf, outsize, &cfg);
}

int64_t lzbench_lzg_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
    return LZG_Decode((const unsigned char*)inbuf, insize, (unsigned char*)outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZHAM
#include "lzham/lzham.h"
#include <memory.h>

int64_t lzbench_lzham_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t dict_size_log, char*, bool)
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

int64_t lzbench_lzham_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t dict_size_log, char*, bool)
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



#ifndef BENCH_REMOVE_LZJB
#include "lzjb/lzjb2010.h"

int64_t lzbench_lzjb_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	return lzjb_compress2010((uint8_t*)inbuf, (uint8_t*)outbuf, insize, outsize, 0); 
}

int64_t lzbench_lzjb_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	return lzjb_decompress2010((uint8_t*)inbuf, (uint8_t*)outbuf, insize, outsize, 0);
}

#endif



#ifndef BENCH_REMOVE_LZLIB
#include "lzlib/lzlib.h"

int64_t lzbench_lzlib_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
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
 

int64_t lzbench_lzlib_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
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
#include "lzma/Alloc.h"
#include "lzma/LzmaDec.h"
#include "lzma/LzmaEnc.h"

#ifndef BENCH_REMOVE_TORNADO
static void *SzAlloc(ISzAllocPtr p, size_t size) { (void)p; return MyAlloc(size); }
static void SzFree(ISzAllocPtr p, void *address) { (void)p; MyFree(address); }
const ISzAlloc g_Alloc = { SzAlloc, SzFree };
#endif

int64_t lzbench_lzma_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	CLzmaEncProps props;
	int res;
    size_t headerSize = LZMA_PROPS_SIZE;
	SizeT out_len = outsize - LZMA_PROPS_SIZE;
	
	LzmaEncProps_Init(&props);
	props.level = level;
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

int64_t lzbench_lzma_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
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



#ifndef BENCH_REMOVE_LZMAT
#include "lzmat/lzmat.h"

int64_t lzbench_lzmat_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	uint32_t complen = outsize;
	if (lzmat_encode((uint8_t*)outbuf, &complen, (uint8_t*)inbuf, insize) != 0)
		return 0;
	return complen;
}

int64_t lzbench_lzmat_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	uint32_t decomplen = outsize;
	if (lzmat_decode((uint8_t*)outbuf, &decomplen, (uint8_t*)inbuf, insize) != 0)
		return 0;
	return decomplen;
}

#endif




#ifndef BENCH_REMOVE_LZO
#include "lzo/lzo1.h"
#include "lzo/lzo1a.h"
#include "lzo/lzo1b.h"
#include "lzo/lzo1c.h"
#include "lzo/lzo1f.h"
#include "lzo/lzo1x.h"
#include "lzo/lzo1y.h"
#include "lzo/lzo1z.h"
#include "lzo/lzo2a.h"

char* lzbench_lzo_init(size_t, size_t, size_t, const std::string&)
{
	lzo_init();

    return (char*) malloc(LZO1B_999_MEM_COMPRESS);
}

void lzbench_lzo_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_lzo1_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
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

int64_t lzbench_lzo1_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	lzo_uint decomplen = 0;

    if (lzo1_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1a_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
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

int64_t lzbench_lzo1a_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	lzo_uint decomplen = 0;

    if (lzo1a_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1b_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
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

int64_t lzbench_lzo1b_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	lzo_uint decomplen = 0;

    if (lzo1b_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1c_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
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

int64_t lzbench_lzo1c_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	lzo_uint decomplen = 0;

    if (lzo1c_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1f_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
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

int64_t lzbench_lzo1f_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	lzo_uint decomplen = 0;

    if (lzo1f_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1x_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
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

int64_t lzbench_lzo1x_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	lzo_uint decomplen = 0;

    if (lzo1x_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1y_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
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

int64_t lzbench_lzo1y_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	lzo_uint decomplen = 0;

    if (lzo1y_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

int64_t lzbench_lzo1z_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
{
	lzo_uint lzo_complen = 0;
	int res;

    if (!workmem)
        return 0;

    res = lzo1z_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    
	if (res != LZO_E_OK) return 0;
		
	return lzo_complen; 
}

int64_t lzbench_lzo1z_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	lzo_uint decomplen = 0;

    if (lzo1z_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}


int64_t lzbench_lzo2a_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
{
	lzo_uint lzo_complen = 0;
	int res;

    if (!workmem)
        return 0;

    res = lzo2a_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem);
    
	if (res != LZO_E_OK) return 0;
		
	return lzo_complen; 
}

int64_t lzbench_lzo2a_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	lzo_uint decomplen = 0;

    if (lzo2a_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0;

	return decomplen; 
}

#endif





#ifndef BENCH_REMOVE_LZRW
extern "C"
{
	#include "lzrw/lzrw.h"
}

char* lzbench_lzrw_init(size_t, size_t, size_t, const std::string&)
{
    return (char*) malloc(lzrw2_req_mem());
}

void lzbench_lzrw_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_lzrw_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
{
    if (!workmem)
        return 0;
        
	uint32_t complen = 0;
	switch (level)
	{
		default:
		case 1: lzrw1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen); break;
		case 2: lzrw1a_compress(COMPRESS_ACTION_COMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen); break;
		case 3: lzrw2_compress(COMPRESS_ACTION_COMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen); break;
		case 4: lzrw3_compress(COMPRESS_ACTION_COMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen); break;
		case 5: lzrw3a_compress(COMPRESS_ACTION_COMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen); break;
	}

	return complen;
}

int64_t lzbench_lzrw_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
{
    if (!workmem)
        return 0;
    
	uint32_t decomplen = 0;
	switch (level)
	{
		default:
		case 1: lzrw1_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen); break;
		case 2: lzrw1a_compress(COMPRESS_ACTION_DECOMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen); break;
		case 3: lzrw2_compress(COMPRESS_ACTION_DECOMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen); break;
		case 4: lzrw3_compress(COMPRESS_ACTION_DECOMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen); break;
		case 5: lzrw3a_compress(COMPRESS_ACTION_DECOMPRESS, (uint8_t*)workmem, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen); break;
	}

	return decomplen;
}

#endif



#ifndef BENCH_REMOVE_LZSSE
#include "lzsse/lzsse2/lzsse2.h"

char* lzbench_lzsse2_init(size_t insize, size_t, size_t, const std::string&)
{
    return (char*) LZSSE2_MakeOptimalParseState(insize);
}

void lzbench_lzsse2_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE2_FreeOptimalParseState((LZSSE2_OptimalParseState*) workmem);
}

int64_t lzbench_lzsse2_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
{
    if (!workmem) return 0;

    return LZSSE2_CompressOptimalParse((LZSSE2_OptimalParseState*) workmem, inbuf, insize, outbuf, outsize, level);
}

int64_t lzbench_lzsse2_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	return LZSSE2_Decompress(inbuf, insize, outbuf, outsize);
}


#include "lzsse/lzsse4/lzsse4.h"

char* lzbench_lzsse4_init(size_t insize, size_t, size_t, const std::string&)
{
    return (char*) LZSSE4_MakeOptimalParseState(insize);
}

void lzbench_lzsse4_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE4_FreeOptimalParseState((LZSSE4_OptimalParseState*) workmem);
}

int64_t lzbench_lzsse4_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
{
    if (!workmem) return 0;

    return LZSSE4_CompressOptimalParse((LZSSE4_OptimalParseState*) workmem, inbuf, insize, outbuf, outsize, level);
}

int64_t lzbench_lzsse4_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
    return LZSSE4_Decompress(inbuf, insize, outbuf, outsize);
}

char* lzbench_lzsse4fast_init(size_t, size_t, size_t, const std::string&)
{
    return (char*) LZSSE4_MakeFastParseState();
}

void lzbench_lzsse4fast_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE4_FreeFastParseState((LZSSE4_FastParseState*) workmem);
}

int64_t lzbench_lzsse4fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char* workmem, bool)
{
    if (!workmem) return 0;

    return LZSSE4_CompressFast((LZSSE4_FastParseState*) workmem, inbuf, insize, outbuf, outsize);
}


#include "lzsse/lzsse8/lzsse8.h"

char* lzbench_lzsse8_init(size_t insize, size_t, size_t, const std::string&)
{
    return (char*) LZSSE8_MakeOptimalParseState(insize);
}

void lzbench_lzsse8_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE8_FreeOptimalParseState((LZSSE8_OptimalParseState*) workmem);
}

int64_t lzbench_lzsse8_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
{
    if (!workmem) return 0;

    return LZSSE8_CompressOptimalParse((LZSSE8_OptimalParseState*) workmem, inbuf, insize, outbuf, outsize, level);
}

int64_t lzbench_lzsse8_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
    return LZSSE8_Decompress(inbuf, insize, outbuf, outsize);
}

char* lzbench_lzsse8fast_init(size_t, size_t, size_t, const std::string&)
{
    return (char*) LZSSE8_MakeFastParseState();
}

void lzbench_lzsse8fast_deinit(char* workmem)
{
    if (!workmem) return;
    LZSSE8_FreeFastParseState((LZSSE8_FastParseState*) workmem);
}

int64_t lzbench_lzsse8fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char* workmem, bool)
{
    if (!workmem) return 0;

    return LZSSE8_CompressFast((LZSSE8_FastParseState*) workmem, inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_PITHY
#include "pithy/pithy.h"

int64_t lzbench_pithy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	return pithy_Compress(inbuf, insize, outbuf, outsize, level);
}

int64_t lzbench_pithy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	size_t res = pithy_Decompress(inbuf, insize, outbuf, outsize);
//	printf("insize=%lld outsize=%lld res=%lld\n", insize, outsize, res);
	if (res)
		return outsize;
	return 0;
}

#endif


#ifndef BENCH_REMOVE_QUICKLZ
#include "quicklz/quicklz151b7.h"
#include "quicklz/quicklz.h"
#define MAX(a,b) ((a)>(b))?(a):(b) 

int64_t lzbench_quicklz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t , char*, bool)
{
    int64_t res;
    qlz150_state_compress* state = (qlz150_state_compress*) calloc(1, MAX(qlz_get_setting_3(1),MAX(qlz_get_setting_1(1), qlz_get_setting_2(1))));
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

int64_t lzbench_quicklz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t , char*, bool)
{
    int64_t res;
    qlz150_state_compress* dstate = (qlz150_state_compress*) calloc(1, MAX(qlz_get_setting_3(2),MAX(qlz_get_setting_1(2), qlz_get_setting_2(2))));
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



#ifndef BENCH_REMOVE_SHRINKER
#include "shrinker/shrinker.h"

int64_t lzbench_shrinker_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	return shrinker_compress(inbuf, outbuf, insize); 
}

int64_t lzbench_shrinker_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	return shrinker_decompress(inbuf, outbuf, outsize); 
}

#endif


#ifndef BENCH_REMOVE_SNAPPY
#include "snappy/snappy.h"

int64_t lzbench_snappy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	snappy::RawCompress(inbuf, insize, outbuf, &outsize);
	return outsize;
}

int64_t lzbench_snappy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	snappy::RawUncompress(inbuf, insize, outbuf);
	return outsize;
}

#endif




#ifndef BENCH_REMOVE_TORNADO
#include "tornado/tor_test.h"

int64_t lzbench_tornado_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	return tor_compress(level, (uint8_t*)inbuf, insize, (uint8_t*)outbuf, outsize); 
}

int64_t lzbench_tornado_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	return tor_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, outsize); 
}

#endif



#ifndef BENCH_REMOVE_UCL
#include "ucl/ucl.h"

int64_t lzbench_ucl_nrv2b_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	ucl_uint complen;
	int res = ucl_nrv2b_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, level, NULL, NULL);

	if (res != UCL_E_OK) return 0;
	return complen;
}

int64_t lzbench_ucl_nrv2b_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	ucl_uint decomplen;
	int res = ucl_nrv2b_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL);

	if (res != UCL_E_OK) return 0;
	return decomplen;
}

int64_t lzbench_ucl_nrv2d_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	ucl_uint complen;
	int res = ucl_nrv2d_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, level, NULL, NULL);

	if (res != UCL_E_OK) return 0;
	return complen;
}

int64_t lzbench_ucl_nrv2d_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	ucl_uint decomplen;
	int res = ucl_nrv2d_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL);

	if (res != UCL_E_OK) return 0;
	return decomplen;
}

int64_t lzbench_ucl_nrv2e_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	ucl_uint complen;
	int res = ucl_nrv2e_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, level, NULL, NULL);

	if (res != UCL_E_OK) return 0;
	return complen;
}

int64_t lzbench_ucl_nrv2e_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	ucl_uint decomplen;
	int res = ucl_nrv2e_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL);

	if (res != UCL_E_OK) return 0;
	return decomplen;
}

#endif



#ifndef BENCH_REMOVE_WFLZ
#include "wflz/wfLZ.h"

char* lzbench_wflz_init(size_t, size_t, size_t, const std::string&)
{
    return (char*) malloc(wfLZ_GetWorkMemSize());
}

void lzbench_wflz_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_wflz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
{
    int64_t res;
    if (!workmem)
        return 0;

    if (level == 0) 
		res = wfLZ_CompressFast((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, (uint8_t*)workmem, 0);
    else
        res = wfLZ_Compress((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, (uint8_t*)workmem, 0);
    
    return res;
}

int64_t lzbench_wflz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
    wfLZ_Decompress((const uint8_t*)inbuf, (uint8_t*)outbuf);
    return outsize;
}

#endif



#ifndef BENCH_REMOVE_XPACK
#include "xpack/lib/libxpack.h" 

typedef struct {
    struct xpack_compressor *xpackc;
    struct xpack_decompressor *xpackd;
} xpack_params_s;

char* lzbench_xpack_init(size_t insize, size_t level, size_t, const std::string&)
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

int64_t lzbench_xpack_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem, bool)
{
    xpack_params_s* xpack_params = (xpack_params_s*) workmem;
    if (!xpack_params || !xpack_params->xpackc) return 0;

    return xpack_compress(xpack_params->xpackc, inbuf, insize, outbuf, outsize);
}

int64_t lzbench_xpack_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char* workmem, bool)
{
    xpack_params_s* xpack_params = (xpack_params_s*) workmem;
    if (!xpack_params || !xpack_params->xpackd) return 0;

    size_t res = xpack_decompress(xpack_params->xpackd, inbuf, insize, outbuf, outsize, NULL);
    if (res != 0) return 0;

    return outsize;
}

#endif



#ifndef BENCH_REMOVE_XZ
#include "xz/alone.h" 

int64_t lzbench_xz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
    return xz_alone_compress(inbuf, insize, outbuf, outsize, level, 0, 0);
}

int64_t lzbench_xz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
    return xz_alone_decompress(inbuf, insize, outbuf, outsize, 0, 0, 0);
}

#endif



#ifndef BENCH_REMOVE_YALZ77
#include "yalz77/lz77.h"

int64_t lzbench_yalz77_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
  lz77::compress_t compress(level, lz77::DEFAULT_BLOCKSIZE);
  std::string compressed = compress.feed((unsigned char*)inbuf, (unsigned char*)inbuf+insize);
  if (compressed.size() > outsize) return 0;
  memcpy(outbuf, compressed.c_str(), compressed.size());
  return compressed.size();
}

int64_t lzbench_yalz77_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
  lz77::decompress_t decompress;
  std::string temp;
  decompress.feed((unsigned char*)inbuf, (unsigned char*)inbuf+insize, temp);
  const std::string& decompressed = decompress.result();
  if (decompressed.size() > outsize) return 0;
  memcpy(outbuf, decompressed.c_str(), decompressed.size());
  return decompressed.size();
}

#endif



#ifndef BENCH_REMOVE_YAPPY
#include "yappy/yappy.hpp"

char* lzbench_yappy_init(size_t insize, size_t level, size_t, const std::string&)
{
	YappyFillTables();
    return NULL;
}

int64_t lzbench_yappy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	return YappyCompress((uint8_t*)inbuf, (uint8_t*)outbuf, insize, level) - (uint8_t*)outbuf; 
}

int64_t lzbench_yappy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	return YappyUnCompress((uint8_t*)inbuf, (uint8_t*)inbuf+insize, (uint8_t*)outbuf) - (uint8_t*)outbuf; 
}

#endif



#ifndef BENCH_REMOVE_ZLIB
#include "zlib/zlib.h"

int64_t lzbench_zlib_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	uLongf zcomplen = insize;
	int err = compress2((uint8_t*)outbuf, &zcomplen, (uint8_t*)inbuf, insize, level);
	if (err != Z_OK)
		return 0;
	return zcomplen;
}

int64_t lzbench_zlib_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	uLongf zdecomplen = outsize;
	int err = uncompress((uint8_t*)outbuf, &zdecomplen, (uint8_t*)inbuf, insize); 
	if (err != Z_OK)
		return 0;
	return outsize;
}

#endif



#if !defined(BENCH_REMOVE_SLZ) && !defined(BENCH_REMOVE_ZLIB)
extern "C"
{
	#include "slz/slz.h"
}

int64_t lzbench_slz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t param2, char*, bool)
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
		blk = MIN(insize, window);

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
int64_t lzbench_slz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t param2, char*, bool)
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



#ifndef BENCH_REMOVE_ZLING
#include "libzling/libzling.h"

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

int64_t lzbench_zling_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	baidu::zling::MemInputter  inputter((uint8_t*)inbuf, insize);
	baidu::zling::MemOutputter outputter((uint8_t*)outbuf, outsize);
	baidu::zling::Encode(&inputter, &outputter, NULL, level);

	return outputter.GetOutputSize();
}

int64_t lzbench_zling_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
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
    ZSTD_DDict* ddict;
} zstd_params_s;

// dictionary could be train follow this link: https://github.com/google/brotli/issues/697
char* lzbench_zstd_init(size_t insize, size_t level, size_t windowLog, const std::string& dictionary)
{
    ZSTD_CDict* cdict = NULL;
    ZSTD_DDict* ddict = NULL;
    zstd_params_s* zstd_params = (zstd_params_s*) malloc(sizeof(zstd_params_s));
    if (!zstd_params) return NULL;
    zstd_params->cctx = ZSTD_createCCtx();
    zstd_params->dctx = ZSTD_createDCtx();
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_compressionLevel, level);
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_windowLog, windowLog);
    if (!dictionary.empty()) {
        cdict = ZSTD_createCDict(dictionary.data(), dictionary.length(), level);;
        ZSTD_CCtx_refCDict(zstd_params->cctx, cdict);
        ddict = ZSTD_createDDict(dictionary.data(), dictionary.length());
        ZSTD_DCtx_refDDict(zstd_params->dctx, ddict);
    }
    zstd_params->cdict = cdict;
    zstd_params->ddict = ddict;

    return (char*) zstd_params;
}

void lzbench_zstd_deinit(char* workmem)
{
    zstd_params_s* zstd_params = (zstd_params_s*) workmem;
    if (!zstd_params) return;
    if (zstd_params->cctx) ZSTD_freeCCtx(zstd_params->cctx);
    if (zstd_params->dctx) ZSTD_freeDCtx(zstd_params->dctx);
    if (zstd_params->cdict) ZSTD_freeCDict(zstd_params->cdict);
    if (zstd_params->ddict) ZSTD_freeDDict(zstd_params->ddict);
    free(workmem);
}

int64_t lzbench_zstd_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char* workmem, bool endstream)
{
    zstd_params_s* zstd_params = (zstd_params_s*) workmem;
    if (!zstd_params || !zstd_params->cctx) return 0;

    // From https://github.com/facebook/zstd/blob/dev/examples/streaming_compression.c
    size_t wsize = 0;
    ZSTD_EndDirective const mode = endstream ? ZSTD_e_end : ZSTD_e_continue;
    ZSTD_inBuffer input = { inbuf, insize, 0 };
    int finished;
    do {
        ZSTD_outBuffer output = { outbuf+wsize, outsize-wsize, 0 };
        const size_t remaining = ZSTD_compressStream2(zstd_params->cctx, &output , &input, mode);
        wsize += output.pos;

        finished = endstream ? (remaining == 0) : (input.pos == input.size);
    } while (!finished);

    return wsize;
}

int64_t lzbench_zstd_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char* workmem, bool)
{
    zstd_params_s* zstd_params = (zstd_params_s*) workmem;
    if (!zstd_params || !zstd_params->dctx) return 0;

    size_t last_ret = 0;
    size_t wsize = 0;
    ZSTD_inBuffer input = { inbuf, insize, 0 };
    while (input.pos < input.size) {
        ZSTD_outBuffer output = { outbuf+wsize, outsize-wsize, 0 };
        const size_t ret = ZSTD_decompressStream(zstd_params->dctx, &output , &input);
        wsize += output.pos;
        last_ret = ret;
    }

    if (last_ret != 0) {
        return 0;
    }

    return wsize;
}

char* lzbench_zstd_LDM_init(size_t insize, size_t level, size_t windowLog, const std::string& dict)
{
    zstd_params_s* zstd_params = (zstd_params_s*) lzbench_zstd_init(insize, level, windowLog, dict);
    if (!zstd_params) return NULL;
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_enableLongDistanceMatching, 1);
    return (char*) zstd_params;
}

int64_t lzbench_zstd_LDM_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t windowLog, char* workmem, bool endstream)
{
    zstd_params_s* zstd_params = (zstd_params_s*) workmem;
    if (!zstd_params || !zstd_params->cctx) return 0;
    ZSTD_CCtx_setParameter(zstd_params->cctx, ZSTD_c_enableLongDistanceMatching, 1);
    return lzbench_zstd_compress(inbuf, insize, outbuf, outsize, level, windowLog, (char*) zstd_params, endstream);
}
#endif


#ifdef BENCH_HAS_NAKAMICHI
#include "nakamichi/nakamichi.h"

int64_t lzbench_nakamichi_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*, bool)
{
	return NakaCompress(outbuf, inbuf, insize);
}

int64_t lzbench_nakamichi_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*, bool)
{
	return NakaDecompress(outbuf, inbuf, insize);
}

#endif

#ifdef BENCH_HAS_CUDA
#include <cuda_runtime.h>

char* lzbench_cuda_init(size_t insize, size_t, size_t, const std::string&)
{
    char* workmem;
    cudaMalloc(& workmem, insize);
    return workmem;
}

void lzbench_cuda_deinit(char* workmem)
{
    cudaFree(workmem);
}

int64_t lzbench_cuda_memcpy(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char* workmem, bool)
{
    cudaMemcpy(workmem, inbuf, insize, cudaMemcpyHostToDevice);
    cudaMemcpy(outbuf, workmem, insize, cudaMemcpyDeviceToHost);
    return insize;
}

int64_t lzbench_cuda_return_0(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char*, bool)
{
    return 0;
}

#ifdef BENCH_HAS_NVCOMP
#include "nvcomp/lz4.h"

typedef struct {
  size_t buffer_size;
  size_t compressed_max_size;
  size_t* compressed_size;
  cudaStream_t stream;
  char* uncompressed_d;
  char* buffer_d;
  char* compressed_d;
  nvcompLZ4FormatOpts opts;
} nvcomp_params_s;

// allocate the host and device memory buffers for the nvcom LZ4 compression and decompression
// the chunk size is configured by the compression level, 0 to 5 inclusive, corresponding to a chunk size from 32 kB to 1 MB
char* lzbench_nvcomp_init(size_t insize, size_t level, size_t, const std::string&)
{
  // allocate the host memory for the algorithm options
  nvcomp_params_s* nvcomp_params = (nvcomp_params_s*) malloc(sizeof(nvcomp_params_s));
  if (!nvcomp_params) return NULL;

  // set the chunk size based on the compression level
  nvcomp_params->opts.chunk_size = 1 << (15 + level);

  int status = 0;

  // create a CUDA stream to run the compression/decompression
  status = cudaStreamCreate(&nvcomp_params->stream);
  assert(status == cudaSuccess);

  // allocate device memory for the data to be compressed
  status = cudaMalloc(&nvcomp_params->uncompressed_d, insize);
  assert(status == cudaSuccess);

  // determine the size of the temporary buffer
  // note that the data type and the data to be compressed are not actually used
  status = nvcompLZ4CompressGetTempSize(nvcomp_params->uncompressed_d, insize, NVCOMP_TYPE_CHAR, &nvcomp_params->opts, &nvcomp_params->buffer_size);
  assert(status == nvcompSuccess);

  // allocate device memory for the temporary buffer
  status = cudaMalloc(&nvcomp_params->buffer_d, nvcomp_params->buffer_size);
  assert(status == cudaSuccess);

  // determine the size of the output buffer
  // note that the data type and the data to be compressed are not actually used
  status = nvcompLZ4CompressGetOutputSize(nvcomp_params->uncompressed_d, insize, NVCOMP_TYPE_CHAR, &nvcomp_params->opts, nvcomp_params->buffer_d, nvcomp_params->buffer_size, &nvcomp_params->compressed_max_size, 0);
  assert(status == nvcompSuccess);

  // allocate device memory for the compressed data
  status = cudaMalloc(&nvcomp_params->compressed_d, nvcomp_params->compressed_max_size);
  assert(status == cudaSuccess);

  // allocate pinned host memory for storing the compressed size from the device
  status = cudaMallocHost(&nvcomp_params->compressed_size, sizeof(size_t));
  assert(status == cudaSuccess);

  return (char*) nvcomp_params;
}

void lzbench_nvcomp_deinit(char* params)
{
  nvcomp_params_s* nvcomp_params = (nvcomp_params_s*) params;

  // free all the device memory
  cudaFree(nvcomp_params->compressed_d);
  cudaFree(nvcomp_params->buffer_d);
  cudaFree(nvcomp_params->uncompressed_d);

  // release the CUDA stream
  cudaStreamDestroy(nvcomp_params->stream);

  // free the host memory for the algorithm options
  free(nvcomp_params);
}

int64_t lzbench_nvcomp_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* params)
{
  nvcomp_params_s* nvcomp_params = (nvcomp_params_s*) params;
  int status = 0;

  // copy the uncompressed data to the device
  status = cudaMemcpyAsync(nvcomp_params->uncompressed_d, inbuf, insize, cudaMemcpyHostToDevice, nvcomp_params->stream);
  assert(status == cudaSuccess);

  // compress the data on the device
  * nvcomp_params->compressed_size = nvcomp_params->compressed_max_size;
  status = nvcompLZ4CompressAsync(
      nvcomp_params->uncompressed_d,
      insize,
      NVCOMP_TYPE_CHAR,
      &nvcomp_params->opts,
      nvcomp_params->buffer_d,
      nvcomp_params->buffer_size,
      nvcomp_params->compressed_d,
      nvcomp_params->compressed_size,
      nvcomp_params->stream);
  assert(status == nvcompSuccess);

  // limit the data to be copied back to the size available on the host
  size_t size = std::min(nvcomp_params->compressed_max_size, outsize);

  // copy the compressed data back to the host
  status = cudaMemcpyAsync(outbuf, nvcomp_params->compressed_d, size, cudaMemcpyDeviceToHost, nvcomp_params->stream);
  assert(status == cudaSuccess);

  // ensure that all operations and copies are complete, and that nvcomp_params->compressed_size is available
  status = cudaStreamSynchronize(nvcomp_params->stream);
  assert(status == cudaSuccess);

  return *nvcomp_params->compressed_size;
}

int64_t lzbench_nvcomp_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char* params)
{
  nvcomp_params_s* nvcomp_params = (nvcomp_params_s*) params;
  int status = 0;

  // check that the device buffer is large enough for the compressed data
  assert(insize <= nvcomp_params->compressed_max_size);

  // copy the compressed data to the device
  status = cudaMemcpyAsync(nvcomp_params->compressed_d, inbuf, insize, cudaMemcpyHostToDevice, nvcomp_params->stream);
  assert(status == cudaSuccess);

  // extract the metadata
  void* metadata_ptr;
  status = nvcompDecompressGetMetadata(nvcomp_params->compressed_d, insize, &metadata_ptr, nvcomp_params->stream);
  assert(status == cudaSuccess);

  // get the temporary buffer size
  size_t buffer_size;
  status = nvcompDecompressGetTempSize(metadata_ptr, &buffer_size);
  assert(status == cudaSuccess);

  // check that the temporary buffer is large enough for the decompression
  assert(buffer_size <= nvcomp_params->buffer_size);

  // get the uncompressed size
  size_t uncompressed_size;
  status = nvcompDecompressGetOutputSize(metadata_ptr, &uncompressed_size);
  assert(status == cudaSuccess);

  // check that the uncompressed buffer is large enough for the uncompressed data
  assert(uncompressed_size == outsize);

  // decompression the data on the device
  status = nvcompDecompressAsync(
      nvcomp_params->compressed_d,
      insize,
      nvcomp_params->buffer_d,
      nvcomp_params->buffer_size,
      metadata_ptr,
      nvcomp_params->uncompressed_d,
      uncompressed_size,
      nvcomp_params->stream);
  assert(status == cudaSuccess);

  // copy the uncompressed data back to the host
  status = cudaMemcpyAsync(outbuf, nvcomp_params->uncompressed_d, uncompressed_size, cudaMemcpyDeviceToHost, nvcomp_params->stream);
  assert(status == cudaSuccess);

  // ensure that all operations and copies are complete
  status = cudaStreamSynchronize(nvcomp_params->stream);
  assert(status == cudaSuccess);

  // destroy the metadata
  nvcompDecompressDestroyMetadata(metadata_ptr);

  return uncompressed_size;
}

#endif  // BENCH_HAS_NVCOMP

#endif  // BENCH_HAS_CUDA
