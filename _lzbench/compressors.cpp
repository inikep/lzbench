#include "compressors.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h> // memcpy


int64_t lzbench_memcpy(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char* )
{
    memcpy(outbuf, inbuf, insize);
    return insize;
}

int64_t lzbench_return_0(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char* )
{
    return 0;
}


#ifndef BENCH_REMOVE_BLOSCLZ
#include "blosclz/blosclz.h"

int64_t lzbench_blosclz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
    return blosclz_compress(level, inbuf, insize, outbuf, outsize, 1);
}

int64_t lzbench_blosclz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t , size_t, char*)
{
    return blosclz_decompress(inbuf, insize, outbuf, outsize);
}

#endif


#ifndef BENCH_REMOVE_BRIEFLZ
#include "brieflz/brieflz.h"

char* lzbench_brieflz_init(size_t insize)
{
    return (char*) malloc(blz_workmem_size(insize));
}

void lzbench_brieflz_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_brieflz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
{
    if (!workmem)
        return 0;

    int64_t res = blz_pack(inbuf, outbuf, insize, (void*)workmem);

    return res;
}

int64_t lzbench_brieflz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
    return blz_depack_safe(inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_BROTLI
#include "brotli/enc/encode.h"
#include "brotli/dec/decode.h"

int64_t lzbench_brotli_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t windowLog, char*)
{
    brotli::BrotliParams p;
	p.quality = level;
    if (windowLog) p.lgwin = windowLog; // sliding window size. Range is 10 to 24.
//	p.lgblock = 24; // maximum input block size. Range is 16 to 24.

    size_t actual_osize = outsize;
    return brotli::BrotliCompressBuffer(p, insize, (const uint8_t*)inbuf, &actual_osize, (uint8_t*)outbuf) == 0 ? 0 : actual_osize;
}
int64_t lzbench_brotli_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
    size_t actual_osize = outsize;
    return BrotliDecompressBuffer(insize, (const uint8_t*)inbuf, &actual_osize, (uint8_t*)outbuf) == 0 ? 0 : actual_osize;
}

#endif




#ifndef BENCH_REMOVE_CRUSH
#include "crush/crush.hpp"

int64_t lzbench_crush_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return crush::compress(level, (uint8_t*)inbuf, insize, (uint8_t*)outbuf);
}

int64_t lzbench_crush_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return crush::decompress((uint8_t*)inbuf, (uint8_t*)outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_CSC
#include "libcsc/csc_enc.h"
#include "libcsc/csc_dec.h"
#include <string.h> // memcpy

struct MemSeqStream
{
    union {
        ISeqInStream is;
        ISeqOutStream os;
    };
    char *buf;
	size_t len;
};


int stdio_read(void *p, void *buf, size_t *size)
{
    MemSeqStream *sss = (MemSeqStream *)p;
//    *size = fread(buf, 1, *size, sss->f);
	if (*size > sss->len)
		*size = sss->len;
	memcpy(buf, sss->buf, *size);
	sss->buf += *size;
	sss->len -= *size;
    return 0;
}

size_t stdio_write(void *p, const void *buf, size_t size)
{
    MemSeqStream *sss = (MemSeqStream *)p;
	
	memcpy(sss->buf, buf, size);
	sss->buf += size;
	sss->len += size;
    return size;
}

int64_t lzbench_csc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t dict_size, char*)
{
	MemSeqStream isss, osss;
	CSCProps p;
	if (!dict_size) dict_size = 1<<26;

	if (insize < dict_size)
		dict_size = insize;

	CSCEncProps_Init(&p, dict_size, level);
	CSCEnc_WriteProperties(&p, (uint8_t*)outbuf, 0);

    isss.is.Read = stdio_read;
    isss.buf = inbuf;
	isss.len = insize;
	
    osss.os.Write = stdio_write;
    osss.buf = outbuf + CSC_PROP_SIZE;
	osss.len = CSC_PROP_SIZE;
	
	CSCEncHandle h = CSCEnc_Create(&p, (ISeqOutStream*)&osss);
	CSCEnc_Encode(h, (ISeqInStream*)&isss, NULL);
	CSCEnc_Encode_Flush(h);
	CSCEnc_Destroy(h);

//	printf("Estimated memory usage: %llu MB\n", CSCEnc_EstMemUsage(&p) / 1048576ull);
//	printf("insize=%lld osss.len=%lld\n", insize, osss.len);
	
	return osss.len;
}

int64_t lzbench_csc_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	MemSeqStream isss, osss;
	CSCProps p;

	CSCDec_ReadProperties(&p, (uint8_t*)inbuf);

    isss.is.Read = stdio_read;
    isss.buf = inbuf + CSC_PROP_SIZE;
	isss.len = insize - CSC_PROP_SIZE;
	
    osss.os.Write = stdio_write;
    osss.buf = outbuf;
	osss.len = 0;

	CSCDecHandle h = CSCDec_Create(&p, (ISeqInStream*)&isss);
	CSCDec_Decode(h, (ISeqOutStream*)&osss, NULL);
	CSCDec_Destroy(h);
	
	return osss.len;
}

#endif



#ifndef BENCH_REMOVE_DENSITY
extern "C"
{
	#include "density/density_api.h"
}

int64_t lzbench_density_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	density_buffer_processing_result result = density_buffer_compress((uint8_t *)inbuf, insize, (uint8_t *)outbuf, outsize, (DENSITY_COMPRESSION_MODE)level, DENSITY_BLOCK_TYPE_DEFAULT, NULL, NULL);
	if (result.state) 
		return 0;
		
	return result.bytesWritten;
}

int64_t lzbench_density_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
    density_buffer_processing_result result = density_buffer_decompress((uint8_t *)inbuf, insize, (uint8_t *)outbuf, outsize, NULL, NULL);
	if (result.state) 
		return 0;
		
	return result.bytesWritten;
}

#endif



#ifndef BENCH_REMOVE_FASTLZ
extern "C"
{
	#include "fastlz/fastlz.h"
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


#ifndef BENCH_REMOVE_GIPFELI
#include "gipfeli/gipfeli.h"

int64_t lzbench_gipfeli_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
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

int64_t lzbench_gipfeli_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
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




#ifndef BENCH_REMOVE_LZ4
#include "lz4/lz4.h"
#include "lz4/lz4hc.h"

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



#ifndef BENCH_REMOVE_LZ5
#include "lz5/lz5.h"
#include "lz5/lz5hc.h"

int64_t lzbench_lz5_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return LZ5_compress_default(inbuf, outbuf, insize, outsize);
}

int64_t lzbench_lz5fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return LZ5_compress_fast(inbuf, outbuf, insize, outsize, level);
}

int64_t lzbench_lz5hc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return LZ5_compress_HC(inbuf, outbuf, insize, outsize, level);
}

int64_t lzbench_lz5_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return LZ5_decompress_safe(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZF
extern "C"
{
	#include "lzf/lzf.h"
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



#ifndef BENCH_REMOVE_LZG
#include "liblzg/lzg.h"

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
#include "lzham/lzham.h"
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



#ifndef BENCH_REMOVE_LZJB
#include "lzjb/lzjb2010.h"

int64_t lzbench_lzjb_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return lzjb_compress2010((uint8_t*)inbuf, (uint8_t*)outbuf, insize, outsize, 0); 
}

int64_t lzbench_lzjb_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return lzjb_decompress2010((uint8_t*)inbuf, (uint8_t*)outbuf, insize, outsize, 0);
}

#endif



#ifndef BENCH_REMOVE_LZLIB
#include "lzlib/lzlib.h"

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
#include "lzma/Alloc.h"
#include "lzma/LzmaDec.h"
#include "lzma/LzmaEnc.h"

static void *SzAlloc(void *p, size_t size) { p = p; return MyAlloc(size); }
static void SzFree(void *p, void *address) { p = p; MyFree(address); }
static ISzAlloc g_Alloc = { SzAlloc, SzFree };

int64_t lzbench_lzma_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
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



#ifndef BENCH_REMOVE_LZMAT
#include "lzmat/lzmat.h"

int64_t lzbench_lzmat_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	uint32_t complen = outsize;
	if (lzmat_encode((uint8_t*)outbuf, &complen, (uint8_t*)inbuf, insize) != 0)
		return 0;
	return complen;
}

int64_t lzbench_lzmat_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
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

char* lzbench_lzo_init(size_t )
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





#ifndef BENCH_REMOVE_LZRW
extern "C"
{
	#include "lzrw/lzrw.h"
}

char* lzbench_lzrw_init(size_t )
{
    return (char*) malloc(lzrw2_req_mem());
}

void lzbench_lzrw_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_lzrw_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
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

int64_t lzbench_lzrw_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
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



#ifndef BENCH_REMOVE_PITHY
#include "pithy/pithy.h"

int64_t lzbench_pithy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return pithy_Compress(inbuf, insize, outbuf, outsize, level);
}

int64_t lzbench_pithy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
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

int64_t lzbench_quicklz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t , char*)
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

int64_t lzbench_quicklz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t , char*)
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

int64_t lzbench_shrinker_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return shrinker_compress(inbuf, outbuf, insize); 
}

int64_t lzbench_shrinker_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return shrinker_decompress(inbuf, outbuf, outsize); 
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



#ifndef BENCH_REMOVE_WFLZ
#include "wflz/wfLZ.h"

char* lzbench_wflz_init(size_t )
{
    return (char*) malloc(wfLZ_GetWorkMemSize());
}

void lzbench_wflz_deinit(char* workmem)
{
    free(workmem);
}

int64_t lzbench_wflz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char* workmem)
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

int64_t lzbench_wflz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
    wfLZ_Decompress((const uint8_t*)inbuf, (uint8_t*)outbuf);
    return outsize;
}

#endif




#ifndef BENCH_REMOVE_XZ
#include "xz/alone_encoder.h" 

int64_t lzbench_xz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
    return xz_alone_compress(inbuf, insize, outbuf, outsize, level, 0, 0);
}

int64_t lzbench_xz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
    return xz_alone_decompress(inbuf, insize, outbuf, outsize, 0, 0, 0);
}

#endif




#ifndef BENCH_REMOVE_YALZ77
#include "yalz77/lz77.h"

int64_t lzbench_yalz77_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
  lz77::compress_t compress(level, lz77::DEFAULT_BLOCKSIZE);
  std::string compressed = compress.feed((unsigned char*)inbuf, (unsigned char*)inbuf+insize);
  if (compressed.size() > outsize) return 0;
  memcpy(outbuf, compressed.c_str(), compressed.size());
  return compressed.size();
}

int64_t lzbench_yalz77_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
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

char* lzbench_yappy_init(size_t insize)
{
	YappyFillTables();
    return NULL;
}

int64_t lzbench_yappy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, char*)
{
	return YappyCompress((uint8_t*)inbuf, (uint8_t*)outbuf, insize, level) - (uint8_t*)outbuf; 
}

int64_t lzbench_yappy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return YappyUnCompress((uint8_t*)inbuf, (uint8_t*)inbuf+insize, (uint8_t*)outbuf) - (uint8_t*)outbuf; 
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
#include "zstd/zstd.h"
#include "zstd/zstd_static.h"

int64_t lzbench_zstd_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t windowLog, char*)
{
    size_t res;
        
    ZSTD_CCtx* cctx = ZSTD_createCCtx();
    if (!cctx) return 0;
  
    ZSTD_parameters params;
    params.cParams = ZSTD_getCParams(level, insize, 0);
    params.fParams.contentSizeFlag = 1;
    ZSTD_adjustCParams(&params.cParams, insize, 0);
    if (windowLog)
    {
        params.cParams.windowLog = windowLog;
        params.cParams.chainLog = windowLog + ((params.cParams.strategy == ZSTD_btlazy2) || (params.cParams.strategy == ZSTD_btopt));
    }

    res = ZSTD_compressBegin_advanced(cctx, NULL, 0, params, insize);
    if (ZSTD_isError(res)) return res;

    res = ZSTD_compressContinue(cctx, outbuf, outsize, inbuf, insize);
    if (ZSTD_isError(res)) return res;

    size_t res2 = ZSTD_compressEnd(cctx, outbuf+res, outsize-res);
    if (ZSTD_isError(res2)) return res2;

    ZSTD_freeCCtx(cctx);
    return res + res2;
//	return ZSTD_compress(outbuf, outsize, inbuf, insize, level);
}

int64_t lzbench_zstd_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, char*)
{
	return ZSTD_decompress(outbuf, outsize, inbuf, insize);
}

#endif
