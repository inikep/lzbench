#include "compressors.h"
#include <stdio.h>
#include <stdint.h>


#ifndef BENCH_REMOVE_BROTLI
#include "brotli/enc/encode.h"
#include "brotli/dec/decode.h"

size_t bench_brotli_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
    brotli::BrotliParams p;
	p.quality = level;
//	p.lgwin = 24; // sliding window size. Range is 10 to 24.
//	p.lgblock = 24; // maximum input block size. Range is 16 to 24. 
    size_t actual_osize = outsize;
    return brotli::BrotliCompressBuffer(p, insize, (const uint8_t*)inbuf, &actual_osize, (uint8_t*)outbuf) == 0 ? 0 : actual_osize;
}
size_t bench_brotli_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
    size_t actual_osize = outsize;
    return BrotliDecompressBuffer(insize, (const uint8_t*)inbuf, &actual_osize, (uint8_t*)outbuf) == 0 ? 0 : actual_osize;
}

#endif




#ifndef BENCH_REMOVE_CRUSH
#include "crush/crush.hpp"

size_t bench_crush_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return crush::compress(level, (uint8_t*)inbuf, insize, (uint8_t*)outbuf);
}

size_t bench_crush_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
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

size_t bench_csc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t dict_size, size_t)
{
	MemSeqStream isss, osss;
	CSCProps p;
//	uint32_t dict_size = 64000000;

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

size_t bench_csc_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
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

size_t bench_density_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	density_buffer_processing_result result = density_buffer_compress((uint8_t *)inbuf, insize, (uint8_t *)outbuf, outsize, (DENSITY_COMPRESSION_MODE)level, DENSITY_BLOCK_TYPE_DEFAULT, NULL, NULL);
	if (result.state) 
		return 0;
		
	return result.bytesWritten;
}

size_t bench_density_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
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

size_t bench_fastlz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return fastlz_compress_level(level, inbuf, insize, outbuf);
}

size_t bench_fastlz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	return fastlz_decompress(inbuf, insize, outbuf, outsize);
}

#endif




#ifndef BENCH_REMOVE_LZ4
#include "lz4/lz4.h"
#include "lz4/lz4hc.h"

size_t bench_lz4_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return LZ4_compress_default(inbuf, outbuf, insize, outsize);
}

size_t bench_lz4fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return LZ4_compress_fast(inbuf, outbuf, insize, outsize, level);
}

size_t bench_lz4hc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return LZ4_compress_HC(inbuf, outbuf, insize, outsize, level);
}

size_t bench_lz4_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	return LZ4_decompress_safe(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZ4
#include "lz5/lz5.h"
#include "lz5/lz5hc.h"

size_t bench_lz5_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return LZ5_compress_default(inbuf, outbuf, insize, outsize);
}

size_t bench_lz5fast_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return LZ5_compress_fast(inbuf, outbuf, insize, outsize, level);
}

size_t bench_lz5hc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return LZ5_compress_HC(inbuf, outbuf, insize, outsize, level);
}

size_t bench_lz5_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	return LZ5_decompress_safe(inbuf, outbuf, insize, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZF
extern "C"
{
	#include "lzf/lzf.h"
}

size_t bench_lzf_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	if (level == 0)
		return lzf_compress(inbuf, insize, outbuf, outsize); 
	return lzf_compress_very(inbuf, insize, outbuf, outsize); 
}

size_t bench_lzf_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	return lzf_decompress(inbuf, insize, outbuf, outsize);
}

#endif



#ifndef BENCH_REMOVE_LZF
#include "lzham/lzham.h"
#include <memory.h>

size_t bench_lzham_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t dict_size, size_t)
{
	lzham_compress_params comp_params;
	memset(&comp_params, 0, sizeof(comp_params));
	comp_params.m_struct_size = sizeof(lzham_compress_params);
	comp_params.m_dict_size_log2 = dict_size;
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

size_t bench_lzham_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t dict_size, size_t)
{
	lzham_uint32 comp_adler32 = 0;
	lzham_decompress_params decomp_params;

	memset(&decomp_params, 0, sizeof(decomp_params));
	decomp_params.m_struct_size = sizeof(decomp_params);
	decomp_params.m_dict_size_log2 = dict_size;

	lzham_decompress_memory(&decomp_params, (uint8_t*)outbuf, &outsize, (const lzham_uint8 *)inbuf, insize, &comp_adler32);
	return outsize;
}

#endif



#ifndef BENCH_REMOVE_LZJB
#include "lzjb/lzjb2010.h"

size_t bench_lzjb_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return lzjb_compress2010((uint8_t*)inbuf, (uint8_t*)outbuf, insize, outsize, 0); 
}

size_t bench_lzjb_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	return lzjb_decompress2010((uint8_t*)inbuf, (uint8_t*)outbuf, insize, outsize, 0);
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

size_t bench_lzma_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	char rs[800] = { 0 };
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

size_t bench_lzma_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	char rs[800] = { 0 };
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

size_t bench_lzmat_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	uint32_t complen = outsize;
	if (lzmat_encode((uint8_t*)outbuf, &complen, (uint8_t*)inbuf, insize) != 0)
		return 0;
	return complen;
}

size_t bench_lzmat_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	uint32_t decomplen = outsize;
	if (lzmat_decode((uint8_t*)outbuf, &decomplen, (uint8_t*)inbuf, insize) != 0)
		return 0;
	return decomplen;
}

#endif




#ifndef BENCH_REMOVE_LZO
#include "lzo/lzo1b.h"
#include "lzo/lzo1c.h"
#include "lzo/lzo1f.h"
#include "lzo/lzo1x.h"
#include "lzo/lzo1y.h"
#include "lzo/lzo1z.h"
#include "lzo/lzo2a.h"

size_t bench_lzo_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t workmem, size_t)
{
	lzo_uint lzo_complen = 0;
	int res;
	
	switch (level)
	{
		default:
		case 1: res = lzo1b_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 9: res = lzo1b_9_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 99: res = lzo1b_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 999: res = lzo1b_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 1001: res = lzo1c_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 1009: res = lzo1c_9_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 1099: res = lzo1c_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 1999: res = lzo1c_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 2001: res = lzo1f_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 2999: res = lzo1f_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 3001: res = lzo1x_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 3999: res = lzo1x_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 4001: res = lzo1y_1_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 4999: res = lzo1y_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 5999: res = lzo1z_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
		case 6999: res = lzo2a_999_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &lzo_complen, (void*)workmem); break;
	}
	
	if (res != LZO_E_OK) return 0;
		
	return lzo_complen; 
}

size_t bench_lzo_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	lzo_uint decomplen = 0;

	switch (level)
	{
		default:
		case 1: case 9: case 99: case 999:
			if (lzo1b_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0; break;
		case 1001: case 1009: case 1099: case 1999:
			if (lzo1c_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0; break;
		case 2001: case 2999:
			if (lzo1f_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0; break;
		case 3001: case 3999:
			if (lzo1x_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0; break;
		case 4001: case 4999:
			if (lzo1y_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0; break;
		case 5999:
			if (lzo1z_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0; break;
		case 6999:
			if (lzo2a_decompress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL) != LZO_E_OK) return 0; break;
	}
	
	return decomplen; 
}

#endif





#ifndef BENCH_REMOVE_LZRW
extern "C"
{
	#include "lzrw/lzrw.h"
}

size_t bench_lzrw_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t workmem, size_t)
{
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

size_t bench_lzrw_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t workmem, size_t)
{
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

size_t bench_pithy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return pithy_Compress(inbuf, insize, outbuf, outsize, level);
}

size_t bench_pithy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
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

size_t bench_quicklz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t state, size_t)
{
	switch (level)
	{
		default:
		case 1:	return qlz_compress_1(inbuf, outbuf, insize, (qlz150_state_compress*)state); break;
		case 2:	return qlz_compress_2(inbuf, outbuf, insize, (qlz150_state_compress*)state); break;
		case 3:	return qlz_compress_3(inbuf, outbuf, insize, (qlz150_state_compress*)state); break;
		case 4:	return qlz_compress(inbuf, outbuf, insize, (qlz_state_compress*)state); break;
	}
}

size_t bench_quicklz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t dstate, size_t)
{
	switch (level)
	{
		default:
		case 1: return qlz_decompress_1(inbuf, outbuf, (qlz150_state_decompress*)dstate); break;
		case 2: return qlz_decompress_2(inbuf, outbuf, (qlz150_state_decompress*)dstate); break;
		case 3: return qlz_decompress_3(inbuf, outbuf, (qlz150_state_decompress*)dstate); break;
		case 4: return qlz_decompress(inbuf, outbuf, (qlz_state_decompress*)dstate); break;
	}
}

#endif



#ifndef BENCH_REMOVE_SHRINKER
#include "shrinker/shrinker.h"

size_t bench_shrinker_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	return shrinker_compress(inbuf, outbuf, insize); 
}

size_t bench_shrinker_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	return shrinker_decompress(inbuf, outbuf, outsize); 
}

#endif


#ifndef BENCH_REMOVE_SNAPPY
#include "snappy/snappy.h"

size_t bench_snappy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	snappy::RawCompress(inbuf, insize, outbuf, &outsize);
	return outsize;
}

size_t bench_snappy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	snappy::RawUncompress(inbuf, insize, outbuf);
	return outsize;
}

#endif




#ifndef BENCH_REMOVE_TORNADO
#include "tornado/tor_test.h"

size_t bench_tornado_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return tor_compress(level, (uint8_t*)inbuf, (uint8_t*)outbuf, insize); 
}

size_t bench_tornado_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	return tor_decompress((uint8_t*)inbuf, (uint8_t*)outbuf, insize); 
}

#endif



#ifndef BENCH_REMOVE_UCL
#include "ucl/ucl.h"

size_t bench_ucl_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t algo, size_t level, size_t)
{
	ucl_uint complen;
	int res;
	
	switch (algo)
	{
		default:
		case 1:	res = ucl_nrv2b_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, level, NULL, NULL); break;
		case 2:	res = ucl_nrv2d_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, level, NULL, NULL); break;
		case 3:	res = ucl_nrv2e_99_compress((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &complen, NULL, level, NULL, NULL); break;
	}

	if (res != UCL_E_OK) return 0;
	return complen;
}

size_t bench_ucl_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t algo, size_t level, size_t)
{
	ucl_uint decomplen;
	int res;
	
	switch (algo)
	{
		default:
		case 1:	res = ucl_nrv2b_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL); break;
		case 2:	res = ucl_nrv2d_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL); break;
		case 3:	res = ucl_nrv2e_decompress_8((uint8_t*)inbuf, insize, (uint8_t*)outbuf, &decomplen, NULL); break;
	}

	if (res != UCL_E_OK) return 0;
	return decomplen;
}

#endif



#ifndef BENCH_REMOVE_WFLZ
#include "wflz/wfLZ.h"

size_t bench_wflz_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t workmem, size_t)
{
    if (level == 1) 
		return wfLZ_CompressFast((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, (uint8_t*)workmem, 0);

    return wfLZ_Compress((const uint8_t*)inbuf, insize, (uint8_t*)outbuf, (uint8_t*)workmem, 0);
}

size_t bench_wflz_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
    wfLZ_Decompress((const uint8_t*)inbuf, (uint8_t*)outbuf);
    return outsize;
}

#endif



#ifndef BENCH_REMOVE_YAPPY
#include "yappy/yappy.hpp"

size_t bench_yappy_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return YappyCompress((uint8_t*)inbuf, (uint8_t*)outbuf, insize, level) - (uint8_t*)outbuf; 
}

size_t bench_yappy_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	return YappyUnCompress((uint8_t*)inbuf, (uint8_t*)inbuf+insize, (uint8_t*)outbuf) - (uint8_t*)outbuf; 
}

#endif



#ifndef BENCH_REMOVE_ZLIB
#include "zlib/zlib.h"

size_t bench_zlib_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	uLongf zcomplen = insize;
	int err = compress2((uint8_t*)outbuf, &zcomplen, (uint8_t*)inbuf, insize, level);
	if (err != Z_OK)
		return 0;
	return zcomplen;
}

size_t bench_zlib_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
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

size_t bench_zling_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	baidu::zling::MemInputter  inputter((uint8_t*)inbuf, insize);
	baidu::zling::MemOutputter outputter((uint8_t*)outbuf, outsize);
	baidu::zling::Encode(&inputter, &outputter, NULL, level);

	return outputter.GetOutputSize();
}

size_t bench_zling_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	baidu::zling::MemInputter  inputter((uint8_t*)inbuf, insize);
	baidu::zling::MemOutputter outputter((uint8_t*)outbuf, outsize);
	baidu::zling::Decode(&inputter, &outputter);

	return outputter.GetOutputSize();
} 

#endif



#ifndef BENCH_REMOVE_ZSTD
#include "zstd/zstd.h"

size_t bench_zstd_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return ZSTD_compress(outbuf, outsize, inbuf, insize);
}

size_t bench_zstd_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	return ZSTD_decompress(outbuf, outsize, inbuf, insize);
}

#endif




#ifndef BENCH_REMOVE_ZSTDHC
#include "zstd/zstdhc.h"

size_t bench_zstdhc_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t level, size_t, size_t)
{
	return ZSTD_HC_compress(outbuf, outsize, inbuf, insize, level);
}

size_t bench_zstdhc_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, size_t, size_t, size_t)
{
	return ZSTD_decompress(outbuf, outsize, inbuf, insize);
}

#endif



