// has to be compiled separated because of collisions with LZMA's 7zTypes.h

#ifndef BENCH_REMOVE_CSC
#include "compressors.h"
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

	CSCEncHandle h = CSCEnc_Create(&p, (ISeqOutStream*)&osss, NULL);
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

	CSCDecHandle h = CSCDec_Create(&p, (ISeqInStream*)&isss, NULL);
	CSCDec_Decode(h, (ISeqOutStream*)&osss, NULL);
	CSCDec_Destroy(h);

	return osss.len;
}

#endif
