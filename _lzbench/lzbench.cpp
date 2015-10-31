/*
(C) 2011-2015 by Przemyslaw Skibinski (inikep@gmail.com)

    LICENSE

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 3 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details at
    Visit <http://www.gnu.org/copyleft/gpl.html>.

*/

#define _CRT_SECURE_NO_WARNINGS
#define PROGNAME "lzbench"
#define PROGVERSION "0.7"

#define MAX(a,b) ((a)>(b))?(a):(b)
#ifndef MIN
	#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(WIN64) || defined(_WIN64)
	#define WINDOWS
#endif

#define __STDC_FORMAT_MACROS // now PRIu64 will work
#include <inttypes.h> // now PRIu64 will work
#define _FILE_OFFSET_BITS 64  // turn off_t into a 64-bit type for ftello() and fseeko()

#include <vector>
#include <numeric>
#include <algorithm> // sort
#include <stdlib.h> 
#include <stdio.h> 
#include <stdint.h> 
#include <string.h> 
#include "compressors.h"
#include "pithy/pithy.h"
#include "lzo/lzo1b.h"
#include "yappy/yappy.hpp"
#include "quicklz/quicklz.h"
#include "wflz/wfLZ.h"
extern "C"
{
	#include "lzrw/lzrw.h"
}

#ifdef WINDOWS
	#include <windows.h>
	#define InitTimer(x) if (!QueryPerformanceFrequency(&x)) { printf("QueryPerformance not present"); };
	#define GetTime(x) QueryPerformanceCounter(&x); 
	#define GetDiffTime(ticksPerSecond, start_ticks, end_ticks) (1000*(end_ticks.QuadPart - start_ticks.QuadPart)/ticksPerSecond.QuadPart)
	void uni_sleep(UINT usec) { Sleep(usec); };
	#ifndef __GNUC__
		#define fseeko64 _fseeki64 
		#define ftello64 _ftelli64
	#endif
	#define PROGOS "Windows"
#else
	#include <time.h>   
	#include <unistd.h>
	#include <sys/resource.h>
	typedef struct timespec LARGE_INTEGER;
	#define InitTimer(x) 
	#define GetTime(x) if(clock_gettime( CLOCK_REALTIME, &x) == -1 ){ printf("clock_gettime error"); };
	#define GetDiffTime(ticksPerSecond, start_ticks, end_ticks) (1000*( end_ticks.tv_sec - start_ticks.tv_sec ) + ( end_ticks.tv_nsec - start_ticks.tv_nsec )/1000000)
	void uni_sleep(uint32_t usec) { usleep(usec * 1000); };
	#define PROGOS "Linux"
#endif

#define ITERS(count) for(int ii=0; ii<count; ii++)

bool show_full_stats = false;
bool turbobench_format = false;

void print_stats(const char* func_name, int level, std::vector<uint32_t> &ctime, std::vector<uint32_t> &dtime, uint32_t insize, uint32_t outsize)
{
	std::sort(ctime.begin(), ctime.end());
	std::sort(dtime.begin(), dtime.end());

	uint32_t cmili_fastest = ctime[0];
	uint32_t dmili_fastest = dtime[0];
	uint32_t cmili_med = ctime[ctime.size()/2];
	uint32_t dmili_med = dtime[dtime.size()/2];
	uint32_t cmili_avg = std::accumulate(ctime.begin(),ctime.end(),0) / ctime.size();
	uint32_t dmili_avg = std::accumulate(dtime.begin(),dtime.end(),0) / dtime.size();

	if (cmili_fastest == 0) cmili_fastest = 1;
	if (dmili_fastest == 0) dmili_fastest = 1;
	if (cmili_med == 0) cmili_med = 1;
	if (dmili_med == 0) dmili_med = 1;
	if (cmili_avg == 0) cmili_avg = 1;
	if (dmili_avg == 0) dmili_avg = 1;


	char desc[256];
	snprintf(desc, sizeof(desc), "%s", func_name);

	int len = strlen(func_name);
	if (level > 0 && len >= 2)
	{
		desc[len - 1] += level % 10;
		desc[len - 2] += level / 10;
	}
	
	if (show_full_stats)
	{
		printf("%-19s fastest %d ms (%d MB/s), %d, %d ms (%d MB/s)\n", desc, cmili_fastest, insize / cmili_fastest / 1024, outsize, dmili_fastest, insize / dmili_fastest / 1024);
		printf("%-19s median  %d ms (%d MB/s), %d, %d ms (%d MB/s)\n", desc, cmili_med, insize / cmili_med / 1024, outsize, dmili_med, insize / dmili_med / 1024);
		printf("%-19s average %d ms (%d MB/s), %d, %d ms (%d MB/s)\n", desc, cmili_avg, insize / cmili_avg / 1024, outsize, dmili_avg, insize / dmili_avg / 1024);
	}
	else
	{
		if (turbobench_format)
			printf("%12d%6.1f%9.2f%9.2f  %s\n", outsize, outsize * 100.0/ insize, insize / cmili_fastest / 1024.0, insize / dmili_fastest / 1024.0, desc);
		else
        {
			printf("| %-27s ", desc);
			if (insize / cmili_fastest / 1024 < 10) printf("|%6.2f MB/s ", insize / cmili_fastest / 1024.0); else printf("|%6d MB/s ", insize / cmili_fastest / 1024); 
			if (insize / dmili_fastest / 1024 < 10) printf("|%6.2f MB/s ", insize / dmili_fastest / 1024.0); else printf("|%6d MB/s ", insize / dmili_fastest / 1024); 
			printf("|%12d |%6.2f |\n", outsize, outsize * 100.0/ insize);
        }
	}

	ctime.clear();
	dtime.clear();
};


uint32_t common(uint8_t *p1, uint8_t *p2, uint32_t count)
{
	uint32_t size = 0;

	while (count-->0)
	{
		if (*(p1++) != *(p2++))
			break;
		size++;
	}

	if (count>0)
		printf("count=%d  %d %d %d %d!=%d %d %d %d\n", count, p1[-1], p1[0], p1[1], p1[2], p2[-1], p2[0], p2[1], p2[2]);

	return size;
}


void check_decompression(uint8_t *inbuf, uint32_t inlen, uint8_t *decomp, uint32_t outlen, std::vector<uint32_t> &ctime, std::vector<uint32_t> &dtime, uint32_t comp_time, uint32_t decomp_time)
{
	ctime.push_back(comp_time);
	dtime.push_back(decomp_time);

	if (inlen != outlen)
		printf("ERROR: inlen[%u] != outlen[%u]\n", inlen, outlen);

	if (memcmp(inbuf, decomp, inlen) != 0)
		printf("ERROR: common=%d\n", common(inbuf, decomp, inlen));

	memset(decomp, 0, inlen); // clear output buffer
	uni_sleep(1); // give processor to other processes
}


size_t bench_compress(compress_func compress, size_t chunk_size, std::vector<size_t> &compr_lens, uint8_t *inbuf, size_t insize, uint8_t *outbuf, size_t outsize, size_t param1, size_t param2, size_t param3)
{
	size_t clen, part, sum = 0;
	compr_lens.clear();
	
	while (insize > 0)
	{
		part = MIN(insize, chunk_size);
		clen = compress((char*)inbuf, part, (char*)outbuf, outsize, param1, param2, param3);
		if (clen <= 0) return 0;

	//	printf("part=%lld clen=%lld\n", part, clen);
		inbuf += part;
		insize -= part;
		outbuf += clen;
		outsize -= clen;
		compr_lens.push_back(clen);
		sum += clen;
	}
	return sum;
}


size_t bench_decompress(compress_func decompress, std::vector<size_t> &compr_lens, uint8_t *inbuf, size_t insize, uint8_t *outbuf, size_t outsize, size_t param1, size_t param2, size_t param3)
{
	int num=0;
	size_t dlen, part, sum = 0;

	while (insize > 0)
	{
		part = compr_lens[num++];
		if (part > insize) return 0;
		dlen = decompress((char*)inbuf, part, (char*)outbuf, outsize, param1, param2, param3);
		if (dlen <= 0) return 0;

	//	printf("part=%lld dlen=%lld\n", part, dlen);
		inbuf += part;
		insize -= part;
		outbuf += dlen;
		outsize -= dlen;
		sum += dlen;
	}
	
	return sum;
}


void bench_test(const char* func_name, int level, compress_func compress, compress_func decompress, size_t chunk_size, int iters, uint8_t *inbuf, size_t insize, uint8_t *compbuf, size_t comprsize, uint8_t *decomp, LARGE_INTEGER ticksPerSecond, size_t param1, size_t param2, size_t param3)
{
	LARGE_INTEGER start_ticks, mid_ticks, end_ticks, start_all;
	size_t complen, decomplen;
	std::vector<uint32_t> ctime, dtime;
	std::vector<size_t> compr_lens;
	
	if (!compress || !decompress) return;
	
	ITERS(iters)
	{
		GetTime(start_ticks);
		complen = bench_compress(compress, chunk_size, compr_lens, inbuf, insize, compbuf, comprsize, param1, param2, param3);
		GetTime(mid_ticks);
		decomplen = bench_decompress(decompress, compr_lens, compbuf, complen, decomp, insize, param1, param2, param3);
		GetTime(end_ticks);
		check_decompression(inbuf, insize, decomp, decomplen, ctime, dtime, GetDiffTime(ticksPerSecond, start_ticks, mid_ticks), GetDiffTime(ticksPerSecond, mid_ticks, end_ticks));
	}
	print_stats(func_name, level, ctime, dtime, insize, complen);
}


void benchmark(FILE* in, int iters, uint32_t chunk_size, int cspeed)
{
	std::vector<uint32_t> ctime, dtime;
	LARGE_INTEGER ticksPerSecond, start_ticks, mid_ticks, end_ticks, start_all;
	uint32_t comprsize, insize;
	uint8_t *inbuf, *compbuf, *decomp, *work;

	InitTimer(ticksPerSecond);

	fseek(in, 0L, SEEK_END);
	insize = ftell(in);
	rewind(in);

#ifndef BENCH_REMOVE_PITHY
	comprsize = pithy_MaxCompressedLength(insize);
#else
	comprsize = insize + 2048;
#endif

//	printf("insize=%lld comprsize=%lld\n", insize, comprsize);
	inbuf = (uint8_t*)malloc(insize + 2048);
	compbuf = (uint8_t*)malloc(comprsize);
	decomp = (uint8_t*)calloc(1, insize + 2048);

	if (!inbuf || !compbuf || !decomp)
	{
		printf("Not enough memory!");
		exit(1);
	}

	insize = fread(inbuf, 1, insize, in);
	if (chunk_size > insize) chunk_size = insize;

	GetTime(start_all);

	ITERS(iters)
	{
		GetTime(start_ticks);
		memcpy(compbuf, inbuf, insize);
		GetTime(mid_ticks);
		memcpy(decomp, compbuf, insize);
		GetTime(end_ticks);
		check_decompression(inbuf, insize, decomp, insize, ctime, dtime, GetDiffTime(ticksPerSecond, start_ticks, mid_ticks), GetDiffTime(ticksPerSecond, mid_ticks, end_ticks));
	}
    printf("| Compressor name             | Compression| Decompress.| Compr. size | Ratio |\n");
	print_stats("memcpy", 0, ctime, dtime, insize, insize);

	qlz150_state_compress* state;
	int state_size, dstate_size;

	if (cspeed <= 200) bench_test("lz5 r131", 0, bench_lz5_compress, bench_lz5_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);
	if (cspeed <= 40) bench_test("lz5hc r131 -1", 0, bench_lz5hc_compress, bench_lz5_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);
	if (cspeed <= 15) bench_test("lz5hc r131 -4", 0, bench_lz5hc_compress, bench_lz5_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 4, 0, 0);
	if (cspeed <= 3) bench_test("lz5hc r131 -9", 0, bench_lz5hc_compress, bench_lz5_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 9, 0, 0);

/*
	if (cspeed <= 487) bench_test("lz4 r131", 0, bench_lz4_compress, bench_lz4_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);
	for (int level = 1; level <= 16; level+=2)
        if (cspeed <= 107) bench_test("lz4hc r131 -00", level, bench_lz4hc_compress, bench_lz4_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, level, 0, 0);


	if (cspeed <= 487) bench_test("lz5 r131", 0, bench_lz5_compress, bench_lz5_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);

	for (int level = 1; level <= 11; level+=2)
        if (cspeed <= 107) bench_test("lz5hc r131 -0", level, bench_lz5hc_compress, bench_lz5_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, level, 0, 0);

	goto done;
*/


	if (cspeed <= 102) bench_test("brotli 2015-10-29 level 0", 0, bench_brotli_compress, bench_brotli_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);
	if (cspeed <= 75) bench_test("brotli 2015-10-29 level 3", 0, bench_brotli_compress, bench_brotli_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3, 0, 0);
	if (cspeed <= 15) bench_test("brotli 2015-10-29 level 6", 0, bench_brotli_compress, bench_brotli_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 6, 0, 0);
	if (cspeed <= 4) bench_test("brotli 2015-10-29 level 9", 0, bench_brotli_compress, bench_brotli_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 9, 0, 0);
	if (cspeed <= 0) bench_test("brotli 2015-10-29 level 11", 0, bench_brotli_compress, bench_brotli_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 11, 0, 0);

	if (cspeed <= 33) bench_test("crush 1.0 level 0", 0, bench_crush_compress, bench_crush_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);
	if (cspeed <= 4) bench_test("crush 1.0 level 1", 0, bench_crush_compress, bench_crush_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);

	if (cspeed <= 20) bench_test("csc 3.3 level 1", 0, bench_csc_compress, bench_csc_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 1<<24, 0);
	if (cspeed <= 13) bench_test("csc 3.3 level 2", 0, bench_csc_compress, bench_csc_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 2, 1<<24, 0);
	if (cspeed <= 8) bench_test("csc 3.3 level 3", 0, bench_csc_compress, bench_csc_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3, 1<<24, 0);
	if (cspeed <= 6) bench_test("csc 3.3 level 4", 0, bench_csc_compress, bench_csc_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 4, 1<<24, 0);
	if (cspeed <= 4) bench_test("csc 3.3 level 5", 0, bench_csc_compress, bench_csc_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 5, 1<<24, 0);

	if (cspeed <= 742) bench_test("density 0.12.5 beta level 1", 0, bench_density_compress, bench_density_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);
	if (cspeed <= 463) bench_test("density 0.12.5 beta level 2", 0, bench_density_compress, bench_density_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 2, 0, 0);
	if (cspeed <= 178) bench_test("density 0.12.5 beta level 3", 0, bench_density_compress, bench_density_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3, 0, 0);

	if (cspeed <= 236) bench_test("fastlz 0.1 level 1", 0, bench_fastlz_compress, bench_fastlz_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);
	if (cspeed <= 255) bench_test("fastlz 0.1 level 2", 0, bench_fastlz_compress, bench_fastlz_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 2, 0, 0);

	if (cspeed <= 487) bench_test("lz4 r131", 0, bench_lz4_compress, bench_lz4_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);
	if (cspeed <= 533) bench_test("lz4fast r131 acc=3", 0, bench_lz4fast_compress, bench_lz4_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3, 0, 0);
	if (cspeed <= 806) bench_test("lz4fast r131 acc=17", 0, bench_lz4fast_compress, bench_lz4_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 17, 0, 0);
	if (cspeed <= 107) bench_test("lz4hc r131 -1", 0, bench_lz4hc_compress, bench_lz4_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);
	if (cspeed <= 60) bench_test("lz4hc r131 -4", 0, bench_lz4hc_compress, bench_lz4_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 4, 0, 0);
	if (cspeed <= 20) bench_test("lz4hc r131 -9", 0, bench_lz4hc_compress, bench_lz4_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 9, 0, 0);

	if (cspeed <= 200) bench_test("lz5 r131", 0, bench_lz5_compress, bench_lz5_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);
	if (cspeed <= 40) bench_test("lz5hc r131 -1", 0, bench_lz5hc_compress, bench_lz5_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);
	if (cspeed <= 15) bench_test("lz5hc r131 -4", 0, bench_lz5hc_compress, bench_lz5_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 4, 0, 0);
	if (cspeed <= 3) bench_test("lz5hc r131 -9", 0, bench_lz5hc_compress, bench_lz5_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 9, 0, 0);

	if (cspeed <= 265) bench_test("lzf level 0", 0, bench_lzf_compress, bench_lzf_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);
	if (cspeed <= 270) bench_test("lzf level 1", 0, bench_lzf_compress, bench_lzf_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);

	if (cspeed <= 7) bench_test("lzham 1.0 -m0d26 -0", 0, bench_lzham_compress, bench_lzham_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 26, 0);
	if (cspeed <= 2) bench_test("lzham 1.0 -m0d26 -0", 1, bench_lzham_compress, bench_lzham_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 26, 0);

	if (cspeed <= 239) bench_test("lzjb 2010", 0, bench_lzjb_compress, bench_lzjb_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);


	if (cspeed <= 18) bench_test("lzma 9.38 level 0", 0, bench_lzma_compress, bench_lzma_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);
	if (cspeed <= 17) bench_test("lzma 9.38 level 1", 0, bench_lzma_compress, bench_lzma_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);
	if (cspeed <= 15) bench_test("lzma 9.38 level 2", 0, bench_lzma_compress, bench_lzma_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 2, 0, 0);
	if (cspeed <= 11) bench_test("lzma 9.38 level 3", 0, bench_lzma_compress, bench_lzma_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3, 0, 0);
	if (cspeed <= 10) bench_test("lzma 9.38 level 4", 0, bench_lzma_compress, bench_lzma_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 4, 0, 0);
	if (cspeed <= 2) bench_test("lzma 9.38 level 5", 0, bench_lzma_compress, bench_lzma_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 5, 0, 0);

	if (cspeed <= 23) bench_test("lzmat 1.01", 0, bench_lzmat_compress, bench_lzmat_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);


	work=(uint8_t*)calloc(1, LZO1B_999_MEM_COMPRESS);
	if (work)
	{
		lzo_init();
		if (cspeed <= 181) bench_test("lzo1b 2.09 -1", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, (size_t)work, 0);
		if (cspeed <= 124) bench_test("lzo1b 2.09 -9", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 9, (size_t)work, 0);
		if (cspeed <= 83) bench_test("lzo1b 2.09 -99", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 99, (size_t)work, 0);
		if (cspeed <= 8) bench_test("lzo1b 2.09 -999", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 999, (size_t)work, 0);
		if (cspeed <= 188) bench_test("lzo1c 2.09 -1", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1001, (size_t)work, 0);
		if (cspeed <= 108) bench_test("lzo1c 2.09 -9", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1009, (size_t)work, 0);
		if (cspeed <= 80) bench_test("lzo1c 2.09 -99", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1099, (size_t)work, 0);
		if (cspeed <= 11) bench_test("lzo1c 2.09 -999", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1999, (size_t)work, 0);
		if (cspeed <= 172) bench_test("lzo1f 2.09 -1", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 2001, (size_t)work, 0);
		if (cspeed <= 10) bench_test("lzo1f 2.09 -999", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 2999, (size_t)work, 0);
		if (cspeed <= 414) bench_test("lzo1x 2.09 -1", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3001, (size_t)work, 0);
		if (cspeed <= 4) bench_test("lzo1x 2.09 -999", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3999, (size_t)work, 0);
		if (cspeed <= 424) bench_test("lzo1y 2.09 -1", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 4001, (size_t)work, 0);
		if (cspeed <= 4) bench_test("lzo1y 2.09 -999", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 4999, (size_t)work, 0);
		if (cspeed <= 4) bench_test("lzo1z 2.09 -999", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 5999, (size_t)work, 0);
		if (cspeed <= 11) bench_test("lzo2a 2.09 -999", 0, bench_lzo_compress, bench_lzo_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 6999, (size_t)work, 0);
		free(work);
	}

	work=(uint8_t*)calloc(1, lzrw2_req_mem());
	if (work)
	{
		if (cspeed <= 179) bench_test("lzrw1", 0, bench_lzrw_compress, bench_lzrw_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, (size_t)work, 0);
		if (cspeed <= 180) bench_test("lzrw1a", 0, bench_lzrw_compress, bench_lzrw_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 2, (size_t)work, 0);
		if (cspeed <= 195) bench_test("lzrw2", 0, bench_lzrw_compress, bench_lzrw_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3, (size_t)work, 0);
		if (cspeed <= 208) bench_test("lzrw3", 0, bench_lzrw_compress, bench_lzrw_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 4, (size_t)work, 0);
		if (cspeed <= 91) bench_test("lzrw3a", 0, bench_lzrw_compress, bench_lzrw_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 5, (size_t)work, 0);
		free(work);
	}
	
	if (cspeed <= 332) bench_test("pithy 2011-12-24 level 0", 0, bench_pithy_compress, bench_pithy_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);
	if (cspeed <= 384) bench_test("pithy 2011-12-24 level 3", 0, bench_pithy_compress, bench_pithy_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3, 0, 0);
	if (cspeed <= 302) bench_test("pithy 2011-12-24 level 6", 0, bench_pithy_compress, bench_pithy_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 6, 0, 0);
	if (cspeed <= 280) bench_test("pithy 2011-12-24 level 9", 0, bench_pithy_compress, bench_pithy_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 9, 0, 0);

	state_size = MAX(qlz_get_setting_3(1),MAX(qlz_get_setting_1(1), qlz_get_setting_2(1)));
	dstate_size = MAX(qlz_get_setting_3(2),MAX(qlz_get_setting_1(2), qlz_get_setting_2(2)));
	state_size = MAX(state_size, dstate_size);
	state = (qlz150_state_compress*) calloc(1, state_size);
//	memset(state,0,state_size);
	if (cspeed <= 359) bench_test("quicklz 1.5.0 -1", 0, bench_quicklz_compress, bench_quicklz_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, (size_t)state, 0);
	if (cspeed <= 172) bench_test("quicklz 1.5.0 -2", 0, bench_quicklz_compress, bench_quicklz_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 2, (size_t)state, 0);
	if (cspeed <= 42) bench_test("quicklz 1.5.0 -3", 0, bench_quicklz_compress, bench_quicklz_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3, (size_t)state, 0);
	if (cspeed <= 383) bench_test("quicklz 1.5.1 b7 -1", 0, bench_quicklz_compress, bench_quicklz_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 4, (size_t)state, 0);
	
	if (cspeed <= 316) bench_test("shrinker", 0, bench_shrinker_compress, bench_shrinker_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);

	if (cspeed <= 345) bench_test("snappy 1.1.3", 0, bench_snappy_compress, bench_snappy_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);



	if (cspeed <= 269) bench_test("tornado 0.6 -1", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);
	if (cspeed <= 221) bench_test("tornado 0.6 -2", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 2, 0, 0);
	if (cspeed <= 135) bench_test("tornado 0.6 -3", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3, 0, 0);
	if (cspeed <= 102) bench_test("tornado 0.6 -4", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 4, 0, 0);
	if (cspeed <= 47) bench_test("tornado 0.6 -5", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 5, 0, 0);
	if (cspeed <= 33) bench_test("tornado 0.6 -6", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 6, 0, 0);
	if (cspeed <= 15) bench_test("tornado 0.6 -7", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 7, 0, 0);
	if (cspeed <= 5) bench_test("tornado 0.6 -10", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 10, 0, 0);
	if (cspeed <= 6) bench_test("tornado 0.6 -13", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 13, 0, 0);
	if (cspeed <= 2) bench_test("tornado 0.6 -16", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 16, 0, 0);

	if (cspeed <= 274) bench_test("tornado 0.6 h16k b1m", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 21, 0, 0);
	if (cspeed <= 240) bench_test("tornado 0.6 h128k b2m", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 22, 0, 0);
	if (cspeed <= 249) bench_test("tornado 0.6 h128k b8m", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 23, 0, 0);
	if (cspeed <= 136) bench_test("tornado 0.6 h4m b8m", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 24, 0, 0);
	if (cspeed <= 218) bench_test("tornado h128k b8m bitio", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 25, 0, 0);
	if (cspeed <= 122) bench_test("tornado h4m b8m bitio", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 26, 0, 0);
	if (cspeed <= 133) bench_test("tornado h4m b32m bitio", 0, bench_tornado_compress, bench_tornado_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 27, 0, 0);


	if (cspeed <= 35) bench_test("ucl_nrv2b 1.03 -1", 0, bench_ucl_compress, bench_ucl_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 1, 0);
	if (cspeed <= 14) bench_test("ucl_nrv2b 1.03 -6", 0, bench_ucl_compress, bench_ucl_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 6, 0);
	if (cspeed <= 34) bench_test("ucl_nrv2d 1.03 -1", 0, bench_ucl_compress, bench_ucl_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 2, 1, 0);
	if (cspeed <= 13) bench_test("ucl_nrv2d 1.03 -6", 0, bench_ucl_compress, bench_ucl_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 2, 6, 0);
	if (cspeed <= 34) bench_test("ucl_nrv2e 1.03 -1", 0, bench_ucl_compress, bench_ucl_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3, 1, 0);
	if (cspeed <= 13) bench_test("ucl_nrv2e 1.03 -6", 0, bench_ucl_compress, bench_ucl_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3, 6, 0);

	work=(uint8_t*)calloc(1, wfLZ_GetWorkMemSize());
	if (work)
	{
		if (cspeed <= 209) bench_test("wflz 2015-09-16", 0, bench_wflz_compress, bench_wflz_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, (size_t)work, 0);
		free(work);
	}
	
	YappyFillTables();
	if (cspeed <= 99) bench_test("yappy 1", 0, bench_yappy_compress, bench_yappy_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);
	if (cspeed <= 76) bench_test("yappy 10", 0, bench_yappy_compress, bench_yappy_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 10, 0, 0);
	if (cspeed <= 53) bench_test("yappy 100", 0, bench_yappy_compress, bench_yappy_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 100, 0, 0);

	if (cspeed <= 66) bench_test("zlib 1.2.8 -1", 0, bench_zlib_compress, bench_zlib_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);
	if (cspeed <= 21) bench_test("zlib 1.2.8 -6", 0, bench_zlib_compress, bench_zlib_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 6, 0, 0);
	if (cspeed <= 6) bench_test("zlib 1.2.8 -9", 0, bench_zlib_compress, bench_zlib_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 9, 0, 0);

	if (cspeed <= 43) bench_test("zling 2015-09-15 level 0", 0, bench_zling_compress, bench_zling_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);
	if (cspeed <= 40) bench_test("zling 2015-09-15 level 1", 0, bench_zling_compress, bench_zling_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);
	if (cspeed <= 37) bench_test("zling 2015-09-15 level 2", 0, bench_zling_compress, bench_zling_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 2, 0, 0);
	if (cspeed <= 33) bench_test("zling 2015-09-15 level 3", 0, bench_zling_compress, bench_zling_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 3, 0, 0);
	if (cspeed <= 30) bench_test("zling 2015-09-15 level 4", 0, bench_zling_compress, bench_zling_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 4, 0, 0);

	if (cspeed <= 260) bench_test("zstd v0.3", 0, bench_zstd_compress, bench_zstd_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 0, 0, 0);
	if (cspeed <= 260) bench_test("zstd_HC v0.3 -1", 0, bench_zstd_compress, bench_zstd_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 1, 0, 0);
	if (cspeed <= 50) bench_test("zstd_HC v0.3 -5", 0, bench_zstd_compress, bench_zstd_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 5, 0, 0);
	if (cspeed <= 20) bench_test("zstd_HC v0.3 -9", 0, bench_zstd_compress, bench_zstd_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 9, 0, 0);
	if (cspeed <= 10) bench_test("zstd_HC v0.3 -13", 0, bench_zstd_compress, bench_zstd_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 13, 0, 0);
	if (cspeed <= 7) bench_test("zstd_HC v0.3 -17", 0, bench_zstd_compress, bench_zstd_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 17, 0, 0);
	if (cspeed <= 4) bench_test("zstd_HC v0.3 -21", 0, bench_zstd_compress, bench_zstd_decompress, chunk_size, iters, inbuf, insize, compbuf, comprsize, decomp, ticksPerSecond, 21, 0, 0);


done:
	//Print_Time("all", 0, &ticksPerSecond, NULL, &start_all, 0, 0, 1);

	free(inbuf);
	free(compbuf);
	free(decomp);

	if (chunk_size > 10 * (1<<20))
		printf("done... (%d iterations, chunk_size=%d MB, min_compr_speed=%d MB)\n", iters, chunk_size >> 20, cspeed);
	else
		printf("done... (%d iterations, chunk_size=%d KB, min_compr_speed=%d MB)\n", iters, chunk_size >> 10, cspeed);
}


int main( int argc, char** argv) 
{
	FILE *in;
	uint32_t iterations, chunk_size, cspeed;

	iterations = 1;
	chunk_size = 1 << 31;
	cspeed = 100;

#ifdef WINDOWS
//	SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
	SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#else
	setpriority(PRIO_PROCESS, 0, -20);
#endif

	printf(PROGNAME " " PROGVERSION " (%d-bit " PROGOS ")   Assembled by P.Skibinski\n", (uint32_t)(8 * sizeof(uint8_t*)));

	while ((argc>1)&&(argv[1][0]=='-')) {
		switch (argv[1][1]) {
	case 'i':
		iterations=atoi(argv[1]+2);
		break;
	case 'b':
		chunk_size = atoi(argv[1] + 2) << 10;
		break;
	case 's':
		cspeed = atoi(argv[1] + 2);
		break;
	case 't':
		turbobench_format = true;
		break;	
	default:
		fprintf(stderr, "unknown option: %s\n", argv[1]);
		exit(1);
		}
		argv++;
		argc--;
	}

	if (argc<2) {
		fprintf(stderr, "usage: " PROGNAME " [options] input\n");
		fprintf(stderr, " -iX: number of iterations (default = %d)\n", iterations);
		fprintf(stderr, " -bX: set block/chunk size to X KB (default = %d KB)\n", chunk_size>>10);
		fprintf(stderr, " -sX: use only compressors with compression speed over X MB (default = %d MB)\n", cspeed);
		exit(1);
	}

	if (!(in=fopen(argv[1], "rb"))) {
		perror(argv[1]);
		exit(1);
	}

	benchmark(in, iterations, chunk_size, cspeed);

	fclose(in);
}



